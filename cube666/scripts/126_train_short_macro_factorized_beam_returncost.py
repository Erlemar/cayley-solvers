"""Fine-tune the factorized short-macro value model on beam-state returns."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_factorized_value import (  # noqa: E402
    MacroFactorizedPrimitiveValueNet,
    load_macro_factorized_value_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--group-ids", type=Path)
    parser.add_argument("--rollouts", type=Path, action="append", required=True)
    parser.add_argument("--action-effects", type=Path)
    parser.add_argument("--action-costs", type=Path)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--rollout-batch-size", type=int, default=2048)
    parser.add_argument("--trace-batch-size", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--teacher-weight", type=float, default=4.0)
    parser.add_argument("--rollout-weight", type=float, default=2.0)
    parser.add_argument("--trace-weight", type=float, default=0.0)
    parser.add_argument("--ranking-weight", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=2.0)
    parser.add_argument("--ranking-temperature", type=float, default=4.0)
    parser.add_argument("--minimum-train-depth", type=int, default=1)
    parser.add_argument("--maximum-train-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=126666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.dataset_dir is not None:
        defaults = {
            "teacher": "teacher.npz",
            "group_ids": "source_state_ids.npy",
            "action_effects": "action_effects.npy",
            "action_costs": "action_costs.npy",
        }
        for name, filename in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, args.dataset_dir / filename)
    missing = [
        name for name in ("teacher", "group_ids") if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "provide --dataset-dir or each dataset path; missing " + ", ".join(missing)
        )
    return args


def combined_value(
    model: MacroFactorizedPrimitiveValueNet,
    outputs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    total, clusters = outputs
    return 0.5 * (
        total.float() * model.config.total_scale
        + clusters.float().sum(dim=1) * model.config.cluster_scale
    )


def build_oracle_rows(
    teacher_states: np.ndarray,
    teacher_actions: np.ndarray,
    action_effects: np.ndarray,
    action_costs: np.ndarray,
    rollout_states: np.ndarray,
    rollout_roots: np.ndarray,
    rollout_depths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align every beam row with its root's exact trace state at that layer."""
    oracle_states = np.empty_like(rollout_states)
    oracle_forward_costs = np.empty(len(rollout_states), dtype=np.float32)
    for root in np.unique(rollout_roots):
        root_rows = np.flatnonzero(rollout_roots == root)
        current = teacher_states[root].copy()
        forward_cost = 0.0
        maximum_depth = int(rollout_depths[root_rows].max())
        for depth in range(1, maximum_depth + 1):
            action = int(teacher_actions[root, depth - 1])
            current = np.take_along_axis(current, action_effects[action], axis=-1)
            forward_cost += float(action_costs[action])
            rows = root_rows[rollout_depths[root_rows] == depth]
            oracle_states[rows] = current
            oracle_forward_costs[rows] = forward_cost
    different = np.any(rollout_states != oracle_states, axis=(1, 2))
    return oracle_states, oracle_forward_costs, np.flatnonzero(different)


def reconstruct_trace_prefixes(
    states: np.ndarray,
    actions: np.ndarray,
    effects: np.ndarray,
    costs: np.ndarray,
    depths: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = states[rows].copy()
    remaining = np.zeros(len(rows), dtype=np.float32)
    for step in range(int(depths[rows].max())):
        target_rows = (depths[rows] > step) & (prefixes <= step)
        remaining[target_rows] += costs[actions[rows[target_rows], step]]
        active = prefixes > step
        active_actions = actions[rows[active], step].astype(np.int64, copy=False)
        current[active] = np.take_along_axis(
            current[active], effects[active_actions], axis=-1
        )
    return current, remaining


@torch.inference_mode()
def evaluate(
    model: MacroFactorizedPrimitiveValueNet,
    states: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int]:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(indices), 4096):
        batch = torch.from_numpy(states[indices[start : start + 4096]]).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions.append(combined_value(model, model(batch)).cpu().numpy())
    predicted = np.concatenate(predictions)
    error = predicted - values[indices]
    return {
        "bias": float(error.mean()),
        "correlation": float(np.corrcoef(predicted, values[indices])[0, 1]),
        "mae": float(np.abs(error).mean()),
        "samples": int(len(indices)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_macro_factorized_value_checkpoint(
        str(args.init_checkpoint), device="cuda"
    )
    with np.load(args.teacher, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        values = payload["search_value_targets"].astype(np.float32, copy=False)
        clusters = payload["search_cluster_targets"].astype(np.float32, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        teacher_actions = (
            payload["teacher_solution_actions"].astype(np.int64, copy=False)
            if args.ranking_weight or args.trace_batch_size
            else None
        )
    groups = np.load(args.group_ids, allow_pickle=False)
    rollout_state_parts: list[np.ndarray] = []
    rollout_target_parts: list[np.ndarray] = []
    rollout_forward_parts: list[np.ndarray] = []
    rollout_depth_parts: list[np.ndarray] = []
    rollout_root_parts: list[np.ndarray] = []
    for rollout_path in args.rollouts:
        with np.load(rollout_path, allow_pickle=False) as payload:
            rollout_state_parts.append(
                payload["states"].astype(np.uint8, copy=False)
            )
            rollout_target_parts.append(
                payload["return_cost_targets"].astype(np.float32, copy=False)
            )
            if args.ranking_weight:
                required = {"forward_costs", "depths", "root_indices"}
                missing = required.difference(payload.files)
                if missing:
                    raise ValueError(
                        f"{rollout_path} lacks ranking arrays: {sorted(missing)}"
                    )
                rollout_forward_parts.append(
                    payload["forward_costs"].astype(np.float32, copy=False)
                )
                rollout_depth_parts.append(
                    payload["depths"].astype(np.int16, copy=False)
                )
                rollout_root_parts.append(
                    payload["root_indices"].astype(np.int64, copy=False)
                )
    rollout_states = np.concatenate(rollout_state_parts)
    rollout_targets = np.concatenate(rollout_target_parts)
    if args.ranking_weight or args.trace_batch_size:
        if args.action_effects is None or args.action_costs is None:
            raise ValueError(
                "ranking/trace training requires --action-effects and --action-costs"
            )
        if teacher_actions is None:
            raise AssertionError("teacher actions missing")
        action_effects = np.load(args.action_effects, allow_pickle=False).astype(
            np.uint8, copy=False
        )
        action_costs = np.load(args.action_costs, allow_pickle=False).astype(
            np.float32, copy=False
        )
        if args.ranking_weight:
            rollout_forward_costs = np.concatenate(rollout_forward_parts)
            rollout_depths = np.concatenate(rollout_depth_parts)
            rollout_roots = np.concatenate(rollout_root_parts)
            oracle_states, oracle_forward_costs, ranking_indices = build_oracle_rows(
                states,
                teacher_actions,
                action_effects,
                action_costs,
                rollout_states,
                rollout_roots,
                rollout_depths,
            )
        else:
            rollout_forward_costs = None
            oracle_states = None
            oracle_forward_costs = None
            ranking_indices = np.arange(len(rollout_states))
    else:
        rollout_forward_costs = None
        oracle_states = None
        oracle_forward_costs = None
        ranking_indices = np.arange(len(rollout_states))
    depth_mask = (depths >= args.minimum_train_depth) & (
        (args.maximum_train_depth == 0) | (depths <= args.maximum_train_depth)
    )
    train_indices = np.flatnonzero((np.mod(groups, 10) != 0) & depth_mask)
    heldout_indices = np.flatnonzero((np.mod(groups, 10) == 0) & depth_mask)
    eval_indices = rng.choice(
        heldout_indices, size=min(8192, len(heldout_indices)), replace=False
    )
    before = evaluate(model, states, values, eval_indices)
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-3)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    started = time.perf_counter()
    last: dict[str, float] = {}
    model.train()
    for step in range(1, args.steps + 1):
        teacher_rows = rng.choice(
            train_indices, size=args.teacher_batch_size, replace=True
        )
        rollout_rows = rng.choice(
            ranking_indices, size=args.rollout_batch_size, replace=True
        )
        teacher_states = torch.from_numpy(states[teacher_rows]).cuda()
        teacher_values = torch.from_numpy(values[teacher_rows]).cuda()
        teacher_clusters = torch.from_numpy(clusters[teacher_rows]).cuda()
        offpath_states = torch.from_numpy(rollout_states[rollout_rows]).cuda()
        offpath_targets = torch.from_numpy(rollout_targets[rollout_rows]).cuda()
        if args.trace_batch_size:
            if teacher_actions is None:
                raise AssertionError("teacher actions missing")
            trace_rows = rng.choice(
                train_indices, size=args.trace_batch_size, replace=True
            )
            trace_prefixes = (
                rng.random(args.trace_batch_size) * depths[trace_rows]
            ).astype(np.int64)
            trace_states_np, trace_targets_np = reconstruct_trace_prefixes(
                states,
                teacher_actions,
                action_effects,
                action_costs,
                depths,
                trace_rows,
                trace_prefixes,
            )
            trace_states = torch.from_numpy(trace_states_np).cuda()
            trace_targets = torch.from_numpy(trace_targets_np).cuda()
        if args.ranking_weight:
            if (
                oracle_states is None
                or oracle_forward_costs is None
                or rollout_forward_costs is None
            ):
                raise AssertionError("ranking arrays missing")
            rank_oracle_states = torch.from_numpy(oracle_states[rollout_rows]).cuda()
            rank_oracle_forward = torch.from_numpy(
                oracle_forward_costs[rollout_rows]
            ).cuda()
            rank_offpath_forward = torch.from_numpy(
                rollout_forward_costs[rollout_rows]
            ).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            teacher_outputs = train_model(teacher_states)
            offpath_outputs = train_model(offpath_states)
            teacher_loss = F.smooth_l1_loss(
                teacher_outputs[0].float(),
                teacher_values / model.config.total_scale,
                beta=0.05,
            ) + F.smooth_l1_loss(
                teacher_outputs[1].float(),
                teacher_clusters / model.config.cluster_scale,
                beta=0.05,
            )
            rollout_prediction = combined_value(model, offpath_outputs)
            rollout_loss = F.smooth_l1_loss(
                rollout_prediction, offpath_targets, beta=2.0
            ) / 72.0
            nonnegative = F.relu(-rollout_prediction).mean() / 72.0
            if args.trace_batch_size:
                trace_outputs = train_model(trace_states)
                trace_prediction = combined_value(model, trace_outputs)
                trace_loss = F.smooth_l1_loss(
                    trace_prediction, trace_targets, beta=2.0
                ) / 72.0
            else:
                trace_loss = torch.zeros((), device="cuda")
            if args.ranking_weight:
                oracle_outputs = train_model(rank_oracle_states)
                oracle_prediction = combined_value(model, oracle_outputs)
                oracle_score = rank_oracle_forward + oracle_prediction
                offpath_score = rank_offpath_forward + rollout_prediction
                ranking_loss = (
                    F.softplus(
                        (
                            oracle_score
                            - offpath_score
                            + args.ranking_margin
                        )
                        / args.ranking_temperature
                    ).mean()
                    * args.ranking_temperature
                    / 72.0
                )
            else:
                ranking_loss = torch.zeros((), device="cuda")
            loss = (
                args.teacher_weight * teacher_loss
                + args.rollout_weight * rollout_loss
                + args.trace_weight * trace_loss
                + args.ranking_weight * ranking_loss
                + 0.05 * nonnegative
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "loss": float(loss.detach()),
            "ranking_loss": float(ranking_loss.detach()),
            "rollout_loss": float(rollout_loss.detach()),
            "teacher_loss": float(teacher_loss.detach()),
            "trace_loss": float(trace_loss.detach()),
        }
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "step": step,
                        **last,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    after = evaluate(model, states, values, eval_indices)
    report = {
        "after": after,
        "before": before,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "last_train_losses": last,
        "model_config": model.config.to_dict(),
        "rollout_samples": len(rollout_states),
        "ranking_samples": int(len(ranking_indices)),
        "ranking_weight": args.ranking_weight,
        "ranking_margin": args.ranking_margin,
        "ranking_temperature": args.ranking_temperature,
        "rollout_weight": args.rollout_weight,
        "teacher_weight": args.teacher_weight,
        "teacher_batch_size": args.teacher_batch_size,
        "rollout_batch_size": args.rollout_batch_size,
        "trace_batch_size": args.trace_batch_size,
        "trace_weight": args.trace_weight,
        "learning_rate": args.learning_rate,
        "source_checkpoint": str(args.init_checkpoint),
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_macro_factorized_primitive_value_v1",
            "factorized_value_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
            "source_payload_kind": checkpoint.get("kind"),
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
