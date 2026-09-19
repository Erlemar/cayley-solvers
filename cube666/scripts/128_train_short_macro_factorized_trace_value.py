"""Fine-tune factorized value on every distribution of exact trace prefixes."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--trace-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--teacher-weight", type=float, default=4.0)
    parser.add_argument("--trace-weight", type=float, default=8.0)
    parser.add_argument("--minimum-train-depth", type=int, default=1)
    parser.add_argument("--maximum-train-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=128666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def combined_value(
    model: MacroFactorizedPrimitiveValueNet,
    outputs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    total, clusters = outputs
    return 0.5 * (
        total.float() * model.config.total_scale
        + clusters.float().sum(dim=1) * model.config.cluster_scale
    )


def reconstruct_prefix_batch(
    states: np.ndarray,
    actions: np.ndarray,
    effects: np.ndarray,
    costs: np.ndarray,
    depths: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = states[rows].copy()
    maximum_depth = int(depths[rows].max())
    remaining = np.zeros(len(rows), dtype=np.float32)
    for step in range(maximum_depth):
        valid = depths[rows] > step
        step_actions = actions[rows[valid], step].astype(np.int64, copy=False)
        remaining[valid & (prefixes <= step)] += costs[
            actions[rows[valid & (prefixes <= step)], step]
        ]
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
    with np.load(args.dataset_dir / "teacher.npz", allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        values = payload["search_value_targets"].astype(np.float32, copy=False)
        clusters = payload["search_cluster_targets"].astype(np.float32, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        actions = payload["teacher_solution_actions"].astype(np.int32, copy=False)
    groups = np.load(args.dataset_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.dataset_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.dataset_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    depth_mask = (depths >= args.minimum_train_depth) & (
        (args.maximum_train_depth == 0) | (depths <= args.maximum_train_depth)
    )
    train_indices = np.flatnonzero((groups % 10 != 0) & depth_mask)
    heldout_indices = np.flatnonzero((groups % 10 == 0) & depth_mask)
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
        trace_rows = rng.choice(train_indices, size=args.trace_batch_size, replace=True)
        prefixes = (
            rng.random(args.trace_batch_size) * depths[trace_rows]
        ).astype(np.int64)
        trace_states_np, trace_targets_np = reconstruct_prefix_batch(
            states, actions, effects, costs, depths, trace_rows, prefixes
        )
        teacher_states = torch.from_numpy(states[teacher_rows]).cuda()
        teacher_values = torch.from_numpy(values[teacher_rows]).cuda()
        teacher_clusters = torch.from_numpy(clusters[teacher_rows]).cuda()
        trace_states = torch.from_numpy(trace_states_np).cuda()
        trace_targets = torch.from_numpy(trace_targets_np).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            teacher_outputs = train_model(teacher_states)
            trace_outputs = train_model(trace_states)
            teacher_loss = F.smooth_l1_loss(
                teacher_outputs[0].float(),
                teacher_values / model.config.total_scale,
                beta=0.05,
            ) + F.smooth_l1_loss(
                teacher_outputs[1].float(),
                teacher_clusters / model.config.cluster_scale,
                beta=0.05,
            )
            trace_prediction = combined_value(model, trace_outputs)
            trace_loss = F.smooth_l1_loss(
                trace_prediction, trace_targets, beta=2.0
            ) / 72.0
            nonnegative = F.relu(-trace_prediction).mean() / 72.0
            loss = (
                args.teacher_weight * teacher_loss
                + args.trace_weight * trace_loss
                + 0.05 * nonnegative
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "loss": float(loss.detach()),
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
        "source_checkpoint": str(args.init_checkpoint),
        "source_payload_kind": checkpoint.get("kind"),
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_macro_factorized_primitive_value_v1",
            "factorized_value_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
