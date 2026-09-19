"""Fine-tune the short-macro model on trace-consistent certified returns.

Independent random-walk labels teach values on the sampled states, but beam search
mostly visits their children.  For a trace state with a verified remaining route of
cost ``V`` and an arbitrary proposed action ``a``, the resulting child has a
certified return route of cost ``V + cost(inverse(a))``: undo ``a`` and follow the
trace.  This trainer preserves ordinary calibration while explicitly supervising
the exact trace child and the model's own high-policy off-trace children.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch.nn import functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--group-ids", type=Path)
    parser.add_argument("--action-effects", type=Path)
    parser.add_argument("--action-costs", type=Path)
    parser.add_argument("--inverse-actions", type=Path)
    parser.add_argument(
        "--train-module",
        type=Path,
        default=Path("cube666/scripts/33_remote_train_macro_effect.py"),
    )
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--trace-batch-size", type=int, default=16)
    parser.add_argument("--hard-actions", type=int, default=128)
    parser.add_argument("--action-chunk-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--minimum-train-depth", type=int, default=1)
    parser.add_argument(
        "--maximum-train-depth",
        type=int,
        default=0,
        help="maximum trace depth to sample; zero keeps every available depth",
    )
    parser.add_argument("--teacher-weight", type=float, default=2.0)
    parser.add_argument("--trace-weight", type=float, default=2.0)
    parser.add_argument("--return-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=0.5)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--retention-margin", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=121666)
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
            "inverse_actions": "inverse_actions.npy",
        }
        for name, filename in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, args.dataset_dir / filename)
    missing = [
        name
        for name in (
            "teacher",
            "group_ids",
            "action_effects",
            "action_costs",
            "inverse_actions",
        )
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "provide --dataset-dir or each dataset path; missing " + ", ".join(missing)
        )
    return args


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("short_macro_trace_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def combined_value(outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    _, total, clusters = outputs
    return 0.5 * (total.float() * 72.0 + clusters.float().sum(dim=1) * 12.0)


@torch.inference_mode()
def policy_topk(
    module: ModuleType,
    logits: torch.Tensor,
    effects: torch.Tensor,
    *,
    topk: int,
    chunk_size: int,
    teacher: torch.Tensor,
    forbidden: torch.Tensor,
) -> torch.Tensor:
    best_scores = torch.empty((len(logits), 0), device=logits.device)
    best_actions = torch.empty(
        (len(logits), 0), dtype=torch.long, device=logits.device
    )
    for start in range(0, len(effects), chunk_size):
        stop = min(start + chunk_size, len(effects))
        candidates = effects[start:stop][None].expand(len(logits), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        action_ids = torch.arange(start, stop, device=logits.device)[None].expand(
            len(logits), -1
        )
        invalid = action_ids.eq(teacher[:, None]) | action_ids.eq(forbidden[:, None])
        scores = scores.masked_fill(invalid, -torch.inf)
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, action_ids), dim=1)
        keep = min(topk, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    return best_actions


def reconstruct_trace_batch(
    states: np.ndarray,
    solution_actions: np.ndarray,
    depths: np.ndarray,
    effects: np.ndarray,
    costs: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = states[rows].copy()
    teacher = np.empty(len(rows), dtype=np.int64)
    previous = np.full(len(rows), -1, dtype=np.int64)
    parent_targets = np.empty(len(rows), dtype=np.float32)
    positive_targets = np.empty(len(rows), dtype=np.float32)
    for position, (row, prefix) in enumerate(zip(rows, prefixes, strict=True)):
        trace = solution_actions[row, : int(depths[row])].astype(np.int64, copy=False)
        for action in trace[:prefix]:
            current[position] = np.take_along_axis(
                current[position], effects[int(action)], axis=-1
            )
        teacher[position] = int(trace[prefix])
        if prefix:
            previous[position] = int(trace[prefix - 1])
        remaining = costs[trace[prefix:]].sum(dtype=np.int64)
        parent_targets[position] = float(remaining)
        positive_targets[position] = float(remaining - costs[teacher[position]])
    return current, teacher, previous, parent_targets, positive_targets


@torch.inference_mode()
def calibration_metrics(
    model: torch.nn.Module,
    states: np.ndarray,
    values: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, float | int]:
    predictions: list[np.ndarray] = []
    for start in range(0, len(indices), 4096):
        batch = torch.from_numpy(states[indices[start : start + 4096]]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = combined_value(model(batch))
        predictions.append(prediction.cpu().numpy())
    predicted = np.concatenate(predictions)
    target = values[indices]
    error = predicted - target
    return {
        "bias": float(error.mean()),
        "correlation": float(np.corrcoef(predicted, target)[0, 1]),
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
    device = torch.device("cuda")
    module = load_module(args.train_module)
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    config = module.Config(action_count=checkpoint["model_config"]["action_count"])
    model = module.MacroEffectPolicyValueNet(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    with np.load(args.teacher, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        labels = payload["teacher_actions"].astype(np.int32, copy=False)
        counts = payload["teacher_action_counts"].astype(np.int16, copy=False)
        values = payload["search_value_targets"].astype(np.float32, copy=False)
        clusters = payload["search_cluster_targets"].astype(np.float32, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        solution_actions = payload["teacher_solution_actions"].astype(
            np.int32, copy=False
        )
    groups = np.load(args.group_ids, allow_pickle=False)
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs_np = np.load(args.action_costs, allow_pickle=False).astype(
        np.int64, copy=False
    )
    inverse_np = np.load(args.inverse_actions, allow_pickle=False).astype(
        np.int64, copy=False
    )
    if np.any(counts != 1) or len(effects_np) != config.action_count:
        raise ValueError("teacher and checkpoint action spaces disagree")
    if np.any(depths <= 0) or np.any(solution_actions[np.arange(len(states)), depths - 1] < 0):
        raise ValueError("teacher solution traces are incomplete")
    depth_mask = depths >= args.minimum_train_depth
    if args.maximum_train_depth:
        depth_mask &= depths <= args.maximum_train_depth
    train_indices = np.flatnonzero((np.mod(groups, 10) != 0) & depth_mask)
    heldout_indices = np.flatnonzero((np.mod(groups, 10) == 0) & depth_mask)
    if not len(train_indices) or not len(heldout_indices):
        raise ValueError("depth filter left an empty training or held-out split")
    eval_indices = rng.choice(
        heldout_indices, size=min(8192, len(heldout_indices)), replace=False
    )
    before = calibration_metrics(model, states, values, eval_indices, device)

    effects = torch.from_numpy(effects_np).to(device)
    costs = torch.from_numpy(costs_np).float().to(device)
    inverse = torch.from_numpy(inverse_np).long().to(device)
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
    for step in range(1, args.steps + 1):
        teacher_rows = rng.choice(
            train_indices, size=args.teacher_batch_size, replace=True
        )
        trace_rows = rng.choice(train_indices, size=args.trace_batch_size, replace=True)
        prefixes = np.asarray(
            [rng.integers(0, int(depths[row])) for row in trace_rows],
            dtype=np.int64,
        )
        (
            trace_states_np,
            teacher_actions_np,
            previous_actions_np,
            parent_targets_np,
            positive_targets_np,
        ) = reconstruct_trace_batch(
            states,
            solution_actions,
            depths,
            effects_np,
            costs_np,
            trace_rows,
            prefixes,
        )
        trace_states = torch.from_numpy(trace_states_np).to(device)
        teacher_actions = torch.from_numpy(teacher_actions_np).to(device)
        previous_actions = torch.from_numpy(previous_actions_np).to(device)
        forbidden = torch.where(
            previous_actions >= 0,
            inverse[previous_actions.clamp_min(0)],
            torch.full_like(previous_actions, -1),
        )
        model.eval()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            trace_logits, _, _ = model(trace_states)
        hard_actions = policy_topk(
            module,
            trace_logits,
            effects,
            topk=args.hard_actions,
            chunk_size=args.action_chunk_size,
            teacher=teacher_actions,
            forbidden=forbidden,
        )
        model.train()

        teacher_states = torch.from_numpy(states[teacher_rows]).to(device)
        teacher_values = torch.from_numpy(values[teacher_rows]).to(device)
        teacher_clusters = torch.from_numpy(clusters[teacher_rows]).to(device)
        parent_targets = torch.from_numpy(parent_targets_np).to(device)
        positive_targets = torch.from_numpy(positive_targets_np).to(device)
        positive_children = trace_states.gather(
            -1, effects[teacher_actions].long()
        )
        hard_children = trace_states[:, None].expand(
            -1, args.hard_actions, -1, -1
        ).gather(-1, effects[hard_actions].long())
        return_targets = parent_targets[:, None] + costs[inverse[hard_actions]]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            teacher_outputs = train_model(teacher_states)
            parent_outputs = train_model(trace_states)
            positive_outputs = train_model(positive_children)
            hard_outputs = train_model(hard_children.flatten(0, 1))
            teacher_total = teacher_outputs[1].float()
            teacher_local = teacher_outputs[2].float()
            teacher_loss = F.smooth_l1_loss(
                teacher_total, teacher_values / 72.0, beta=0.05
            ) + F.smooth_l1_loss(
                teacher_local, teacher_clusters / 12.0, beta=0.05
            )
            parent_value = combined_value(parent_outputs)
            positive_value = combined_value(positive_outputs)
            hard_value = combined_value(hard_outputs).reshape(
                args.trace_batch_size, args.hard_actions
            )
            trace_loss = F.smooth_l1_loss(
                parent_value, parent_targets, beta=2.0
            ) + F.smooth_l1_loss(
                positive_value, positive_targets, beta=2.0
            )
            return_loss = F.smooth_l1_loss(
                hard_value, return_targets, beta=2.0
            )
            positive_policy = module.score_candidates(
                parent_outputs[0], effects[teacher_actions][:, None]
            )[:, 0]
            negative_policy = module.score_candidates(
                parent_outputs[0], effects[hard_actions]
            )
            policy_loss = (
                torch.logsumexp(
                    torch.cat((positive_policy[:, None], negative_policy), dim=1),
                    dim=1,
                )
                - positive_policy
            ).mean()
            positive_f = costs[teacher_actions] + positive_value
            hard_f = costs[hard_actions] + hard_value
            retention_loss = F.relu(
                args.retention_margin + positive_f[:, None] - hard_f
            ).mean() / 72.0
            nonnegative = F.relu(
                -torch.cat(
                    (parent_value, positive_value, hard_value.flatten()), dim=0
                )
            ).mean() / 72.0
            loss = (
                args.teacher_weight * teacher_loss
                + args.trace_weight * trace_loss / 72.0
                + args.return_weight * return_loss / 72.0
                + args.policy_weight * policy_loss
                + args.retention_weight * retention_loss
                + 0.05 * nonnegative
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "retention_loss": float(retention_loss.detach()),
            "return_loss": float(return_loss.detach()),
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

    after = calibration_metrics(model, states, values, eval_indices, device)
    report = {
        "action_count": len(effects_np),
        "action_digest": checkpoint["action_digest"],
        "after": after,
        "before": before,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "hard_actions": args.hard_actions,
        "last_train_losses": last,
        "maximum_train_depth": args.maximum_train_depth,
        "minimum_train_depth": args.minimum_train_depth,
        "model_config": config.as_dict(),
        "steps": args.steps,
        "trace_batch_size": args.trace_batch_size,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "action_digest": checkpoint["action_digest"],
            "model_config": config.as_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
