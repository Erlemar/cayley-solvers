"""Fine-tune a calibrated macro value model on certified return-path costs.

For a teacher state with remaining cost V and an arbitrary proposed macro of cost c,
the child always has a verified solution of cost V + c: undo the macro, then follow
the teacher suffix.  These globally comparable targets are deliberately different
from a parent-local ranking loss.
"""

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

from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--hard-batch-size", type=int, default=32)
    parser.add_argument("--hard-actions", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--teacher-weight", type=float, default=4.0)
    parser.add_argument("--cluster-weight", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--return-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=96666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def choose_labels(labels: np.ndarray, counts: np.ndarray, costs: np.ndarray) -> np.ndarray:
    chosen = np.full(len(labels), -1, dtype=np.int32)
    for row in range(len(labels)):
        valid = [
            int(action)
            for action in labels[row, : int(counts[row])]
            if action >= 0 and costs[int(action)] <= 14
        ]
        if valid:
            chosen[row] = valid[0]
    return chosen


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    states: np.ndarray,
    targets: np.ndarray,
    clusters: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    cluster_predictions: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = torch.from_numpy(states[selected]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, local = model(batch)
        predictions.append((total.float() * model.config.total_scale).cpu().numpy())
        cluster_predictions.append(
            (local.float() * model.config.cluster_scale).cpu().numpy()
        )
    prediction = np.concatenate(predictions)
    cluster_prediction = np.concatenate(cluster_predictions)
    errors = prediction - targets[indices]
    return {
        "bias": float(errors.mean()),
        "cluster_mae": float(
            np.abs(cluster_prediction - clusters[indices]).mean()
        ),
        "correlation": float(np.corrcoef(prediction, targets[indices])[0, 1]),
        "mae": float(np.abs(errors).mean()),
        "prediction_maximum": float(prediction.max()),
        "prediction_minimum": float(prediction.min()),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_macro_factorized_value_checkpoint(
        args.init_checkpoint, device
    )
    config = model.config
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        all_states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        all_values = teacher["search_value_targets"].astype(np.float32, copy=False)
        all_clusters = teacher["search_cluster_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(np.int32)
    effects = torch.from_numpy(
        np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(np.uint8)
    ).to(device)
    chosen = choose_labels(labels, counts, costs)
    with np.load(args.hardneg, allow_pickle=False) as hard:
        hard_rows = hard["row_indices"].astype(np.int64, copy=False)
        proposals = hard["proposals"].astype(np.int32, copy=False)
    if np.any(chosen[hard_rows] < 0) or np.any(proposals < 0):
        raise ValueError("hard-negative rows/proposals are incomplete")
    train_indices = np.flatnonzero(np.mod(groups, args.folds) != args.fold)
    heldout_indices = np.flatnonzero(np.mod(groups, args.folds) == args.fold)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        teacher_selected = train_indices[
            rng.integers(len(train_indices), size=args.teacher_batch_size)
        ]
        hard_positions = rng.integers(len(hard_rows), size=args.hard_batch_size)
        selected_rows = hard_rows[hard_positions]
        columns = rng.integers(
            proposals.shape[1], size=(args.hard_batch_size, args.hard_actions)
        )
        selected_actions_np = proposals[hard_positions[:, None], columns]
        positive_np = chosen[selected_rows]
        duplicate = selected_actions_np == positive_np[:, None]
        if np.any(duplicate):
            replacement = proposals[
                hard_positions[:, None], (columns + 97) % proposals.shape[1]
            ]
            selected_actions_np = np.where(duplicate, replacement, selected_actions_np)

        teacher_states = torch.from_numpy(all_states[teacher_selected]).to(device)
        teacher_targets = torch.from_numpy(all_values[teacher_selected]).to(device)
        teacher_clusters = torch.from_numpy(all_clusters[teacher_selected]).to(device)
        parent_states = torch.from_numpy(all_states[selected_rows]).to(device)
        positive_actions = torch.from_numpy(positive_np).to(device)
        hard_actions = torch.from_numpy(selected_actions_np).to(device)
        positive_children = parent_states.gather(-1, effects[positive_actions].long())
        hard_children = parent_states[:, None].expand(
            -1, args.hard_actions, -1, -1
        ).gather(-1, effects[hard_actions].long())
        parent_targets = torch.from_numpy(all_values[selected_rows]).to(device)
        positive_costs = torch.from_numpy(costs[positive_np]).to(device)
        hard_costs = torch.from_numpy(costs[selected_actions_np]).to(device)
        positive_targets = (parent_targets - positive_costs).clamp_min(0)
        return_targets = parent_targets[:, None] + hard_costs

        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(
                args.steps - args.warmup_steps, 1
            )
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            teacher_scaled, cluster_scaled = train_model(teacher_states)
            positive_scaled, _ = train_model(positive_children)
            return_scaled, _ = train_model(hard_children.flatten(0, 1))
            return_scaled = return_scaled.reshape(args.hard_batch_size, args.hard_actions)
            teacher_loss = F.smooth_l1_loss(
                teacher_scaled.float(), teacher_targets / config.total_scale, beta=0.05
            )
            cluster_loss = F.smooth_l1_loss(
                cluster_scaled.float(), teacher_clusters / config.cluster_scale, beta=0.05
            )
            positive_loss = F.smooth_l1_loss(
                positive_scaled.float(), positive_targets / config.total_scale, beta=0.05
            )
            return_loss = F.smooth_l1_loss(
                return_scaled.float(), return_targets / config.total_scale, beta=0.05
            )
            nonnegative = F.relu(
                -torch.cat(
                    (
                        teacher_scaled.float(),
                        positive_scaled.float(),
                        return_scaled.float().flatten(),
                    )
                )
            ).mean()
            loss = (
                args.teacher_weight * teacher_loss
                + args.cluster_weight * cluster_loss
                + args.positive_weight * positive_loss
                + args.return_weight * return_loss
                + 0.05 * nonnegative
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "cluster_loss": float(cluster_loss.detach()),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "loss": float(loss.detach()),
                "positive_loss": float(positive_loss.detach()),
                "return_loss": float(return_loss.detach()),
                "step": step,
                "teacher_loss": float(teacher_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    eval_count = min(8192, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    heldout = evaluate(
        model,
        all_states,
        all_values,
        all_clusters,
        eval_indices,
        args.eval_batch_size,
        device,
    )
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "hard_actions": args.hard_actions,
        "hard_rows": len(hard_rows),
        "heldout": heldout,
        "history": history,
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_macro_factorized_primitive_value_returncost_v1",
            "model_state_dict": model.state_dict(),
            "returncost_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
