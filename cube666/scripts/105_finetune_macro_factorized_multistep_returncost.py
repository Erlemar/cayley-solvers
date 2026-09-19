"""Fine-tune the calibrated factorized value on multi-step return routes."""

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
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--rollout-batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--teacher-weight", type=float, default=4.0)
    parser.add_argument("--cluster-weight", type=float, default=1.0)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=105666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    states: np.ndarray,
    values: np.ndarray,
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
    errors = prediction - values[indices]
    return {
        "bias": float(errors.mean()),
        "cluster_mae": float(np.abs(cluster_prediction - clusters[indices]).mean()),
        "correlation": float(np.corrcoef(prediction, values[indices])[0, 1]),
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
        teacher_states = teacher["states"].astype(np.uint8, copy=False)
        teacher_values = teacher["search_value_targets"].astype(np.float32, copy=False)
        teacher_clusters = teacher["search_cluster_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    with np.load(args.rollouts, allow_pickle=False) as rollout:
        rollout_states = rollout["states"].astype(np.uint8, copy=False)
        rollout_targets = rollout["return_cost_targets"].astype(np.float32, copy=False)
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
        rollout_selected = rng.integers(
            len(rollout_states), size=args.rollout_batch_size
        )
        exact_states = torch.from_numpy(teacher_states[teacher_selected]).to(device)
        exact_values = torch.from_numpy(teacher_values[teacher_selected]).to(device)
        exact_clusters = torch.from_numpy(teacher_clusters[teacher_selected]).to(device)
        offpath_states = torch.from_numpy(rollout_states[rollout_selected]).to(device)
        offpath_targets = torch.from_numpy(rollout_targets[rollout_selected]).to(device)
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            exact_scaled, cluster_scaled = train_model(exact_states)
            offpath_scaled, _ = train_model(offpath_states)
            teacher_loss = F.smooth_l1_loss(
                exact_scaled.float(), exact_values / config.total_scale, beta=0.05
            )
            cluster_loss = F.smooth_l1_loss(
                cluster_scaled.float(), exact_clusters / config.cluster_scale, beta=0.05
            )
            rollout_loss = F.smooth_l1_loss(
                offpath_scaled.float(), offpath_targets / config.total_scale, beta=0.05
            )
            nonnegative = F.relu(
                -torch.cat((exact_scaled.float(), offpath_scaled.float()))
            ).mean()
            loss = (
                args.teacher_weight * teacher_loss
                + args.cluster_weight * cluster_loss
                + args.rollout_weight * rollout_loss
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
                "rollout_loss": float(rollout_loss.detach()),
                "step": step,
                "teacher_loss": float(teacher_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    eval_count = min(8192, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    heldout = evaluate(
        model,
        teacher_states,
        teacher_values,
        teacher_clusters,
        eval_indices,
        args.eval_batch_size,
        device,
    )
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "heldout": heldout,
        "history": history,
        "rollout_samples": len(rollout_states),
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_macro_factorized_multistep_returncost_v1",
            "model_state_dict": model.state_dict(),
            "multistep_returncost_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
