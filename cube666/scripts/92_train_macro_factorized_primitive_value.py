"""Train the shared-cluster value model on audited primitive cost-to-go labels."""

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
    MacroFactorizedValueConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--cluster-weight", type=float, default=1.0)
    parser.add_argument("--minimum-train-depth", type=int, default=1)
    parser.add_argument("--maximum-train-depth", type=int, default=0)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=92666)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    model: MacroFactorizedPrimitiveValueNet,
    states: np.ndarray,
    targets: np.ndarray,
    cluster_targets: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    cluster_predictions: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, clusters = model(batch)
        predictions.append((total.float() * model.config.total_scale).cpu().numpy())
        cluster_predictions.append(
            (clusters.float() * model.config.cluster_scale).cpu().numpy()
        )
    prediction = np.concatenate(predictions)
    cluster_prediction = np.concatenate(cluster_predictions)
    errors = prediction - targets
    return {
        "bias": float(errors.mean()),
        "cluster_mae": float(np.abs(cluster_prediction - cluster_targets).mean()),
        "correlation": float(np.corrcoef(prediction, targets)[0, 1]),
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
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        targets = teacher["search_value_targets"].astype(np.float32, copy=False)
        cluster_targets = teacher["search_cluster_targets"].astype(np.float32, copy=False)
        depths = (
            teacher["walk_depths"].astype(np.int16, copy=False)
            if "walk_depths" in teacher
            else np.ones(len(states), dtype=np.int16)
        )
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    depth_mask = depths >= args.minimum_train_depth
    if args.maximum_train_depth:
        depth_mask &= depths <= args.maximum_train_depth
    train_indices = np.flatnonzero(
        (np.mod(groups, args.folds) != args.fold) & depth_mask
    )
    heldout_indices = np.flatnonzero(
        (np.mod(groups, args.folds) == args.fold) & depth_mask
    )
    config = MacroFactorizedValueConfig()
    model = MacroFactorizedPrimitiveValueNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    training_values = targets[train_indices]
    high_weights = np.sqrt(training_values / max(float(training_values.mean()), 1e-6))
    high_cdf = np.cumsum(high_weights, dtype=np.float64)
    high_cdf /= high_cdf[-1]
    high_count = int(round(args.batch_size * args.high_value_fraction))
    uniform_count = args.batch_size - high_count
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        parts: list[np.ndarray] = []
        if uniform_count:
            parts.append(train_indices[rng.integers(len(train_indices), size=uniform_count)])
        if high_count:
            positions = np.searchsorted(high_cdf, rng.random(high_count), side="right")
            parts.append(train_indices[positions])
        selected = np.concatenate(parts)
        rng.shuffle(selected)
        batch_states = torch.from_numpy(states[selected]).to(device)
        total_targets = torch.from_numpy(targets[selected]).to(device) / config.total_scale
        local_targets = torch.from_numpy(cluster_targets[selected]).to(device) / config.cluster_scale
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, clusters = train_model(batch_states)
            total_loss = F.smooth_l1_loss(
                total.float(), total_targets.float(), beta=0.05
            )
            cluster_loss = F.smooth_l1_loss(
                clusters.float(), local_targets.float(), beta=0.05
            )
            nonnegative = F.relu(-total.float()).mean()
            loss = total_loss + args.cluster_weight * cluster_loss + 0.05 * nonnegative
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "cluster_loss": float(cluster_loss.detach()),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "loss": float(loss.detach()),
                "step": step,
                "total_loss": float(total_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    eval_count = min(8192, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    heldout = evaluate(
        model,
        states[eval_indices],
        targets[eval_indices],
        cluster_targets[eval_indices],
        args.eval_batch_size,
        device,
    )
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "heldout": heldout,
        "heldout_samples": int(len(heldout_indices)),
        "history": history,
        "model_config": config.to_dict(),
        "maximum_train_depth": args.maximum_train_depth,
        "minimum_train_depth": args.minimum_train_depth,
        "steps": args.steps,
        "train_samples": int(len(train_indices)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_macro_factorized_primitive_value_v1",
            "factorized_value_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
