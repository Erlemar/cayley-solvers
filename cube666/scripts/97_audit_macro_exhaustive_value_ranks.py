"""Audit teacher ranks under exhaustive cost-plus-value scoring of all macros."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--pids", default="")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=97666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    states: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, clusters = model(batch)
        moves = 0.5 * (
            total.float() * model.config.total_scale
            + clusters.float().sum(dim=1) * model.config.cluster_scale
        )
        values.append(moves.clamp_min(0).cpu().numpy())
    return np.concatenate(values)


def summarize(ranks: np.ndarray) -> dict[str, float]:
    return {
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "p90_rank": float(np.quantile(ranks, 0.9)),
        "top1": float(np.mean(ranks <= 1)),
        "top16": float(np.mean(ranks <= 16)),
        "top64": float(np.mean(ranks <= 64)),
        "top256": float(np.mean(ranks <= 256)),
        "top1024": float(np.mean(ranks <= 1024)),
        "top4096": float(np.mean(ranks <= 4096)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model, _ = load_macro_factorized_value_checkpoint(args.checkpoint, device)
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    requested = {int(token) for token in args.pids.split(",") if token.strip()}
    if requested:
        eligible = np.flatnonzero(np.isin(groups, list(requested)))
    else:
        eligible = np.flatnonzero(np.mod(groups, args.folds) == args.fold)
    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.samples:
        # Preserve the hard tail while retaining coverage of the full value range.
        order = eligible[np.argsort(values[eligible])]
        positions = np.linspace(0, len(order) - 1, args.samples).round().astype(np.int64)
        selected = order[positions]
        rng.shuffle(selected)
    else:
        selected = eligible
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for number, row in enumerate(selected, 1):
        children = np.take_along_axis(
            np.broadcast_to(states[row], (len(effects),) + states[row].shape),
            effects,
            axis=-1,
        )
        child_values = predict(model, children, args.inference_batch_size, device)
        scores = costs + child_values
        teacher_ids = labels[row, : int(counts[row])]
        teacher_scores = scores[teacher_ids]
        teacher_slot = int(np.argmin(teacher_scores))
        teacher_action = int(teacher_ids[teacher_slot])
        teacher_score = float(teacher_scores[teacher_slot])
        rank = 1 + int(np.sum(scores < teacher_score))
        rows.append(
            {
                "best_action": int(np.argmin(scores)),
                "best_score": float(scores.min()),
                "pid": int(groups[row]),
                "rank": rank,
                "row": int(row),
                "teacher_action": teacher_action,
                "teacher_action_cost": int(costs[teacher_action]),
                "teacher_score": teacher_score,
                "teacher_value": float(values[row]),
            }
        )
        if number == 1 or number % 16 == 0 or number == len(selected):
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "rank": rank,
                        "samples": number,
                    }
                ),
                flush=True,
            )
    ranks = np.asarray([row["rank"] for row in rows], dtype=np.int64)
    value_bins = (0, 20, 40, 80, 160, 10000)
    by_value: dict[str, dict[str, float]] = {}
    for low, high in zip(value_bins, value_bins[1:]):
        positions = [
            index
            for index, row in enumerate(rows)
            if low <= float(row["teacher_value"]) < high
        ]
        if positions:
            by_value[f"{low}-{high - 1}"] = summarize(ranks[positions]) | {
                "samples": len(positions)
            }
    report = {
        "action_count": len(effects),
        "checkpoint": str(args.checkpoint),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "overall": summarize(ranks) | {"samples": len(rows)},
        "by_value": by_value,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
