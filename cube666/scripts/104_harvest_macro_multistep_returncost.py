"""Harvest multi-step proposer rollouts with certified return-to-teacher costs."""

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

from cube666.macro_dual_policy import load_macro_dual_policy_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--roots", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--action-batch-size", type=int, default=4096)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=104666)
    return parser.parse_args()


@torch.inference_mode()
def retrieve(
    model: torch.nn.Module,
    states: np.ndarray,
    keys: torch.Tensor,
    topk: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            queries = model.encode_states(batch)
        scores = queries @ keys.transpose(0, 1)
        output.append(scores.topk(topk, dim=1).indices.cpu().numpy())
    return np.concatenate(output).astype(np.int32, copy=False)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    model, _ = load_macro_dual_policy_checkpoint(args.checkpoint, device)
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        all_states = teacher["states"].astype(np.uint8, copy=False)
        all_values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.int32, copy=False
    )
    eligible = np.flatnonzero(np.mod(groups, args.folds) != args.fold)
    high_count = int(round(args.roots * args.high_value_fraction))
    uniform_count = args.roots - high_count
    parts: list[np.ndarray] = []
    if uniform_count:
        parts.append(rng.choice(eligible, size=uniform_count, replace=False))
    if high_count:
        weights = np.sqrt(
            all_values[eligible] / max(float(all_values[eligible].mean()), 1e-6)
        )
        weights /= weights.sum()
        parts.append(rng.choice(eligible, size=high_count, replace=False, p=weights))
    root_rows = np.unique(np.concatenate(parts))
    if len(root_rows) < args.roots:
        remaining = np.setdiff1d(eligible, root_rows, assume_unique=False)
        root_rows = np.concatenate(
            (root_rows, rng.choice(remaining, size=args.roots - len(root_rows), replace=False))
        )
    rng.shuffle(root_rows)

    key_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(effects), args.action_batch_size):
            stop = min(start + args.action_batch_size, len(effects))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                key_parts.append(
                    model.encode_actions(
                        torch.from_numpy(effects[start:stop]).to(device),
                        torch.from_numpy(costs[start:stop]).to(device),
                    )
                )
    action_keys = torch.cat(key_parts)
    current = all_states[root_rows].copy()
    root_values = all_values[root_rows].copy()
    cumulative_cost = np.zeros(len(root_rows), dtype=np.int32)
    harvested_states: list[np.ndarray] = []
    harvested_targets: list[np.ndarray] = []
    harvested_depths: list[np.ndarray] = []
    harvested_costs: list[np.ndarray] = []
    started = time.perf_counter()
    for depth in range(1, args.steps + 1):
        proposed = retrieve(
            model, current, action_keys, args.topk, args.batch_size, device
        )
        # Rank-squared sampling emphasizes the policy head while retaining its tail.
        rank_weights = 1.0 / np.square(np.arange(1, args.topk + 1, dtype=np.float64))
        rank_weights /= rank_weights.sum()
        ranks = rng.choice(args.topk, size=len(current), p=rank_weights)
        actions = proposed[np.arange(len(current)), ranks]
        current = np.take_along_axis(current, effects[actions], axis=-1)
        cumulative_cost += costs[actions]
        harvested_states.append(current.copy())
        harvested_targets.append(root_values + cumulative_cost)
        harvested_depths.append(np.full(len(current), depth, dtype=np.int16))
        harvested_costs.append(cumulative_cost.copy())
        print(
            json.dumps(
                {
                    "depth": depth,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "mean_path_cost": float(cumulative_cost.mean()),
                }
            ),
            flush=True,
        )
    states = np.concatenate(harvested_states)
    targets = np.concatenate(harvested_targets).astype(np.float32, copy=False)
    depths = np.concatenate(harvested_depths)
    path_costs = np.concatenate(harvested_costs)
    source_rows = np.tile(root_rows, args.steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        states=states,
        return_cost_targets=targets,
        depths=depths,
        path_costs=path_costs,
        source_rows=source_rows,
    )
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "maximum_target": float(targets.max()),
        "mean_target": float(targets.mean()),
        "roots": len(root_rows),
        "samples": len(states),
        "steps": args.steps,
        "topk": args.topk,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
