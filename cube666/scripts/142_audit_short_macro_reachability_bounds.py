"""Audit theorem-backed bounded-reachability labels on harvested cube666 frontiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--solved-completions", type=Path)
    parser.add_argument(
        "--teacher",
        type=Path,
        help="optional teacher.npz; exact retained teacher-prefix states are positives",
    )
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--remaining-depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.solved_completions is None and args.teacher is None:
        parser.error("provide --solved-completions and/or --teacher")
    return args


def cycle_distance_vectors(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.uint8)
    flat = states.reshape(-1, 6, 24)
    output = np.empty((len(flat), 6), dtype=np.float32)
    for row in range(len(flat)):
        for cluster in range(6):
            permutation = flat[row, cluster]
            seen = np.zeros(24, dtype=np.bool_)
            cycles = 0
            for start in range(24):
                if seen[start]:
                    continue
                cycles += 1
                cursor = start
                while not seen[cursor]:
                    seen[cursor] = True
                    cursor = int(permutation[cursor])
            output[row, cluster] = (24 - cycles) / 2.0
    return output.reshape(states.shape[:-2] + (6,))


def build_subset_step_dual_weights(action_effects: np.ndarray) -> np.ndarray:
    action_vectors = cycle_distance_vectors(action_effects)
    weights: list[np.ndarray] = []
    for mask in range(1, 1 << 6):
        selected = np.asarray([(mask >> bit) & 1 for bit in range(6)], dtype=bool)
        distance = action_vectors[:, selected].sum(axis=1)
        active = distance > 0
        ratio = float(np.min(1.0 / distance[active]))
        weight = np.zeros(6, dtype=np.float32)
        weight[selected] = ratio
        weights.append(weight)
    output = np.stack(weights)
    if np.any(action_vectors @ output.T > 1.0 + 1e-5):
        raise AssertionError("constructed an infeasible macro-step dual")
    return output


@torch.inference_mode()
def cycle_lower_bounds(
    states: torch.Tensor, dual_weights: torch.Tensor
) -> torch.Tensor:
    flat = states.reshape(-1, 6, 24)
    starts = torch.arange(24, device=states.device).view(1, 1, 24)
    starts = starts.expand(len(flat), 6, -1)
    cursor = starts
    minimum = starts
    for _ in range(24):
        cursor = flat.gather(-1, cursor.long()).long()
        minimum = torch.minimum(minimum, cursor)
    cycles = minimum.eq(starts).sum(dim=2)
    vectors = (24 - cycles).float() * 0.5
    return (vectors @ dual_weights.T).max(dim=1).values


def quantiles(values: np.ndarray) -> dict[str, float | int]:
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    effects = np.load(args.action_effects, allow_pickle=False).astype(
        np.uint8, copy=False
    )
    with np.load(args.frontier, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        roots = payload["root_source_indices"].astype(np.int64, copy=False)
        paths = payload["paths"].astype(np.int64, copy=False)
    solved = np.zeros(len(states), dtype=np.bool_)
    completion_length_by_state = np.full(len(states), -1, dtype=np.int16)
    if args.solved_completions is not None:
        with np.load(args.solved_completions, allow_pickle=False) as payload:
            solved_indices = payload["frontier_indices"].astype(
                np.int64, copy=False
            )
            completion_lengths = payload["completion_lengths"].astype(
                np.int16, copy=False
            )
        solved[solved_indices] = True
        completion_length_by_state[solved_indices] = completion_lengths
    if args.teacher is not None:
        with np.load(args.teacher, allow_pickle=False) as payload:
            teacher_states = payload["states"].astype(np.uint8, copy=False)
            teacher_depths = payload["walk_depths"].astype(np.int16, copy=False)
            teacher_actions = payload["teacher_solution_actions"].astype(
                np.int64, copy=False
            )
        prefix_depth = paths.shape[1]
        for index in range(len(states)):
            root = int(roots[index])
            if int(teacher_depths[root]) < prefix_depth:
                continue
            teacher_state = teacher_states[root].copy()
            for action in teacher_actions[root, :prefix_depth]:
                teacher_state = np.take_along_axis(
                    teacher_state, effects[int(action)], axis=-1
                )
            if np.array_equal(states[index], teacher_state):
                solved[index] = True
                completion_length_by_state[index] = int(teacher_depths[root]) - prefix_depth
    solved_indices = np.flatnonzero(solved)
    completion_lengths = completion_length_by_state[solved_indices]

    dual = build_subset_step_dual_weights(effects)
    dual_device = torch.from_numpy(dual).cuda()
    lower_parts: list[np.ndarray] = []
    for start in range(0, len(states), args.batch_size):
        batch = torch.from_numpy(states[start : start + args.batch_size]).cuda()
        lower_parts.append(cycle_lower_bounds(batch, dual_device).cpu().numpy())
    lower = np.concatenate(lower_parts)
    if np.any(lower[solved_indices] > completion_lengths + 1e-4):
        raise AssertionError("macro-step lower bound exceeds a replayed completion")
    certified_negative = lower > args.remaining_depth + 1e-5
    if np.any(certified_negative & solved):
        raise AssertionError("a certified negative also has a bounded completion")

    paired_roots = 0
    certified_pairs = 0
    for root in np.unique(roots):
        root_mask = roots == root
        positives = int((root_mask & solved).sum())
        negatives = int((root_mask & certified_negative).sum())
        certified_pairs += positives * negatives
        paired_roots += int(positives > 0 and negatives > 0)
    report = {
        "actions": int(len(effects)),
        "certified_negative_fraction": float(certified_negative.mean()),
        "certified_negatives": int(certified_negative.sum()),
        "certified_pairs": certified_pairs,
        "frontier_states": int(len(states)),
        "paired_root_fraction": paired_roots / max(int(np.unique(roots).size), 1),
        "paired_roots": paired_roots,
        "remaining_depth": args.remaining_depth,
        "solved": int(solved.sum()),
        "solved_lower_bounds": quantiles(lower[solved]),
        "unknown": int((~solved & ~certified_negative).sum()),
        "unknown_lower_bounds": quantiles(lower[~solved & ~certified_negative]),
        "certified_negative_lower_bounds": quantiles(lower[certified_negative]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
