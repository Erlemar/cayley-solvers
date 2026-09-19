"""Measure how much recent primitive history predicts the next teacher action."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-model",
        type=Path,
        default=PROJECT / "models/cube666_trajectory_memory_portfolio_v1/model.npz",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def extract_sequences(
    raw_states: np.ndarray,
    offsets: np.ndarray,
    masks: np.ndarray,
    generators: np.ndarray,
) -> tuple[list[list[tuple[tuple[int, ...], int, int]]], int]:
    sequences: list[list[tuple[tuple[int, ...], int, int]]] = []
    discontinuities = 0
    for pid in range(len(offsets) - 1):
        start, end = int(offsets[pid]), int(offsets[pid + 1])
        history: list[int] = []
        rows: list[tuple[tuple[int, ...], int, int]] = []
        for index in range(start, end - 1):
            candidates = [
                action
                for action in range(len(generators))
                if int(masks[index]) & (1 << action)
            ]
            matches = [
                action
                for action in candidates
                if np.array_equal(raw_states[index][generators[action]], raw_states[index + 1])
            ]
            if not matches:
                discontinuities += 1
                history.clear()
                continue
            action = matches[0]
            rows.append((tuple(history[-16:]), action, int(end - index - 1)))
            history.append(action)
        sequences.append(rows)
    return sequences, discontinuities


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    generators = np.asarray(
        [puzzle.generators[name] for name in puzzle.move_names], dtype=np.int64
    )
    with np.load(args.memory_model, allow_pickle=False) as memory:
        raw_states = memory["states"].astype(np.uint8, copy=False)
        offsets = memory["offsets"].astype(np.int64, copy=False)
        masks = memory["action_masks"].astype(np.uint64, copy=False)
    sequences, discontinuities = extract_sequences(
        raw_states, offsets, masks, generators
    )
    report: dict[str, object] = {"history_discontinuities": discontinuities, "orders": {}}
    for order in (0, 1, 2, 4, 8, 16):
        counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        global_counts: Counter[int] = Counter()
        for pid, rows in enumerate(sequences):
            if pid % args.folds == args.fold:
                continue
            for history, action, _ in rows:
                key = history[-order:] if order else ()
                counts[key][action] += 1
                global_counts[action] += 1
        totals = Counter()
        hits = {1: Counter(), 4: Counter(), 8: Counter(), 16: Counter()}
        fallback = [action for action, _ in global_counts.most_common()]
        for pid, rows in enumerate(sequences):
            if pid % args.folds != args.fold:
                continue
            for history, action, distance in rows:
                key = history[-order:] if order else ()
                ranking = [candidate for candidate, _ in counts.get(key, Counter()).most_common()]
                if not ranking:
                    ranking = fallback
                bin_name = next(
                    name
                    for lower, upper, name in (
                        (1, 20, "1-20"),
                        (21, 40, "21-40"),
                        (41, 80, "41-80"),
                        (81, 120, "81-120"),
                        (121, 160, "121-160"),
                        (161, 10_000, "161+"),
                    )
                    if lower <= distance <= upper
                )
                totals["overall"] += 1
                totals[bin_name] += 1
                for width in hits:
                    if action in ranking[:width]:
                        hits[width]["overall"] += 1
                        hits[width][bin_name] += 1
        report["orders"][str(order)] = {
            name: {
                "count": total,
                **{
                    f"top{width}": hits[width][name] / total
                    for width in hits
                },
            }
            for name, total in totals.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
