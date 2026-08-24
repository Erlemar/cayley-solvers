"""Measure reusable normalized macro transitions in a solved cube666 submission."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macros import analyze_corner_fixing_macro  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT / "cayley-py-666-cube",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "submission_macro_boundaries.json",
    )
    return parser.parse_args()


def describe(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "minimum": None, "mean": None, "median": None, "maximum": None}
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    states_by_id = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
    known_moves = set(puzzle.generators)

    with open(args.submission, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"initial_state_id", "path"}:
        raise ValueError("submission must contain initial_state_id,path")

    segment_lengths: list[int] = []
    improvements: list[int] = []
    reducing_segment_lengths: list[int] = []
    boundary_counts: list[int] = []
    path_lengths: list[int] = []
    effect_counts: Counter[bytes] = Counter()
    reducing_effect_counts: Counter[bytes] = Counter()
    length_histogram: Counter[int] = Counter()
    improving_length_histogram: Counter[int] = Counter()
    paths_with_transition = 0
    paths_with_improving_transition = 0
    invalid_macros = 0

    for row in rows:
        state_id = str(int(row["initial_state_id"]))
        state = states_by_id.get(state_id)
        if state is None:
            raise ValueError(f"unknown state id {state_id}")
        moves = tuple(token for token in row["path"].split(".") if token)
        unknown = set(moves).difference(known_moves)
        if unknown:
            raise ValueError(f"state {state_id} has unknown moves {sorted(unknown)}")
        path_lengths.append(len(moves))

        current = state
        boundaries: list[tuple[int, int]] = []
        for prefix in range(len(moves) + 1):
            corners_solved = all(
                current[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            )
            if corners_solved:
                report = residual_report(current, puzzle.solved_state, decomposition)
                if report.all_even:
                    if report.unrestricted_three_cycles is None:
                        raise AssertionError("even normalized state has no exact residual")
                    boundaries.append((prefix, report.unrestricted_three_cycles))
            if prefix < len(moves):
                current = puzzle.apply_move(current, moves[prefix])
        if current != puzzle.solved_state:
            raise AssertionError(f"state {state_id} failed exact replay")
        boundary_counts.append(len(boundaries))
        if len(boundaries) >= 2:
            paths_with_transition += 1
        path_has_improvement = False
        for (before, before_cost), (after, after_cost) in zip(
            boundaries,
            boundaries[1:],
            strict=False,
        ):
            segment = moves[before:after]
            if not segment:
                continue
            improvement = before_cost - after_cost
            segment_lengths.append(len(segment))
            improvements.append(improvement)
            length_histogram[len(segment)] += 1
            try:
                macro = analyze_corner_fixing_macro(
                    segment,
                    puzzle.generators,
                    decomposition,
                )
            except ValueError:
                invalid_macros += 1
                continue
            effect_key = np.asarray(macro.cluster_permutations, dtype=np.uint8).tobytes()
            effect_counts[effect_key] += 1
            if improvement > 0:
                path_has_improvement = True
                reducing_segment_lengths.append(len(segment))
                reducing_effect_counts[effect_key] += 1
                improving_length_histogram[len(segment)] += 1
        if path_has_improvement:
            paths_with_improving_transition += 1

    report = {
        "boundary_counts": describe(boundary_counts),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "improvements": describe(improvements),
        "improving_length_histogram": {
            str(key): value for key, value in sorted(improving_length_histogram.items())
        },
        "invalid_macros": invalid_macros,
        "length_histogram": {str(key): value for key, value in sorted(length_histogram.items())},
        "path_lengths": describe(path_lengths),
        "paths": len(rows),
        "paths_with_improving_transition": paths_with_improving_transition,
        "paths_with_transition": paths_with_transition,
        "reducing_segment_lengths": describe(reducing_segment_lengths),
        "repeated_reducing_effects": sum(count > 1 for count in reducing_effect_counts.values()),
        "segment_lengths": describe(segment_lengths),
        "submission": str(args.submission),
        "transitions": len(segment_lengths),
        "unique_effects": len(effect_counts),
        "unique_reducing_effects": len(reducing_effect_counts),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
