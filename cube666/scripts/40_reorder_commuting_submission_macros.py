"""Shorten solved paths by reordering exact commuting normalized macro segments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macros import (  # noqa: E402
    analyze_corner_fixing_macro,
    reduce_commuting_quarter_turn_path,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "submissions" / "cube666_commuting_macro_reordered.csv",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "commuting_macro_reordered.json",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def effects_commute(left: np.ndarray, right: np.ndarray) -> bool:
    left_then_right = np.take_along_axis(left, right, axis=-1)
    right_then_left = np.take_along_axis(right, left, axis=-1)
    return bool(np.array_equal(left_then_right, right_then_left))


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    states = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
    with open(args.submission, newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    output_rows: list[dict[str, str]] = []
    improved_rows: list[dict[str, int]] = []
    replay_verified = 0
    source_total = 0
    output_total = 0
    commuting_pairs = 0
    accepted_swaps = 0
    for row in source_rows:
        state_id = str(int(row["initial_state_id"]))
        pid = int(state_id)
        source_path = tuple(token for token in row["path"].split(".") if token)
        current = states[state_id]
        boundaries: list[tuple[int, tuple[int, ...]]] = []
        for prefix in range(len(source_path) + 1):
            if all(
                current[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            ):
                report = residual_report(current, puzzle.solved_state, decomposition)
                if report.all_even:
                    boundaries.append((prefix, current))
            if prefix < len(source_path):
                current = puzzle.apply_move(current, source_path[prefix])
        if current != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: source path failed replay")
        if len(boundaries) < 2:
            output_path = source_path
        else:
            prefix_path = source_path[: boundaries[0][0]]
            suffix_path = source_path[boundaries[-1][0] :]
            segments: list[tuple[str, ...]] = []
            effects: list[np.ndarray] = []
            for before, after in zip(boundaries, boundaries[1:], strict=False):
                segment = source_path[before[0] : after[0]]
                macro = analyze_corner_fixing_macro(
                    segment, puzzle.generators, decomposition
                )
                segments.append(segment)
                effects.append(
                    np.asarray(macro.cluster_permutations, dtype=np.uint8)
                )
            output_path = reduce_commuting_quarter_turn_path(
                prefix_path + tuple(move for segment in segments for move in segment) + suffix_path
            )
            while True:
                best: tuple[int, int, tuple[str, ...]] | None = None
                for index in range(len(segments) - 1):
                    if not effects_commute(effects[index], effects[index + 1]):
                        continue
                    commuting_pairs += 1
                    swapped = list(segments)
                    swapped[index], swapped[index + 1] = (
                        swapped[index + 1],
                        swapped[index],
                    )
                    candidate = reduce_commuting_quarter_turn_path(
                        prefix_path
                        + tuple(move for segment in swapped for move in segment)
                        + suffix_path
                    )
                    if len(candidate) >= len(output_path):
                        continue
                    proposal = (len(candidate), index, candidate)
                    if best is None or proposal < best:
                        best = proposal
                if best is None:
                    break
                _, index, output_path = best
                segments[index], segments[index + 1] = (
                    segments[index + 1],
                    segments[index],
                )
                effects[index], effects[index + 1] = (
                    effects[index + 1],
                    effects[index],
                )
                accepted_swaps += 1
        if puzzle.apply_path(states[state_id], output_path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: reordered path failed replay")
        replay_verified += 1
        source_total += len(source_path)
        output_total += len(output_path)
        if len(output_path) < len(source_path):
            improved_rows.append(
                {
                    "output_moves": len(output_path),
                    "pid": pid,
                    "saving": len(source_path) - len(output_path),
                    "source_moves": len(source_path),
                }
            )
        output_rows.append(
            {"initial_state_id": state_id, "path": ".".join(output_path)}
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(args.out)
    report = {
        "accepted_swaps": accepted_swaps,
        "commuting_pairs_considered": commuting_pairs,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "improved_pids": len(improved_rows),
        "output": str(args.out),
        "output_total": output_total,
        "replay_verified": replay_verified,
        "rows": improved_rows,
        "source": str(args.submission),
        "source_total": source_total,
        "total_saving": source_total - output_total,
    }
    atomic_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
