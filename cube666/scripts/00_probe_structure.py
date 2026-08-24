"""Probe the exact cluster/parity structure of the local 666 competition data."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import (  # noqa: E402
    apply_path,
    build_decomposition,
    parity_repair_path,
    residual_report,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="directory containing puzzle_info.json and test.csv",
    )
    parser.add_argument("--limit", type=int, default=None, help="only scan the first N test states")
    parser.add_argument("--json-out", type=Path, default=None, help="optional machine-readable report")
    return parser.parse_args()


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "mean": round(statistics.fmean(values), 4),
        "median": statistics.median(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    puzzle.verify_inverse_pairs()
    decomposition = build_decomposition(puzzle.generators)
    solver_build_start = time.perf_counter()
    corner_coordinates = CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    corner_solver = ExactCornerSolver.build(corner_coordinates)
    corner_solver_build_seconds = time.perf_counter() - solver_build_start

    initial_parity_pattern_counts: Counter[str] = Counter()
    post_corner_parity_pattern_counts: Counter[str] = Counter()
    repair_length_counts: Counter[int] = Counter()
    residual_counts: list[int] = []
    corner_lengths: list[int] = []
    setup_lengths: list[int] = []
    state_count = 0

    for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv"):
        if args.limit is not None and state_count >= args.limit:
            break
        before = residual_report(state, puzzle.solved_state, decomposition)
        corner_path = corner_solver.solve(state, puzzle.solved_state)
        corner_solved_state = apply_path(state, puzzle.generators, corner_path)
        post_corner = residual_report(corner_solved_state, puzzle.solved_state, decomposition)
        corner_stickers_solved = all(
            corner_solved_state[position] == puzzle.solved_state[position]
            for position in decomposition.corner_orbit
        )
        if not corner_stickers_solved or post_corner.corner_parity:
            raise AssertionError(f"corner solve failed for state {state_id}")

        repair = parity_repair_path(post_corner.parity_vector, decomposition)
        parity_repaired_state = apply_path(corner_solved_state, puzzle.generators, repair)
        after = residual_report(parity_repaired_state, puzzle.solved_state, decomposition)
        if not after.all_even:
            raise AssertionError(f"parity repair failed for state {state_id}")

        initial_parity_pattern_counts["".join(str(bit) for bit in before.parity_vector)] += 1
        post_corner_parity_pattern_counts[
            "".join(str(bit) for bit in post_corner.parity_vector)
        ] += 1
        repair_length_counts[len(repair)] += 1
        assert after.unrestricted_three_cycles is not None
        residual_counts.append(after.unrestricted_three_cycles)
        corner_lengths.append(len(corner_path))
        setup_lengths.append(len(repair) + len(corner_path))
        state_count += 1

    unique_effects = {
        vector for _, vector in decomposition.parity_effects if any(vector)
    }
    report = {
        "state_size": puzzle.size,
        "directed_moves": len(puzzle.move_names),
        "orbit_sizes": [
            len(decomposition.corner_orbit),
            *[len(orbit) for orbit in decomposition.center_orbits],
            *[len(pair.left) for pair in decomposition.wing_pairs],
            *[len(pair.right) for pair in decomposition.wing_pairs],
        ],
        "corner_orbit": list(decomposition.corner_orbit),
        "center_orbits": [list(orbit) for orbit in decomposition.center_orbits],
        "wing_pairs": [
            {"left": list(pair.left), "right": list(pair.right)}
            for pair in decomposition.wing_pairs
        ],
        "inner_moves": list(decomposition.inner_move_names),
        "nonzero_unique_parity_effects": [list(vector) for vector in sorted(unique_effects)],
        "states_scanned": state_count,
        "corner_solver_build_seconds": round(corner_solver_build_seconds, 4),
        "initial_parity_patterns": dict(sorted(initial_parity_pattern_counts.items())),
        "post_corner_parity_patterns": dict(sorted(post_corner_parity_pattern_counts.items())),
        "parity_repair_lengths": dict(sorted(repair_length_counts.items())),
        "optimal_corner_qtm": describe(corner_lengths),
        "parity_plus_corner_setup_qtm": describe(setup_lengths),
        "post_corner_unrestricted_three_cycles": describe(residual_counts),
        "post_corner_unrestricted_three_cycles_total": sum(residual_counts),
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.json_out)


if __name__ == "__main__":
    main()
