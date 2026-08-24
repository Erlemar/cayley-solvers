"""Build replay-valid classical 666 solutions with exact inserted 3-cycles."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import apply_path, build_decomposition, parity_repair_path, residual_report  # noqa: E402
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402
from cube666.macros import (  # noqa: E402
    build_isolated_three_cycle_library,
    enumerate_basic_corner_fixing_commutators,
    enumerate_conjugated_macros,
    enumerate_nested_corner_fixing_commutators,
    finish_with_inserted_three_cycles,
    greedy_macro_reduce,
    load_three_cycle_library,
    reduce_quarter_turn_path,
    save_three_cycle_library,
    state_cluster_permutations,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "cube666" / "results" / "classical_baseline.csv",
    )
    parser.add_argument(
        "--finisher-cache",
        type=Path,
        default=PROJECT / "cube666" / "artifacts" / "three_cycle_library.json",
    )
    parser.add_argument(
        "--no-bulk-greedy",
        action="store_true",
        help="skip the depth-1 greedy parallel-commutator phase",
    )
    parser.add_argument("--max-bulk-macros", type=int, default=100)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "classical_baseline.json",
    )
    return parser.parse_args()


def describe(values: list[int]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "mean": round(statistics.fmean(values), 4),
        "median": statistics.median(values),
        "maximum": max(values),
        "total": sum(values),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    if args.finisher_cache.exists():
        library = load_three_cycle_library(
            args.finisher_cache,
            puzzle.generators,
            decomposition,
        )
    else:
        nested = enumerate_nested_corner_fixing_commutators(
            puzzle.generators,
            decomposition,
            inner_conjugator_depth=1,
        )
        library = build_isolated_three_cycle_library(
            nested,
            puzzle.generators,
            decomposition,
            max_conjugator_depth=3,
        )
        save_three_cycle_library(args.finisher_cache, library, puzzle.generators)
    if len(library) != 6 * 4048:
        raise ValueError(f"finisher library is incomplete: {len(library)} != 24288")
    bulk_macros = ()
    if not args.no_bulk_greedy:
        bulk_macros = enumerate_conjugated_macros(
            enumerate_basic_corner_fixing_commutators(
                puzzle.generators,
                decomposition,
            ),
            puzzle.generators,
            decomposition,
            max_conjugator_depth=1,
        )

    output_rows: list[dict[str, str]] = []
    report_rows = []
    lengths: list[int] = []
    for index, (state_id, state) in enumerate(
        puzzle.iter_test_states(args.data_dir / "test.csv")
    ):
        if index < args.start_index:
            continue
        if index >= args.start_index + args.limit:
            break
        corner_path = corner_solver.solve(state, puzzle.solved_state)
        corner_state = apply_path(state, puzzle.generators, corner_path)
        post_corner = residual_report(corner_state, puzzle.solved_state, decomposition)
        parity_path = parity_repair_path(post_corner.parity_vector, decomposition)
        normalized = apply_path(corner_state, puzzle.generators, parity_path)
        setup_path = reduce_quarter_turn_path(corner_path + parity_path)
        bulk_path: tuple[str, ...] = ()
        bulk_initial_cost = residual_report(
            normalized,
            puzzle.solved_state,
            decomposition,
        ).unrestricted_three_cycles
        if bulk_initial_cost is None:
            raise AssertionError("normalized state has odd cluster parity")
        bulk_final_cost = bulk_initial_cost
        bulk_macro_count = 0
        bulk_state = normalized
        if bulk_macros:
            bulk_result = greedy_macro_reduce(
                state_cluster_permutations(normalized, puzzle.solved_state, decomposition),
                bulk_macros,
                max_macros=args.max_bulk_macros,
            )
            bulk_path = bulk_result.path
            bulk_final_cost = bulk_result.final_cost
            bulk_macro_count = bulk_result.macro_count
            bulk_state = apply_path(normalized, puzzle.generators, bulk_path)
        clusters = state_cluster_permutations(
            bulk_state,
            puzzle.solved_state,
            decomposition,
        )
        initial_path = reduce_quarter_turn_path(setup_path + bulk_path)
        finished = finish_with_inserted_three_cycles(
            clusters,
            initial_path,
            library,
            puzzle.generators,
            decomposition,
        )
        final_state = apply_path(state, puzzle.generators, finished.path)
        if final_state != puzzle.solved_state:
            raise AssertionError(f"final replay failed for state {state_id}")

        output_rows.append({"initial_state_id": state_id, "path": ".".join(finished.path)})
        lengths.append(len(finished.path))
        report_rows.append(
            {
                "state_id": state_id,
                "corner_moves": len(corner_path),
                "parity_moves": len(parity_path),
                "residual_three_cycles": finished.initial_cost,
                "pre_bulk_three_cycles": bulk_initial_cost,
                "bulk_final_three_cycles": bulk_final_cost,
                "bulk_macros": bulk_macro_count,
                "bulk_moves_before_insertion": len(bulk_path),
                "finisher_macros": finished.macro_count,
                "final_moves": len(finished.path),
            }
        )
        print(
            f"state={state_id} setup={len(setup_path)} bulk={bulk_initial_cost}->{bulk_final_cost} "
            f"final_moves={len(finished.path)}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(temporary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary_csv.replace(args.out)

    report = {
        "states": len(output_rows),
        "finisher_library_size": len(library),
        "solution_lengths": describe(lengths),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "rows": report_rows,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.report_out.with_suffix(args.report_out.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_report.replace(args.report_out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
