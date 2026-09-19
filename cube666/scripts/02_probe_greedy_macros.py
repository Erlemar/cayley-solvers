"""Run a first greedy bulk-commutator reduction on real 666 states."""

from __future__ import annotations

import argparse
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
    enumerate_basic_corner_fixing_commutators,
    enumerate_conjugated_macros,
    greedy_macro_reduce,
    state_cluster_permutations,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--conjugator-depth", type=int, default=1)
    parser.add_argument("--max-macros", type=int, default=100)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "greedy_macro_probe.json",
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
    coordinates = CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    corner_solver = ExactCornerSolver.build(coordinates)
    base_macros = enumerate_basic_corner_fixing_commutators(
        puzzle.generators,
        decomposition,
    )
    macros = enumerate_conjugated_macros(
        base_macros,
        puzzle.generators,
        decomposition,
        max_conjugator_depth=args.conjugator_depth,
    )

    initial_costs: list[int] = []
    final_costs: list[int] = []
    macro_counts: list[int] = []
    macro_move_counts: list[int] = []
    rows = []
    for index, (state_id, state) in enumerate(
        puzzle.iter_test_states(args.data_dir / "test.csv")
    ):
        if index >= args.limit:
            break
        corner_path = corner_solver.solve(state, puzzle.solved_state)
        corner_state = apply_path(state, puzzle.generators, corner_path)
        post_corner = residual_report(corner_state, puzzle.solved_state, decomposition)
        parity_path = parity_repair_path(post_corner.parity_vector, decomposition)
        normalized = apply_path(corner_state, puzzle.generators, parity_path)
        cluster_permutations = state_cluster_permutations(
            normalized,
            puzzle.solved_state,
            decomposition,
        )
        result = greedy_macro_reduce(
            cluster_permutations,
            macros,
            max_macros=args.max_macros,
        )
        replayed = apply_path(normalized, puzzle.generators, result.path)
        replay_report = residual_report(replayed, puzzle.solved_state, decomposition)
        if replay_report.unrestricted_three_cycles != result.final_cost:
            raise AssertionError(f"macro replay mismatch for state {state_id}")

        initial_costs.append(result.initial_cost)
        final_costs.append(result.final_cost)
        macro_counts.append(result.macro_count)
        macro_move_counts.append(len(result.path))
        rows.append(
            {
                "state_id": state_id,
                "initial_cost": result.initial_cost,
                "final_cost": result.final_cost,
                "macro_count": result.macro_count,
                "macro_moves": len(result.path),
                "trajectory": list(result.cost_trajectory),
            }
        )
        print(
            f"state={state_id} residual={result.initial_cost}->{result.final_cost} "
            f"macros={result.macro_count} moves={len(result.path)}",
            flush=True,
        )

    report = {
        "states": len(rows),
        "macro_library_size": len(macros),
        "conjugator_depth": args.conjugator_depth,
        "initial_residual": describe(initial_costs),
        "greedy_final_residual": describe(final_costs),
        "greedy_reduction_total": sum(initial_costs) - sum(final_costs),
        "macro_counts": describe(macro_counts),
        "macro_move_counts": describe(macro_move_counts),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "rows": rows,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.json_out)


if __name__ == "__main__":
    main()
