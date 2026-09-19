"""Assemble, replay, and min-merge model-scout plus KMC frontier solutions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import (  # noqa: E402
    apply_path,
    build_decomposition,
    parity_repair_path,
    residual_report,
)
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macros import (  # noqa: E402
    reduce_commuting_quarter_turn_path,
    reduce_quarter_turn_path,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "submissions" / "cube666_model_scout_kmc_merged.csv",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "model_scout_kmc_merged.json",
    )
    return parser.parse_args()


def load_submission(path: Path) -> dict[int, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): tuple(
                token for token in row["path"].split(".") if token
            )
            for row in csv.DictReader(handle)
        }


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library, puzzle.generators, decomposition
    )
    frontier_report = json.loads(
        (args.frontier_dir / "report.json").read_text(encoding="utf-8")
    )
    if frontier_report["action_digest"] != table.digest:
        raise ValueError("frontier report and action library digests differ")
    incumbent = load_submission(args.incumbent)
    states = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )

    candidates: dict[int, tuple[str, ...]] = {}
    rows: list[dict[str, object]] = []
    for frontier_row in frontier_report["rows"]:
        pid = int(frontier_row["pid"])
        query_id = str(frontier_row["query_id"])
        state = states[pid]
        if "prefix_path" in frontier_row:
            setup_path = tuple(str(move) for move in frontier_row["prefix_path"])
        else:
            corner_path = corner_solver.solve(state, puzzle.solved_state)
            corner_state = apply_path(state, puzzle.generators, corner_path)
            parity_path = parity_repair_path(
                residual_report(
                    corner_state, puzzle.solved_state, decomposition
                ).parity_vector,
                decomposition,
            )
            setup_path = reduce_quarter_turn_path(corner_path + parity_path)
        scout_path = reduce_quarter_turn_path(
            tuple(
                move
                for action in frontier_row["beam_actions"]
                for move in table.paths[int(action)]
            )
        )
        query_path_file = args.query_dir / "paths" / f"{query_id}.path.txt"
        if not query_path_file.exists():
            continue
        query_path = tuple(
            token
            for token in query_path_file.read_text(encoding="utf-8").strip().split(".")
            if token
        )
        candidate = reduce_commuting_quarter_turn_path(
            setup_path + scout_path + query_path
        )
        if puzzle.apply_path(state, candidate) != puzzle.solved_state:
            raise AssertionError(f"{query_id}: assembled candidate failed replay")
        old = candidates.get(pid)
        if old is None or (len(candidate), candidate) < (len(old), old):
            candidates[pid] = candidate
        rows.append(
            {
                "candidate_moves": len(candidate),
                "delta_vs_incumbent": len(candidate) - len(incumbent[pid]),
                "incumbent_moves": len(incumbent[pid]),
                "kmc_moves": len(query_path),
                "pid": pid,
                "query_id": query_id,
                "replay_verified": True,
                "scout_moves": len(scout_path),
                "setup_moves": len(setup_path),
            }
        )

    output_rows: list[dict[str, str]] = []
    source_counts = {"incumbent": 0, "model_scout_kmc": 0}
    output_total = 0
    for pid in sorted(incumbent):
        path = incumbent[pid]
        candidate = candidates.get(pid)
        source = "incumbent"
        if candidate is not None and (len(candidate), candidate) < (len(path), path):
            path = candidate
            source = "model_scout_kmc"
        if puzzle.apply_path(states[pid], path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: merged winner failed replay")
        source_counts[source] += 1
        output_total += len(path)
        output_rows.append(
            {"initial_state_id": str(pid), "path": ".".join(path)}
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(args.out)

    incumbent_total = sum(len(path) for path in incumbent.values())
    report = {
        "candidate_mean": statistics.fmean(
            int(row["candidate_moves"]) for row in rows
        ),
        "candidate_rows": rows,
        "incumbent": str(args.incumbent),
        "incumbent_total": incumbent_total,
        "output": str(args.out),
        "output_total": output_total,
        "replay_verified_candidates": len(rows),
        "replay_verified_output": len(output_rows),
        "source_counts": source_counts,
        "total_saving": incumbent_total - output_total,
    }
    atomic_json(args.report_out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
