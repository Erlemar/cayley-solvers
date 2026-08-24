"""Materialize and strict-min merge replay-verified root macro+KMC rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--full-query-path",
        action="store_true",
        help=(
            "Treat each query path as a complete solution from the original scramble. "
            "Use this with 29_query_kmc_frontiers.py --use-prefix-context."
        ),
    )
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def replay_verify(
    puzzle: Cube666Puzzle,
    states: dict[int, tuple[int, ...]],
    pid: int,
    path: tuple[str, ...],
) -> None:
    if set(path).difference(puzzle.generators):
        raise ValueError(f"PID {pid}: candidate contains an unknown move")
    if puzzle.apply_path(states[pid], path) != puzzle.solved_state:
        raise ValueError(f"PID {pid}: candidate failed exact replay")


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    states = {
        int(pid): state
        for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    frontier = json.loads(
        (args.frontier_dir / "report.json").read_text(encoding="utf-8")
    )
    pid = int(frontier["pid"])
    setup_path = tuple(str(move) for move in frontier["setup_path"])
    candidates: list[dict[str, object]] = []
    for row in frontier["rows"]:
        query_id = str(row["query_id"])
        path_file = args.query_dir / "paths" / f"{query_id}.path.txt"
        if not path_file.exists():
            continue
        kmc_path = parse_path(path_file.read_text(encoding="utf-8"))
        macro_path = tuple(str(move) for move in row["macro_path"])
        raw_path = kmc_path if args.full_query_path else setup_path + macro_path + kmc_path
        reduced = reduce_commuting_quarter_turn_path(raw_path)
        replay_verify(puzzle, states, pid, reduced)
        candidates.append(
            {
                **row,
                "kmc_moves": len(kmc_path),
                "raw_total_moves": len(raw_path),
                "reduced_path": reduced,
                "reduced_total_moves": len(reduced),
            }
        )
    if not candidates:
        raise RuntimeError("no completed KMC rollout paths were found")
    candidates.sort(
        key=lambda row: (int(row["reduced_total_moves"]), str(row["query_id"]))
    )
    best = candidates[0]

    with open(args.incumbent, newline="", encoding="utf-8") as handle:
        incumbent_rows = list(csv.DictReader(handle))
    if not incumbent_rows or set(incumbent_rows[0]) != {"initial_state_id", "path"}:
        raise ValueError("incumbent must contain initial_state_id,path")
    incumbent_total = sum(len(parse_path(row["path"])) for row in incumbent_rows)
    incumbent_pid_moves = next(
        len(parse_path(row["path"]))
        for row in incumbent_rows
        if int(row["initial_state_id"]) == pid
    )
    accepted = int(best["reduced_total_moves"]) < incumbent_pid_moves
    output_rows = []
    for row in incumbent_rows:
        row_pid = int(row["initial_state_id"])
        if accepted and row_pid == pid:
            output_rows.append(
                {
                    "initial_state_id": str(pid),
                    "path": ".".join(best["reduced_path"]),
                }
            )
        else:
            output_rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["initial_state_id", "path"])
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(args.out)
    output_total = incumbent_total - (
        incumbent_pid_moves - int(best["reduced_total_moves"]) if accepted else 0
    )
    serializable_candidates = [
        {key: value for key, value in row.items() if key != "reduced_path"}
        for row in candidates
    ]
    report = {
        "accepted": accepted,
        "best": {key: value for key, value in best.items() if key != "reduced_path"},
        "candidates": serializable_candidates,
        "incumbent": str(args.incumbent),
        "incumbent_pid_moves": incumbent_pid_moves,
        "incumbent_total": incumbent_total,
        "output": str(args.out),
        "output_total": output_total,
        "pid": pid,
        "saving": incumbent_total - output_total,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    report_temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_temporary.replace(args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
