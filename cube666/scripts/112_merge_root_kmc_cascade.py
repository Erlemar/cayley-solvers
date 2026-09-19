"""Strict-min merge wide-completed root macro rollouts for multiple PIDs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
    parser.add_argument("--wide-query-dir", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--full-query-path",
        action="store_true",
        help=(
            "Treat query paths as complete solutions from the original scramble. "
            "Use this with production-context frontier queries."
        ),
    )
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


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
    candidates: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for row in frontier["rows"]:
        query_id = str(row["query_id"])
        path_file = args.wide_query_dir / "paths" / f"{query_id}.path.txt"
        if not path_file.exists():
            continue
        pid = int(row["pid"])
        setup_path = tuple(str(move) for move in row["setup_path"])
        macro_path = tuple(str(move) for move in row["macro_path"])
        kmc_path = parse_path(path_file.read_text(encoding="utf-8"))
        raw_path = kmc_path if args.full_query_path else setup_path + macro_path + kmc_path
        reduced = reduce_commuting_quarter_turn_path(raw_path)
        if puzzle.apply_path(states[pid], reduced) != puzzle.solved_state:
            raise ValueError(f"{query_id}: reduced wide path failed exact replay")
        candidates[pid].append(
            {
                "action_id": int(row["action_id"]),
                "macro_primitive_moves": int(row["macro_primitive_moves"]),
                "query_id": query_id,
                "raw_total_moves": len(raw_path),
                "reduced_path": reduced,
                "reduced_total_moves": len(reduced),
                "wide_kmc_moves": len(kmc_path),
            }
        )
    if not candidates:
        raise RuntimeError("no completed wide rollout paths were found")
    winners = {
        pid: min(
            rows,
            key=lambda row: (int(row["reduced_total_moves"]), str(row["query_id"])),
        )
        for pid, rows in candidates.items()
    }

    with open(args.incumbent, newline="", encoding="utf-8") as handle:
        incumbent_rows = list(csv.DictReader(handle))
    if not incumbent_rows or set(incumbent_rows[0]) != {"initial_state_id", "path"}:
        raise ValueError("incumbent must contain initial_state_id,path")
    incumbent_lengths = {
        int(row["initial_state_id"]): len(parse_path(row["path"]))
        for row in incumbent_rows
    }
    incumbent_total = sum(incumbent_lengths.values())
    accepted: dict[int, dict[str, object]] = {
        pid: winner
        for pid, winner in winners.items()
        if int(winner["reduced_total_moves"]) < incumbent_lengths[pid]
    }
    output_rows = []
    for row in incumbent_rows:
        pid = int(row["initial_state_id"])
        if pid in accepted:
            output_rows.append(
                {
                    "initial_state_id": str(pid),
                    "path": ".".join(accepted[pid]["reduced_path"]),
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
    saving = sum(
        incumbent_lengths[pid] - int(row["reduced_total_moves"])
        for pid, row in accepted.items()
    )
    serializable = lambda row: {  # noqa: E731
        key: value for key, value in row.items() if key != "reduced_path"
    }
    report = {
        "accepted": {str(pid): serializable(row) for pid, row in accepted.items()},
        "candidate_counts": {str(pid): len(rows) for pid, rows in candidates.items()},
        "incumbent": str(args.incumbent),
        "incumbent_total": incumbent_total,
        "output": str(args.out),
        "output_total": incumbent_total - saving,
        "saving": saving,
        "winners": {str(pid): serializable(row) for pid, row in winners.items()},
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
