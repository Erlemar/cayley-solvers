"""Select each PID's top cheap-KMC root rollouts for wide completion."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--cheap-query-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--indices-file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    states = {
        int(pid): state
        for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    frontier = json.loads(
        (args.frontier_dir / "report.json").read_text(encoding="utf-8")
    )
    by_pid: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(frontier["rows"]):
        query_id = str(row["query_id"])
        path_file = args.cheap_query_dir / "paths" / f"{query_id}.path.txt"
        if not path_file.exists():
            continue
        pid = int(row["pid"])
        setup_path = tuple(str(move) for move in row["setup_path"])
        macro_path = tuple(str(move) for move in row["macro_path"])
        cheap_path = parse_path(path_file.read_text(encoding="utf-8"))
        reduced = reduce_commuting_quarter_turn_path(
            setup_path + macro_path + cheap_path
        )
        if puzzle.apply_path(states[pid], reduced) != puzzle.solved_state:
            raise ValueError(f"{query_id}: reduced cheap path failed exact replay")
        by_pid[pid].append(
            {
                "action_id": int(row["action_id"]),
                "cheap_kmc_moves": len(cheap_path),
                "cheap_reduced_total_moves": len(reduced),
                "index": index,
                "macro_primitive_moves": int(row["macro_primitive_moves"]),
                "query_id": query_id,
            }
        )
    expected_pids = {int(pid) for pid in frontier["pids"]}
    if set(by_pid) != expected_pids:
        missing = sorted(expected_pids.difference(by_pid))
        raise RuntimeError(f"cheap completion is missing PIDs {missing}")
    selections: dict[str, list[dict[str, object]]] = {}
    indices: list[int] = []
    for pid in sorted(by_pid):
        ranked = sorted(
            by_pid[pid],
            key=lambda row: (
                int(row["cheap_reduced_total_moves"]),
                str(row["query_id"]),
            ),
        )
        chosen = ranked[: args.topk]
        selections[str(pid)] = chosen
        indices.extend(int(row["index"]) for row in chosen)
    indices.sort()
    args.indices_file.parent.mkdir(parents=True, exist_ok=True)
    indices_temporary = args.indices_file.with_suffix(args.indices_file.suffix + ".tmp")
    indices_temporary.write_text(
        "\n".join(str(index) for index in indices) + "\n", encoding="utf-8"
    )
    indices_temporary.replace(args.indices_file)
    report = {
        "cheap_query_dir": str(args.cheap_query_dir),
        "frontier_dir": str(args.frontier_dir),
        "indices": indices,
        "indices_file": str(args.indices_file),
        "pids": sorted(by_pid),
        "selected": len(indices),
        "selections": selections,
        "topk": args.topk,
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
