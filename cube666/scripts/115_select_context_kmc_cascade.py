"""Select production-context macro prefixes for wide KMC completion.

Each cheap query path is already a complete solution from the original scramble.
Candidates are compared only with the no-op control from the same PID and symmetry
frame.  The selected frontier indices are emitted per frame so the expensive pass
can reproduce the matching solver geometry exactly.
"""

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
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--query-prefix", required=True)
    parser.add_argument("--query-suffix", required=True)
    parser.add_argument(
        "--frames",
        default="3_18_35",
        help="underscore- or comma-separated symmetry frame indices",
    )
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--min-cheap-improvement", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    frames = [
        int(token)
        for token in args.frames.replace("_", ",").split(",")
        if token.strip()
    ]
    if not frames or len(set(frames)) != len(frames):
        raise ValueError("frames must be non-empty and unique")

    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    states = {
        int(pid): state
        for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    frontier = json.loads(
        (args.frontier_dir / "report.json").read_text(encoding="utf-8")
    )
    rows = list(frontier["rows"])
    observations: list[dict[str, object]] = []
    by_pid_frame: defaultdict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)

    for frame in frames:
        query_dir = args.query_root / (
            f"{args.query_prefix}{frame:02d}{args.query_suffix}"
        )
        for index, row in enumerate(rows):
            query_id = str(row["query_id"])
            path_file = query_dir / "paths" / f"{query_id}.path.txt"
            if not path_file.exists():
                raise FileNotFoundError(f"missing cheap result {path_file}")
            pid = int(row["pid"])
            path = parse_path(path_file.read_text(encoding="utf-8"))
            reduced = reduce_commuting_quarter_turn_path(path)
            if puzzle.apply_path(states[pid], reduced) != puzzle.solved_state:
                raise ValueError(f"{query_id} frame {frame}: replay failed")
            observation = {
                "action_id": int(row["action_id"]),
                "frame": frame,
                "index": index,
                "macro_primitive_moves": int(row["macro_primitive_moves"]),
                "pid": pid,
                "query_id": query_id,
                "raw_moves": len(path),
                "reduced_moves": len(reduced),
            }
            observations.append(observation)
            by_pid_frame[(pid, frame)].append(observation)

    noops: dict[tuple[int, int], int] = {}
    for key, group in by_pid_frame.items():
        controls = [row for row in group if int(row["action_id"]) == -1]
        if len(controls) != 1:
            raise ValueError(f"PID/frame {key} has {len(controls)} no-op controls")
        noops[key] = int(controls[0]["reduced_moves"])

    eligible_by_pid: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        if int(row["action_id"]) < 0:
            continue
        baseline = noops[(int(row["pid"]), int(row["frame"]))]
        improvement = baseline - int(row["reduced_moves"])
        enriched = {
            **row,
            "cheap_control_moves": baseline,
            "cheap_improvement": improvement,
        }
        if improvement >= args.min_cheap_improvement:
            eligible_by_pid[int(row["pid"])].append(enriched)

    selected: list[dict[str, object]] = []
    selections: dict[str, list[dict[str, object]]] = {}
    for pid in sorted({int(row["pid"]) for row in rows}):
        ranked = sorted(
            eligible_by_pid[pid],
            key=lambda row: (
                -int(row["cheap_improvement"]),
                int(row["reduced_moves"]),
                int(row["frame"]),
                str(row["query_id"]),
            ),
        )
        chosen = ranked[: args.topk]
        selected.extend(chosen)
        selections[str(pid)] = chosen

    args.out_dir.mkdir(parents=True, exist_ok=True)
    indices_by_frame: dict[str, list[int]] = {}
    for frame in frames:
        indices = sorted(
            {int(row["index"]) for row in selected if int(row["frame"]) == frame}
        )
        indices_by_frame[str(frame)] = indices
        (args.out_dir / f"frame{frame:02d}_indices.txt").write_text(
            "" if not indices else "\n".join(str(index) for index in indices) + "\n",
            encoding="utf-8",
        )

    report = {
        "eligible": sum(len(group) for group in eligible_by_pid.values()),
        "frames": frames,
        "frontier_dir": str(args.frontier_dir),
        "indices_by_frame": indices_by_frame,
        "min_cheap_improvement": args.min_cheap_improvement,
        "observations": len(observations),
        "selected": len(selected),
        "selections": selections,
        "topk": args.topk,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "selections"}, indent=2))
    for pid, chosen in selections.items():
        summary = ", ".join(
            f"f{int(row['frame']):02d}:{row['query_id']}({int(row['cheap_improvement']):+d})"
            for row in chosen
        )
        print(f"pid={pid} selected={summary or 'none'}")


if __name__ == "__main__":
    main()
