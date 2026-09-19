"""Merge per-state 666 paths, commute-reduce them, and replay every winner."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        default=[],
        help="result directory containing zero-padded *.path.txt artifacts; repeatable",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        action="append",
        default=[],
        help="submission-shaped candidate CSV; repeatable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "classical_merged.json",
    )
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_sample(path: Path) -> dict[str, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row["initial_state_id"]): parse_path(str(row["path"]))
            for row in rows
        }


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    test_rows = list(puzzle.iter_test_states(args.data_dir / "test.csv"))
    candidates: dict[str, list[tuple[str, tuple[str, ...]]]] = {
        state_id: [("sample", path)]
        for state_id, path in load_sample(args.data_dir / "sample_submission.csv").items()
    }
    for csv_path in args.candidate_csv:
        source = csv_path.stem
        for state_id, path in load_sample(csv_path).items():
            candidates.setdefault(state_id, []).append((source, path))
    for directory in args.candidate_dir:
        source = directory.name
        for path_file in sorted(directory.glob("*.path.txt")):
            state_id = str(int(path_file.name.removesuffix(".path.txt")))
            candidates.setdefault(state_id, []).append(
                (source, parse_path(path_file.read_text(encoding="utf-8")))
            )

    winners: list[tuple[str, tuple[str, ...], str]] = []
    source_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    reductions = 0
    for state_id, state in test_rows:
        options = candidates.get(state_id)
        if not options:
            raise ValueError(f"state {state_id}: no candidate path")
        verified: list[tuple[int, tuple[str, ...], str]] = []
        for source, raw_path in options:
            reduced = reduce_commuting_quarter_turn_path(raw_path)
            candidate_counts[source] += 1
            if len(reduced) < len(raw_path):
                reductions += len(raw_path) - len(reduced)
            final = puzzle.apply_path(state, reduced)
            if final != puzzle.solved_state:
                mismatches = sum(
                    left != right
                    for left, right in zip(final, puzzle.solved_state, strict=True)
                )
                raise ValueError(
                    f"state {state_id}: candidate {source!r} fails replay with "
                    f"{mismatches} mismatches"
                )
            verified.append((len(reduced), reduced, source))
        _, winner, source = min(verified)
        winners.append((state_id, winner, source))
        source_counts[source] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        for state_id, path, _ in winners:
            writer.writerow({"initial_state_id": state_id, "path": ".".join(path)})
    temporary.replace(args.output)

    lengths = [len(path) for _, path, _ in winners]
    report = {
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "commuting_moves_removed_across_candidates": reductions,
        "maximum": max(lengths),
        "mean": round(statistics.fmean(lengths), 6),
        "median": statistics.median(lengths),
        "minimum": min(lengths),
        "output": str(args.output.resolve()),
        "replay_verified_states": len(winners),
        "source_counts": dict(sorted(source_counts.items())),
        "total_score": sum(lengths),
    }
    atomic_write(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
