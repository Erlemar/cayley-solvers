"""Add exact one-step insertion/cancellation features to rough-word rows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macros import (  # noqa: E402
    load_three_cycle_library,
    summarize_reducing_insertion_context,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_oracle_gate16_v1/unique_rough_rows.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_oracle_gate16_v1/"
            "unique_rough_rows_insertion_context.json"
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=PROJECT / "cube666/artifacts/three_cycle_library.json",
    )
    parser.add_argument("--descriptor-limit", type=int, default=512)
    parser.add_argument("--position-bins", type=int, default=8)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("max-rows must be positive")
        rows = rows[: args.max_rows]

    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    library = load_three_cycle_library(args.library, puzzle.generators, decomposition)
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        summary = summarize_reducing_insertion_context(
            row["cluster_permutations"],
            row["rough_path"],
            library,
            puzzle.generators,
            decomposition,
            descriptor_limit=args.descriptor_limit,
            position_bins=args.position_bins,
        )
        if summary.residual_cost != int(row["residual_three_cycles"]):
            raise AssertionError(
                f"PID {row['pid']}: residual mismatch "
                f"{summary.residual_cost} != {row['residual_three_cycles']}"
            )
        row["insertion_context"] = asdict(summary)
        print(
            f"features={index}/{len(rows)} pid={row['pid']} "
            f"path={summary.path_length} residual={summary.residual_cost} "
            f"best_delta={summary.added_lengths[0] if summary.added_lengths else 0}",
            flush=True,
        )

    atomic_write(args.out, rows)
    print(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "descriptor_limit": args.descriptor_limit,
                "elapsed_seconds": time.perf_counter() - started,
                "out": str(args.out),
                "position_bins": args.position_bins,
                "rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
