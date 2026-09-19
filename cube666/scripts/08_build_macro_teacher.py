"""Build exact greedy teacher labels for the 6x6x6 bulk-macro policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import MacroActionTable, generate_random_walk_teacher  # noqa: E402
from cube666.macros import (  # noqa: E402
    enumerate_basic_corner_fixing_commutators,
    enumerate_conjugated_macros,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--max-walk-depth", type=int, default=40)
    parser.add_argument("--max-labels", type=int, default=32)
    parser.add_argument("--score-chunk-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "cube666" / "training" / "macro_teacher_v1.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    macros = enumerate_conjugated_macros(
        enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition),
        puzzle.generators,
        decomposition,
        max_conjugator_depth=1,
    )
    table = MacroActionTable.from_macros(macros)
    dataset = generate_random_walk_teacher(
        table,
        sample_count=args.samples,
        max_walk_depth=args.max_walk_depth,
        max_labels=args.max_labels,
        seed=args.seed,
        score_chunk_size=args.score_chunk_size,
    )
    dataset.save(args.out)
    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "max_walk_depth": args.max_walk_depth,
        "output": str(args.out),
        "samples": dataset.sample_count,
        "seed": args.seed,
        "teacher_improvement_mean": round(
            float(
                (
                    dataset.cluster_cost_targets.sum(axis=1)
                    - dataset.teacher_next_costs
                ).mean()
            ),
            6,
        ),
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
