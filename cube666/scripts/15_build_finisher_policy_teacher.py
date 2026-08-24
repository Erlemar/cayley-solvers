"""Build isolated-3-cycle actions and exact geodesic policy datasets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    generate_geodesic_macro_teacher,
    save_macro_action_library,
)
from cube666.macros import load_three_cycle_library  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument(
        "--finisher-library",
        type=Path,
        default=PROJECT / "cube666" / "artifacts" / "three_cycle_library.json",
    )
    parser.add_argument("--train-samples", type=int, default=100_000)
    parser.add_argument("--eval-samples", type=int, default=5_000)
    parser.add_argument("--max-depth", type=int, default=68)
    parser.add_argument("--max-labels", type=int, default=16)
    parser.add_argument("--label-candidates", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=666)
    parser.add_argument("--eval-seed", type=int, default=667)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "training" / "finisher_policy_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    library = load_three_cycle_library(
        args.finisher_library,
        puzzle.generators,
        decomposition,
    )
    macros = tuple(
        macro
        for _, macro in sorted(
            library.items(),
            key=lambda item: (item[0][0], item[0][1]),
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = save_macro_action_library(args.out_dir / "action_library.json", macros)
    train = generate_geodesic_macro_teacher(
        table,
        sample_count=args.train_samples,
        max_depth=args.max_depth,
        max_labels=args.max_labels,
        label_candidates=args.label_candidates,
        seed=args.train_seed,
    )
    train.save(args.out_dir / "train.npz")
    evaluation = generate_geodesic_macro_teacher(
        table,
        sample_count=args.eval_samples,
        max_depth=args.max_depth,
        max_labels=args.max_labels,
        label_candidates=args.label_candidates,
        seed=args.eval_seed,
    )
    evaluation.save(args.out_dir / "eval.npz")
    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "eval_samples": evaluation.sample_count,
        "eval_seed": args.eval_seed,
        "max_depth": args.max_depth,
        "max_labels": args.max_labels,
        "label_candidates": args.label_candidates,
        "train_samples": train.sample_count,
        "train_seed": args.train_seed,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
