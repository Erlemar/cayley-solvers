"""Combine one or more candidate submissions with a fallback, pick shortest per puzzle.

    python scripts/combine_submissions.py \\
        --candidates submissions/medium_b4k_full.csv \\
        --fallback data/kociemba_fallback.csv \\
        --out submissions/combined.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube
from cayley.submit import build_submission
from cayley.verify import verify_submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True, type=Path)
    ap.add_argument("--fallback", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    stats = build_submission(
        puzzle,
        test_csv=PROJECT / "data" / "test.csv",
        candidate_csvs=args.candidates,
        fallback_csv=args.fallback,
        out_path=args.out,
    )
    print(f"stats: {stats}")
    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
