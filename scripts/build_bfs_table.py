"""Build and persist a BFS lookup table. One-time cost; reused by post-processing.

    python scripts/build_bfs_table.py --depth 5 --out data/bfs_table_d5.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.bfs_table import build_bfs_table
from cayley.puzzle import PictureCube


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    t0 = time.time()
    table = build_bfs_table(puzzle, max_depth=args.depth, verbose=True)
    elapsed = time.time() - t0
    print(f"total: {len(table.table):,} states at depth <= {args.depth} ({elapsed:.1f}s)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.save(args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"saved: {args.out} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
