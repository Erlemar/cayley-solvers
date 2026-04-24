"""Build a memory-efficient Megaminx BFS table (bytes keys).

    python megaminx/scripts/build_bfs_bytes.py --depth 6 --out megaminx/data/bfs_bytes_d6.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.bfs_bytes import build_bfs_bytes
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-canonical", action="store_true")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"building BFS (bytes-keyed) to depth {args.depth}", flush=True)
    table = build_bfs_bytes(puzzle, max_depth=args.depth, canonical=not args.no_canonical, verbose=True)
    print(f"states: {len(table.table):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB on disk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
