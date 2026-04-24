"""Build a Megaminx BFS lookup table from the solved identity state.

    python megaminx/scripts/build_bfs_table.py --depth 5 --out megaminx/data/bfs_table_d5.pkl

Each entry maps a state tuple -> tuple of generator indices (shortest path from identity).
Used for:
 - MITM target set (beam search connects to this shell for free-optimal tail)
 - Window replacement post-processing (replace a window with the shortest equivalent)

Megaminx state sizes (branching 24, non-backtracking):
  d3 ~   6,208 states   (~10 MB, < 1 s)
  d4 ~  90,144 states   (~90 MB, ~5 s)
  d5 ~ 1,280,160 states (~1.5 GB, ~1-2 min)
  d6 ~17,975,460 states (~20 GB at tuple-keyed dict, **likely OOM on 16 GB laptop**)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.bfs_table import build_bfs_table
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-canonical", action="store_true",
                    help="disable the skip-inverse-of-last pruning (slower, fuller coverage)")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"puzzle: {len(puzzle.move_names)} moves, state_size {len(puzzle.solved_state)}")
    print(f"building BFS to depth {args.depth} "
          f"(canonical={'off' if args.no_canonical else 'on'})")

    table = build_bfs_table(
        puzzle, max_depth=args.depth, canonical=not args.no_canonical, verbose=True
    )
    # BfsTable is puzzle-name-tagged to picture_cube_333 by default; patch the tag.
    table.puzzle_name = "megaminx"
    print(f"total unique states: {len(table.table):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.save(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB on disk)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
