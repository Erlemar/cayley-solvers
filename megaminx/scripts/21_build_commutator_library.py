"""T1.1 prep: build a commutator library for Megaminx.

A starter version of the speedcubing-macro library: short commutators
[A, B] = A B A^{-1} B^{-1} and conjugates X [A, B] X^{-1} enumerated by
brute-force over generator words. Each commutator is a permutation; the
table maps perm -> shortest word that produces it.

Library purpose:
1. Window-replacement post-processing: any contiguous slice of a beam-search
   path whose net permutation lands in the library can be replaced by the
   (potentially shorter) library word.
2. Future: macro-augmented beam search — treat each library word as a
   "meta-action" the beam can apply.

Caveat: the IHES (3x3) version of this saved 0 moves on top of BFS-d5
window replacement (the structured permutations beam-search produces did NOT
land in the d4/d6 commutator subset). Megaminx has 24 generators (vs IHES's
18) and longer paths (avg 85 vs 25), so the prior pessimism may not apply.
This script tests the empirical answer.

Output: `data/commutator_table.pkl` — a `BfsTable` (bytes-keyed via the
existing `BfsTable.save` interface).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.commutator import build_commutator_library, merge_tables
from cayley.bfs_table import BfsTable
from megaminx.puzzle import Megaminx


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"loaded {len(puzzle.move_names)} generators")

    t0 = time.time()
    table = build_commutator_library(
        puzzle, include_d4=True, include_d6=True, include_d8=True, verbose=True
    )
    print(f"\ncommutator library built: {len(table.table):,} entries, "
          f"max_depth={table.max_depth}, total_time={time.time() - t0:.1f}s")

    # Depth distribution
    from collections import Counter
    depth_dist = Counter(len(w) for w in table.table.values())
    print(f"\ndepth distribution (perm count by word length):")
    for d in sorted(depth_dist):
        print(f"  d={d:>2}: {depth_dist[d]:,}")

    # Save
    out_path = PROJECT / "data" / "commutator_table.pkl"
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump({
            "table": table.table,
            "max_depth": table.max_depth,
            "puzzle_name": "megaminx_commutators",
        }, f)
    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Optional: merge with BFS-d6 and report combined size.
    bfs6_path = PROJECT / "data" / "bfs_bytes_d6.pkl"
    if bfs6_path.exists():
        from megaminx.bfs_bytes import BfsBytesTable
        print(f"\nloading BFS-d6 ({bfs6_path.stat().st_size / 1024 / 1024:.0f} MB)...")
        bfs6 = BfsBytesTable.load(bfs6_path)
        print(f"  bfs6: {len(bfs6.table):,} entries (max_depth={bfs6.max_depth})")
        # Count overlap (commutator perms also in BFS-d6)
        n_overlap = sum(1 for k in table.table if bytes(k) in bfs6.table)
        n_new = len(table.table) - n_overlap
        print(f"  commutator perms also in BFS-d6: {n_overlap:,}")
        print(f"  commutator perms NEW to library: {n_new:,}")
        if n_new == 0:
            print("  → commutator library adds 0 NEW perms beyond BFS-d6")
            print("    (consistent with IHES finding; window-replacement post-proc unlikely to help)")
        else:
            print(f"  → {n_new:,} new perms; window-replacement may save moves on these")

    return 0


if __name__ == "__main__":
    sys.exit(main())
