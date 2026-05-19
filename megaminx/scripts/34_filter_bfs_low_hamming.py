"""T1.1 Phase 2 (revised) — filter BFS-d6 perms by low hamming distance.

The brute-force commutator table has minimum hamming = 18 (no 3-cycles).
BFS-d6 has 19.4M permutations reachable in <=6 gen turns. Some of these
ARE 3-cycles or other small-hamming permutations. Filter them out as a
candidate macro pool.

3-corner-cycle: hamming 9 (3 corners x 3 stickers) — but megaminx corners
share stickers with adjacent faces in non-trivial ways, so the actual
hamming may differ. We'll see empirically.

Output: commutator-table-format pickle that 27_path_sa.py can load via
--commutator-table flag.

Run: PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/34_filter_bfs_low_hamming.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfs", type=Path, default=PROJECT / "data" / "bfs_bytes_d6.pkl")
    ap.add_argument("--max-hamming", type=int, default=18,
                    help="Keep perms with hamming <= this. d=6 commutator floor was 18; "
                         "lower values capture 3-cycle-like minimal-disruption macros.")
    ap.add_argument("--max-len", type=int, default=8,
                    help="Keep perms with d <= this in BFS-d6.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = PROJECT / "data" / f"bfs_macros_h_le_{args.max_hamming}_d_le_{args.max_len}.pkl"

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    solved_arr = np.array(puzzle.solved_state, dtype=np.int32)

    print(f"loading BFS-d6 {args.bfs} ({args.bfs.stat().st_size / 1024 / 1024:.0f} MB) ...")
    bfs = BfsBytesTable.load(args.bfs)
    print(f"  {len(bfs.table):,} entries")

    print(f"\nscanning for hamming <= {args.max_hamming}, d <= {args.max_len} ...")
    import time
    from collections import Counter

    t0 = time.time()
    kept: dict[tuple[int, ...], tuple[int, ...]] = {}
    hd_counts: Counter[int] = Counter()
    len_counts: Counter[int] = Counter()
    n_scanned = 0
    n_kept = 0

    for state_bytes, path_bytes in bfs.table.items():
        n_scanned += 1
        if n_scanned % 2_000_000 == 0:
            print(f"  scanned {n_scanned:,} / {len(bfs.table):,} ({100*n_scanned/len(bfs.table):.1f}%) "
                  f"kept {n_kept:,}, elapsed {time.time()-t0:.1f}s")
        d = len(path_bytes)
        if d == 0 or d > args.max_len:
            continue
        # Hamming distance vs solved (using bytes for speed)
        s = np.frombuffer(state_bytes, dtype=np.uint8)
        hd = int((s.astype(np.int32) != solved_arr).sum())
        if hd > args.max_hamming:
            continue
        # Keep
        perm = tuple(s.astype(np.int32).tolist())
        word = tuple(int(b) for b in path_bytes)
        kept[perm] = word
        hd_counts[hd] += 1
        len_counts[d] += 1
        n_kept += 1

    elapsed = time.time() - t0
    print(f"\nscan done in {elapsed:.1f}s. Kept {n_kept:,} / {n_scanned:,}")

    if not kept:
        print("ERROR: 0 macros after filter. Try --max-hamming higher.")
        return 1

    print(f"\nkept hamming distribution:")
    for h in sorted(hd_counts):
        bar = "#" * min(60, hd_counts[h] // max(1, sum(hd_counts.values()) // 100))
        print(f"  h={h:>3}: {hd_counts[h]:>6,d} {bar}")
    print(f"\nkept length distribution:")
    for L in sorted(len_counts):
        print(f"  len={L:>2}: {len_counts[L]:>6,d}")

    out_d = {
        "table": kept,
        "max_depth": max(len_counts.keys()),
        "puzzle_name": "megaminx_bfs_low_hamming",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out_d, f)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
