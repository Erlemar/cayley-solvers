"""T1.1 Phase 2 — hamming-filter the brute-force commutator library.

The existing data/commutator_table.pkl has 4,200 commutators (240 depth-4,
3,960 depth-6). MacroInsert in 27_path_sa.py with these brute-force macros
got 7.8% accept rate. Hypothesis: macros that disrupt MORE pieces (high
hamming) dilute the candidate pool; macros that disrupt FEWER pieces
(low hamming, like speedcuber 3-cycles) should be more useful.

This script:
  1. Loads the existing commutator table
  2. Computes hamming distance (= # pieces moved from solved) for each macro
  3. Saves a filtered table at data/commutator_table_hamming_le_<H>.pkl

Output format matches the existing table so 27_path_sa.py loads it
unchanged via --commutator-table flag.

Run: PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/33_hamming_filter_macros.py
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.puzzle import Megaminx


def hamming_to_solved(perm: tuple[int, ...], solved: tuple[int, ...]) -> int:
    return sum(1 for a, b in zip(perm, solved) if a != b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path,
                    default=PROJECT / "data" / "commutator_table.pkl")
    ap.add_argument("--max-hamming", type=int, default=12,
                    help="Keep only macros with hamming <= this (default 12 = ~3-corner-cycle level)")
    ap.add_argument("--max-len", type=int, default=8,
                    help="Keep only macros of length <= this (default 8)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output pickle (default: data/commutator_table_hamming_le_<H>.pkl)")
    args = ap.parse_args()

    if args.out is None:
        args.out = PROJECT / "data" / f"commutator_table_hamming_le_{args.max_hamming}.pkl"

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    solved = puzzle.solved_state

    print(f"loading {args.src}")
    with open(args.src, "rb") as f:
        d = pickle.load(f)
    src_table = d["table"]
    print(f"  {len(src_table):,} entries (max_depth={d['max_depth']})")

    # Hamming distribution
    print("\ncomputing hamming distances ...")
    from collections import Counter
    hd_counts: Counter[int] = Counter()
    len_counts: Counter[int] = Counter()
    by_perm_hd: dict[tuple[int, ...], tuple[int, int]] = {}  # perm -> (hd, word_len)

    for perm, word in src_table.items():
        if perm == solved:
            continue  # the empty word / identity entry
        hd = hamming_to_solved(perm, solved)
        hd_counts[hd] += 1
        len_counts[len(word)] += 1
        by_perm_hd[perm] = (hd, len(word))

    print(f"\nhamming distribution (across all {sum(hd_counts.values()):,} non-id macros):")
    for h in sorted(hd_counts):
        bar = "#" * min(80, hd_counts[h] // max(1, sum(hd_counts.values()) // 200))
        print(f"  h={h:>3}: {hd_counts[h]:>5,d} {bar}")

    print(f"\nlength distribution:")
    for L in sorted(len_counts):
        print(f"  len={L:>2}: {len_counts[L]:,}")

    # Filter
    filtered: dict[tuple[int, ...], tuple[int, ...]] = {}
    for perm, word in src_table.items():
        if perm == solved:
            continue
        if len(word) == 0 or len(word) > args.max_len:
            continue
        if hamming_to_solved(perm, solved) > args.max_hamming:
            continue
        filtered[perm] = word

    print(f"\n=== filter (hamming <= {args.max_hamming}, len <= {args.max_len}) ===")
    print(f"  kept: {len(filtered):,} / {len(src_table):,} ({100*len(filtered)/max(len(src_table),1):.1f}%)")
    if filtered:
        f_lens = Counter(len(w) for w in filtered.values())
        print(f"  length distribution (kept):")
        for L in sorted(f_lens):
            print(f"    len={L:>2}: {f_lens[L]:,}")
        f_hd = Counter(hamming_to_solved(p, solved) for p in filtered)
        print(f"  hamming distribution (kept):")
        for h in sorted(f_hd):
            print(f"    h={h:>3}: {f_hd[h]:,}")

    # Save in same format
    out_d = {
        "table": filtered,
        "max_depth": max((len(w) for w in filtered.values()), default=0),
        "puzzle_name": d.get("puzzle_name", "megaminx_commutators_filtered"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out_d, f)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
