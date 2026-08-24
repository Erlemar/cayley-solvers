"""Verify a 4x4x4 submission CSV: replay every path and check it reaches solved.

    python3 cube444/scripts/00_verify.py --submission cube444/community/submission_54754_merge33.csv

Also prints the per-difficulty-band profile, which is the only breakdown that
matters for this competition (see ANALYSIS_AND_PLAN.md section 2.1).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube444.puzzle import Cube444


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--puzzle-info", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--test", type=Path, default=PROJECT / "data" / "test.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    puz = Cube444.load(args.puzzle_info)
    puz.verify_inverse_pairs()
    central = np.array(puz.solved_state, dtype=np.int16)
    G = {k: np.array(v, dtype=np.int64) for k, v in puz.generators.items()}

    tests, kinds = {}, {}
    with open(args.test, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = int(r["initial_state_id"])
            tests[pid] = np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int16)
            m = re.match(r"generated rw, len=(\d+)$", r["comment"])
            kinds[pid] = int(m.group(1)) if m else -1  # -1 => santa 2023

    with open(args.submission, encoding="utf-8") as f:
        sub = {int(r["initial_state_id"]): r["path"] for r in csv.DictReader(f)}

    missing = sorted(set(tests) - set(sub))
    if missing:
        print(f"FAIL: {len(missing)} puzzle ids missing from submission, e.g. {missing[:5]}")
        return 1

    lens, bad = {}, []
    for pid, s in tests.items():
        moves = puz.parse_path(sub[pid])
        unknown = [m for m in moves if m not in G]
        if unknown:
            bad.append((pid, f"unknown moves {unknown[:3]}"))
            continue
        cur = s
        for m in moves:
            cur = cur[G[m]]
        if not np.array_equal(cur, central):
            bad.append((pid, "does not reach solved"))
        lens[pid] = len(moves)

    total = sum(lens.values())
    print(f"submission: {args.submission}")
    print(f"  puzzles   : {len(tests)}")
    print(f"  solved    : {len(tests) - len(bad)}")
    print(f"  TOTAL MOVES: {total:,}   (mean {total/len(tests):.2f})")
    if bad:
        print(f"  FAILURES  : {len(bad)}  e.g. {bad[:5]}")
        return 1

    if not args.quiet:
        rw = [(kinds[p], lens[p]) for p in lens if kinds[p] > 0]
        santa = [lens[p] for p in lens if kinds[p] == -1]
        print()
        print("  per-difficulty profile")
        print(f"  {'rw_len band':>14} | {'n':>4} | {'mean':>7} | {'min':>4} | {'max':>4}")
        bands = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 100),
                 (101, 250), (251, 500), (501, 1000)]
        for lo, hi in bands:
            v = [L for k, L in rw if lo <= k <= hi]
            if v:
                print(f"  {f'{lo}-{hi}':>14} | {len(v):4d} | {np.mean(v):7.2f} | "
                      f"{min(v):4d} | {max(v):4d}")
        if santa:
            print(f"  {'santa2023':>14} | {len(santa):4d} | {np.mean(santa):7.2f} | "
                  f"{min(santa):4d} | {max(santa):4d}")
        hard = [L for k, L in rw if k > 40] + santa
        print()
        print(f"  hard band (rw_len>40 + santa): {len(hard)} puzzles, "
              f"mean {np.mean(hard):.2f}  (counting lower bound 36.8)")
    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
