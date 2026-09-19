"""N-way per-pid minimum merge of 4x4x4 submission CSVs. Rule 26.

    python3 cube444/scripts/06_merge.py --out cube444/submissions/best.csv \
        cube444/community/submission_54754_merge33.csv cube444/submissions/run1.csv

Every path is REPLAYED and verified before it can win a pid, so a corrupt input
CSV can never poison the merge.

Rule 26 exists because on megaminx a "+17 moves saved" bridge result turned out
to be +6 once compared against the per-pid min over ALL available CSVs -- the
community had already contributed 5 of the 6 wins. Always merge against every
CSV on disk before claiming an improvement, and read the per-source attribution
this script prints.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube444.puzzle import Cube444


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--test", type=Path, default=PROJECT / "data" / "test.csv")
    args = ap.parse_args()

    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    central = np.array(puz.solved_state, dtype=np.int64)
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}

    with open(args.test, encoding="utf-8") as f:
        tests = {int(r["initial_state_id"]):
                 np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                 for r in csv.DictReader(f)}

    best: dict[int, tuple[int, str, str]] = {}   # pid -> (len, path, source)
    for src in args.inputs:
        if not src.exists():
            print(f"  SKIP (missing): {src}")
            continue
        with open(src, encoding="utf-8") as f:
            rows = {int(r["initial_state_id"]): r["path"] for r in csv.DictReader(f)}
        n_ok = n_bad = n_win = 0
        for pid, path in rows.items():
            if pid not in tests:
                continue
            moves = path.split(".")
            if any(m not in G for m in moves):
                n_bad += 1
                continue
            cur = tests[pid]
            for m in moves:
                cur = cur[G[m]]
            if not np.array_equal(cur, central):
                n_bad += 1
                continue
            n_ok += 1
            if pid not in best or len(moves) < best[pid][0]:
                best[pid] = (len(moves), path, src.name)
                n_win += 1
        tot = sum(len(p.split(".")) for p in rows.values())
        print(f"  {src.name}: {len(rows)} pids, {tot:,} moves, verified {n_ok}, "
              f"invalid {n_bad}, currently-best {n_win}")

    missing = sorted(set(tests) - set(best))
    if missing:
        print(f"\nFAIL: {len(missing)} pids have no valid path, e.g. {missing[:5]}")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(best):
            w.writerow([pid, best[pid][1]])

    total = sum(v[0] for v in best.values())
    print(f"\nmerged -> {args.out}")
    print(f"  {len(best)} pids, TOTAL {total:,} moves (mean {total/len(best):.2f})")
    print("\n  per-source attribution (pids where this source is strictly the best):")
    from collections import Counter
    cnt = Counter(v[2] for v in best.values())
    for src, n in cnt.most_common():
        moves = sum(v[0] for v in best.values() if v[2] == src)
        print(f"    {src:50s} {n:5d} pids  {moves:7,} moves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
