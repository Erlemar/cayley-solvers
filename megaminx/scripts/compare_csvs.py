"""Per-pid compare + min-merge analyzer for megaminx submission CSVs.

Replaces the ad-hoc `python -c` one-liners written repeatedly when evaluating a new
solver config / rescue / bridge run. For each CSV (format: initial_state_id,path with
dot-separated moves) computes per-pid move counts over the COMMON pids, then reports:

  - each CSV's total
  - the per-pid min-merge total over ALL given CSVs + gain vs the best single CSV
  - the CANDIDATE CSV's UNIQUE wins over the per-pid min of the OTHERS (its real
    deployment contribution -- the binding number, not standalone-vs-own-baseline; see
    megaminx_gotchas "A new solver CONFIG's standalone win can be merge-redundant")
  - for exactly 2 CSVs: head-to-head wins each way + the differing-pid table

Usage:
    .venv/Scripts/python.exe megaminx/scripts/compare_csvs.py A.csv B.csv [C.csv ...]
        [--candidate IDX]   # which CSV is the "new" one (default: last); its unique
                            # wins over the min of the others are reported
        [--pids 100,200]    # restrict to a pid subset
        [--show N]          # show up to N pids per win/loss list (default 40)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> dict[int, int]:
    """pid -> move count (len of dot-separated path). Skips empty/missing paths."""
    out: dict[int, int] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # tolerate either initial_state_id,path or id,path; path is the last column
        for row in reader:
            if len(row) < 2 or not row[-1].strip():
                continue
            out[int(row[0])] = len(row[-1].split("."))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path, help="2+ submission CSVs")
    ap.add_argument("--candidate", type=int, default=None,
                    help="0-based index of the 'new' CSV whose unique wins to report "
                         "(default: last)")
    ap.add_argument("--pids", type=str, default=None, help="comma list to restrict to")
    ap.add_argument("--show", type=int, default=40, help="max pids to list per group")
    args = ap.parse_args()

    if len(args.csvs) < 2:
        ap.error("need at least 2 CSVs to compare")

    data = [load(p) for p in args.csvs]
    common = set(data[0])
    for d in data[1:]:
        common &= set(d)
    if args.pids:
        want = {int(p) for p in args.pids.split(",")}
        common &= want
    pids = sorted(common)
    if not pids:
        print("no common pids to compare")
        return 1

    print(f"common pids compared: {len(pids)}")
    totals = [sum(d[p] for p in pids) for d in data]
    for path, t in zip(args.csvs, totals):
        print(f"  {t:>9}  {path.name}")

    merged = sum(min(d[p] for d in data) for p in pids)
    best_single = min(totals)
    print(f"\nper-pid min-merge over all {len(data)} CSVs: {merged}  "
          f"(gain vs best single: {merged - best_single:+d})")

    # Candidate's unique contribution over the min of the OTHERS.
    cand = args.candidate if args.candidate is not None else len(data) - 1
    if not (0 <= cand < len(data)):
        ap.error(f"--candidate {cand} out of range [0,{len(data)-1}]")
    others = [d for i, d in enumerate(data) if i != cand]
    cd = data[cand]
    if others:
        def omin(p):  # min over the other CSVs
            return min(d[p] for d in others)
        others_total = sum(omin(p) for p in pids)
        with_cand = sum(min(cd[p], omin(p)) for p in pids)
        wins = [p for p in pids if cd[p] < omin(p)]
        saved = sum(omin(p) - cd[p] for p in wins)
        print(f"\ncandidate = {args.csvs[cand].name}")
        print(f"  others' per-pid min total: {others_total}")
        print(f"  + candidate min-merged:    {with_cand}  "
              f"(candidate's unique contribution: {with_cand - others_total:+d})")
        print(f"  candidate beats others-min on {len(wins)}/{len(pids)} pids, saving {saved}")
        if wins:
            shown = wins[: args.show]
            print("  unique-win pids (pid: others_min -> candidate): "
                  + str({p: (omin(p), cd[p]) for p in shown})
                  + (" ..." if len(wins) > args.show else ""))

    # Head-to-head for exactly 2 CSVs.
    if len(data) == 2:
        a, b = data
        na, nb = args.csvs[0].name, args.csvs[1].name
        diff = [p for p in pids if a[p] != b[p]]
        aw = [p for p in diff if a[p] < b[p]]
        bw = [p for p in diff if b[p] < a[p]]
        print(f"\nhead-to-head ({na} vs {nb}): {len(diff)} differ | "
              f"{na} better: {len(aw)} (saves {sum(b[p]-a[p] for p in aw)}) | "
              f"{nb} better: {len(bw)} (saves {sum(a[p]-b[p] for p in bw)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
