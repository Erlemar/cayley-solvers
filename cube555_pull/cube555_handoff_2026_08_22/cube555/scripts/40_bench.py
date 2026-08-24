"""Compare solution CSVs: totals, matched intersections, per-pid deltas, sign test, merges.

    python cube555/scripts/40_bench.py base=a.csv new=b.csv --compare --merge
    python cube555/scripts/40_bench.py run=x.csv --santa

Port of cube444/scripts/56_bench_table.py, which the playbook says to have on day 1.
Why it exists, in one line each:

  * A STANDALONE TOTAL IS THE WRONG STATISTIC. A 444 run scored strictly worse standalone
    and was worth ~1,300 moves in the merge. Judge on the per-pid min.
  * A TOTAL WITHOUT A SIGN TEST IS UNREADABLE. On 444, two checkpoints 2 moves apart over
    53 pids had 38 pids emitting IDENTICAL lengths -- p = 0.50, not a result.
  * NEVER EXTRAPOLATE A PARTIAL RUN BY PID COUNT. Only matched intersections are printed.

SANTA REPORTING. The paper scores 19 of the 35 santa instances in this package and does
not say which, and its Santa baseline (96.58) does not match this package's baseline over
all 35 (93.60). So `--santa` reports all 35 AND the hardest-19 and easiest-19 subsets by
the shipped baseline, and beating 92.16 means beating it on all three.
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from math import comb
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
N_SANTA = 35
PAPER = 92.16
PAPER_SANTA_BASELINE = 96.58


def load(path: Path) -> dict[int, int]:
    with open(path, encoding="utf-8") as f:
        return {
            int(r["initial_state_id"]): len(r["path"].split("."))
            for r in csv.DictReader(f)
        }


def sign_test(a: dict[int, int], b: dict[int, int]) -> tuple[int, int, int, float]:
    """Two-sided sign test over the matched pids. Ties excluded, as the test requires."""
    keys = sorted(set(a) & set(b))
    w = sum(1 for k in keys if b[k] < a[k])
    losses = sum(1 for k in keys if b[k] > a[k])
    t = len(keys) - w - losses
    n = w + losses
    if n == 0:
        return w, losses, t, 1.0
    lo = min(w, losses)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2.0**n)
    return w, losses, t, min(1.0, 2 * tail)


def santa_report(name: str, d: dict[int, int], baseline: dict[int, int]) -> None:
    santa = [p for p in range(N_SANTA) if p in d]
    if not santa:
        print(f"  {name}: no santa pids present")
        return
    order = sorted(santa, key=lambda p: -baseline[p])
    hard19, easy19 = order[:19], order[-19:]
    rows = [("all 35", santa), ("hardest 19", hard19), ("easiest 19", easy19)]
    print(f"\n  {name}  (n santa = {len(santa)})")
    print(
        f"    {'subset':<12} {'n':>3} {'mean':>8} {'baseline':>9} {'vs base':>8} "
        f"{'vs 92.16':>9}"
    )
    for label, sel in rows:
        if not sel:
            continue
        m = sum(d[p] for p in sel) / len(sel)
        b = sum(baseline[p] for p in sel) / len(sel)
        print(
            f"    {label:<12} {len(sel):>3} {m:8.3f} {b:9.3f} {m-b:+8.3f} {m-PAPER:+9.3f}"
            f"{'   BEATS PAPER' if m < PAPER else ''}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path.csv")
    ap.add_argument(
        "--compare", action="store_true", help="pairwise deltas + sign test"
    )
    ap.add_argument("--merge", action="store_true", help="every per-pid min subset")
    ap.add_argument("--santa", action="store_true", help="santa subsets vs the paper")
    ap.add_argument(
        "--baseline", type=Path, default=PROJECT / "data" / "sample_submission.csv"
    )
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"expected name=path.csv, got {spec!r}")
        name, path = spec.split("=", 1)
        runs[name] = load(Path(path))
    base = load(args.baseline)

    print(f"{'run':<22} {'n':>6} {'total':>10} {'mean':>9}")
    for name, d in runs.items():
        tot = sum(d.values())
        print(f"{name:<22} {len(d):>6} {tot:>10,} {tot/max(1,len(d)):>9.3f}")

    if args.santa:
        print("\n=== SANTA (the target). paper = 92.16 over an unknown 19 of these 35;")
        print("    the paper's own Santa-community baseline for its 19 was 96.58. ===")
        santa_report("shipped baseline", {p: base[p] for p in range(N_SANTA)}, base)
        for name, d in runs.items():
            santa_report(name, d, base)
        if len(runs) > 1:
            merged = {}
            for p in range(N_SANTA):
                vals = [d[p] for d in runs.values() if p in d]
                if vals:
                    merged[p] = min(vals)
            santa_report(f"MIN-MERGE of {len(runs)}", merged, base)

    if args.compare and len(runs) > 1:
        print("\n=== pairwise, on the MATCHED intersection only ===")
        names = list(runs)
        for a, b in combinations(names, 2):
            da, db = runs[a], runs[b]
            keys = sorted(set(da) & set(db))
            if not keys:
                continue
            ta = sum(da[k] for k in keys)
            tb = sum(db[k] for k in keys)
            w, ls, t, p = sign_test(da, db)
            print(
                f"  {b} vs {a}: n={len(keys)}  {tb:,} vs {ta:,}  delta {tb-ta:+,} "
                f"({(tb-ta)/len(keys):+.3f}/pid)  {w}W/{ls}L/{t}T  p = {p:.3g}"
            )
            if p > 0.05:
                print("      NOT SIGNIFICANT -- do not report this as a win or a loss.")

    if args.merge and len(runs) > 1:
        print(
            "\n=== per-pid min over every subset (judge a member on its CONTRIBUTION) ==="
        )
        names = list(runs)
        common = set.intersection(*(set(d) for d in runs.values()))
        full = sum(min(runs[n][k] for n in names) for k in common)
        for r in range(1, len(names) + 1):
            for sub in combinations(names, r):
                tot = sum(min(runs[n][k] for n in sub) for k in common)
                mark = "  <- full set" if len(sub) == len(names) else ""
                print(
                    f"  {'+'.join(sub):<44} {tot:>8,}  mean {tot/len(common):7.3f}"
                    f"  vs full {tot-full:+5}{mark}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
