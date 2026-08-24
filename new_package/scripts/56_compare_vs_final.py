"""Score a beam result against the standing best submission, on the SAME pids.

Standing reference (user decision, 2026-08-02): every result gets compared to
`tetraminx/submissions/FINAL_tetraminx_28561.csv` -- the n-way per-pid min over 30
submission CSVs plus the raw beam JSONs, 1000/1000 pids, replay-verified.

Why a script rather than arithmetic in the chat: a total is only meaningful against
the same pid set, and the two evaluation sets in play here have very different
difficulty (the 15-pid set averages ~29.9 excluding the trivial pid 0; the 30-pid
discriminating set averages ~31.3). Restricting the reference to exactly the pids a
run covered is the only honest comparison, and doing it by hand invites the
mismatched-baseline error that cost most of 2026-08-02.

**The `diff` column is NOT a head-to-head.** The reference is a min-merge over ~30 CSVs
whose paths come largely from 8M-beam x 4-frame ResMLP runs -- roughly 8x the beam
compute per pid of a 1-frame 4M run, plus the merge advantage of many independent
attempts. A 1-frame run sitting a couple of moves above it is at PARITY on quality per
unit compute, not behind. Read `diff` as "distance to a much-higher-compute artifact".

**`merge` is the column that matters.** It is what min(ours, reference) would bank, i.e.
the only number that turns into a better submission.

Reports total / avg / avg>1 for both sides, the per-pid win-loss split, and the
bankable merge.

Usage:
  python tetraminx/scripts/56_compare_vs_final.py results.json [more.json ...]
  python tetraminx/scripts/56_compare_vs_final.py out.csv --label "4M ep1200"
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_REF = PROJECT / "tetraminx" / "submissions" / "FINAL_tetraminx_28561.csv"


def load_csv(path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = int(row["initial_state_id"])
            path_str = (row.get("path") or "").strip()
            out[p] = len(path_str.split(".")) if path_str else 0
    return out


def load_json(path: Path) -> dict[int, int]:
    """Beam JSONs hold one record per (pid, frame); keep the best verified length."""
    best: dict[int, int] = {}
    for r in json.load(open(path, encoding="utf-8")):
        if r.get("found") and r.get("verify_ok"):
            p = int(r["pid"])
            best[p] = min(best.get(p, 10**9), int(r["path_len"]))
    return best


def stats(d: dict[int, int], pids) -> tuple[int, float, float, int]:
    vals = [d[p] for p in pids]
    nz = [v for v in vals if v > 1]
    tot = sum(vals)
    return (tot,
            tot / len(vals) if vals else 0.0,
            sum(nz) / len(nz) if nz else 0.0,
            len(nz))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--label", default=None)
    ap.add_argument("--per-pid", action="store_true", help="print the per-pid table")
    args = ap.parse_args()

    ref = load_csv(args.ref)
    print(f"reference: {args.ref.name} ({len(ref)} pids, {sum(ref.values()):,} moves total)\n")
    print(f"{'run':34s} {'n':>4s} {'ours':>6s} {'ref':>6s} {'diff':>6s} "
          f"{'avg':>6s} {'avg>1':>6s} {'W':>3s} {'L':>3s} {'T':>3s} {'merge':>6s}")

    for path in args.results:
        got = load_json(path) if path.suffix == ".json" else load_csv(path)
        pids = sorted(p for p in got if p in ref)
        if not pids:
            print(f"{path.name:34s}  no overlap with reference")
            continue
        o_tot, o_avg, o_avg1, _ = stats(got, pids)
        r_tot, r_avg, r_avg1, _ = stats(ref, pids)
        w = sum(1 for p in pids if got[p] < ref[p])
        l = sum(1 for p in pids if got[p] > ref[p])
        t = len(pids) - w - l
        merged = sum(min(got[p], ref[p]) for p in pids)
        label = args.label or path.stem
        print(f"{label:34s} {len(pids):4d} {o_tot:6d} {r_tot:6d} {o_tot - r_tot:+6d} "
              f"{o_avg:6.2f} {o_avg1:6.2f} {w:3d} {l:3d} {t:3d} {merged - r_tot:+6d}")

        if args.per_pid:
            print(f"    {'pid':>6s} {'ours':>5s} {'ref':>5s}")
            for p in pids:
                mark = "  <-- win" if got[p] < ref[p] else ("  (lose)" if got[p] > ref[p] else "")
                print(f"    {p:6d} {got[p]:5d} {ref[p]:5d}{mark}")

    print("\nW/L/T = per-pid wins / losses / ties vs the reference.")
    print("merge = what min(ours, ref) would bank over the reference on these pids "
          "-- the only number that translates into a submission.")
    print("NOTE: `diff` is NOT a like-for-like comparison. The reference is min-merged "
          "and mostly 8M x 4-frame ResMLP, i.e. ~8x the beam compute per pid of a "
          "1-frame run. A small positive diff at 1 frame means PARITY per unit "
          "compute, not a regression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
