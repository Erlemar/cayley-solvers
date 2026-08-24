"""Convert TPU driver results.json (list of {pid, path_idx}) to a submission CSV,
verifying every path against central_state.

    python3 cube444/tpu/json_to_csv.py --results r0.json r1.json --out sub.csv

Multiple results files are merged per-pid-min (so shard outputs and re-runs
combine). Paths are move-index lists from the driver; this maps them to move
names and replays each against the original scramble. An unverifiable path is
dropped with a warning -- it never reaches the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cube444.puzzle import Cube444  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", type=Path, required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--puzzle-info", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--test", type=Path, default=PROJECT / "data" / "test.csv")
    args = ap.parse_args()

    puz = Cube444.load(args.puzzle_info)
    names = list(puz.move_names)
    central = np.array(puz.solved_state, dtype=np.int64)
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}
    with open(args.test, encoding="utf-8") as f:
        tests = {int(r["initial_state_id"]):
                 np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
                 for r in csv.DictReader(f)}

    best: dict[int, list[str]] = {}
    n_bad = 0
    for rf in args.results:
        recs = json.load(open(rf, encoding="utf-8"))
        n_ok = 0
        for rec in recs:
            if not rec.get("verify") or not rec.get("path_idx"):
                continue
            pid = int(rec["pid"])
            path = [names[int(m)] for m in rec["path_idx"]]
            cur = tests[pid].copy()
            for m in path:
                cur = cur[G[m]]
            if not np.array_equal(cur, central):
                n_bad += 1
                continue
            n_ok += 1
            if pid not in best or len(path) < len(best[pid]):
                best[pid] = path
        print(f"  {rf.name}: {len(recs)} records, {n_ok} verified")
    if n_bad:
        print(f"  WARNING: dropped {n_bad} records that failed host re-verification")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(best):
            w.writerow([pid, puz.format_path(best[pid])])
    tot = sum(len(p) for p in best.values())
    print(f"wrote {args.out}: {len(best)} pids, {tot:,} moves "
          f"(mean {tot/max(1,len(best)):.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
