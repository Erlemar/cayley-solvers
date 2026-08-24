"""Write the per-pid min over several solution CSVs -- the multi-agent merge.

    python cube555/scripts/41_merge.py out.csv a.csv b.csv c.csv [--include-baseline]

The paper's 92.16 is a 69-agent per-pid min, so the honest counterpart is our own min over
decorrelated scorers. Judge a member on its CONTRIBUTION to this, never on its standalone
total: on 444 a run that was strictly worse standalone was worth ~1,300 moves in the merge,
and a fine-tune that was a statistical tie (p = 0.66) still contributed -10 and a unique
under-floor pid.

`--include-baseline` folds in the shipped sample_submission. Keep that OFF for the number
compared to the paper -- that number must be model-only -- and report it separately as
"what would ship".

Every merged path is REPLAYED against test.csv before it is written. A path picked from
the wrong file, or from a run with a translation bug, is otherwise invisible.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube555.puzzle import Cube555  # noqa: E402

N_SANTA = 35


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--include-baseline", action="store_true")
    args = ap.parse_args()

    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}
    central = np.array(puz.solved_state, dtype=np.int64)
    tests = {
        int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64
        )
        for r in csv.DictReader(open(PROJECT / "data" / "test.csv", encoding="utf-8"))
    }

    srcs = list(args.inputs)
    if args.include_baseline:
        srcs.append(PROJECT / "data" / "sample_submission.csv")

    best: dict[int, tuple[int, str, str]] = {}
    per_src_wins: dict[str, int] = {}
    for p in srcs:
        name = p.name
        per_src_wins.setdefault(name, 0)
        for r in csv.DictReader(open(p, encoding="utf-8")):
            pid = int(r["initial_state_id"])
            moves = [m for m in r["path"].split(".") if m]
            cur = tests[pid].copy()
            for m in moves:
                cur = cur[G[m]]
            if not np.array_equal(cur, central):
                print(f"  DROPPED {name} pid {pid}: does not solve")
                continue
            if pid not in best or len(moves) < best[pid][0]:
                best[pid] = (len(moves), r["path"], name)

    for _, _, name in best.values():
        per_src_wins[name] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(best):
            w.writerow([pid, best[pid][1]])

    lens = [v[0] for v in best.values()]
    santa = [best[p][0] for p in range(N_SANTA) if p in best]
    print(f"wrote {args.out}: {len(best)} pids, {sum(lens):,} moves")
    if santa:
        print(f"  santa: n={len(santa)} mean {np.mean(santa):.3f}")
    print("  per-source pids won (this is the contribution to judge a member on):")
    for name, n in sorted(per_src_wins.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<50} {n:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
