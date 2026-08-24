"""Replay every path in a submission against test.csv. Run this before quoting a number.

    python cube555/scripts/50_verify.py cube555/submissions/final.csv

The path argument is POSITIONAL. Expect `VERDICT : PASS`. Never trust a file's own
reported total -- on 444 this check is standing rule 6 and it exists because a
translated-frame path can look plausible and not solve.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube555.puzzle import Cube555  # noqa: E402

N_SANTA = 35


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: 50_verify.py <submission.csv>   (positional!)")
    sub = Path(sys.argv[1])
    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}
    central = np.array(puz.solved_state, dtype=np.int64)

    tests = {
        int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64
        )
        for r in csv.DictReader(open(PROJECT / "data" / "test.csv", encoding="utf-8"))
    }
    rows = list(csv.DictReader(open(sub, encoding="utf-8")))
    ok, bad, lens, santa_lens = 0, [], [], []
    for r in rows:
        pid = int(r["initial_state_id"])
        moves = [m for m in r["path"].split(".") if m]
        unknown = [m for m in moves if m not in G]
        if unknown:
            bad.append((pid, f"unknown move(s) {sorted(set(unknown))[:3]}"))
            continue
        cur = tests[pid].copy()
        for m in moves:
            cur = cur[G[m]]
        if np.array_equal(cur, central):
            ok += 1
            lens.append(len(moves))
            if pid < N_SANTA:
                santa_lens.append(len(moves))
        else:
            bad.append((pid, f"does not solve ({len(moves)} moves)"))

    print(f"{sub}: {ok}/{len(rows)} paths replay to solved")
    print(f"  total {sum(lens):,}   mean {np.mean(lens):.3f}" if lens else "  no paths")
    if santa_lens:
        print(f"  santa subset: n={len(santa_lens)} mean {np.mean(santa_lens):.3f}")
    for pid, why in bad[:20]:
        print(f"  FAIL pid {pid}: {why}")
    print(f"VERDICT : {'PASS' if not bad else 'FAIL'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
