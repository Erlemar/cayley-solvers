"""Independent replay verification of a cube444 submission CSV.

Deliberately does NOT reuse the move tables or hashing of the rewriting scripts: it goes
through `src/cube444/puzzle.py` (tuple-based, pure Python) so a bug in the GPU pipeline
cannot verify itself. Checks pid coverage against test.csv, the move alphabet, and that
every path replays to the solved colouring.

    python cube444/scripts/73_verify_submission.py cube444/submissions/leader_r5.csv
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "cube444" / "src"))

from cube444.puzzle import Cube444  # noqa: E402

DATA = PROJECT / "cube444" / "data"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: 73_verify_submission.py <submission.csv> [baseline.csv]")
        return 2
    sub_path = Path(sys.argv[1])
    puzzle = Cube444.load(DATA / "puzzle_info.json")
    puzzle.verify_inverse_pairs()

    tests = {}
    with io.open(DATA / "test.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            tests[int(r["initial_state_id"])] = tuple(
                int(x) for x in r["initial_state"].split(","))

    rows = {}
    with io.open(sub_path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        assert rd.fieldnames[:2] == ["initial_state_id", "path"], \
            f"bad header {rd.fieldnames}"
        for r in rd:
            rows[int(r["initial_state_id"])] = puzzle.parse_path(r["path"].strip())

    missing = sorted(set(tests) - set(rows))
    extra = sorted(set(rows) - set(tests))
    bad_moves, unsolved = [], []
    total = 0
    for pid, path in rows.items():
        for m in path:
            if m not in puzzle.generators:
                bad_moves.append((pid, m))
                break
        total += len(path)
        if not puzzle.is_solved(puzzle.apply_path(tests[pid], path)):
            unsolved.append(pid)

    print(f"file      : {sub_path}")
    print(f"pids      : {len(rows)} (test has {len(tests)})")
    print(f"missing   : {len(missing)}  extra: {len(extra)}")
    print(f"bad moves : {len(bad_moves)}")
    print(f"unsolved  : {len(unsolved)}")
    print(f"TOTAL     : {total:,}   mean {total/max(1,len(rows)):.3f}")

    if len(sys.argv) > 3 or len(sys.argv) == 3:
        base = {}
        with io.open(sys.argv[2], encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                base[int(r["initial_state_id"])] = puzzle.parse_path(r["path"].strip())
        bt = sum(len(v) for v in base.values())
        diff = [(pid, len(base[pid]), len(rows[pid])) for pid in rows
                if pid in base and len(rows[pid]) != len(base[pid])]
        print(f"\nbaseline  : {sys.argv[2]}  {bt:,}")
        print(f"delta     : {total - bt:+d}   changed pids: {len(diff)}")
        for pid, a, b in sorted(diff, key=lambda t: t[2] - t[1])[:20]:
            print(f"   pid {pid}: {a} -> {b}  ({b-a:+d})")
        worse = [d for d in diff if d[2] > d[1]]
        if worse:
            print(f"WARNING: {len(worse)} pids got LONGER")

    ok = not missing and not bad_moves and not unsolved
    print("\nVERDICT   :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
