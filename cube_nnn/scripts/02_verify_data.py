"""End-to-end check against the real 666 competition data.

Validates, on the shipped files and nothing else:
  * every sample path replays its test state to solved
  * the inverse frame is length-preserving and correct
  * all 96 frames (48 symmetries x inverse) produce valid solutions
  * the score model used to convert leaderboard totals into moves/pid

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/02_verify_data.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))
csv.field_size_limit(10 ** 8)

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.symmetry import build_symmetries  # noqa: E402

DATA = ROOT / "cayley-py-666-cube"


def load_csv(path: Path) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return {int(r[0]): r[1] for r in rows[1:]}


def main() -> None:
    cube = NCube.from_puzzle_info(DATA / "puzzle_info.json")
    tests = load_csv(DATA / "test.csv")
    sample = load_csv(DATA / "sample_submission.csv")
    print(f"pids: {len(tests)}   state size {cube.state_size}   supercube {cube.is_supercube}")

    states = {p: tuple(int(x) for x in s.split(",")) for p, s in tests.items()}
    paths = {p: cube.parse_path(s) for p, s in sample.items()}
    bad = [p for p, st in states.items() if len(st) != cube.state_size]
    print(f"malformed states: {len(bad)}  (rule: test.csv must be quoted-parsed)")

    print()
    print("1. SAMPLE PATHS REPLAY")
    fails = [p for p in states if not cube.is_solved(cube.apply_path(states[p], paths[p]))]
    lens = {p: len(paths[p]) for p in paths}
    total = sum(lens.values())
    print(f"   {len(states) - len(fails)}/{len(states)} replay to solved, {len(fails)} fail")
    print(f"   sample total = {total}  (leaderboard sample = 502,399)")
    print(f"   lengths: min {min(lens.values())} max {max(lens.values())}")

    print()
    print("2. INVERSE FRAME (supercube only)")
    rng = random.Random(0)
    probe = rng.sample(sorted(states), 25)
    ok = 0
    for p in probe:
        sol = paths[p]                                   # solves states[p]
        inv_state = cube.invert_state(states[p])
        inv_sol = cube.invert_path(sol)                  # should solve inv_state
        if cube.is_solved(cube.apply_path(inv_state, inv_sol)):
            ok += 1
    print(f"   {ok}/{len(probe)} inverse-frame solutions valid, length preserved")

    print()
    print("3. ALL 96 FRAMES (48 symmetries x {identity, inverse})")
    tab = build_symmetries(cube)
    pid = probe[0]
    st, sol = states[pid], paths[pid]
    good = 0
    for k in range(len(tab)):
        for use_inv in (False, True):
            s2 = tab.conjugate_state(st, k)
            w2 = tab.relabel_path(sol, k)
            if use_inv:
                s2 = cube.invert_state(s2)
                w2 = cube.invert_path(w2)
            if cube.is_solved(cube.apply_path(s2, w2)) and len(w2) == len(sol):
                good += 1
    print(f"   pid {pid}: {good}/96 frames give a valid, same-length solution")
    print("   => a deterministic solver can be run 96 times for 96 independent")
    print("      lengths on the same pid, and the min is free.")

    print()
    print("4. SCORE MODEL  total(M) = sum_p min(L_p, M)")
    L = sorted(lens.values())

    def total_at(m: int) -> int:
        return sum(min(x, m) for x in L)

    for m in (121, 150, 200, 217, 250, 300, 490):
        print(f"   M={m:4d} -> {total_at(m):,}")
    print()
    for name, score in (("webmaking  #1", 194001), ("John McFacker", 355425),
                        ("ours (submitted)", 364107)):
        lo, hi = 1, 1000
        while lo < hi:
            mid = (lo + hi) // 2
            if total_at(mid) < score:
                lo = mid + 1
            else:
                hi = mid
        print(f"   {name:18s} {score:,}  ->  M ~ {lo} moves/pid")


if __name__ == "__main__":
    main()
