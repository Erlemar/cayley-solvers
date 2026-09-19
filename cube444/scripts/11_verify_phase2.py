"""Verify the 4x4x4 -> 3x3x3 reduction map and the classical phase-2 solver.

The frame derivation needs only `twophase.defs/cubie/face` (cheap). The full
solve round-trip needs `twophase.solver`, whose pruning tables take ~30 min to
build on first use -- pass --no-solve to skip that part.

Run:  .venv/Scripts/python.exe cube444/scripts/11_verify_phase2.py [--no-solve]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube444" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from cube444.orbits import is_reduced, is_outer_move, reduction_defects  # noqa: E402
from cube444.phase2 import (  # noqa: E402
    Phase2Solver, build_reduction_map, derive_cubie_groups, opposite_face_pairs,
    outer_gen_face, qtm_cost,
)
from cube444.puzzle import Cube444  # noqa: E402

PUZZLE = ROOT / "cube444" / "data" / "puzzle_info.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-solve", action="store_true",
                    help="skip the parts that need twophase.solver tables")
    ap.add_argument("--n-solve", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=1.0)
    args = ap.parse_args()

    puz = Cube444.load(PUZZLE)
    gens = {k: np.array(v, dtype=np.int64) for k, v in puz.generators.items()}
    solved = np.array(puz.solved_state, dtype=np.int64)
    rng = np.random.default_rng(0)

    print("=== cubie membership ===")
    groups = derive_cubie_groups(gens)
    print(f"  corners {len(groups['corner'])}x3  wings {len(groups['wing'])}x2  "
          f"centers {len(groups['center'])}x1")
    axes = opposite_face_pairs(groups["corner"])
    print(f"  opposite face pairs: {axes}")
    for name in sorted(g for g in gens if not g.startswith("-") and is_outer_move(g)):
        print(f"    {name} turns face {outer_gen_face(gens, name)}")

    print("\n=== frame derivation ===")
    rmap = build_reduction_map(gens, solved, verbose=True)

    print("\n=== facelet round trip on reduced states ===")
    from cube444.phase2 import facelet_is_valid
    outer = [g for g in gens if is_outer_move(g)]
    bad = 0
    for _ in range(200):
        s = solved.copy()
        for _ in range(int(rng.integers(1, 40))):
            s = s[gens[outer[rng.integers(len(outer))]]]
        assert bool(is_reduced(s))
        if not facelet_is_valid(rmap.to_facelets(s)):
            bad += 1
    print(f"  200 random outer-walk states: {200 - bad} valid, {bad} invalid")
    assert bad == 0
    print(f"  solved -> {rmap.to_facelets(solved)}")

    print("\n=== move translation ===")
    for letter, gen in sorted(rmap.letter_cw_gen.items()):
        print(f"  {letter} (CW) = {gen}")
    # applying a translated HTM word must equal applying the generators directly
    for letter, gen in rmap.letter_cw_gen.items():
        inv = gen[1:] if gen.startswith("-") else "-" + gen
        s1 = solved[gens[gen]][gens[gen]]
        s2 = solved
        for m in rmap.translate(letter + "2"):
            s2 = s2[gens[m]]
        assert np.array_equal(s1, s2), f"{letter}2 translation mismatch"
        s3 = solved[gens[inv]]
        s4 = solved
        for m in rmap.translate(letter + "3"):
            s4 = s4[gens[m]]
        assert np.array_equal(s3, s4), f"{letter}3 translation mismatch"
    print("  all 6 letters: n=1/2/3 translations replay correctly")

    if args.no_solve:
        print("\n--no-solve: skipping solver round trip")
        return 0

    print(f"\n=== full solve round trip ({args.n_solve} states) ===")
    print("  (first import builds pruning tables -- can take ~30 min)")
    t0 = time.time()
    p2 = Phase2Solver(gens, solved, timeout=args.timeout)
    print(f"  solver ready in {time.time() - t0:.1f}s")

    lens, fails = [], 0
    for i in range(args.n_solve):
        s = solved.copy()
        for _ in range(int(rng.integers(20, 60))):
            s = s[gens[outer[rng.integers(len(outer))]]]
        t1 = time.time()
        word = p2.solve(s)
        if word is None:
            fails += 1
            print(f"  [{i}] UNSOLVABLE (unexpected for an outer-only walk)")
            continue
        final = p2.apply_word(s, word)
        ok = np.array_equal(final, solved)
        lens.append(len(word))
        print(f"  [{i}] scramble_defects={int(reduction_defects(s))} "
              f"qtm={len(word):2d} verify={ok} ({time.time() - t1:.2f}s)")
        assert ok, "phase-2 word did not solve the state"
    if lens:
        print(f"\n  mean QTM {np.mean(lens):.2f}  max {max(lens)}  fails {fails}")
        print("  (3x3x3 QTM God's number is 26; random states average ~21)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
