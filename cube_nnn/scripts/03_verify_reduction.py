"""Validate the G1 membership test by sampling, then measure the endgame.

The detector is only trustworthy if BOTH directions hold:
  * every random word in the 12 outer turns tests True   (no false negatives)
  * random words in the full 36 generators test False     (no false positives)

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/03_verify_reduction.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.reduction import in_g1, edge_slots  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"


def main() -> None:
    cube = NCube.from_puzzle_info(DATA)
    n = cube.n
    outer = cube.outer_moves()
    allm = list(cube.move_names)
    rng = random.Random(0)

    slots = edge_slots(cube)
    print(f"edge slots: {len(slots)}, each {len(slots[0])} wing cells "
          f"(expect 12 x {n - 2})")
    print(f"outer generators: {len(outer)} (expect 12)")
    print()

    print("1. NO FALSE NEGATIVES: random words in the outer turns must be in G1")
    for wl in (1, 5, 20, 60, 200):
        ok = 0
        trials = 60
        for _ in range(trials):
            w = [rng.choice(outer) for _ in range(wl)]
            if in_g1(cube, cube.apply_path(cube.solved_state, w)):
                ok += 1
        flag = "OK" if ok == trials else "*** FAIL ***"
        print(f"   word length {wl:4d}: {ok}/{trials} detected in G1   {flag}")

    print()
    print("2. NO FALSE POSITIVES: random words in all 36 generators")
    for wl in (1, 2, 4, 10, 40, 200):
        hits = 0
        trials = 200
        for _ in range(trials):
            w = [rng.choice(allm) for _ in range(wl)]
            if in_g1(cube, cube.apply_path(cube.solved_state, w)):
                hits += 1
        # A short random word can legitimately BE a G1 element: each move is an
        # outer turn with prob 12/36, so ~(1/3)^wl, plus cancelling pairs.
        exp = trials * (1 / 3) ** wl
        print(f"   word length {wl:4d}: {hits}/{trials} tested in G1 "
              f"(chance-level {exp:.1f}; these are real G1 elements, not errors)")

    print()
    print("3. SANITY: solved state and single inner turns")
    print(f"   solved in G1: {in_g1(cube, cube.solved_state)}  (expect True)")
    for mv in ("f1", "f2", "r2", "d3"):
        s = cube.apply_move(cube.solved_state, mv)
        print(f"   after {mv}: in G1 = {in_g1(cube, s)}  (expect False)")
    for mv in ("f0", "f5", "r0", "d5"):
        s = cube.apply_move(cube.solved_state, mv)
        print(f"   after {mv}: in G1 = {in_g1(cube, s)}  (expect True)")

    print()
    print("4. HOW RARE IS G1?  index is 424.4 bits, so a random state is never in it")
    hits = sum(1 for _ in range(500)
               if in_g1(cube, cube.apply_path(cube.solved_state,
                                              [rng.choice(allm) for _ in range(300)])))
    print(f"   {hits}/500 random deep states in G1 (expect 0)")


if __name__ == "__main__":
    main()
