"""Validate G1 coordinates, then probe how deep a beam solves the endgame.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/05_endgame.py --width 4096
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.beam import Engine, beam_solve  # noqa: E402
from cube_nnn.reduction import in_g1  # noqa: E402
from cube_nnn.endgame import g1_coords, piece_mismatch_heuristic  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    cube = NCube.from_puzzle_info(DATA)
    outer = [m for m in cube.move_names if m.lstrip("-") in
             {"f0", "f5", "r0", "r5", "d0", "d5"}]
    assert len(outer) == 12, outer
    eng = Engine(cube, outer)
    solved = np.array(cube.solved_state, dtype=np.int16)
    rng = np.random.default_rng(0)

    print("1. COORDINATE EXTRACTION")
    c0 = g1_coords(cube, tuple(solved))
    print(f"   solved coords is_solved={c0.is_solved()}  (expect True)")
    print(f"   corners {len(c0.corner_perm)}  edges {len(c0.edge_perm)}  "
          f"faces {len(c0.face_rot)}  (expect 8 / 12 / 6)")
    # every G1 element must yield valid coords, and only solved is solved
    bad = 0
    for _ in range(200):
        w = [outer[i] for i in rng.integers(0, 12, size=rng.integers(1, 40))]
        st = cube.apply_path(tuple(solved), w)
        assert in_g1(cube, st)
        c = g1_coords(cube, st)
        if -1 in c.face_rot or sorted(c.corner_perm) != list(range(8)):
            bad += 1
        if c.is_solved() and tuple(st) != tuple(solved):
            bad += 1
    print(f"   200 random G1 elements: {bad} malformed/false-solved (expect 0)")

    print()
    print("2. A SINGLE OUTER TURN CHANGES THE FACE ROTATION (supercube part)")
    for mv in ("f0", "r0", "d5"):
        c = g1_coords(cube, cube.apply_move(tuple(solved), mv))
        print(f"   after {mv}: face_rot {c.face_rot}  corners moved "
              f"{sum(1 for i, p in enumerate(c.corner_perm) if i != p)}")

    print()
    print(f"3. BEAM HORIZON  (12 outer moves, width {args.width}, "
          f"max {args.steps} steps, {args.trials} trials)")
    h = piece_mismatch_heuristic(cube)
    proj = np.arange(cube.state_size)
    print(f"   {'scramble':>9s} {'solved':>7s} {'mean len':>9s} {'best_h':>7s} {'s':>6s}")
    for j in (5, 10, 15, 20, 25, 30, 40, 60):
        lens, ok, bh, t0 = [], 0, [], time.time()
        for t in range(args.trials):
            w = [outer[i] for i in rng.integers(0, 12, size=j)]
            st = cube.apply_path(tuple(solved), w)
            r = beam_solve(eng, np.array(st, dtype=np.int16), proj,
                           width=args.width, max_steps=args.steps, seed=t, heuristic=h)
            if r.solved:
                ok += 1
                lens.append(len(r.path))
            else:
                bh.append(r.best_h)
        ml = f"{np.mean(lens):.1f}" if lens else "-"
        mb = f"{np.mean(bh):.1f}" if bh else "-"
        print(f"   {j:>9d} {ok}/{args.trials:<5d} {ml:>9s} {mb:>7s} {time.time() - t0:6.0f}")


if __name__ == "__main__":
    main()
