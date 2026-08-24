"""Endgame solver: validate PDB heuristics, then measure the beam horizon.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/07_endgame_pdb.py --width 4096
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
from cube_nnn.g1_action import build_action, outer_moves  # noqa: E402
from cube_nnn.g1_heuristic import CornerPDB, build_edge_pdb  # noqa: E402
from cube_nnn.endgame import piece_mismatch_heuristic  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
TAB = ROOT / "cube_nnn" / "tables" / "corner_pdb.npy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    cube = NCube.from_puzzle_info(DATA)
    act = build_action(cube)
    moves = list(act.moves)
    eng = Engine(cube, moves)
    solved = np.array(cube.solved_state, dtype=np.int16)
    rng = np.random.default_rng(0)

    cp = CornerPDB.load(cube, TAB)
    print(f"corner PDB: {cp.table.size:,} entries, max depth {cp.table.max()}")
    print(f"  h(solved) = {cp(solved.reshape(1, -1))[0]}  (expect 0)")

    t0 = time.time()
    egroups = [np.array(g) for g in ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11])]
    epdbs = [build_edge_pdb(cube, act, g) for g in egroups]
    print(f"edge PDBs: 3 x k=4, {epdbs[0].table.size:,} entries each, "
          f"max depths {[int(e.table.max()) for e in epdbs]}, {time.time() - t0:.0f}s")
    for e in epdbs:
        assert e(solved.reshape(1, -1))[0] == 0, "edge PDB h(solved) != 0"
    print("  all edge PDBs h(solved) = 0")

    # calibration: h vs scramble depth
    print()
    print("CALIBRATION (mean h by scramble depth in the 12 outer moves)")
    print(f"  {'scr':>4s} {'corner':>7s} {'edgeMax':>8s} {'sumAll':>7s}")
    for j in (2, 4, 6, 8, 12, 16, 24, 40):
        sts = []
        for _ in range(30):
            w = [moves[i] for i in rng.integers(0, 12, size=j)]
            sts.append(cube.apply_path(tuple(solved), w))
        S = np.array(sts, dtype=np.int16)
        c = cp(S)
        em = np.stack([e(S) for e in epdbs]).max(0)
        sm = c + np.stack([e(S) for e in epdbs]).sum(0)
        print(f"  {j:>4d} {c.mean():7.2f} {em.mean():8.2f} {sm.mean():7.2f}")

    print()
    print(f"BEAM HORIZON (width {args.width}, max {args.steps} steps, "
          f"{args.trials} trials)")
    hmis = piece_mismatch_heuristic(cube)

    def h_max(S):
        return np.maximum(cp(S), np.stack([e(S) for e in epdbs]).max(0))

    def h_sum(S):
        return cp(S) + np.stack([e(S) for e in epdbs]).sum(0)

    def h_mix(S):
        return cp(S) * 3 + np.stack([e(S) for e in epdbs]).sum(0)

    proj = np.arange(cube.state_size)
    for tag, h in (("mismatch(ctrl)", hmis), ("max", h_max),
                   ("sum", h_sum), ("3*corner+sum", h_mix)):
        line = []
        for j in (10, 20, 30, 50):
            ok, lens, t0 = 0, [], time.time()
            for t in range(args.trials):
                w = [moves[i] for i in rng.integers(0, 12, size=j)]
                st = np.array(cube.apply_path(tuple(solved), w), dtype=np.int16)
                r = beam_solve(eng, st, proj, width=args.width,
                               max_steps=args.steps, seed=t, heuristic=h)
                if r.solved:
                    ok += 1
                    lens.append(len(r.path))
            ml = f"{np.mean(lens):.0f}" if lens else "-"
            line.append(f"scr{j}:{ok}/{args.trials}({ml})")
        print(f"  {tag:14s} " + "  ".join(line), flush=True)


if __name__ == "__main__":
    main()
