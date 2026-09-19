"""Rung 1 of the classical ladder: solve ONE FACE's centres. Then measure rung 2.

This is the BIGCUBES_PLAN R5 measurement that "sets everything" and has never
been made: does a rung cost its INCREMENTAL bits or its CUMULATIVE bits?

Why a face and not an orbit. A face's 16 centres are 4 pieces from each of the 4
centre orbits, and an exact k=4 PDB per orbit measures EXACTLY those 4 pieces --
the heuristic components match the target. Partitioning all 24 pieces of one
orbit into 6 groups while targeting all 24 (the earlier attempt) gives
components that each measure the wrong thing, and it stalls.

    face 1 centres : 4 orbits x log2(24*23*22*21) = 72.0 bits = 14.8 moves
    faces 1-2      : 4 x log2(24!/16!) = 139 bits = 28.6 moves (cumulative)

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/08_face_rung.py --width 16384
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.beam import Engine, beam_solve  # noqa: E402
from cube_nnn.pdb import local_generators, build_pdb  # noqa: E402
from cube_nnn.reduction import centre_positions  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
B6 = math.log2(29.023)


def centre_orbits(cube: NCube) -> list[np.ndarray]:
    pieces = cube.pieces()
    centre = {s[0] for s in pieces.values() if len(s) == 1}
    return [np.array(sorted(o), dtype=np.int64) for o in cube.sticker_orbits()
            if set(o) <= centre]


class FaceScorer:
    """Sum / max of exact k-piece PDBs, one per orbit, for a chosen target set."""

    def __init__(self, cube, orbits, targets, mode="sum"):
        self.mode = mode
        self.orbits, self.tables, self.locmaps = [], [], []
        for orb, tgt in zip(orbits, targets):
            if len(tgt) == 0:
                continue
            lg = local_generators(cube, orb)
            loc = {int(g): i for i, g in enumerate(orb)}
            pieces = np.array([loc[int(t)] for t in tgt], dtype=np.int64)
            self.orbits.append(np.asarray(orb))
            self.tables.append(build_pdb(lg, pieces))
            m = -np.ones(cube.state_size, dtype=np.int64)
            m[orb] = np.arange(len(orb))
            self.locmaps.append(m)

    def __call__(self, states):
        vals = []
        for orb, tab, loc in zip(self.orbits, self.tables, self.locmaps):
            occ = loc[states[:, orb]]
            pos = np.empty_like(occ)
            np.put_along_axis(pos, occ, np.broadcast_to(np.arange(orb.size), occ.shape), axis=1)
            vals.append(tab.lookup(pos[:, tab.pieces]).astype(np.int32))
        V = np.stack(vals)
        return V.sum(0) if self.mode == "sum" else V.max(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    cube = NCube.from_puzzle_info(DATA)
    eng = Engine(cube, list(cube.move_names))
    orbits = centre_orbits(cube)
    cpos = centre_positions(cube.n)
    rng = np.random.default_rng(0)
    names = list(cube.move_names)

    def rand_state(L=400):
        st = np.array(cube.solved_state, dtype=np.int16)
        for i in rng.choice(len(names), size=L):
            st = st[np.array(cube.generators[names[int(i)]], dtype=np.int64)]
        return st

    print(f"width {args.width}, {args.trials} trials, max {args.steps} steps")
    print(f"{'rung':26s} {'bits':>6s} {'bound':>6s} {'mode':>4s} {'result':>22s} {'s':>6s}")
    print("-" * 78)

    faces_done: list[int] = []
    for nface in (1, 2, 3):
        faces_done = list(range(nface))
        targets = []
        npieces = 0
        for orb in orbits:
            tgt = [p for f in faces_done for p in cpos[f] if p in set(orb.tolist())]
            targets.append(np.array(tgt, dtype=np.int64))
            npieces += len(tgt)
        per_orbit = len(targets[0])
        bits = 4 * sum(math.log2(24 - i) for i in range(per_orbit))
        bound = bits / B6
        proj = np.concatenate([t for t in targets if len(t)])
        for mode in ("sum", "max"):
            if per_orbit > 6:
                print(f"{'faces ' + str(nface) + f' ({npieces} pieces)':26s} "
                      f"{bits:6.0f} {bound:6.1f} {mode:>4s} "
                      f"{'k>6 PDB too big':>22s} {0:6.0f}")
                continue
            t0 = time.time()
            sc = FaceScorer(cube, orbits, targets, mode=mode)
            lens = []
            for t in range(args.trials):
                st = rand_state()
                r = beam_solve(eng, st, proj, width=args.width,
                               max_steps=args.steps, seed=t, heuristic=sc)
                lens.append(len(r.path) if r.solved else None)
            got = [x for x in lens if x is not None]
            desc = (f"{np.mean(got):.0f} ({min(got)}-{max(got)}) {len(got)}/{args.trials}"
                    if got else f"UNSOLVED 0/{args.trials}")
            print(f"{'faces ' + str(nface) + f' ({npieces} pieces)':26s} "
                  f"{bits:6.0f} {bound:6.1f} {mode:>4s} {desc:>22s} "
                  f"{time.time() - t0:6.0f}", flush=True)


if __name__ == "__main__":
    main()
