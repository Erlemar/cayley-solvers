"""Exact corner pattern database for the G1 endgame. 8! * 3^7 = 88,179,840.

This is the "corners exact BFS table" that HANDOFF_666 s6 lists as worth
building and that was never built.

WHY IT IS FAST
--------------
The move acts on the two coordinates independently:

    ncp = cp[csrc[m]]                 depends only on the permutation
    nco = (co[csrc[m]] + ctw[m]) % 3  depends only on the orientation

so the whole transition is two lookups into tables of size 12 x 40,320 and
12 x 2,187. No per-state decode/encode, and the BFS is a couple of gathers per
move per level.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/06_build_corner_pdb.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.g1_action import build_action, perm_rank, perm_unrank  # noqa: E402
from cube_nnn.endgame import g1_coords  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
OUT = ROOT / "cube_nnn" / "tables"
NPERM, NORI = 40320, 2187


def check_orientation_sum(cube: NCube, act, trials: int = 300) -> bool:
    rng = np.random.default_rng(0)
    solved = tuple(cube.solved_state)
    for _ in range(trials):
        w = [act.moves[i] for i in rng.integers(0, 12, size=int(rng.integers(0, 30)))]
        c = g1_coords(cube, cube.apply_path(solved, w))
        if sum(c.corner_orient) % 3 != 0:
            return False
    return True


def build_move_tables(act):
    """perm_move (12, 40320) and ori_move (12, 2187)."""
    all_pr = np.arange(NPERM, dtype=np.int64)
    all_perm = perm_unrank(all_pr, 8)                       # (40320, 8)
    perm_move = np.empty((12, NPERM), dtype=np.int32)
    for m in range(12):
        perm_move[m] = perm_rank(all_perm[:, act.csrc[m]])

    all_or = np.arange(NORI, dtype=np.int64)
    co = np.empty((NORI, 8), dtype=np.int8)
    t = all_or.copy()
    for i in range(7):
        co[:, i] = (t % 3).astype(np.int8)
        t //= 3
    co[:, 7] = (-co[:, :7].sum(axis=1)) % 3
    ori_move = np.empty((12, NORI), dtype=np.int32)
    pw = (3 ** np.arange(7)).astype(np.int64)
    for m in range(12):
        nco = (co[:, act.csrc[m]] + act.ctwist[m]) % 3
        ori_move[m] = (nco[:, :7].astype(np.int64) * pw).sum(axis=1)
    return perm_move, ori_move


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cube = NCube.from_puzzle_info(DATA)
    act = build_action(cube)
    print(f"corner orientation sum == 0 mod 3: {check_orientation_sum(cube, act)}"
          "  (justifies indexing 3^7, not 3^8)")

    t0 = time.time()
    perm_move, ori_move = build_move_tables(act)
    print(f"move tables built in {time.time() - t0:.1f}s  "
          f"perm {perm_move.shape} ori {ori_move.shape}")

    size = NPERM * NORI
    table = np.full(size, 255, dtype=np.uint8)
    solved_c = g1_coords(cube, tuple(cube.solved_state))
    pr = int(perm_rank(np.array([solved_c.corner_perm], dtype=np.int8))[0])
    orank = int(sum(o * 3 ** i for i, o in enumerate(solved_c.corner_orient[:7])))
    start = pr * NORI + orank
    table[start] = 0
    frontier = np.array([start], dtype=np.int64)
    total, depth, t0 = 1, 0, time.time()
    print(f"BFS over {size:,} corner states")
    while frontier.size:
        depth += 1
        pr_f, or_f = np.divmod(frontier, NORI)
        outs = []
        for m in range(12):
            outs.append(perm_move[m][pr_f].astype(np.int64) * NORI + ori_move[m][or_f])
        cand = np.unique(np.concatenate(outs))
        cand = cand[table[cand] == 255]
        if cand.size == 0:
            break
        table[cand] = min(depth, 254)
        total += cand.size
        print(f"  depth {depth:2d}: {cand.size:>12,}   total {total:>12,} "
              f"({100 * total / size:5.1f}%)  {time.time() - t0:6.0f}s", flush=True)
        frontier = cand

    filled = int((table != 255).sum())
    print(f"\nfilled {filled:,} / {size:,}   max depth {int(table[table != 255].max())}")
    np.save(OUT / "corner_pdb.npy", table)
    print(f"saved {OUT / 'corner_pdb.npy'}  ({table.nbytes / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
