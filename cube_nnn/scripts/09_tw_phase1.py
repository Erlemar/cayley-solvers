"""Thistlethwaite phase 1 for the G1 endgame: reach H = <U, D, L, R, F2, B2>.

EDGE ORIENTATION, derived not assumed
-------------------------------------
On a 6x6 the wings are ORIENTATION-FREE (2 piece-orbits of 24, pure
permutation), so the 3x3 "edge flip" is really a wing permutation and the
standard per-slot-facelet convention does not transfer -- all 4096 such
assignments fail. The invariant was instead derived by asking, for every
(home group h, slot i), which flips are reachable inside H:

    0 of 144 pairs admit BOTH flips  ->  a separable invariant exists
    c(h,i) is separable              ->  c(h,i) = sigma(h) XOR sigma(i)
    sigma = [0,1,1,0,0,1,1,1,0,0,0,1]

    o(piece) = flip XOR sigma(home) XOR sigma(slot)      o(solved) = 0

PHASE 1 TARGET
--------------
All o = 0, AND the F and B face rotations even: inside H the only F/B moves are
half turns, which shift those rotations by 2, so odd values can never be
repaired later. That is the supercube part, and forgetting it is trap 2 in
BIGCUBES_PLAN.

    coordinate = 12 orientation bits x F parity x B parity = 16,384 states

Small enough to BFS exactly, so phase 1 becomes optimal table lookup.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/09_tw_phase1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.g1_action import build_action  # noqa: E402
from cube_nnn.endgame import g1_coords  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
TAB = ROOT / "cube_nnn" / "tables"
F_FACE, B_FACE = 1, 3


def sigma_from_c(c: np.ndarray) -> np.ndarray:
    return (c[0] ^ c[0][0]).astype(np.int8)


def orient_by_slot(coords, sig) -> np.ndarray:
    ep = np.array(coords.edge_perm)
    ef = np.array(coords.edge_flip)
    return (ef ^ sig[ep] ^ sig[np.arange(12)]).astype(np.int8)


def main() -> None:
    cube = NCube.from_puzzle_info(DATA)
    act = build_action(cube)
    c = np.load(TAB / "edge_orient_c.npy")
    sig = sigma_from_c(c)
    print(f"sigma = {list(map(int, sig))}")

    solved = tuple(cube.solved_state)
    assert not orient_by_slot(g1_coords(cube, solved), sig).any(), "o(solved) != 0"

    # ---- verify the invariant on random states, per PIECE, under H ----------
    rng = np.random.default_rng(0)
    G = {m: np.array(cube.generators[m], dtype=np.int64) for m in act.moves}
    Hm = {"U": ["d5"], "Ui": ["-d5"], "D": ["d0"], "Di": ["-d0"],
          "R": ["r0"], "Ri": ["-r0"], "L": ["r5"], "Li": ["-r5"],
          "F2": ["f0", "f0"], "B2": ["f5", "f5"]}

    def by_piece(st):
        co = g1_coords(cube, tuple(st))
        o = orient_by_slot(co, sig)
        out = np.zeros(12, dtype=np.int8)
        out[np.array(co.edge_perm)] = o
        return out

    bad = ok = 0
    for _ in range(150):
        st = np.array(solved, dtype=np.int16)
        for i in rng.integers(0, len(act.moves), size=int(rng.integers(1, 30))):
            st = st[G[act.moves[int(i)]]]
        base = by_piece(st)
        for seq in Hm.values():
            t = st.copy()
            for nm in seq:
                t = t[G[nm]]
            if np.array_equal(base, by_piece(t)):
                ok += 1
            else:
                bad += 1
    print(f"invariance under H: {ok} ok / {bad} bad   (bad must be 0)")
    assert bad == 0, "edge orientation is not H-invariant"

    # ---- coordinate action -------------------------------------------------
    # o_by_slot_new[j] = o_by_slot_old[esrc[j]] XOR eflip[j] XOR sig[esrc[j]] XOR sig[j]
    esrc = act.esrc.astype(np.int64)
    tog = (act.eflip ^ sig[esrc] ^ sig[np.arange(12)][None, :]).astype(np.int8)

    SIZE = 4096 * 4
    bits = ((np.arange(4096)[:, None] >> np.arange(12)) & 1).astype(np.int8)
    trans = np.empty((12, SIZE), dtype=np.int32)
    pw = (1 << np.arange(12)).astype(np.int64)
    for m in range(12):
        nb = bits[:, esrc[m]] ^ tog[m]
        no = (nb.astype(np.int64) * pw).sum(axis=1)
        dF = int(act.dfr[m][F_FACE]) % 2
        dB = int(act.dfr[m][B_FACE]) % 2
        for fp in (0, 1):
            for bp in (0, 1):
                src = (fp * 2 + bp) * 4096 + np.arange(4096)
                dst = (((fp ^ dF) * 2 + (bp ^ dB)) * 4096) + no
                trans[m][src] = dst

    # ---- BFS from the TARGET SET (o all zero, F/B parities even) -----------
    table = np.full(SIZE, 255, dtype=np.uint8)
    goal = np.array([0 * 4096 + 0], dtype=np.int64)   # o=0, Fpar=0, Bpar=0
    table[goal] = 0
    frontier, depth, total = goal, 0, 1
    while frontier.size:
        depth += 1
        nxt = []
        for m in range(12):
            s = trans[m][frontier]
            s = s[table[s] == 255]
            if s.size:
                s = np.unique(s)
                s = s[table[s] == 255]
                table[s] = depth
                nxt.append(s)
        if not nxt:
            break
        frontier = np.concatenate(nxt)
        total += frontier.size
        print(f"  depth {depth:2d}: {frontier.size:>6,}  total {total:>6,}")
    filled = int((table != 255).sum())
    print(f"\nphase-1 table: {filled:,} / {SIZE:,} reachable, max depth "
          f"{int(table[table != 255].max())}")
    print(f"(2^11 * 4 = {2048 * 4:,} expected if edge orientation has even parity)")
    np.save(TAB / "tw_phase1.npy", table)
    np.save(TAB / "tw_phase1_trans.npy", trans)
    print(f"saved {TAB / 'tw_phase1.npy'}")


if __name__ == "__main__":
    main()
