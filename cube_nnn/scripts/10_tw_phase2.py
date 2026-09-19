"""Thistlethwaite phase 2: from H = <U,D,L,R,F2,B2> reach G2 = <U,D,L2,R2,F2,B2>.

COORDINATE (all three parts verified, and they transform INDEPENDENTLY)
    corner orientation, U/D convention   3^7  = 2187   (480/480 + 960/960 checked)
    UD-slice occupancy, C(12,4)                = 495
    L and R face-rotation parity               =   4   (supercube part: inside G2
                                                        L/R get only half turns,
                                                        so odd is unrepairable)
    total 2187 * 495 * 4 = 4,330,260  -- exactly BFS-able

Because the three factors are independent, the move action is three small
lookups, exactly as for the corner PDB.

COST MODEL
    Inside H the available moves are U,U',D,D',L,L',R,R' at 1 quarter turn each
    and F2,B2 at 2. BFS with unit cost would minimise H-moves, not moves, so
    this uses Dial's algorithm (bucket queue) to get true QTM-optimal distances.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/10_tw_phase2.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.g1_action import build_action  # noqa: E402
from cube_nnn.tw_coords import TWCoords  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
TAB = ROOT / "cube_nnn" / "tables"
L_FACE, R_FACE = 4, 2
NCO, NSL, NPAR = 2187, 495, 4


def slice_rank_table():
    combos = list(combinations(range(12), 4))
    assert len(combos) == NSL
    rank = {c: i for i, c in enumerate(combos)}
    return combos, rank


def main() -> None:
    cube = NCube.from_puzzle_info(DATA)
    act = build_action(cube)
    tw = TWCoords(cube)
    solved = np.array(cube.solved_state, dtype=np.int16)
    G = {m: np.array(cube.generators[m], dtype=np.int64) for m in act.moves}

    # H moves as (quarter-turn cost, sticker permutation sequence)
    HMOVES = [("U", 1, ["d5"]), ("U'", 1, ["-d5"]), ("D", 1, ["d0"]), ("D'", 1, ["-d0"]),
              ("L", 1, ["r5"]), ("L'", 1, ["-r5"]), ("R", 1, ["r0"]), ("R'", 1, ["-r0"]),
              ("F2", 2, ["f0", "f0"]), ("B2", 2, ["f5", "f5"])]

    # ---- derive each move's action on the three factors --------------------
    csrc, ctw, esrc, dpar = [], [], [], []
    for name, cost, seq in HMOVES:
        st = solved.copy()
        for nm in seq:
            st = st[G[nm]]
        # corner slot source map + twist, read off the solved state
        co = tw.corner_orient(st.reshape(1, -1))[0]
        occ = st[tw.cfirst]
        csrc.append(tw.chome[occ])
        ctw.append(co)
        # edge slot source map
        eocc = st[tw.eref]
        esrc.append(tw.ehome[eocc])
        # L/R rotation parity delta
        from cube_nnn.endgame import face_rotations
        fr = face_rotations(cube, tuple(st))
        dpar.append((fr[L_FACE] % 2) * 2 + (fr[R_FACE] % 2))
    csrc = np.array(csrc, dtype=np.int64)
    ctw = np.array(ctw, dtype=np.int64)
    esrc = np.array(esrc, dtype=np.int64)
    dpar = np.array(dpar, dtype=np.int64)
    print("derived move actions for", [h[0] for h in HMOVES])

    # ---- corner-orientation transition, 2187 -> 2187 -----------------------
    allo = np.arange(NCO, dtype=np.int64)
    co = np.empty((NCO, 8), dtype=np.int64)
    t = allo.copy()
    for i in range(7):
        co[:, i] = t % 3
        t //= 3
    co[:, 7] = (-co[:, :7].sum(axis=1)) % 3
    pw3 = (3 ** np.arange(7)).astype(np.int64)
    co_move = np.empty((len(HMOVES), NCO), dtype=np.int32)
    for m in range(len(HMOVES)):
        nc = (co[:, csrc[m]] + ctw[m]) % 3
        co_move[m] = (nc[:, :7] * pw3).sum(axis=1)

    # ---- UD-slice transition, 495 -> 495 -----------------------------------
    combos, rank = slice_rank_table()
    sl_move = np.empty((len(HMOVES), NSL), dtype=np.int32)
    inv = np.empty_like(esrc)
    for m in range(len(HMOVES)):
        inv[m][esrc[m]] = np.arange(12)
    for m in range(len(HMOVES)):
        for i, c in enumerate(combos):
            sl_move[m][i] = rank[tuple(sorted(int(inv[m][j]) for j in c))]

    # ---- combined BFS with Dial's algorithm --------------------------------
    SIZE = NCO * NSL * NPAR
    dist = np.full(SIZE, 255, dtype=np.uint8)
    goal_sl = rank[tuple(sorted(int(x) for x in tw.ud_slice))]
    goal = (0 * NSL + goal_sl) * NPAR + 0
    dist[goal] = 0
    buckets = {0: np.array([goal], dtype=np.int64)}
    d = 0
    total = 1
    maxd = 0
    while buckets:
        if d not in buckets:
            d += 1
            if d > 60:
                break
            continue
        cur = buckets.pop(d)
        cur = cur[dist[cur] == d]
        if cur.size == 0:
            d += 1
            continue
        c_co, rest = np.divmod(cur, NSL * NPAR)
        c_sl, c_pa = np.divmod(rest, NPAR)
        for m, (name, cost, _) in enumerate(HMOVES):
            n_co = co_move[m][c_co].astype(np.int64)
            n_sl = sl_move[m][c_sl].astype(np.int64)
            n_pa = c_pa ^ dpar[m]
            nxt = (n_co * NSL + n_sl) * NPAR + n_pa
            nd = d + cost
            if nd > 254:
                continue
            fresh = nxt[dist[nxt] > nd]
            if fresh.size:
                fresh = np.unique(fresh)
                fresh = fresh[dist[fresh] > nd]
                dist[fresh] = nd
                buckets[nd] = np.concatenate([buckets.get(nd, np.empty(0, np.int64)), fresh])
                total += fresh.size
                maxd = max(maxd, nd)
        d += 1
    filled = int((dist != 255).sum())
    print(f"phase-2 table: {filled:,} / {SIZE:,} reachable, max QTM depth {maxd}")
    print(f"  mean depth {dist[dist != 255].mean():.2f}")
    np.save(TAB / "tw_phase2.npy", dist)
    np.save(TAB / "tw_phase2_moves.npy",
            np.array([co_move, ], dtype=object), allow_pickle=True)
    print(f"saved {TAB / 'tw_phase2.npy'}")


if __name__ == "__main__":
    main()
