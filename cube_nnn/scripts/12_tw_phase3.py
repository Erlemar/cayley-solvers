"""Thistlethwaite phase 3: from G2 = <U,D,L2,R2,F2,B2> reach G3 = <U2,D2,L2,R2,F2,B2>.

G3's structure was DERIVED by random walk inside it, not assumed:

    corner tetrads   {0,3,5,6} and {1,2,4,7}
    edge orbits      {0,2,6,10}, {1,3,8,11}, {4,5,7,9}   (the last IS the UD-slice)
    face rotations   every face in {0,2} only, all 64 combinations
                     -> half turns can never reach an odd rotation, so U and D
                        rotations must already be even on entering G3

COORDINATE (four independent factors)
    corner tetrad assignment    C(8,4)                          =  70
    edge tetrad among the 8 NON-slice slots  C(8,4)             =  70
        (phase 2 already pins the UD-slice, so the full
         12!/(4!4!4!) coordinate is unnecessary)
    rotation code   U,D in Z/4 (16) x L,R,F,B in {0,2} (16)     = 256
    global permutation parity                                   =   2
                                                          total = 2,508,800

    ROTATIONS MUST REACH 0, NOT MERELY EVEN. Phase 4's BFS reaches only 8 of
    the 64 even-rotation codes, so leaving the rotations at e.g. (2,2,2,2,2,2)
    lands outside G3's orbit and phase 4 reads distance 255. G2 can fix this:
    U/D quarter turns move those rotations by 1, L2/R2/F2/B2 move theirs by 2.

Moves are G2's generators: U,U',D,D' at cost 1 and L2,R2,F2,B2 at cost 2, so
distances are computed with Dial's algorithm to stay QTM-exact.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/12_tw_phase3.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.tw_coords import TWCoords  # noqa: E402
from cube_nnn.endgame import face_rotations  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
TAB = ROOT / "cube_nnn" / "tables"
U_FACE, D_FACE = 0, 5
ROT_FACES = (1, 2, 3, 4)   # F, R, B, L -- even-only under G2
NONSLICE = (0, 1, 2, 3, 6, 8, 10, 11)

CT_A = (0, 3, 5, 6)                      # corner tetrad A
EO = ((0, 2, 6, 10), (1, 3, 8, 11), (4, 5, 7, 9))   # edge orbits

G2MOVES = [("U", 1, ["d5"]), ("U'", 1, ["-d5"]), ("D", 1, ["d0"]), ("D'", 1, ["-d0"]),
           ("L2", 2, ["r5", "r5"]), ("R2", 2, ["r0", "r0"]),
           ("F2", 2, ["f0", "f0"]), ("B2", 2, ["f5", "f5"])]


def parity(p) -> int:
    p = list(p)
    seen = [False] * len(p)
    tr = 0
    for i in range(len(p)):
        if seen[i]:
            continue
        j, ln = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            ln += 1
        tr += ln - 1
    return tr % 2


def main() -> None:
    cube = NCube.from_puzzle_info(DATA)
    tw = TWCoords(cube)
    G = {m: np.array(cube.generators[m], dtype=np.int64) for m in cube.move_names}
    solved = np.array(cube.solved_state, dtype=np.int16)

    ct_combos = list(combinations(range(8), 4))
    ct_rank = {c: i for i, c in enumerate(ct_combos)}
    NCT = len(ct_combos)

    # edge orbit assignment: label each slot 0/1/2, rank the multiset arrangement
    eo_label = np.empty(12, dtype=np.int64)
    for k, grp in enumerate(EO):
        for j in grp:
            eo_label[j] = k
    et_combos = list(combinations(range(8), 4))
    et_rank = {c: i for i, c in enumerate(et_combos)}
    NEO = len(et_combos)
    ns_pos = {s: k for k, s in enumerate(NONSLICE)}
    NROT = 256
    print(f"coordinate sizes: corner tetrad {NCT}, edge tetrad {NEO}, "
          f"rot {NROT}, total {NCT * NEO * NROT * 2:,}")

    # per-move slot permutations, read off the solved state
    csrc, esrc, dpar, mpar, spar = [], [], [], [], []
    for name, cost, seq in G2MOVES:
        st = solved.copy()
        for nm in seq:
            st = st[G[nm]]
        occ = st[tw.cfirst]
        cs = tw.chome[occ]
        csrc.append(cs)
        eocc = st[tw.eref]
        es = tw.ehome[eocc]
        esrc.append(es)
        fr = face_rotations(cube, tuple(st))
        dpar.append((fr[U_FACE] % 4, fr[D_FACE] % 4,
                     tuple((fr[f] // 2) & 1 for f in ROT_FACES)))
        # CORNER parity alone, not the sum. Every G2 move has even TOTAL
        # parity, so the sum is constant 0 and carries no information --
        # while G3 needs parity(corners)=0 on its own, and U/D flip it.
        mpar.append(parity(cs) % 2)
        # induced permutation on the 4 UD-slice slots (G2 preserves the slice)
        pos = {sl: k for k, sl in enumerate(EO[2])}
        spar.append(parity([pos[int(es[EO[2][k]])] for k in range(4)]))
    csrc = np.array(csrc, dtype=np.int64)
    esrc = np.array(esrc, dtype=np.int64)
    mpar = np.array(mpar, dtype=np.int64)
    spar = np.array(spar, dtype=np.int64)
    print('slice-parity flip per move:', dict(zip([m[0] for m in G2MOVES], spar.tolist())))
    print("move parities:", dict(zip([m[0] for m in G2MOVES], mpar.tolist())))

    NM = len(G2MOVES)
    cinv = np.empty_like(csrc)
    einv = np.empty_like(esrc)
    for m in range(NM):
        cinv[m][csrc[m]] = np.arange(8)
        einv[m][esrc[m]] = np.arange(12)

    ct_move = np.empty((NM, NCT), dtype=np.int32)
    for m in range(NM):
        for i, c in enumerate(ct_combos):
            ct_move[m][i] = ct_rank[tuple(sorted(int(cinv[m][j]) for j in c))]

    eo_move = np.empty((NM, NEO), dtype=np.int32)
    for m in range(NM):
        for i, c in enumerate(et_combos):
            nc = []
            for k in c:
                dst = int(einv[m][NONSLICE[k]])
                nc.append(ns_pos[dst])
            eo_move[m][i] = et_rank[tuple(sorted(nc))]

    # rotation-code transition: code = ((u*4 + d) * 16) + bits(F,R,B,L)
    rot_move = np.empty((NM, 256), dtype=np.int32)
    for m in range(NM):
        du, dd, dbits = dpar[m]
        dmask = sum(b << i for i, b in enumerate(dbits))
        for code in range(256):
            ud, bits_ = divmod(code, 16)
            u, d = divmod(ud, 4)
            rot_move[m][code] = ((((u + du) % 4) * 4 + ((d + dd) % 4)) * 16
                                 + (bits_ ^ dmask))

    SIZE = NCT * NEO * NROT * 2 * 2
    dist = np.full(SIZE, 255, dtype=np.uint8)
    goal_et = et_rank[tuple(sorted(ns_pos[j] for j in EO[0]))]
    # MULTI-SOURCE. Targeting rot == 0 specifically is wrong: only 32 of the
    # 8 x 64 (parity-signature, rotation) pairs are G3-reachable, so for half
    # the signature classes rot == 0 is unreachable and phase 4 reads 255 no
    # matter how phase 3 is retried. Seed every ALL-EVEN rotation instead and
    # let the retry loop pick a rotation compatible with the signature.
    goals = []
    for u in (0, 2):
        for d in (0, 2):
            for bits_ in range(16):
                rot = ((u * 4 + d) * 16) + bits_
                goals.append((((ct_rank[CT_A] * NEO + goal_et) * NROT + rot) * 2 + 0) * 2 + 0)
    goals = np.array(sorted(set(goals)), dtype=np.int64)
    print(f"phase-3 goal set: {goals.size} all-even-rotation targets")
    dist[goals] = 0
    buckets = {0: goals}
    d, total, maxd = 0, 1, 0
    while d <= 60:
        cur = buckets.pop(d, None)
        if cur is None:
            d += 1
            continue
        cur = cur[dist[cur] == d]
        if cur.size == 0:
            d += 1
            continue
        rest, sp = np.divmod(cur, 2)
        rest, pa = np.divmod(rest, 2)
        rest, rp = np.divmod(rest, NROT)
        ct, eo = np.divmod(rest, NEO)
        for m in range(NM):
            nd = d + G2MOVES[m][1]
            if nd > 254:
                continue
            nxt = ((((ct_move[m][ct].astype(np.int64) * NEO + eo_move[m][eo]) * NROT
                     + rot_move[m][rp]) * 2 + (pa ^ mpar[m])) * 2 + (sp ^ spar[m]))
            fresh = nxt[dist[nxt] > nd]
            if fresh.size:
                fresh = np.unique(fresh)
                fresh = fresh[dist[fresh] > nd]
                dist[fresh] = nd
                buckets[nd] = np.concatenate(
                    [buckets.get(nd, np.empty(0, np.int64)), fresh])
                total += fresh.size
                maxd = max(maxd, nd)
        d += 1
    filled = int((dist != 255).sum())
    print(f"phase-3 table: {filled:,} / {SIZE:,} reachable, max QTM depth {maxd}, "
          f"mean {dist[dist != 255].mean():.2f}")
    np.save(TAB / "tw_phase3.npy", dist)
    print(f"saved {TAB / 'tw_phase3.npy'}")


if __name__ == "__main__":
    main()
