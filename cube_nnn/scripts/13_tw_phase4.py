"""Thistlethwaite phase 4: inside G3 = <U2,D2,L2,R2,F2,B2>, finish the cube.

Everything left is a permutation WITHIN the derived tetrads/orbits, plus the six
even face rotations:

    corner perm within the 2 tetrads   4! * 4!        =     576
    edge perm within the 3 orbits      4!^3           =  13,824
    face rotations, each in {0,2}      2^6            =      64
                                             product  = 509,607,936

Only a few percent is reachable (parity constraints), but the dense index keeps
the transition arithmetic trivial. Every move is a half turn costing 2 QTM, so
plain BFS in half turns gives QTM distance = 2 * depth.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/13_tw_phase4.py
"""
from __future__ import annotations

import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.tw_coords import TWCoords  # noqa: E402
from cube_nnn.endgame import face_rotations  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
TAB = ROOT / "cube_nnn" / "tables"

CT = ((0, 3, 5, 6), (1, 2, 4, 7))
EO = ((0, 2, 6, 10), (1, 3, 8, 11), (4, 5, 7, 9))
G3MOVES = [("U2", ["d5", "d5"]), ("D2", ["d0", "d0"]), ("L2", ["r5", "r5"]),
           ("R2", ["r0", "r0"]), ("F2", ["f0", "f0"]), ("B2", ["f5", "f5"])]

P4 = list(permutations(range(4)))
P4R = {p: i for i, p in enumerate(P4)}


def main() -> None:
    cube = NCube.from_puzzle_info(DATA)
    tw = TWCoords(cube)
    G = {m: np.array(cube.generators[m], dtype=np.int64) for m in cube.move_names}
    solved = np.array(cube.solved_state, dtype=np.int16)
    NM = len(G3MOVES)

    csrc, esrc, drot = [], [], []
    for name, seq in G3MOVES:
        st = solved.copy()
        for nm in seq:
            st = st[G[nm]]
        csrc.append(tw.chome[st[tw.cfirst]])
        esrc.append(tw.ehome[st[tw.eref]])
        fr = face_rotations(cube, tuple(st))
        drot.append(sum(((fr[f] // 2) & 1) << f for f in range(6)))
    csrc = np.array(csrc, dtype=np.int64)
    esrc = np.array(esrc, dtype=np.int64)
    drot = np.array(drot, dtype=np.int64)

    # within-group permutation action for each move
    def group_move(src, groups, gsize=4):
        """For each move, the induced permutation index map on each group."""
        inv = np.empty_like(src)
        for m in range(NM):
            inv[m][src[m]] = np.arange(src.shape[1])
        out = []
        for gi, grp in enumerate(groups):
            pos = {s: k for k, s in enumerate(grp)}
            tabs = np.empty((NM, 24), dtype=np.int32)
            for m in range(NM):
                # slot grp[k] receives what was at src[m][grp[k]]
                perm = [pos[int(src[m][grp[k]])] for k in range(gsize)]
                for i, p in enumerate(P4):
                    tabs[m][i] = P4R[tuple(p[perm[k]] for k in range(gsize))]
            out.append(tabs)
        return out

    cmove = group_move(csrc, CT)
    emove = group_move(esrc, EO)
    print("derived within-group actions for", [m[0] for m in G3MOVES])

    NC, NE, NR = 576, 13824, 64
    SIZE = NC * NE * NR
    print(f"index space {SIZE:,} ({SIZE / 1e6:.0f} MB table)")
    dist = np.full(SIZE, 255, dtype=np.uint8)
    ident = P4R[(0, 1, 2, 3)]
    goal = ((ident * 24 + ident) * NE
            + ((ident * 24 + ident) * 24 + ident)) * NR + 0
    dist[goal] = 0
    frontier = np.array([goal], dtype=np.int64)
    depth, total = 0, 1
    while frontier.size:
        depth += 1
        rest, rot = np.divmod(frontier, NR)
        cpart, epart = np.divmod(rest, NE)
        cA, cB = np.divmod(cpart, 24)
        e01, e2 = np.divmod(epart, 24)
        e0, e1 = np.divmod(e01, 24)
        nxt = []
        for m in range(NM):
            n = ((((cmove[0][m][cA].astype(np.int64) * 24 + cmove[1][m][cB]) * NE)
                  + ((emove[0][m][e0].astype(np.int64) * 24 + emove[1][m][e1]) * 24
                     + emove[2][m][e2])) * NR) + (rot ^ drot[m])
            n = n[dist[n] == 255]
            if n.size:
                n = np.unique(n)
                n = n[dist[n] == 255]
                dist[n] = depth
                nxt.append(n)
        if not nxt:
            break
        frontier = np.concatenate(nxt)
        total += frontier.size
        print(f"  half-turn depth {depth:2d} (QTM {2 * depth:2d}): "
              f"+{frontier.size:>9,}  total {total:>10,}", flush=True)
    filled = int((dist != 255).sum())
    md = int(dist[dist != 255].max())
    mean = float(dist[dist != 255].mean())
    print(f"\nphase-4 table: {filled:,} reachable, max {md} half turns "
          f"= {2 * md} QTM, mean {2 * mean:.2f} QTM")
    np.save(TAB / "tw_phase4.npy", dist)
    print(f"saved {TAB / 'tw_phase4.npy'}")


if __name__ == "__main__":
    main()
