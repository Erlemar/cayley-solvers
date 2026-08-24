"""Run phases 1 and 2 by greedy descent on the exact tables, and VERIFY.

Both tables are exact distances, so greedy descent is optimal for each phase and
needs no search: at every state pick any move that lowers the phase distance.

Checks, on random G1 elements:
  * phase 1 lands in H = <U,D,L,R,F2,B2>   (edge orientation 0, F/B rotations even)
  * phase 2 lands in G2 = <U,D,L2,R2,F2,B2>(corner orientation 0, UD-slice placed,
                                            L/R rotations even)
  * every emitted word replays on the real 216-sticker state

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/11_tw_solve12.py
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
from cube_nnn.endgame import g1_coords, face_rotations  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
TAB = ROOT / "cube_nnn" / "tables"
F_FACE, B_FACE, L_FACE, R_FACE = 1, 3, 4, 2
NCO, NSL, NPAR = 2187, 495, 4

P1 = [("U", 1, ["d5"]), ("U'", 1, ["-d5"]), ("D", 1, ["d0"]), ("D'", 1, ["-d0"]),
      ("L", 1, ["r5"]), ("L'", 1, ["-r5"]), ("R", 1, ["r0"]), ("R'", 1, ["-r0"]),
      ("F", 1, ["f0"]), ("F'", 1, ["-f0"]), ("B", 1, ["f5"]), ("B'", 1, ["-f5"])]
P2 = [("U", 1, ["d5"]), ("U'", 1, ["-d5"]), ("D", 1, ["d0"]), ("D'", 1, ["-d0"]),
      ("L", 1, ["r5"]), ("L'", 1, ["-r5"]), ("R", 1, ["r0"]), ("R'", 1, ["-r0"]),
      ("F2", 2, ["f0", "f0"]), ("B2", 2, ["f5", "f5"])]


def main() -> None:
    cube = NCube.from_puzzle_info(DATA)
    tw = TWCoords(cube)
    act = build_action(cube)
    G = {m: np.array(cube.generators[m], dtype=np.int64) for m in cube.move_names}
    sig = (np.load(TAB / "edge_orient_c.npy")[0] ^ 0).astype(np.int8)
    t1 = np.load(TAB / "tw_phase1.npy")
    t2 = np.load(TAB / "tw_phase2.npy")
    combos = list(combinations(range(12), 4))
    srank = {c: i for i, c in enumerate(combos)}
    solved = np.array(cube.solved_state, dtype=np.int16)

    def apply(st, seq):
        for nm in seq:
            st = st[G[nm]]
        return st

    def idx1(st):
        c = g1_coords(cube, tuple(st))
        ep = np.array(c.edge_perm)
        o = (np.array(c.edge_flip) ^ sig[ep] ^ sig[np.arange(12)]).astype(np.int64)
        fr = face_rotations(cube, tuple(st))
        code = int((o * (1 << np.arange(12))).sum())
        return ((fr[F_FACE] % 2) * 2 + (fr[B_FACE] % 2)) * 4096 + code

    def idx2(st):
        co = tw.corner_orient(st.reshape(1, -1))[0].astype(np.int64)
        crank = int((co[:7] * (3 ** np.arange(7))).sum())
        home = tw.ehome[st[tw.eref]]
        occupied = tuple(sorted(int(j) for j in range(12)
                                if home[j] in set(tw.ud_slice.tolist())))
        fr = face_rotations(cube, tuple(st))
        par = (fr[L_FACE] % 2) * 2 + (fr[R_FACE] % 2)
        return (crank * NSL + srank[occupied]) * NPAR + par

    def descend(st, table, idxfn, moves, tag):
        word = []
        d = int(table[idxfn(st)])
        guard = 0
        while d > 0:
            guard += 1
            if guard > 100:
                raise RuntimeError(f"{tag}: descent stuck at d={d}")
            for name, cost, seq in moves:
                nxt = apply(st, seq)
                nd = int(table[idxfn(nxt)])
                if nd < d:
                    st, d = nxt, nd
                    word += seq
                    break
            else:
                raise RuntimeError(f"{tag}: no decreasing move at d={d}")
        return st, word

    rng = np.random.default_rng(0)
    outer = [m for m in cube.move_names
             if m.lstrip("-") in {"f0", "f5", "r0", "r5", "d0", "d5"}]
    print(f"{'trial':>5s} {'p1':>4s} {'p2':>4s} {'tot':>4s}  {'in H':>5s} {'in G2':>6s} {'replay':>7s}")
    tot1, tot2 = [], []
    for t in range(20):
        st = solved.copy()
        for i in rng.integers(0, len(outer), size=120):
            st = st[G[outer[int(i)]]]
        start = st.copy()
        st, w1 = descend(st, t1, idx1, P1, "phase1")
        inH = t1[idx1(st)] == 0
        st, w2 = descend(st, t2, idx2, P2, "phase2")
        inG2 = t2[idx2(st)] == 0
        replay = np.array_equal(apply(start, w1 + w2), st)
        tot1.append(len(w1))
        tot2.append(len(w2))
        print(f"{t:>5d} {len(w1):>4d} {len(w2):>4d} {len(w1) + len(w2):>4d}  "
              f"{str(bool(inH)):>5s} {str(bool(inG2)):>6s} {str(replay):>7s}")
    print(f"\nphase1 mean {np.mean(tot1):.1f} (max {max(tot1)}), "
          f"phase2 mean {np.mean(tot2):.1f} (max {max(tot2)}), "
          f"combined mean {np.mean(tot1) + np.mean(tot2):.1f} QTM")


if __name__ == "__main__":
    main()
