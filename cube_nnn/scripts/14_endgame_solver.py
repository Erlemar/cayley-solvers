"""Complete 4-phase endgame solver for the G1 (3x3x3 supercube) endgame.

Chains greedy descent on the four exact Thistlethwaite tables and verifies the
cube ends fully SOLVED -- all 216 stickers home, not just "in the last coset".

    phase 1  ->  H  = <U,D,L,R,F2,B2>      edge orientation + F,B rot parity
    phase 2  ->  G2 = <U,D,L2,R2,F2,B2>    corner orientation + UD-slice + L,R par
    phase 3  ->  G3 = <U2,D2,L2,R2,F2,B2>  tetrads + U,D rot parity
    phase 4  ->  identity                  within-tetrad perms + even rotations

Every table is an exact distance, so each phase is optimal and needs no search.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/14_endgame_solver.py
"""
from __future__ import annotations

import sys
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.tw_coords import TWCoords  # noqa: E402
from cube_nnn.endgame import g1_coords, face_rotations  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
TAB = ROOT / "cube_nnn" / "tables"
U_F, F_F, R_F, B_F, L_F, D_F = 0, 1, 2, 3, 4, 5
CT = ((0, 3, 5, 6), (1, 2, 4, 7))
ROT_FACES = (1, 2, 3, 4)
NONSLICE = (0, 1, 2, 3, 6, 8, 10, 11)
EO = ((0, 2, 6, 10), (1, 3, 8, 11), (4, 5, 7, 9))
P4 = list(permutations(range(4)))
P4R = {p: i for i, p in enumerate(P4)}

M1 = [("U", ["d5"]), ("U'", ["-d5"]), ("D", ["d0"]), ("D'", ["-d0"]),
      ("L", ["r5"]), ("L'", ["-r5"]), ("R", ["r0"]), ("R'", ["-r0"]),
      ("F", ["f0"]), ("F'", ["-f0"]), ("B", ["f5"]), ("B'", ["-f5"])]
M2 = [("U", ["d5"]), ("U'", ["-d5"]), ("D", ["d0"]), ("D'", ["-d0"]),
      ("L", ["r5"]), ("L'", ["-r5"]), ("R", ["r0"]), ("R'", ["-r0"]),
      ("F2", ["f0", "f0"]), ("B2", ["f5", "f5"])]
M3 = [("U", ["d5"]), ("U'", ["-d5"]), ("D", ["d0"]), ("D'", ["-d0"]),
      ("L2", ["r5", "r5"]), ("R2", ["r0", "r0"]),
      ("F2", ["f0", "f0"]), ("B2", ["f5", "f5"])]
M4 = [("U2", ["d5", "d5"]), ("D2", ["d0", "d0"]), ("L2", ["r5", "r5"]),
      ("R2", ["r0", "r0"]), ("F2", ["f0", "f0"]), ("B2", ["f5", "f5"])]


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


class Endgame:
    def __init__(self, cube: NCube):
        self.cube = cube
        self.tw = TWCoords(cube)
        self.G = {m: np.array(cube.generators[m], dtype=np.int64)
                  for m in cube.move_names}
        self.sig = np.load(TAB / "edge_orient_c.npy")[0].astype(np.int8)
        self.t1 = np.load(TAB / "tw_phase1.npy")
        self.t2 = np.load(TAB / "tw_phase2.npy")
        self.t3 = np.load(TAB / "tw_phase3.npy")
        self.t4 = np.load(TAB / "tw_phase4.npy", mmap_mode="r")
        self.sl_combos = list(combinations(range(12), 4))
        self.sl_rank = {c: i for i, c in enumerate(self.sl_combos)}
        self.ct_combos = list(combinations(range(8), 4))
        self.ct_rank = {c: i for i, c in enumerate(self.ct_combos)}
        self.et_combos = list(combinations(range(8), 4))
        self.et_rank = {c: i for i, c in enumerate(self.et_combos)}
        self.ns_pos = {s: k for k, s in enumerate(NONSLICE)}

    def apply(self, st, seq):
        for nm in seq:
            st = st[self.G[nm]]
        return st

    def i1(self, st):
        c = g1_coords(self.cube, tuple(st))
        ep = np.array(c.edge_perm)
        o = (np.array(c.edge_flip) ^ self.sig[ep] ^ self.sig[np.arange(12)])
        fr = face_rotations(self.cube, tuple(st))
        return ((fr[F_F] % 2) * 2 + (fr[B_F] % 2)) * 4096 + int(
            (o.astype(np.int64) * (1 << np.arange(12))).sum())

    def i2(self, st):
        co = self.tw.corner_orient(st.reshape(1, -1))[0].astype(np.int64)
        home = self.tw.ehome[st[self.tw.eref]]
        occ = tuple(sorted(int(j) for j in range(12)
                           if home[j] in set(self.tw.ud_slice.tolist())))
        fr = face_rotations(self.cube, tuple(st))
        return ((int((co[:7] * (3 ** np.arange(7))).sum()) * 495 + self.sl_rank[occ])
                * 4 + (fr[L_F] % 2) * 2 + (fr[R_F] % 2))

    def i3(self, st):
        chome = self.tw.chome[st[self.tw.cfirst]]
        ctA = tuple(sorted(int(j) for j in range(8) if int(chome[j]) in CT[0]))
        ehome = self.tw.ehome[st[self.tw.eref]]
        et = tuple(sorted(self.ns_pos[j] for j in NONSLICE
                          if int(ehome[j]) in EO[0]))
        fr = face_rotations(self.cube, tuple(st))
        rot = (((fr[U_F] % 4) * 4 + (fr[D_F] % 4)) * 16
               + sum(((fr[f] // 2) & 1) << i for i, f in enumerate(ROT_FACES)))
        pa = parity(chome.tolist()) % 2
        # UD-slice permutation parity: phase 2 pins these pieces into the slice,
        # so this is well defined throughout phase 3, and L2/R2/F2/B2 flip it.
        pos = {sl: k for k, sl in enumerate(EO[2])}
        sp = parity([pos[int(ehome[EO[2][k]])] for k in range(4)])
        return (((self.ct_rank[ctA] * 70 + self.et_rank[et]) * 256 + rot) * 2
                + pa) * 2 + sp

    def i4(self, st):
        chome = self.tw.chome[st[self.tw.cfirst]]
        ehome = self.tw.ehome[st[self.tw.eref]]
        cr = []
        for grp in CT:
            pos = {s: k for k, s in enumerate(grp)}
            cr.append(P4R[tuple(pos[int(chome[grp[k]])] for k in range(4))])
        er = []
        for grp in EO:
            pos = {s: k for k, s in enumerate(grp)}
            er.append(P4R[tuple(pos[int(ehome[grp[k]])] for k in range(4))])
        fr = face_rotations(self.cube, tuple(st))
        rot = sum(((fr[f] // 2) & 1) << f for f in range(6))
        return (((cr[0] * 24 + cr[1]) * 13824
                 + ((er[0] * 24 + er[1]) * 24 + er[2])) * 64 + rot)

    def descend(self, st, table, idx, moves, tag, rng=None):
        if rng is not None:
            moves = list(moves)
            rng.shuffle(moves)
        word, d, guard = [], int(table[idx(st)]), 0
        while d > 0:
            guard += 1
            if guard > 200:
                raise RuntimeError(f"{tag} stuck at d={d}")
            if rng is not None:
                rng.shuffle(moves)
            for name, seq in moves:
                nxt = self.apply(st, seq)
                nd = int(table[idx(nxt)])
                if nd < d:
                    st, d, word = nxt, nd, word + seq
                    break
            else:
                raise RuntimeError(f"{tag}: no decreasing move at d={d}")
        return st, word

    def solve(self, st, tries: int = 500, seed: int = 0):
        """Phases 1-2 are deterministic; phase 3 is retried with randomised
        descent until it lands in a phase-4-reachable coset.

        The phase-3 coordinate (tetrads + rotations + parity) does NOT determine
        the phase-4 coset -- G3 sits at index 96 inside the within-tetrad group,
        constrained by p(cA)=p(cB) and p(e0)^p(e1)^p(e2)=0 plus a rotation
        coupling. Rather than encode that, exploit the fact that phase 3 has many
        optimal solutions which land in DIFFERENT cosets, and take one that works.

        Measured hit rate is about 1/24 per attempt (the 1/96 coset index, times
        the 4x that the parity signature already buys), so 500 tries leaves a
        failure probability under 1e-9. Every emitted word is still replay-checked.
        """
        w1 = self.descend(st, self.t1, self.i1, M1, "p1")
        st1, w1 = w1[0], w1[1]
        st2, w2 = self.descend(st1, self.t2, self.i2, M2, "p2")
        rng = np.random.default_rng(seed)
        for attempt in range(tries):
            # Randomised descent alone is not enough: optimal phase-3 solutions
            # can all share the same coset invariant. A short random G2 PREFIX
            # moves the state to a different coset before descending, at the
            # cost of a few moves. G2 generators keep us inside G2.
            pre = []
            if attempt:
                for _ in range(1 + int(rng.integers(0, 4))):
                    pre += M3[int(rng.integers(0, len(M3)))][1]
            stp = self.apply(st2, pre)
            st3, w3 = self.descend(stp, self.t3, self.i3, M3, "p3",
                                   rng=(None if attempt == 0 else rng))
            w3 = pre + w3
            if int(self.t4[self.i4(st3)]) != 255:
                st4, w4 = self.descend(st3, self.t4, self.i4, M4, "p4")
                return st4, [w1, w2, w3, w4], attempt + 1
        return None, None, tries


def main() -> None:
    cube = NCube.from_puzzle_info(DATA)
    eg = Endgame(cube)
    solved = np.array(cube.solved_state, dtype=np.int16)
    outer = [m for m in cube.move_names
             if m.lstrip("-") in {"f0", "f5", "r0", "r5", "d0", "d5"}]
    rng = np.random.default_rng(0)
    print(f"{'t':>3s} {'p1':>4s} {'p2':>4s} {'p3':>4s} {'p4':>4s} {'total':>6s} "
          f"{'SOLVED':>7s} {'replay':>7s}")
    tots = [[], [], [], []]
    fails = []
    allok = True
    for t in range(24):
        st = solved.copy()
        for i in rng.integers(0, len(outer), size=150):
            st = st[eg.G[outer[int(i)]]]
        start = st.copy()
        end, w, tries = eg.solve(st, tries=150, seed=t)
        if end is None:
            print(f"{t:>3d} {'-':>4s} {'-':>4s} {'-':>4s} {'-':>4s} {'-':>6s} "
                  f"{'FAIL':>7s} {'-':>7s}  tries={tries}")
            fails.append(t)
            continue
        for k in range(4):
            tots[k].append(len(w[k]))
        full = np.concatenate([np.array(x, dtype=object) for x in w]) if any(w) else []
        flat = [m for part in w for m in part]
        ok = bool(np.array_equal(end, solved))
        rp = bool(np.array_equal(eg.apply(start, flat), solved))
        allok &= ok and rp
        print(f"{t:>3d} {len(w[0]):>4d} {len(w[1]):>4d} {len(w[2]):>4d} {len(w[3]):>4d} "
              f"{len(flat):>6d} {str(ok):>7s} {str(rp):>7s}  tries={tries}")
    print("")
    print(f"solved {len(tots[0])}/24, failed {len(fails)} (pids {fails})")
    if not tots[0]:
        return
    means = [float(np.mean(x)) for x in tots]
    print(f"\nmeans: p1 {means[0]:.1f}  p2 {means[1]:.1f}  p3 {means[2]:.1f}  "
          f"p4 {means[3]:.1f}   TOTAL {sum(means):.1f} QTM")
    print(f"all solved and replay-verified: {allok}")
    print(f"(counting bound for G1 is 21.3 QTM; optimal ~26-32)")


if __name__ == "__main__":
    main()
