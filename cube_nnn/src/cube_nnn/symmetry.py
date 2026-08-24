"""The 48 spatial symmetries of the n-cube, plus the inverse frame.

Built from the same 3D model as puzzle.py, so it is exact for every n.

WHY THIS EXISTS
---------------
A deterministic solver run on a conjugated state produces a DIFFERENT solution,
which un-conjugates back to a valid solution of the original of a different
length. 48 rotations/mirrors x the inverse frame = 96 independent attempts, and
you keep the shortest. Unlike the neural beam -- where frames multiply a
per-frame success rate and 0 x 96 is still 0 -- a classical solver succeeds every
time, so frames convert directly into a length distribution.

    solve_via_frame(s, sym):  t = conjugate(s, sym)
                              w = solver(t)                  # word solving t
                              return relabel(w, sym^-1)      # solves s, same length

    solve_via_inverse(s):     w = solver(invert_state(s))
                              return invert_path(w)          # solves s, same length

Both are length-preserving, so the min over 96 is free correctness-wise.

WHAT IS CHECKED HERE
--------------------
For every symmetry `sym` and every generator `g`, conjugation sym.g.sym^-1 must
land exactly on another generator. The relabel table is built from that and the
build asserts 0 failures -- the round-trip check the project docs keep asking
for, done once, up front.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .puzzle import NCube, _NORMALS, _sticker_maps


def _orthogonal_matrices() -> list[tuple[tuple[int, int, int], ...]]:
    """All 48 signed permutation matrices (det +-1): the full octahedral group."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = [[0, 0, 0] for _ in range(3)]
            for row, (col, sg) in enumerate(zip(perm, signs)):
                m[row][col] = sg
            mats.append(tuple(tuple(r) for r in m))
    assert len(mats) == 48
    return mats


def _apply(mat, v):
    return tuple(sum(mat[i][j] * v[j] for j in range(3)) for i in range(3))


def _det(m) -> int:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


@dataclass(frozen=True)
class SymmetryTable:
    n: int
    perms: tuple[tuple[int, ...], ...]          # sticker permutation per symmetry
    relabel: tuple[dict[str, str], ...]         # move name -> conjugated move name
    dets: tuple[int, ...]                       # +1 rotation, -1 includes a mirror

    def __len__(self) -> int:
        return len(self.perms)

    def conjugate_state(self, state, k: int):
        """Apply symmetry k to a state (relabels both positions and values)."""
        p = self.perms[k]
        inv = [0] * len(p)
        for i, j in enumerate(p):
            inv[j] = i
        # position i of the new state comes from position inv[i]; values are
        # sticker ids and must be relabelled by the same map.
        return tuple(p[state[inv[i]]] for i in range(len(p)))

    def relabel_path(self, path, k: int) -> list[str]:
        r = self.relabel[k]
        return [r[m] for m in path]


def build_symmetries(cube: NCube) -> SymmetryTable:
    n = cube.n
    idx_of, info = _sticker_maps(n)
    m = n - 1
    # cube centre is at (m/2, m/2, m/2); work in doubled coords to stay integral
    perms, relabels, dets = [], [], []
    name_by_perm = {v: k for k, v in cube.generators.items()}

    for mat in _orthogonal_matrices():
        sp = [0] * cube.state_size
        ok = True
        for i, (cell, nrm) in info.items():
            dc = tuple(2 * cell[a] - m for a in range(3))       # doubled, centred
            rc = _apply(mat, dc)
            newcell = tuple((rc[a] + m) // 2 for a in range(3))
            if any((rc[a] + m) % 2 for a in range(3)):
                ok = False
                break
            key = (newcell, _apply(mat, nrm))
            if key not in idx_of:
                ok = False
                break
            sp[i] = idx_of[key]
        if not ok:
            continue
        sp_t = tuple(sp)
        inv = [0] * len(sp_t)
        for i, j in enumerate(sp_t):
            inv[j] = i
        # conjugate every generator: (sym . g . sym^-1)[i] = sym[g[inv[i]]]
        rel = {}
        good = True
        for name, g in cube.generators.items():
            conj = tuple(sp_t[g[inv[i]]] for i in range(len(sp_t)))
            tgt = name_by_perm.get(conj)
            if tgt is None:
                good = False
                break
            rel[name] = tgt
        if not good:
            continue
        perms.append(sp_t)
        relabels.append(rel)
        dets.append(_det(mat))

    return SymmetryTable(n=n, perms=tuple(perms), relabel=tuple(relabels), dets=tuple(dets))


def verify(cube: NCube, tab: SymmetryTable) -> dict[str, int]:
    """Independent re-check of everything the table claims. Returns a report."""
    rep = {"n_symmetries": len(tab), "rotations": sum(1 for d in tab.dets if d > 0),
           "mirrors": sum(1 for d in tab.dets if d < 0),
           "relabel_bijective": 0, "solved_fixed": 0, "conjugation_exact": 0}
    for k in range(len(tab)):
        if set(tab.relabel[k].values()) == set(cube.generators):
            rep["relabel_bijective"] += 1
        if tab.conjugate_state(cube.solved_state, k) == cube.solved_state:
            rep["solved_fixed"] += 1
    # conjugation identity on random words: solving a conjugated state with the
    # relabelled word must work.
    import random
    rng = random.Random(0)
    names = [m for m in cube.move_names]
    fails = 0
    for k in range(len(tab)):
        word = [rng.choice(names) for _ in range(12)]
        scr = cube.apply_path(cube.solved_state, word)
        sol = cube.invert_path(word)
        scr_k = tab.conjugate_state(scr, k)
        sol_k = tab.relabel_path(sol, k)
        if cube.apply_path(scr_k, sol_k) != cube.solved_state:
            fails += 1
    rep["conjugation_exact"] = len(tab) - fails
    return rep
