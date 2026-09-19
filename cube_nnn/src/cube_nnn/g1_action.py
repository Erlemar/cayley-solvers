"""The action of the 12 outer turns on 3x3x3-supercube coordinates.

Needed only to BUILD pattern databases: a PDB is a BFS in an abstract
coordinate space, so the abstract move action has to exist. Searching itself
still runs on full 216-sticker states.

DERIVED, NOT HAND-WRITTEN
-------------------------
Every delta is read off by applying the move to the SOLVED state and extracting
coordinates. That is exact, because a move's effect on (slot, orientation) is a
property of the move alone:

    new_cp[i] = cp[src[i]]                      slot source map
    new_co[i] = (co[src[i]] + twist[i]) % 3     at solved co==0, so co after
                                                 the move IS twist
    new_ep[i] = ep[esrc[i]]
    new_ef[i] = ef[esrc[i]] ^ eflip[i]
    new_fr[f] = (fr[f] + dfr[f]) % 4            outer turns never permute faces

`verify_action` re-checks all of it against the real 216-sticker moves on random
G1 elements, which is the only thing that makes the derivation trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .endgame import g1_coords
from .puzzle import NCube

OUTER_BASE = ("f0", "f5", "r0", "r5", "d0", "d5")


def outer_moves(cube: NCube) -> list[str]:
    m = cube.n - 1
    base = [f"{a}{i}" for a in ("f", "r", "d") for i in (0, m)]
    return [x for b in base for x in (b, "-" + b)]


@dataclass(frozen=True)
class G1Action:
    moves: tuple[str, ...]
    csrc: np.ndarray      # (M, 8)  corner slot source map
    ctwist: np.ndarray    # (M, 8)
    esrc: np.ndarray      # (M, 12) edge slot source map
    eflip: np.ndarray     # (M, 12)
    dfr: np.ndarray       # (M, 6)  face rotation delta mod 4


def build_action(cube: NCube) -> G1Action:
    moves = outer_moves(cube)
    solved = tuple(cube.solved_state)
    csrc, ctw, esrc, efl, dfr = [], [], [], [], []
    for mv in moves:
        c = g1_coords(cube, cube.apply_move(solved, mv))
        csrc.append(c.corner_perm)
        ctw.append(c.corner_orient)
        esrc.append(c.edge_perm)
        efl.append(c.edge_flip)
        dfr.append(c.face_rot)
    return G1Action(
        moves=tuple(moves),
        csrc=np.array(csrc, dtype=np.int8),
        ctwist=np.array(ctw, dtype=np.int8),
        esrc=np.array(esrc, dtype=np.int8),
        eflip=np.array(efl, dtype=np.int8),
        dfr=np.array(dfr, dtype=np.int8),
    )


def apply_coords(act: G1Action, m: int, cp, co, ep, ef, fr):
    s = act.csrc[m]
    ncp = cp[s]
    nco = (co[s] + act.ctwist[m]) % 3
    e = act.esrc[m]
    nep = ep[e]
    nef = ef[e] ^ act.eflip[m]
    nfr = (fr + act.dfr[m]) % 4
    return ncp, nco, nep, nef, nfr


def verify_action(cube: NCube, act: G1Action, trials: int = 300, seed: int = 0) -> dict:
    """Coordinate action must agree with the real 216-sticker move, always."""
    rng = np.random.default_rng(seed)
    solved = tuple(cube.solved_state)
    bad = 0
    for _ in range(trials):
        word = [act.moves[i] for i in rng.integers(0, len(act.moves),
                                                   size=int(rng.integers(0, 30)))]
        st = cube.apply_path(solved, word)
        c = g1_coords(cube, st)
        cp = np.array(c.corner_perm, dtype=np.int8)
        co = np.array(c.corner_orient, dtype=np.int8)
        ep = np.array(c.edge_perm, dtype=np.int8)
        ef = np.array(c.edge_flip, dtype=np.int8)
        fr = np.array(c.face_rot, dtype=np.int8)
        m = int(rng.integers(0, len(act.moves)))
        cp2, co2, ep2, ef2, fr2 = apply_coords(act, m, cp, co, ep, ef, fr)
        truth = g1_coords(cube, cube.apply_move(st, act.moves[m]))
        if (tuple(cp2) != truth.corner_perm or tuple(co2) != truth.corner_orient
                or tuple(ep2) != truth.edge_perm or tuple(ef2) != truth.edge_flip
                or tuple(fr2) != truth.face_rot):
            bad += 1
    return {"trials": trials, "mismatches": bad}


# ---------------------------------------------------------------- ranking ---

def perm_rank(p: np.ndarray) -> np.ndarray:
    """Lehmer rank of permutations. p: (N, k) -> (N,) in [0, k!)."""
    n, k = p.shape
    out = np.zeros(n, dtype=np.int64)
    for i in range(k):
        smaller = (p[:, i + 1:] < p[:, i:i + 1]).sum(axis=1)
        out = out * (k - i) + smaller
    return out


def perm_unrank(r: np.ndarray, k: int) -> np.ndarray:
    """Inverse of perm_rank. r: (N,) -> (N, k)."""
    n = r.shape[0]
    out = np.empty((n, k), dtype=np.int8)
    rem = r.copy()
    for i in range(k - 1, -1, -1):
        f = k - i
        out[:, i] = rem % f
        rem //= f
    # convert Lehmer digits to a permutation
    for i in range(k - 1, -1, -1):
        for j in range(i + 1, k):
            out[:, j] += (out[:, j] >= out[:, i]).astype(np.int8)
    return out
