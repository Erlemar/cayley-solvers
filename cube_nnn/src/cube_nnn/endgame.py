"""The endgame: solve a REDUCED state using only the 12 outer-layer turns.

Measured in 01_verify_structure.py: outer turns move zero centre stickers
between faces, so G1 = <12 outer turns> is exactly the 3x3x3 supercube group.

    |G1| = |3x3x3| * 4^6/2 = 8.858e22 = 76.23 bits
    12 generators, quarter turns only  ->  counting bound 21.3 moves
    realistic optimal is ~26-32 in this metric (the counting bound is loose)

Because the beam engine already runs on full 216-sticker states, searching G1
needs no abstract coordinate at all -- just restrict the generator set. The
3x3-level coordinates below exist only to build heuristics and to verify that
what we think is happening is happening.

COORDINATES (extracted, not assumed -- validated in 05_endgame.py)
------------------------------------------------------------------
    corners     8 slots, which home corner sits in each + 3-fold orientation
    edges      12 slots, which home edge GROUP sits in each + reversal flag
    faces       6 rotations in Z/4  (this is the supercube part; a colour cube
                would not have it, and ignoring it yields "solved except a
                whole-face rotation" -- trap 2 in BIGCUBES_PLAN)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .puzzle import NCube, _sticker_maps
from .reduction import centre_positions, edge_slots, _face_of, _rot90


@dataclass(frozen=True)
class G1Coords:
    corner_perm: tuple[int, ...]
    corner_orient: tuple[int, ...]
    edge_perm: tuple[int, ...]
    edge_flip: tuple[int, ...]
    face_rot: tuple[int, ...]

    def is_solved(self) -> bool:
        return (self.corner_perm == tuple(range(8))
                and all(o == 0 for o in self.corner_orient)
                and self.edge_perm == tuple(range(len(self.edge_perm)))
                and all(f == 0 for f in self.edge_flip)
                and all(r == 0 for r in self.face_rot))


def corner_cells(cube: NCube) -> list[tuple[int, int, int]]:
    return sorted(c for c, s in cube.pieces().items() if len(s) == 3)


def corner_sticker_order(cube: NCube) -> dict[tuple[int, int, int], list[int]]:
    """Each corner's 3 stickers in RIGHT-HANDED order (det of normals > 0).

    Ordering them by face index instead flips handedness between corners, so
    orientation stops composing additively and the derived move action is wrong
    for about half of them (measured: 235/400 mismatches). A rotation preserves
    handedness, so a right-handed cyclic order composes as (o + twist) % 3.
    """
    _, info = _sticker_maps(cube.n)
    out = {}
    for cell, stickers in cube.pieces().items():
        if len(stickers) != 3:
            continue
        ss = sorted(stickers)
        nrm = [info[s][1] for s in ss]
        det = (nrm[0][0] * (nrm[1][1] * nrm[2][2] - nrm[1][2] * nrm[2][1])
               - nrm[0][1] * (nrm[1][0] * nrm[2][2] - nrm[1][2] * nrm[2][0])
               + nrm[0][2] * (nrm[1][0] * nrm[2][1] - nrm[1][1] * nrm[2][0]))
        out[cell] = ss if det > 0 else [ss[0], ss[2], ss[1]]
    return out


def face_rotations(cube: NCube, state) -> tuple[int, ...]:
    """For each face, k in Z/4 with that face's centre block = solved rotated k."""
    n = cube.n
    out = []
    for f, positions in centre_positions(n).items():
        vals = [state[p] for p in positions]
        k_found = -1
        src = list(positions)
        for k in range(4):
            if vals == src:
                k_found = k
                break
            src = [_rot90(p, n) for p in src]
        out.append(k_found)
    return tuple(out)


def g1_coords(cube: NCube, state) -> G1Coords:
    """Extract 3x3x3-supercube coordinates. Assumes `state` is in G1."""
    n = cube.n
    pieces = cube.pieces()
    ccells = corner_cells(cube)
    cindex = {c: i for i, c in enumerate(ccells)}
    home_cell = {}
    for cell, stickers in pieces.items():
        for s in stickers:
            home_cell[s] = cell

    order = corner_sticker_order(cube)
    cperm, corient = [], []
    for cell in ccells:
        occ0 = state[order[cell][0]]
        hcell = home_cell[occ0]
        cperm.append(cindex[hcell])
        corient.append(order[hcell].index(occ0))

    slots = edge_slots(cube)
    scell_of = {tuple(sl): i for i, sl in enumerate(map(tuple, slots))}
    where = {cell: (i, j) for i, sl in enumerate(slots) for j, cell in enumerate(sl)}
    eperm, eflip = [], []
    for sl in slots:
        seen = []
        for cell in sl:
            sticker = pieces[cell][0]
            seen.append(where[home_cell[state[sticker]]])
        eperm.append(seen[0][0])
        eflip.append(0 if [p for _, p in seen] == list(range(len(sl))) else 1)

    return G1Coords(tuple(cperm), tuple(corient), tuple(eperm), tuple(eflip),
                    face_rotations(cube, state))


def piece_mismatch_heuristic(cube: NCube):
    """Cheap 3x3-level heuristic: misplaced corners + edge groups + face rotations.

    Far more informative than raw sticker mismatch, because in G1 the 16 centres
    of a face move as one unit -- counting them individually multiplies one error
    by 16 and drowns the corner/edge signal.
    """
    n = cube.n
    pieces = cube.pieces()
    ccells = corner_cells(cube)
    corner_stickers = np.array([pieces[c][0] for c in ccells], dtype=np.int64)
    slots = edge_slots(cube)
    edge_stickers = np.array([pieces[sl[0]][0] for sl in slots], dtype=np.int64)
    # one representative centre per face is enough to detect a face rotation
    face_rep = np.array([centre_positions(n)[f][0] for f in range(6)], dtype=np.int64)
    probe = np.concatenate([corner_stickers, edge_stickers, face_rep])
    home = np.asarray(cube.solved_state, dtype=np.int16)[probe]

    def h(states: np.ndarray) -> np.ndarray:
        return (states[:, probe] != home).sum(axis=1)

    return h
