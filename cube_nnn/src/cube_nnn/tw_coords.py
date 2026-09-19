"""Thistlethwaite coordinates for the G1 endgame, with the STANDARD conventions.

The coordinates in endgame.py use arbitrary sticker references, which is fine
for identifying a state but useless for a subgroup ladder: measured, corner
orientation there is preserved only by U and edge flip by nothing. Thistlethwaite
needs references tied to the cube's axes:

    corner orientation   measured on the U/D axis
                         -> preserved by  <U, D, L2, R2, F2, B2>
    edge orientation     measured on the F/B axis
                         -> preserved by  <U, D, L, R, F2, B2>

Both are DEFINED here geometrically and then CHECKED against the real moves
(`verify_conventions`). Assuming a convention is how the corner-chirality bug got
in (235/400 mismatches), so nothing here is trusted until the check passes.

Face names in this repo's generators:  U=d5  D=d0  R=r0  L=r5  F=f0  B=f5
"""

from __future__ import annotations

import numpy as np

from .puzzle import NCube, _sticker_maps
from .reduction import edge_slots

U, F, R, B, L, D = 0, 1, 2, 3, 4, 5
FACE_OF_MOVE = {"d5": U, "d0": D, "r0": R, "r5": L, "f0": F, "f5": B}
NAME_OF_FACE = {U: "U", D: "D", R: "R", L: "L", F: "F", B: "B"}


def _det(a, b, c) -> int:
    return (a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0]))


def corner_orders_ud(cube: NCube) -> dict:
    """Per corner cell: its 3 stickers, y-axis (U/D) sticker FIRST, right-handed."""
    _, info = _sticker_maps(cube.n)
    out = {}
    for cell, st in cube.pieces().items():
        if len(st) != 3:
            continue
        st = sorted(st)
        nrm = {s: info[s][1] for s in st}
        ax = [s for s in st if nrm[s][1] != 0]
        assert len(ax) == 1, f"corner {cell} has {len(ax)} y-stickers"
        rest = [s for s in st if s != ax[0]]
        trio = [ax[0], rest[0], rest[1]]
        if _det(nrm[trio[0]], nrm[trio[1]], nrm[trio[2]]) < 0:
            trio = [ax[0], rest[1], rest[0]]
        out[cell] = trio
    return out


def edge_refs_fb(cube: NCube) -> dict:
    """Per edge slot: the sticker of its first wing cell on the REFERENCE face.

    Reference face = U or D if the slot touches one, else F or B. This is the
    standard F/B edge-orientation convention.
    """
    _, info = _sticker_maps(cube.n)
    n = cube.n
    pieces = cube.pieces()
    out = {}
    for si, sl in enumerate(edge_slots(cube)):
        cell = sl[0]
        st = pieces[cell]
        faces = [s // (n * n) for s in st]
        pick = None
        for want in (U, D, F, B):
            if want in faces:
                pick = st[faces.index(want)]
                break
        assert pick is not None, f"edge slot {si} faces {faces}"
        out[si] = pick
    return out


class TWCoords:
    """Standard-convention corner orientation, edge orientation, UD-slice."""

    def __init__(self, cube: NCube):
        self.cube = cube
        n = cube.n
        pieces = cube.pieces()
        self.corder = corner_orders_ud(cube)
        ccells = sorted(c for c, s in pieces.items() if len(s) == 3)
        self.ccells = ccells
        self.cfirst = np.array([self.corder[c][0] for c in ccells], dtype=np.int64)
        self.chome = np.full(cube.state_size, -1, dtype=np.int64)
        self.cpos = np.zeros(cube.state_size, dtype=np.int64)
        for ci, c in enumerate(ccells):
            for k, s in enumerate(self.corder[c]):
                self.chome[s] = ci
                self.cpos[s] = k

        slots = edge_slots(cube)
        self.slots = slots
        refs = edge_refs_fb(cube)
        self.eref = np.array([refs[i] for i in range(len(slots))], dtype=np.int64)
        # sticker -> (home edge slot, is that sticker the home slot's reference?)
        self.ehome = np.full(cube.state_size, -1, dtype=np.int64)
        self.eisref = np.zeros(cube.state_size, dtype=np.int64)
        for si, sl in enumerate(slots):
            for cell in sl:
                for s in pieces[cell]:
                    self.ehome[s] = si
            self.eisref[refs[si]] = 1
        # which slots are the UD-slice (E-layer): touch neither U nor D
        self.ud_slice = np.array(
            [si for si, sl in enumerate(slots)
             if not ({s // (n * n) for s in pieces[sl[0]]} & {U, D})], dtype=np.int64)
        assert self.ud_slice.size == 4, self.ud_slice

    def corner_orient(self, states: np.ndarray) -> np.ndarray:
        """(B,8) orientation indexed by SLOT."""
        occ = states[:, self.cfirst]
        return self.cpos[occ]

    def corner_orient_by_piece(self, states: np.ndarray) -> np.ndarray:
        """(B,8) orientation indexed by PIECE -- the invariant a subgroup preserves.

        A U turn permutes the corners, so the slot-indexed array permutes with
        them; only the piece-indexed one is literally unchanged.
        """
        occ = states[:, self.cfirst]
        out = np.empty_like(occ)
        np.put_along_axis(out, self.chome[occ], self.cpos[occ], axis=1)
        return out

    def edge_orient(self, states: np.ndarray) -> np.ndarray:
        """(B,12) orientation indexed by SLOT."""
        occ = states[:, self.eref]
        return 1 - self.eisref[occ]

    def edge_orient_by_piece(self, states: np.ndarray) -> np.ndarray:
        occ = states[:, self.eref]
        out = np.empty_like(occ)
        np.put_along_axis(out, self.ehome[occ], 1 - self.eisref[occ], axis=1)
        return out

    def edge_home(self, states: np.ndarray) -> np.ndarray:
        return self.ehome[states[:, self.eref]]    # (B,12) home slot of piece in slot j

    def ud_slice_mask(self, states: np.ndarray) -> np.ndarray:
        """(B,12) bool: is the group now in slot j one of the 4 UD-slice groups."""
        home = self.edge_home(states)
        return np.isin(home, self.ud_slice)


def verify_conventions(cube: NCube, tw: TWCoords, trials: int = 200, seed: int = 0):
    """Check each coordinate is preserved by exactly the subgroup it should be."""
    rng = np.random.default_rng(seed)
    solved = np.array(cube.solved_state, dtype=np.int16)
    moves = [m for m in cube.move_names
             if m.lstrip("-") in {"f0", "f5", "r0", "r5", "d0", "d5"}]
    quarters = {m: FACE_OF_MOVE[m.lstrip("-")] for m in moves}

    def rand_state():
        st = solved.copy()
        for i in rng.integers(0, len(moves), size=int(rng.integers(1, 30))):
            st = st[np.array(cube.generators[moves[int(i)]], dtype=np.int64)]
        return st

    report = {}
    for name, getter, keep_q, keep_h in (
            ("corner_orient", tw.corner_orient_by_piece, {U, D}, {L, R, F, B}),
            ("edge_orient", tw.edge_orient_by_piece, {U, D, L, R}, {F, B})):
        ok_q = ok_h = bad_q = bad_h = 0
        for _ in range(trials):
            st = rand_state()
            base = getter(st.reshape(1, -1))[0]
            for mv, f in quarters.items():
                g = np.array(cube.generators[mv], dtype=np.int64)
                one = getter(st[g].reshape(1, -1))[0]
                two = getter(st[g][g].reshape(1, -1))[0]
                if f in keep_q:
                    ok_q += int(np.array_equal(base, one))
                    bad_q += int(not np.array_equal(base, one))
                if f in keep_h:
                    ok_h += int(np.array_equal(base, two))
                    bad_h += int(not np.array_equal(base, two))
        report[name] = {"quarter_preserved_ok": ok_q, "quarter_preserved_bad": bad_q,
                        "half_preserved_ok": ok_h, "half_preserved_bad": bad_h}
    return report
