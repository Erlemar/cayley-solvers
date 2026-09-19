"""Is a state REDUCED, i.e. does it lie in G1 = <the 12 outer-layer turns>?

This is the gate for the endgame re-solve. Measured in 01_verify_structure.py:
outer turns move ZERO centre stickers between faces, so G1 acts as

    centres : each face's (n-2)^2 centre block rotates rigidly, k*90 degrees
    wings   : each edge's (n-2) wing pieces move as one rigid group
    corners : as 3x3x3 corners

which makes G1 the 3x3x3 supercube group (8.858e22, 76.23 bits) and reduction a
coset problem of index 424.4 bits. A reduced state is therefore at most ~21-30
moves from solved using outer turns ONLY -- so any solution path that passes
through a reduced state can have its whole tail replaced by a 3x3x3 solve.

`in_g1` is validated by sampling in 03_verify_reduction.py: every random word in
the outer turns must test True, and random words in the full generator set must
test False.
"""

from __future__ import annotations

from typing import Sequence

from .puzzle import NCube, piece_of_sticker


def _face_of(idx: int, n: int) -> int:
    return idx // (n * n)


def _rc(idx: int, n: int) -> tuple[int, int]:
    k = idx % (n * n)
    return divmod(k, n)


def centre_positions(n: int) -> dict[int, list[int]]:
    """face -> its (n-2)^2 centre sticker indices."""
    out: dict[int, list[int]] = {}
    for f in range(6):
        out[f] = [f * n * n + r * n + c
                  for r in range(1, n - 1) for c in range(1, n - 1)]
    return out


def _rot90(idx: int, n: int) -> int:
    """Rotate a sticker index within its own face by 90 degrees (r,c)->(c,n-1-r)."""
    f = _face_of(idx, n)
    r, c = _rc(idx, n)
    return f * n * n + c * n + (n - 1 - r)


def edge_slots(cube: NCube) -> list[list[int]]:
    """Each edge slot as its (n-2) wing CELLS, ordered along the edge.

    An edge cell is a cell carrying exactly 2 stickers; cells are grouped by the
    unordered pair of faces they show, then sorted along the varying coordinate.
    """
    n = cube.n
    pieces = cube.pieces()
    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for cell, stickers in pieces.items():
        if len(stickers) != 2:
            continue
        faces = tuple(sorted(_face_of(s, n) for s in stickers))
        groups.setdefault(faces, []).append(cell)
    slots = []
    for faces, cells in sorted(groups.items()):
        # the edge runs along whichever coordinate varies
        varying = [a for a in range(3) if len({c[a] for c in cells}) > 1]
        assert len(varying) == 1, f"edge {faces} varies in {varying}"
        slots.append(sorted(cells, key=lambda c: c[varying[0]]))
    return slots


def in_g1(cube: NCube, state: Sequence[int]) -> bool:
    """True iff `state` is reachable from solved using only the 12 outer turns."""
    n = cube.n
    if not cube.is_supercube:
        raise ValueError("in_g1 needs a supercube (state == group element)")

    # --- centres: each face must be a rigid rotation of its own solved block
    for f, positions in centre_positions(n).items():
        vals = [state[p] for p in positions]
        if any(_face_of(v, n) != f for v in vals):
            return False
        for k in range(4):
            src = positions
            for _ in range(k):
                src = [_rot90(p, n) for p in src]
            if vals == src:
                break
        else:
            return False

    # --- wings: each slot must hold one home edge group, order-consistent
    home_cell = piece_of_sticker(n)
    cell_stickers = cube.pieces()
    slots = edge_slots(cube)
    home_of_cell = {tuple(sorted(cs)): i for i, slot in enumerate(slots)
                    for cs in [slot]}
    # map each edge cell -> (slot index, position along that slot)
    where: dict[tuple[int, int, int], tuple[int, int]] = {}
    for si, slot in enumerate(slots):
        for pos, cell in enumerate(slot):
            where[cell] = (si, pos)

    for slot in slots:
        seen = []
        for cell in slot:
            sticker = cell_stickers[cell][0]
            piece_home_cell = home_cell[state[sticker]]
            seen.append(where[piece_home_cell])
        home_slots = {s for s, _ in seen}
        if len(home_slots) != 1:
            return False
        order = [p for _, p in seen]
        m = len(order)
        if order != list(range(m)) and order != list(range(m - 1, -1, -1)):
            return False
    return True


def first_reduced_index(cube: NCube, state: Sequence[int], path: Sequence[str],
                        stride: int = 1) -> int | None:
    """Index i such that applying path[:i] to `state` lands in G1, or None.

    The tail path[i:] can then be replaced by any 3x3x3 supercube solve of that
    reduced state -- bound 21.3 moves, realistically 24-30 -- which is the
    'endgame re-solve' win. Non-worsening: only splice if shorter.
    """
    cur = tuple(state)
    if in_g1(cube, cur):
        return 0
    for i, mv in enumerate(path, start=1):
        cur = cube.apply_move(cur, mv)
        if i % stride == 0 and in_g1(cube, cur):
            return i
    return None
