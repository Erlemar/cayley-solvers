"""FMC insertion-solver foundations for megaminx.

Decode a state into piece permutations + orientations, and classify the unsolved
"residue" (what's left to solve). The solved state is the identity
(central_state = range(120)). A piece is a set of stickers sharing a face-set:
  - corners: |face_set| = 3  -> 20 pieces, orientation in Z/3
  - edges:   |face_set| = 2  -> 30 pieces, orientation in Z/2

decode() reads, for each slot, which home piece currently sits there and its
orientation (same home_lookup trick as corner_coord.py / edge_coord.py).
residue() turns that into the cycle structure the insertion finder needs:
a "single 3-cycle" residue (3 pieces of one type cyclically permuted, everything
else solved) is the case one commutator insertion can finish.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _face_sets(generators: dict[str, list[int]]) -> dict[int, frozenset]:
    n = len(next(iter(generators.values())))
    out = {}
    for i in range(n):
        faces = set()
        for name, perm in generators.items():
            if perm[i] != i:
                faces.add(name.lstrip("-"))
        out[i] = frozenset(faces)
    return out


def _build_piece_tables(generators: dict[str, list[int]], piece_size: int):
    """Return (slots, home_lookup) for pieces with |face_set| == piece_size.

    slots[k] = tuple of sticker positions (sorted numerically) for slot k.
    home_lookup[sorted(sticker values)] = home piece id.
    """
    fs = _face_sets(generators)
    pieces: dict[frozenset, list[int]] = {}
    for i, f in fs.items():
        if len(f) == piece_size:
            pieces.setdefault(f, []).append(i)
    sorted_face_sets = sorted(pieces.keys(), key=lambda f: tuple(sorted(f)))
    slots = [tuple(sorted(pieces[f])) for f in sorted_face_sets]
    home_lookup = {tuple(sorted(s)): idx for idx, s in enumerate(slots)}
    return slots, home_lookup


def perm_cycles(perm: np.ndarray) -> list[tuple[int, ...]]:
    """Non-trivial cycles (length >= 2) of a permutation given as perm[slot]=home."""
    n = len(perm)
    seen = [False] * n
    cycles = []
    for start in range(n):
        if seen[start] or perm[start] == start:
            seen[start] = True
            continue
        cyc = []
        j = start
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = int(perm[j])
        if len(cyc) >= 2:
            cycles.append(tuple(cyc))
    return cycles


@dataclass
class PieceState:
    corner_perm: np.ndarray
    corner_ori: np.ndarray
    edge_perm: np.ndarray
    edge_ori: np.ndarray


@dataclass
class Residue:
    kind: str  # 'solved' | 'corner_3cycle' | 'edge_3cycle' | 'other'
    n_corner_unsolved: int
    n_edge_unsolved: int
    corner_cycles: list = field(default_factory=list)
    edge_cycles: list = field(default_factory=list)
    corner_twist_only: list = field(default_factory=list)  # in-place, mis-oriented
    edge_flip_only: list = field(default_factory=list)

    @property
    def is_single_3cycle(self) -> bool:
        return self.kind in ("corner_3cycle", "edge_3cycle")


class PieceModel:
    """State <-> piece-coordinate decoder + residue classifier."""

    def __init__(self, puzzle):
        gens = {n: list(g) for n, g in puzzle.generators.items()}
        self.corner_slots, self.corner_home = _build_piece_tables(gens, 3)
        self.edge_slots, self.edge_home = _build_piece_tables(gens, 2)
        assert len(self.corner_slots) == 20, f"expected 20 corners, got {len(self.corner_slots)}"
        assert len(self.edge_slots) == 30, f"expected 30 edges, got {len(self.edge_slots)}"

    def _decode(self, state, slots, home_lookup, ori_mod):
        n = len(slots)
        perm = np.empty(n, dtype=np.int64)
        ori = np.empty(n, dtype=np.int64)
        for k, positions in enumerate(slots):
            vals = tuple(state[p] for p in positions)
            home = home_lookup.get(tuple(sorted(vals)))
            if home is None:
                raise ValueError(f"slot {k} values {vals} are not a valid piece")
            home_canon = slots[home]
            r = None
            for rr in range(ori_mod):
                rotated = tuple(home_canon[(rr + k2) % ori_mod] for k2 in range(ori_mod))
                if rotated == vals:
                    r = rr
                    break
            if r is None:
                raise ValueError(f"slot {k}: no orientation match {vals} vs {home_canon}")
            perm[k] = home
            ori[k] = r
        return perm, ori

    def decode(self, state) -> PieceState:
        state = tuple(int(x) for x in state)
        cp, co = self._decode(state, self.corner_slots, self.corner_home, 3)
        ep, eo = self._decode(state, self.edge_slots, self.edge_home, 2)
        return PieceState(cp, co, ep, eo)

    def residue(self, state) -> Residue:
        ps = self.decode(state)
        cidx = np.arange(20)
        eidx = np.arange(30)
        c_misperm = ps.corner_perm != cidx
        c_misori = (ps.corner_ori != 0) & ~c_misperm
        e_misperm = ps.edge_perm != eidx
        e_misori = (ps.edge_ori != 0) & ~e_misperm
        n_c = int(c_misperm.sum() + c_misori.sum())
        n_e = int(e_misperm.sum() + e_misori.sum())
        c_cycles = perm_cycles(ps.corner_perm)
        e_cycles = perm_cycles(ps.edge_perm)
        twist_only = [int(k) for k in np.where(c_misori)[0]]
        flip_only = [int(k) for k in np.where(e_misori)[0]]

        if n_c == 0 and n_e == 0:
            kind = "solved"
        elif (n_e == 0 and len(c_cycles) == 1 and len(c_cycles[0]) == 3
              and not twist_only):
            # exactly one 3-cycle of corners, all edges solved, no stray twists
            # (the 3 cycled corners may carry orientation; a commutator handles it)
            kind = "corner_3cycle"
        elif (n_c == 0 and len(e_cycles) == 1 and len(e_cycles[0]) == 3
              and not flip_only):
            kind = "edge_3cycle"
        else:
            kind = "other"
        return Residue(
            kind=kind, n_corner_unsolved=n_c, n_edge_unsolved=n_e,
            corner_cycles=c_cycles, edge_cycles=e_cycles,
            corner_twist_only=twist_only, edge_flip_only=flip_only,
        )
