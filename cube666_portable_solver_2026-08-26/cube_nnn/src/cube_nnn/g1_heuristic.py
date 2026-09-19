"""PDB-backed heuristics for the G1 endgame, evaluated on full 216-sticker states.

Two table families:

  CornerPDB  exact, all 8!*3^7 = 88,179,840 corner states, max depth 14.
             Too large for a precomputed transition table, but it does not need
             one: permutation and orientation transform independently, so the
             two small move tables suffice.

  EdgePDB    k tracked edge groups among 12 slots, with flips: 12^k * 2^k.
             k=4 is 331,776 entries, so the FULL transition table fits (16 MB)
             and BFS is instant. Position and flip do NOT factor apart here
             (a group's flip toggles based on the slot it lands in), which is
             why these stay small rather than following the corner trick.

Aggregation is left to the caller: `max` is admissible, `sum` is not but is far
more discriminative, and which one actually drives the beam is an empirical
question -- measured in 07_endgame_pdb.py rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .endgame import corner_cells, corner_sticker_order
from .g1_action import G1Action, perm_rank
from .puzzle import NCube
from .reduction import edge_slots

NPERM, NORI = 40320, 2187


@dataclass
class CornerPDB:
    table: np.ndarray
    first: np.ndarray        # (8,) representative sticker per corner slot
    home_of: np.ndarray      # sticker -> home corner slot index
    orient_of: np.ndarray    # sticker -> its index in the home cell's RH order
    pw3: np.ndarray

    @classmethod
    def load(cls, cube: NCube, path) -> "CornerPDB":
        table = np.load(path)
        assert table.size == NPERM * NORI, table.size
        ccells = corner_cells(cube)
        cindex = {c: i for i, c in enumerate(ccells)}
        order = corner_sticker_order(cube)
        first = np.array([order[c][0] for c in ccells], dtype=np.int64)
        home_of = np.full(cube.state_size, -1, dtype=np.int64)
        orient_of = np.zeros(cube.state_size, dtype=np.int64)
        for cell in ccells:
            for k, s in enumerate(order[cell]):
                home_of[s] = cindex[cell]
                orient_of[s] = k
        return cls(table, first, home_of, orient_of,
                   (3 ** np.arange(7)).astype(np.int64))

    def index(self, states: np.ndarray) -> np.ndarray:
        occ = states[:, self.first]                    # (B, 8)
        cp = self.home_of[occ].astype(np.int8)
        co = self.orient_of[occ]
        return perm_rank(cp) * NORI + (co[:, :7] * self.pw3).sum(axis=1)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        return self.table[self.index(states)].astype(np.int32)


@dataclass
class EdgePDB:
    tracked: np.ndarray      # (k,) home edge-slot indices being tracked
    table: np.ndarray
    rep: np.ndarray          # (12,) representative sticker per edge slot
    home_slot: np.ndarray    # sticker -> home edge slot
    home_pos: np.ndarray     # sticker -> position within its home slot
    k: int

    def index_from_posflip(self, pos: np.ndarray, flip: np.ndarray) -> np.ndarray:
        idx = np.zeros(pos.shape[0], dtype=np.int64)
        for i in range(self.k - 1, -1, -1):
            idx = idx * 12 + pos[:, i]
        f = np.zeros(pos.shape[0], dtype=np.int64)
        for i in range(self.k - 1, -1, -1):
            f = f * 2 + flip[:, i]
        return f * (12 ** self.k) + idx

    def posflip(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each tracked edge group: which slot it is in, and whether reversed."""
        occ = states[:, self.rep]                       # (B,12) sticker in slot j pos 0
        who = self.home_slot[occ]                       # (B,12) home group now in slot j
        rev = (self.home_pos[occ] != 0).astype(np.int64)
        B = states.shape[0]
        pos = np.empty((B, 12), dtype=np.int64)
        flip = np.empty((B, 12), dtype=np.int64)
        slots = np.broadcast_to(np.arange(12), who.shape)
        np.put_along_axis(pos, who, slots, axis=1)
        np.put_along_axis(flip, who, rev, axis=1)
        return pos[:, self.tracked], flip[:, self.tracked]

    def __call__(self, states: np.ndarray) -> np.ndarray:
        pos, flip = self.posflip(states)
        return self.table[self.index_from_posflip(pos, flip)].astype(np.int32)


def build_edge_pdb(cube: NCube, act: G1Action, tracked, verbose: bool = False) -> EdgePDB:
    slots = edge_slots(cube)
    pieces = cube.pieces()
    rep = np.array([pieces[sl[0]][0] for sl in slots], dtype=np.int64)
    home_slot = np.full(cube.state_size, -1, dtype=np.int64)
    home_pos = np.zeros(cube.state_size, dtype=np.int64)
    for i, sl in enumerate(slots):
        for j, cell in enumerate(sl):
            for s in pieces[cell]:
                home_slot[s] = i
                home_pos[s] = j

    tracked = np.asarray(tracked, dtype=np.int64)
    k = len(tracked)
    size = (12 ** k) * (2 ** k)
    # positions move by the inverse of esrc; flip toggles by eflip at the DESTINATION
    esrcinv = np.empty_like(act.esrc)
    for m in range(12):
        esrcinv[m][act.esrc[m]] = np.arange(12, dtype=np.int8)

    pw12 = (12 ** np.arange(k)).astype(np.int64)
    pw2 = (2 ** np.arange(k)).astype(np.int64)
    all_idx = np.arange(size, dtype=np.int64)
    fpart, ppart = np.divmod(all_idx, 12 ** k)
    pos = np.stack([(ppart // pw12[i]) % 12 for i in range(k)], axis=1)
    flp = np.stack([(fpart // pw2[i]) % 2 for i in range(k)], axis=1)
    valid = np.ones(size, dtype=bool)
    for i in range(k):
        for j in range(i + 1, k):
            valid &= pos[:, i] != pos[:, j]

    trans = np.empty((12, size), dtype=np.int32)
    for m in range(12):
        npos = esrcinv[m][pos]
        nflip = flp ^ act.eflip[m][npos]
        trans[m] = ((nflip * pw2).sum(axis=1) * (12 ** k)
                    + (npos.astype(np.int64) * pw12).sum(axis=1))

    table = np.full(size, 255, dtype=np.uint8)
    start = int((np.arange(k) * 0).sum())  # placeholder, computed below
    start_pos = tracked
    start_idx = int((start_pos * pw12).sum())
    table[start_idx] = 0
    frontier = np.array([start_idx], dtype=np.int64)
    depth, total = 0, 1
    while frontier.size:
        depth += 1
        cand = np.unique(np.concatenate([trans[m][frontier] for m in range(12)]))
        cand = cand[table[cand] == 255]
        cand = cand[valid[cand]]
        if cand.size == 0:
            break
        table[cand] = min(depth, 254)
        total += cand.size
        if verbose:
            print(f"    edge depth {depth:2d}: {cand.size:>10,}  total {total:>10,}")
        frontier = cand
    table[~valid] = 0
    return EdgePDB(tracked, table, rep, home_slot, home_pos, k)
