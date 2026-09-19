"""Turn a set of orbit PDBs into a beam heuristic over full 216-sticker states.

Bridges pdb.py (which works in an orbit's local 24-slot coordinates) and beam.py
(which works on full states).

The position coordinate the PDB wants is `pos[p] = current local slot of the
piece whose home slot is p`, i.e. the INVERSE of "what is sitting in slot j".
Getting that backwards silently produces a plausible-looking but wrong
heuristic, so it is computed once here and reused.
"""

from __future__ import annotations

import numpy as np

from .pdb import OrbitPDB, build_pdb, local_generators
from .puzzle import NCube


class OrbitScorer:
    def __init__(self, cube: NCube, orbits: list[np.ndarray], k: int = 4,
                 verbose: bool = False):
        self.cube = cube
        self.orbits = [np.asarray(o, dtype=np.int64) for o in orbits]
        self.k = k
        self.tables: list[list[OrbitPDB]] = []
        self.loc_of_sticker = -np.ones(cube.state_size, dtype=np.int64)
        for o in self.orbits:
            self.loc_of_sticker[o] = np.arange(len(o))
        for o in self.orbits:
            lg = local_generators(cube, o)
            subs = [np.arange(i, min(i + k, 24)) for i in range(0, 24, k)]
            self.tables.append([build_pdb(lg, s, verbose=verbose) for s in subs])

    def positions(self, states: np.ndarray, oi: int) -> np.ndarray:
        """(B, 24) pos[p] = current local slot of the piece whose home slot is p."""
        o = self.orbits[oi]
        occupant_home = self.loc_of_sticker[states[:, o]]      # (B,24) per slot
        pos = np.empty_like(occupant_home)
        slots = np.broadcast_to(np.arange(o.size), occupant_home.shape)
        np.put_along_axis(pos, occupant_home, slots, axis=1)
        return pos

    def __call__(self, states: np.ndarray) -> np.ndarray:
        total = np.zeros(states.shape[0], dtype=np.int32)
        for oi in range(len(self.orbits)):
            pos = self.positions(states, oi)
            for t in self.tables[oi]:
                total += t.lookup(pos[:, t.pieces]).astype(np.int32)
        return total

    def solved_projection(self) -> np.ndarray:
        return np.concatenate(self.orbits)
