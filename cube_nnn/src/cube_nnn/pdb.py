"""Exact pattern databases on a 24-slot sticker orbit. No training required.

The misplaced-count heuristic plateaus (measured: stalls at h=9-12 on one centre
orbit), which is what forced a neural scorer in the earlier attempts. A PDB is
exact, needs no data, and is built once in minutes.

CONSTRUCTION
------------
A generator restricted to an orbit is a permutation of that orbit's 24 slots
(orbits are closed under the group, so this is exact). Track the positions of k
chosen pieces; index them as a base-24 number with k digits, so the index space
is 24^k -- slightly larger than the 24!/(24-k)! reachable states but needing no
ranking function, which makes the whole BFS a few vectorised numpy lines.

    k=4       331,776 entries      0.3 MB
    k=5     7,962,624 entries      8   MB
    k=6   191,102,976 entries    191   MB

Disjoint subsets are combined by SUM for beam search (informative, not
admissible) or by MAX if you need admissibility for IDA*.

POSITION TRANSITION
-------------------
With the repo's source-map convention `new[j] = old[gl[j]]`, the piece sitting
at local slot `gl[j]` lands at slot `j`, so POSITIONS move by the inverse of the
local generator. `_local_gens` returns that inverse directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .puzzle import NCube


def local_generators(cube: NCube, orbit: np.ndarray) -> np.ndarray:
    """(M, 24) inverse-local generators: how PIECE POSITIONS move."""
    orbit = np.asarray(orbit, dtype=np.int64)
    pos_of = {int(g): i for i, g in enumerate(orbit)}
    out = []
    for name in cube.move_names:
        g = cube.generators[name]
        gl = np.array([pos_of[int(g[int(p)])] for p in orbit], dtype=np.int16)
        glinv = np.empty_like(gl)
        glinv[gl] = np.arange(len(gl), dtype=np.int16)
        out.append(glinv)
    return np.array(out, dtype=np.int16)


@dataclass
class OrbitPDB:
    k: int
    pieces: np.ndarray        # local indices of the tracked pieces (their HOME slots)
    table: np.ndarray         # uint8 distances over the 24^k index space
    radix: int = 24

    def index_of(self, positions: np.ndarray) -> np.ndarray:
        """positions: (B, k) local slots of the tracked pieces -> (B,) index."""
        idx = np.zeros(positions.shape[0], dtype=np.int64)
        for i in range(self.k - 1, -1, -1):
            idx = idx * self.radix + positions[:, i].astype(np.int64)
        return idx

    def lookup(self, positions: np.ndarray) -> np.ndarray:
        return self.table[self.index_of(positions)]


def _successors(idx: np.ndarray, glinv: np.ndarray, k: int, radix: int = 24) -> np.ndarray:
    """Vectorised successor index under one local generator."""
    out = np.zeros_like(idx)
    tmp = idx.copy()
    mult = 1
    for _ in range(k):
        d = (tmp % radix).astype(np.int64)
        tmp //= radix
        out += glinv[d].astype(np.int64) * mult
        mult *= radix
    return out


def build_pdb(lgens: np.ndarray, pieces: np.ndarray, verbose: bool = False) -> OrbitPDB:
    """Exact BFS over the placement of `pieces` (their home local slots)."""
    k = len(pieces)
    radix = 24
    size = radix ** k
    table = np.full(size, 255, dtype=np.uint8)
    start = np.zeros((1, k), dtype=np.int64)
    start[0] = pieces
    pdb = OrbitPDB(k=k, pieces=np.asarray(pieces), table=table, radix=radix)
    cur = pdb.index_of(start)
    table[cur] = 0
    depth = 0
    total = 1
    while cur.size:
        depth += 1
        # Mark move-by-move. Concatenating all successors first allocates
        # n_moves * |frontier| * 8 bytes at once, which OOMs for k=6
        # (a 30M frontier x 24 moves is ~5.8 GB).
        pieces_new = []
        d = min(depth, 254)
        for m in range(lgens.shape[0]):
            succ = _successors(cur, lgens[m], k, radix)
            succ = succ[table[succ] == 255]
            if succ.size:
                succ = np.unique(succ)
                succ = succ[table[succ] == 255]
                table[succ] = d
                pieces_new.append(succ)
        if not pieces_new:
            break
        cand = np.concatenate(pieces_new)
        total += cand.size
        if verbose:
            print(f"    depth {depth:2d}: {cand.size:>12,}  total {total:>12,}")
        cur = cand
    return pdb


def reachable_count(k: int) -> int:
    n = 1
    for i in range(k):
        n *= 24 - i
    return n
