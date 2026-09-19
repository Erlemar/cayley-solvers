"""Batched move engine + beam search over projected (coset) targets.

Everything here works on numpy arrays of shape (B, state_size) of uint8/int16,
so a beam step is a gather plus an argpartition -- no Python loop over states.

A move is a SOURCE map, so applying generator g to a batch is exactly

    new = states[:, g]

which is why the whole engine is three lines. Targets are COSETS: you give a
set of sticker positions ("the projection") and the search stops when every one
of them holds its home sticker. That is what makes the phase ladder work --
distance to a big set is much shorter than distance to the identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .puzzle import NCube


@dataclass
class Engine:
    cube: NCube
    moves: list[str]
    gen: np.ndarray = field(init=False)      # (M, state_size) source maps
    inv_of: np.ndarray = field(init=False)   # move index -> index of its inverse

    def __post_init__(self) -> None:
        self.gen = np.array([self.cube.generators[m] for m in self.moves], dtype=np.int16)
        idx = {m: i for i, m in enumerate(self.moves)}
        self.inv_of = np.array(
            [idx.get(self.cube.inverse_name(m), -1) for m in self.moves], dtype=np.int16)

    @property
    def n_moves(self) -> int:
        return len(self.moves)

    def apply(self, states: np.ndarray, m: int) -> np.ndarray:
        return states[:, self.gen[m]]

    def expand(self, states: np.ndarray) -> np.ndarray:
        """(B, S) -> (M, B, S) all children."""
        return states[:, self.gen]  # (B, M, S)

    def solved_row(self) -> np.ndarray:
        return np.array(self.cube.solved_state, dtype=np.int16)


def _hash_rows(a: np.ndarray, key: np.ndarray) -> np.ndarray:
    """64-bit hash per row, for dedup only."""
    return (a.astype(np.uint64) * key).sum(axis=1, dtype=np.uint64)


@dataclass
class BeamResult:
    solved: bool
    path: list[str]
    steps: int
    best_h: int
    history: list[tuple[int, int, float]]  # (step, best h, mean h)


def beam_solve(
    engine: Engine,
    start: np.ndarray,
    projection: np.ndarray,
    width: int = 1 << 14,
    max_steps: int = 200,
    seed: int = 0,
    heuristic=None,
    level_cap: float = 0.25,
    verbose: bool = False,
) -> BeamResult:
    """Beam search until every position in `projection` holds its home sticker.

    heuristic(states) -> float array. Default: count of projected positions that
    are not home. Ties are broken by a fixed random key so the search does not
    degenerate into scanning one subtree.
    """
    cube = engine.cube
    S = cube.state_size
    home = np.asarray(cube.solved_state, dtype=np.int16)[projection]
    rng = np.random.default_rng(seed)
    hkey = rng.integers(1, 2**63, size=S, dtype=np.uint64)

    if heuristic is None:
        def heuristic(st: np.ndarray) -> np.ndarray:
            return (st[:, projection] != home).sum(axis=1)

    states = start.reshape(1, S).astype(np.int16)
    # backpointers: for each level, (parent index, move index)
    back: list[tuple[np.ndarray, np.ndarray]] = []
    last_mv = np.full(1, -1, dtype=np.int16)
    history = []

    h0 = heuristic(states)
    if h0[0] == 0:
        return BeamResult(True, [], 0, 0, history)

    for step in range(1, max_steps + 1):
        B = states.shape[0]
        kids = states[:, engine.gen]                       # (B, M, S)
        kids = kids.reshape(B * engine.n_moves, S)
        parent = np.repeat(np.arange(B, dtype=np.int32), engine.n_moves)
        mv = np.tile(np.arange(engine.n_moves, dtype=np.int16), B)

        h = heuristic(kids).astype(np.int64)
        # Non-backtracking: never undo the move that produced the parent.
        # The sentinel MUST stay small enough that h*4096 below cannot overflow
        # int64 -- with int64max//4 it wraps NEGATIVE, so banned children sort
        # FIRST and the beam fills with backtracking moves. That bug plateaus
        # the search 3-4 moves short of the target.
        BAN = 1 << 40
        lm = last_mv[parent]
        banned = np.where(lm >= 0, engine.inv_of[np.maximum(lm, 0)], -1)
        h[banned == mv] = BAN
        best = int(h.min())
        if best == 0:
            j = int(np.argmin(h))
            path_idx = [(int(parent[j]), int(mv[j]))]
            p = int(parent[j])
            for lvl in range(len(back) - 1, -1, -1):
                pp, pm = back[lvl]
                path_idx.append((int(pp[p]), int(pm[p])))
                p = int(pp[p])
            path = [engine.moves[m] for _, m in reversed(path_idx)]
            history.append((step, 0, float(h.mean())))
            return BeamResult(True, path, step, 0, history)

        # Prune first (cheap), THEN dedup only the survivors. Doing it the other
        # way round costs a unique() over B*M rows every step; at width 65536
        # that is 2.4M rows and it dominates the whole search.
        # RANDOM TIE-BREAK. h is integer and ties are massive; without jitter
        # argpartition keeps a structurally similar set every step, beam
        # diversity collapses, and the search plateaus well short of the goal.
        key = h * 4096 + rng.integers(0, 4096, size=h.size, dtype=np.int64)
        cap = min(4 * width, key.size)
        cand = np.argpartition(key, cap - 1)[:cap] if cap < key.size else np.arange(key.size)
        hh = _hash_rows(kids[cand], hkey)
        _, first = np.unique(hh, return_index=True)
        cand = cand[first]
        # STRATIFIED SELECTION. Strict top-B traps the search: near a local
        # minimum every beam state sits at the same h, escaping needs a
        # temporarily WORSE state, and top-B never keeps one. Capping how many
        # states any single h-level may occupy forces the beam to spill into
        # higher levels. Measured: sum-of-PDBs descends 22 -> 2 then plateaus
        # forever under strict top-B.
        k = min(width, cand.size)
        if level_cap >= 1.0 or cand.size <= k:
            sel = cand[np.argpartition(key[cand], k - 1)[:k]] if k < cand.size else cand
        else:
            order = cand[np.argsort(key[cand], kind="stable")]
            hs = h[order]
            starts = np.flatnonzero(np.r_[True, hs[1:] != hs[:-1]])
            runlen = np.diff(np.r_[starts, hs.size])
            rank = np.arange(hs.size) - np.repeat(starts, runlen)
            cap_n = max(1, int(level_cap * width))
            take = rank < cap_n
            sel = order[take][:k]
            if sel.size < k:
                # BACKFILL. Capping per level without refilling leaves the beam
                # holding far fewer than `width` states, which starves the
                # search instead of diversifying it (measured: level_cap 0.01
                # ran 10x faster because the beam had collapsed).
                rest = order[~take]
                sel = np.concatenate([sel, rest[:k - sel.size]])

        states = kids[sel]
        last_mv = mv[sel]
        back.append((parent[sel], mv[sel]))
        history.append((step, best, float(h.mean())))
        if verbose and step % 5 == 0:
            print(f"    step {step:3d}  best_h {best:3d}  mean_h {h.mean():6.2f}  beam {states.shape[0]}")

    return BeamResult(False, [], max_steps, int(best), history)
