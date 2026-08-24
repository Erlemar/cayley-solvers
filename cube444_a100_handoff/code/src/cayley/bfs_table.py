"""BFS lookup table: shortest word for every permutation reachable within depth K.

Used by `post_process.reduce_factor` to find shorter substitutions for windows of the
solution. We build the table once (a few seconds for depth 5, longer for 6+), then
lookups are O(1).

State representation: tuple of 72 ints (the permutation). For 3x3x3-class puzzles the
unique-state count grows roughly 18^d / overcounting at depth d.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import pickle
import time

import numpy as np

from cayley.puzzle import PictureCube


@dataclass
class BfsTable:
    """Lookup: canonical permutation tuple -> (tuple of generator indices)."""

    table: dict[tuple[int, ...], tuple[int, ...]]
    max_depth: int
    puzzle_name: str

    def lookup(self, perm: tuple[int, ...]) -> tuple[int, ...] | None:
        return self.table.get(perm)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)

    @classmethod
    def load(cls, path: str | Path) -> "BfsTable":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_bfs_table(
    puzzle: PictureCube,
    max_depth: int,
    canonical: bool = True,
    verbose: bool = False,
) -> BfsTable:
    """BFS from identity, recording the shortest (canonical) word to each permutation.

    `canonical=True` prunes the obvious redundancies (skip the inverse of the last move).
    This still enumerates all distinct permutations — a move and its inverse produce
    different permutations so pruning their composition doesn't lose coverage.
    """
    gen_names = list(puzzle.move_names)
    n_gen = len(gen_names)
    gens = [np.array(puzzle.generators[n], dtype=np.int64) for n in gen_names]
    inv_idx = np.array(
        [gen_names.index(puzzle.inverse_name(n)) for n in gen_names], dtype=np.int64
    )

    identity = tuple(range(len(puzzle.solved_state)))
    table: dict[tuple[int, ...], tuple[int, ...]] = {identity: ()}

    # BFS level by level so we can prune using the depth.
    frontier: list[tuple[tuple[int, ...], tuple[int, ...]]] = [(identity, ())]
    for d in range(max_depth):
        t0 = time.time()
        next_frontier: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for state, word in frontier:
            last = word[-1] if word else -1
            for gi in range(n_gen):
                if canonical and last >= 0 and gi == inv_idx[last]:
                    continue
                # Apply generator gi to state: new_state[i] = state[gens[gi][i]]
                arr = np.array(state, dtype=np.int64)
                new_arr = arr[gens[gi]]
                new_state = tuple(new_arr.tolist())
                if new_state not in table:
                    new_word = word + (gi,)
                    table[new_state] = new_word
                    next_frontier.append((new_state, new_word))
        frontier = next_frontier
        if verbose:
            print(f"  depth {d + 1}: +{len(next_frontier):,} new; total={len(table):,} ({time.time() - t0:.1f}s)")

    return BfsTable(table=table, max_depth=max_depth, puzzle_name="picture_cube_333")
