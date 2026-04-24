"""Memory-efficient BFS table for Megaminx using `bytes` keys instead of tuple[int, ...].

Tuple of 120 Python ints = ~3.6 KB per state (Python object overhead dominates).
Bytes of 120 uint8 = 120 B per state. 30x smaller; enables depth 6 (18M states, ~4 GB)
on a 16 GB laptop where the tuple-keyed version would need ~65 GB.

Drop-in for `cayley.post_process.reduce_factor_via_bfs_table` via the bytes_lookup shim.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class BfsBytesTable:
    """bytes-keyed table. `table[bytes_state] = path_bytes` where path_bytes are
    raw uint8 generator indices (one byte per move)."""

    table: dict[bytes, bytes]
    max_depth: int
    puzzle_name: str
    move_names: tuple[str, ...]  # move_names[gi] -> name

    def lookup_tuple(self, perm: tuple[int, ...]) -> tuple[int, ...] | None:
        """Accept a tuple of ints (same interface as BfsTable.lookup). Return the gen-index
        tuple for the shortest word to produce that permutation, or None."""
        key = bytes(perm)
        pb = self.table.get(key)
        return tuple(pb) if pb is not None else None

    # Alias so `reduce_factor_via_bfs_table(..., table)` works without modification
    # (it calls `table.lookup(perm)`).
    def lookup(self, perm: tuple[int, ...]) -> tuple[int, ...] | None:
        return self.lookup_tuple(perm)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)

    @classmethod
    def load(cls, path: str | Path) -> "BfsBytesTable":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_bfs_bytes(
    puzzle,
    max_depth: int,
    canonical: bool = True,
    verbose: bool = False,
) -> BfsBytesTable:
    """BFS from identity, storing keys as bytes (state) and values as bytes (generator
    indices). Uses numpy for the gather operation — much faster than tuple iteration
    in Python.
    """
    gen_names = list(puzzle.move_names)
    n_gen = len(gen_names)
    gens = np.stack(
        [np.array(puzzle.generators[n], dtype=np.int8) for n in gen_names], axis=0
    )  # (n_gen, state_size)
    inv_idx = np.array(
        [gen_names.index(puzzle.inverse_name(n)) for n in gen_names], dtype=np.int8
    )

    state_size = len(puzzle.solved_state)
    identity_arr = np.arange(state_size, dtype=np.int8)
    identity_key = identity_arr.tobytes()

    table: dict[bytes, bytes] = {identity_key: b""}
    frontier: list[tuple[np.ndarray, bytes]] = [(identity_arr, b"")]

    for d in range(max_depth):
        t0 = time.time()
        next_frontier: list[tuple[np.ndarray, bytes]] = []
        for state, word in frontier:
            last = word[-1] if word else -1
            for gi in range(n_gen):
                if canonical and last >= 0 and gi == inv_idx[last]:
                    continue
                new_state = state[gens[gi]]  # numpy fancy indexing, still int8
                key = new_state.tobytes()
                if key not in table:
                    new_word = word + bytes((gi,))
                    table[key] = new_word
                    next_frontier.append((new_state, new_word))
        frontier = next_frontier
        if verbose:
            print(f"  depth {d + 1}: +{len(next_frontier):,} new; total={len(table):,} "
                  f"({time.time() - t0:.1f}s)", flush=True)

    return BfsBytesTable(
        table=table, max_depth=max_depth,
        puzzle_name=getattr(puzzle, "__class__", type(puzzle)).__name__.lower(),
        move_names=tuple(gen_names),
    )
