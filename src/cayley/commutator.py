"""Commutator library for extended window-replacement post-processing.

A commutator is a short word of the form [A, B] = A B A^{-1} B^{-1} (length 4), or its
conjugated extensions X [A, B] X^{-1} (length 6), [A, B C D] (length 8), etc. Each
commutator is a permutation; for window-replacement post-processing we want a table
{permutation -> shortest commutator-style word}.

Why not just use BFS-d6? BFS-d6 has ~11M states and needs ~6 GB RAM. The commutator
library with depths {4, 6, 8} has at most ~50K entries — ~100 MB, tractable in-memory.
The tradeoff: BFS-d6 covers EVERY permutation of diameter 6; the commutator library
covers a sparse subset (those representable as commutators). But this subset is
precisely the "structured" permutations that appear in long beam-search solutions.

When merged with our existing BFS-d5 table, the combined table strictly dominates
BFS-d5 alone and adds O(10K) more permutations at depth 6-8.

The library is used via the existing `reduce_factor_via_bfs_table` pass — the library
is a `BfsTable` with the same interface.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from cayley.bfs_table import BfsTable
from cayley.puzzle import PictureCube


def _apply_word(state: tuple[int, ...], word_idx: tuple[int, ...], gens: list[np.ndarray]) -> tuple[int, ...]:
    """Apply a sequence of generator indices to a state, return the resulting permutation."""
    cur = np.array(state, dtype=np.int64)
    for gi in word_idx:
        cur = cur[gens[gi]]
    return tuple(cur.tolist())


def _invert_word(word_idx: tuple[int, ...], inv_idx: np.ndarray) -> tuple[int, ...]:
    """Reverse a word and invert each move (length-preserving word inverse)."""
    return tuple(int(inv_idx[m]) for m in reversed(word_idx))


def _commutator_word(a: tuple[int, ...], b: tuple[int, ...], inv_idx: np.ndarray) -> tuple[int, ...]:
    """Return the word representing the commutator [A, B] = A B A^{-1} B^{-1}."""
    return a + b + _invert_word(a, inv_idx) + _invert_word(b, inv_idx)


def _conjugate_word(x: tuple[int, ...], w: tuple[int, ...], inv_idx: np.ndarray) -> tuple[int, ...]:
    """Return X W X^{-1}."""
    return x + w + _invert_word(x, inv_idx)


def _add_if_shorter(table: dict[tuple[int, ...], tuple[int, ...]], perm: tuple[int, ...], word: tuple[int, ...]) -> None:
    """Insert (perm -> word) into table if absent or new word is strictly shorter."""
    existing = table.get(perm)
    if existing is None or len(word) < len(existing):
        table[perm] = word


def build_commutator_library(
    puzzle: PictureCube,
    include_d4: bool = True,
    include_d6: bool = True,
    include_d8: bool = False,
    verbose: bool = False,
) -> BfsTable:
    """Enumerate commutators and return a `BfsTable` with {perm -> shortest commutator word}.

    - d4: [A, B] for all generator pairs (324 words).
    - d6: X [A, B] X^{-1} (5832 words) and [A B, C] = A B C A^{-1} B^{-1} C^{-1} (5832 words).
    - d8: X [A, B C] X^{-1} and [A B, C D] subsets. Larger; skipped by default.

    The table.max_depth is set to the longest included word length so callers know
    the window cap to scan.
    """
    gen_names = list(puzzle.move_names)
    n_gen = len(gen_names)
    gens = [np.array(puzzle.generators[n], dtype=np.int64) for n in gen_names]
    inv_idx = np.array(
        [gen_names.index(puzzle.inverse_name(n)) for n in gen_names], dtype=np.int64
    )

    identity = tuple(range(len(puzzle.solved_state)))
    table: dict[tuple[int, ...], tuple[int, ...]] = {identity: ()}

    max_depth = 0

    def add_word(word: tuple[int, ...]) -> None:
        nonlocal max_depth
        perm = _apply_word(identity, word, gens)
        _add_if_shorter(table, perm, word)
        if len(word) > max_depth:
            max_depth = len(word)

    if include_d4:
        t0 = time.time()
        for a in range(n_gen):
            for b in range(n_gen):
                if a == b or a == int(inv_idx[b]):
                    continue  # [A, A] = id; [A, A^{-1}] = id
                w = _commutator_word((a,), (b,), inv_idx)
                add_word(w)
        if verbose:
            print(f"d4 [A,B]: added, table={len(table):,} ({time.time() - t0:.2f}s)")

    if include_d6:
        t0 = time.time()
        # X [A, B] X^{-1}: conjugate a depth-4 commutator by a single generator.
        for x in range(n_gen):
            for a in range(n_gen):
                for b in range(n_gen):
                    if a == b or a == int(inv_idx[b]):
                        continue
                    w = _conjugate_word((x,), _commutator_word((a,), (b,), inv_idx), inv_idx)
                    add_word(w)
        if verbose:
            print(f"d6 X[A,B]X': added, table={len(table):,} ({time.time() - t0:.2f}s)")

        # [A B, C] = A B C A^{-1} B^{-1} C^{-1} (length 6).
        t0 = time.time()
        for a in range(n_gen):
            for b in range(n_gen):
                if a == int(inv_idx[b]):
                    continue  # A B cancels
                for c in range(n_gen):
                    if c == int(inv_idx[b]):
                        continue  # B C cancels
                    w = _commutator_word((a, b), (c,), inv_idx)
                    add_word(w)
        if verbose:
            print(f"d6 [AB,C]: added, table={len(table):,} ({time.time() - t0:.2f}s)")

    if include_d8:
        t0 = time.time()
        # X [A B, C] X^{-1} (length 8).
        for x in range(n_gen):
            for a in range(n_gen):
                for b in range(n_gen):
                    if a == int(inv_idx[b]):
                        continue
                    for c in range(n_gen):
                        if c == int(inv_idx[b]):
                            continue
                        base = _commutator_word((a, b), (c,), inv_idx)
                        w = _conjugate_word((x,), base, inv_idx)
                        add_word(w)
        if verbose:
            print(f"d8 X[AB,C]X': added, table={len(table):,} ({time.time() - t0:.2f}s)")

    return BfsTable(table=table, max_depth=max_depth, puzzle_name="picture_cube_333_commutators")


def merge_tables(primary: BfsTable, secondary: BfsTable) -> BfsTable:
    """Return a new BfsTable: perm -> shorter of primary[perm], secondary[perm].

    Puzzle name and max_depth are taken to be the max over the two inputs. Caller
    decides which table is "primary" (usually BFS-d5); secondary supplements it.
    """
    merged: dict[tuple[int, ...], tuple[int, ...]] = dict(primary.table)
    added = 0
    improved = 0
    for perm, word in secondary.table.items():
        existing = merged.get(perm)
        if existing is None:
            merged[perm] = word
            added += 1
        elif len(word) < len(existing):
            merged[perm] = word
            improved += 1
    return BfsTable(
        table=merged,
        max_depth=max(primary.max_depth, secondary.max_depth),
        puzzle_name=f"{primary.puzzle_name}+{secondary.puzzle_name}",
    )
