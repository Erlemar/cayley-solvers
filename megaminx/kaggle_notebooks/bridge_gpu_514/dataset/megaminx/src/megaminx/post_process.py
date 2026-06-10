"""Megaminx-specific post-processing.

Every face generator has rotation order 5, so contiguous same-face runs can be shortened:
  X^5 = identity            (saves 5 moves)
  X^4 = -X                  (saves 3 moves)
  X^3 = -X . -X             (saves 1 move)
  -X^3 = X . X              (saves 1 move)

A "run" is a contiguous subsequence of moves that all act on the same face (X or -X).
The net rotation of the run is k mod 5 where k = (# CW) - (# CCW).
We rewrite the run as the shortest of:
  k CW moves     (length |k|)  when 0 < k ≤ 2
  (5-k) CCW moves (length 5-k) when 2 < k < 5
  — taking k mod 5 into the range [0, 4].

Correctness: each rewrite keeps the cumulative permutation unchanged.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def _base_face(name: str) -> str:
    return name[1:] if name.startswith("-") else name


def reduce_same_face_runs(path: Sequence[str]) -> list[str]:
    """Rewrite each contiguous same-face run with the minimum-length equivalent."""
    out: list[str] = []
    i = 0
    n = len(path)
    while i < n:
        face = _base_face(path[i])
        j = i
        net = 0  # CW count - CCW count, modulo 5 later
        while j < n and _base_face(path[j]) == face:
            net += -1 if path[j].startswith("-") else 1
            j += 1
        k = net % 5  # 0..4
        if k == 0:
            pass
        elif k <= 2:
            out.extend([face] * k)
        else:
            out.extend(["-" + face] * (5 - k))
        i = j
    return out


def cancel_adjacent_inverses(path: Sequence[str]) -> list[str]:
    """Generic inverse-pair cancellation (mirrors cayley.post_process for puzzle-independence)."""
    out: list[str] = []
    for m in path:
        inv = m[1:] if m.startswith("-") else "-" + m
        if out and out[-1] == inv:
            out.pop()
        else:
            out.append(m)
    return out


def full_post_process(
    path: Sequence[str],
    puzzle=None,
    bfs_table=None,
    max_window: int | None = None,
) -> list[str]:
    """Apply all Megaminx post-passes to a fixpoint.

    If `puzzle` and `bfs_table` are provided, also runs BFS-table window replacement
    (`cayley.post_process.reduce_factor_via_bfs_table`) between passes. Otherwise only
    same-face reduction + adjacent-inverse cancellation.
    """
    prev: list[str] = list(path)
    while True:
        cur = reduce_same_face_runs(prev)
        cur = cancel_adjacent_inverses(cur)
        if puzzle is not None and bfs_table is not None:
            from cayley.post_process import reduce_factor_via_bfs_table
            cur = reduce_factor_via_bfs_table(cur, puzzle, bfs_table, max_window=max_window)
        if cur == prev:
            return cur
        prev = cur
