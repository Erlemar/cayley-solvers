"""Post-processing passes that shorten a valid move sequence.

Phase 1 includes the cheap passes; heavier shortcutting (D2-style ReduceFactor) arrives
in Phase 4.

All passes are correctness-preserving: if `path` solves a state, the output also solves
that state. Callers should still re-verify via `verify.verify_path` before submitting.
"""

from __future__ import annotations

from typing import Sequence

from cayley.bfs_table import BfsTable
from cayley.puzzle import PictureCube


def cancel_adjacent_inverses(path: Sequence[str], puzzle: PictureCube) -> list[str]:
    """Remove every adjacent (move, inverse_of_move) pair, iteratively to fixpoint."""
    out: list[str] = []
    for move in path:
        if out and out[-1] == puzzle.inverse_name(move):
            out.pop()
        else:
            out.append(move)
    return out


def shortcut_via_state_hash(
    path: Sequence[str],
    initial_state: tuple[int, ...],
    puzzle: PictureCube,
) -> list[str]:
    """Replay the path, hashing every intermediate state. From each position, try every
    generator — if it produces a state that reappears later in the trajectory, jump to
    that later position. Greedy but very fast.
    """
    states: list[tuple[int, ...]] = [tuple(initial_state)]
    cur = tuple(initial_state)
    for m in path:
        cur = puzzle.apply_move(cur, m)
        states.append(cur)

    state_to_latest_idx: dict[tuple[int, ...], int] = {}
    for i, s in enumerate(states):
        state_to_latest_idx[s] = i  # latest occurrence wins

    out: list[str] = []
    i = 0
    while i < len(path):
        current = states[i]
        best_jump = i + 1
        best_move = path[i]
        for move_name in puzzle.move_names:
            gen = puzzle.generators[move_name]
            next_state = tuple(current[g] for g in gen)
            j = state_to_latest_idx.get(next_state)
            if j is not None and j > best_jump:
                best_jump = j
                best_move = move_name
        out.append(best_move)
        i = best_jump
    return out


def shortcut_two_step(
    path: Sequence[str],
    initial_state: tuple[int, ...],
    puzzle: PictureCube,
) -> list[str]:
    """Like shortcut_via_state_hash but tries every 2-move pair (18*18=324 combos) at each
    position. If some pair (a, b) produces a state that appears later in the trajectory,
    we jump forward — replacing a longer segment with just (a, b).

    Only accepts replacements that STRICTLY shorten the path.
    """
    states: list[tuple[int, ...]] = [tuple(initial_state)]
    cur = tuple(initial_state)
    for m in path:
        cur = puzzle.apply_move(cur, m)
        states.append(cur)

    state_to_latest_idx: dict[tuple[int, ...], int] = {}
    for i, s in enumerate(states):
        state_to_latest_idx[s] = i

    names = puzzle.move_names

    out: list[str] = []
    i = 0
    N = len(path)
    while i < N:
        current = states[i]
        best_jump = -1
        best_moves: tuple[str, ...] = ()

        # window=1 (1 move skipping)
        for m in names:
            next_state = tuple(current[g] for g in puzzle.generators[m])
            j = state_to_latest_idx.get(next_state, -1)
            if j > i + 1 and j - i > 1 + (1 if best_moves else 0):
                # saves at least 1 move over taking `path[i]` alone
                if j > best_jump:
                    best_jump = j
                    best_moves = (m,)

        # window=2: compose a . b
        if i + 2 < N:  # only worth trying if the path is at least 2 steps past here
            for a in names:
                s1 = tuple(current[g] for g in puzzle.generators[a])
                for b in names:
                    s2 = tuple(s1[g] for g in puzzle.generators[b])
                    j = state_to_latest_idx.get(s2, -1)
                    if j > i + 2 and j - i > 2 + (0 if best_moves else 0):
                        if j - i - 2 > best_jump - i - len(best_moves):
                            best_jump = j
                            best_moves = (a, b)

        if best_moves and best_jump > i + len(best_moves):
            out.extend(best_moves)
            i = best_jump
        else:
            out.append(path[i])
            i += 1
    return out


def shortcut_three_step(
    path: Sequence[str],
    initial_state: tuple[int, ...],
    puzzle: PictureCube,
) -> list[str]:
    """Three-move-ahead shortcut (N7 from NEW_IDEAS_SYNTHESIS). At each position, try
    18^3 = 5832 three-move sequences; if any produces a state appearing later, replace.

    Slower than shortcut_two_step (~18× more candidates) but catches cases where no
    1- or 2-move jump exists but a 3-move commutator-style insertion does.
    """
    states: list[tuple[int, ...]] = [tuple(initial_state)]
    cur = tuple(initial_state)
    for m in path:
        cur = puzzle.apply_move(cur, m)
        states.append(cur)

    state_to_latest_idx: dict[tuple[int, ...], int] = {}
    for i, s in enumerate(states):
        state_to_latest_idx[s] = i

    names = puzzle.move_names
    # Precompute generator tuples once.
    gens = {n: puzzle.generators[n] for n in names}

    out: list[str] = []
    i = 0
    N = len(path)
    while i < N:
        current = states[i]
        best_jump = -1
        best_moves: tuple[str, ...] = ()

        for a in names:
            s1 = tuple(current[g] for g in gens[a])
            for b in names:
                s2 = tuple(s1[g] for g in gens[b])
                # 2-move shortcut first
                j = state_to_latest_idx.get(s2, -1)
                if j > i + 2 and j - i - 2 > best_jump - i - len(best_moves):
                    best_jump = j
                    best_moves = (a, b)
                for c in names:
                    s3 = tuple(s2[g] for g in gens[c])
                    j = state_to_latest_idx.get(s3, -1)
                    if j > i + 3 and j - i - 3 > best_jump - i - len(best_moves):
                        best_jump = j
                        best_moves = (a, b, c)

        if best_moves and best_jump > i + len(best_moves):
            out.extend(best_moves)
            i = best_jump
        else:
            out.append(path[i])
            i += 1
    return out


def reduce_factor_via_bfs_table(
    path: Sequence[str],
    puzzle: PictureCube,
    table: BfsTable,
    max_window: int | None = None,
) -> list[str]:
    """Sliding-window replacement. For each window of length w (2 ≤ w ≤ min(max_window,
    path_len)), compute its net permutation; if the BFS table has a shorter word for the
    same permutation, substitute.

    The algorithm picks the FIRST improvement found per pass and restarts. Multiple passes
    until no change. Complexity: O(N × max_window × 72) per pass.
    """
    name_to_idx = {n: i for i, n in enumerate(puzzle.move_names)}
    idx_to_name = puzzle.move_names
    identity = tuple(range(len(puzzle.solved_state)))

    path_idx = [name_to_idx[n] for n in path]
    gens = {i: puzzle.generators[puzzle.move_names[i]] for i in range(len(puzzle.move_names))}

    if max_window is None:
        # Wider windows find more replacements: a 10-move window whose net perm is
        # reachable in 4 moves saves 6 moves. Cap at 12 as diminishing returns set in.
        max_window = max(table.max_depth + 1, 12)

    improved = True
    while improved:
        improved = False
        N = len(path_idx)
        # Scan windows from LARGEST to smallest so we prefer big savings first.
        for w in range(min(max_window, N), 1, -1):
            for i in range(0, N - w + 1):
                # Compute net permutation of path_idx[i : i + w].
                state = identity
                for mi in path_idx[i : i + w]:
                    state = tuple(state[g] for g in gens[mi])
                replacement = table.lookup(state)
                if replacement is not None and len(replacement) < w:
                    # Substitute. Re-verify by caller — correctness is guaranteed
                    # because we replaced with a shorter word producing the same perm.
                    path_idx = path_idx[:i] + list(replacement) + path_idx[i + w :]
                    improved = True
                    break
            if improved:
                break

    return [idx_to_name[i] for i in path_idx]


def full_post_process(
    path: Sequence[str],
    initial_state: tuple[int, ...],
    puzzle: PictureCube,
    max_iterations: int = 5,
    use_two_step: bool = True,
    use_three_step: bool = False,
    bfs_table: BfsTable | None = None,
) -> list[str]:
    """Run cancellation and shortcutting passes to fixpoint. If `bfs_table` is supplied,
    also runs window-based ReduceFactor replacement. `use_three_step` adds 3-move N7
    insertion-timing shortcut (~18× slower than 2-step; useful for long paths)."""
    current: list[str] = list(path)
    for _ in range(max_iterations):
        prev_len = len(current)
        current = cancel_adjacent_inverses(current, puzzle)
        current = shortcut_via_state_hash(current, initial_state, puzzle)
        if use_two_step:
            current = shortcut_two_step(current, initial_state, puzzle)
        if use_three_step:
            current = shortcut_three_step(current, initial_state, puzzle)
        if bfs_table is not None:
            current = reduce_factor_via_bfs_table(current, puzzle, bfs_table)
        current = cancel_adjacent_inverses(current, puzzle)
        if len(current) == prev_len:
            break
    return current
