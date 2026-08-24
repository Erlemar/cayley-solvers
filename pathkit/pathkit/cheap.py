"""The two O(L) passes. Always run them; never expect much from them.

Both are strictly subsumed by a radius-1 ball sweep (window.py), so their only reason to
exist is that they cost nothing and need no ball machinery: use them inside a solver loop,
on a machine with no GPU, or as the first thing applied to a freshly produced CSV.

Measured worth on this family of puzzles: 30-80 moves out of ~30,000, i.e. 0.2-0.3 pct.
That number is itself informative -- it says beam output is already locally tight, which
is why the interesting methods are the ones that look further than one move.
"""
from __future__ import annotations


def cancel_inverses(puzzle, word):
    """Remove every adjacent (move, inverse) pair, iteratively to fixpoint."""
    out = []
    for m in word:
        if out and out[-1] == int(puzzle.inv_move[m]):
            out.pop()
        else:
            out.append(int(m))
    return out


def shortcut_via_state_hash(puzzle, s0, word):
    """Replay the path; from each position try every generator and jump to the LATEST
    later position holding the resulting state.

    Greedy and one move deep, but it catches the case adjacent-inverse cancellation
    misses: a state revisited many moves later, which happens when a beam wanders.
    """
    states = puzzle.path_states(s0, word)
    latest = {}
    for i in range(states.shape[0]):
        latest[states[i].tobytes()] = i          # later occurrence overwrites earlier

    out = []
    i = 0
    L = len(word)
    while i < L:
        best_j, best_m = i + 1, int(word[i])
        cur = states[i]
        for m in range(puzzle.n_gen):
            j = latest.get(cur[puzzle.gens[m]].tobytes())
            if j is not None and j > best_j:
                best_j, best_m = j, m
        out.append(best_m)
        i = best_j
    return out


def full_cheap(puzzle, s0, word, max_rounds=8):
    """Alternate both passes until neither shortens. Verified before returning."""
    cur = list(word)
    for _ in range(max_rounds):
        nxt = cancel_inverses(puzzle, shortcut_via_state_hash(puzzle, s0, cur))
        if len(nxt) >= len(cur):
            break
        cur = nxt
    assert puzzle.solves(s0, cur), "cheap post-processing broke the path"
    return cur


def sweep(puzzle, tests, paths):
    """Apply full_cheap to a whole submission. Returns (new_paths, report)."""
    out = {pid: full_cheap(puzzle, tests[pid], w) for pid, w in paths.items()}
    before = sum(len(v) for v in paths.values())
    after = sum(len(v) for v in out.values())
    return out, {"before": before, "after": after, "saved": before - after,
                 "pids": len(paths),
                 "changed": sum(1 for p in paths if len(out[p]) < len(paths[p]))}
