"""Exact window shortening: replace any path window by a shorter word with the same effect.

MECHANISM. For a path s_0 .. s_L, build the ball of radius r around EVERY s_i at once and
look for a state reached both from s_i (at depth d1) and from s_j (at depth d2, j > i).
The two half-words join into a replacement for the window [i, j):

    new length = i + d1 + d2 + (L - j)        saving = (j - d2) - (i + d1)

Every window of length <= 2r is covered by one sweep, and the join is a single group
reduce over the ball entries, so the cost is one BFS per path -- not one search per
window. This is the GPU port of the classic meet-in-the-middle rewrite, generalised from
permutations to states (see balls.py for why that matters).

WHAT IT COSTS. Each rung of r multiplies the ball by the branching factor: measured 19.21
on cube444 (level sizes 1, 24, 468, 9000, 172914), so r=5 is ~19x r=4. Radius 3 sweeps
1000 paths in seconds; radius 5 is minutes per path on a GPU. Budget accordingly and
prefer to spend the next rung only where the paths are long.

WHAT IT PAYS. On loose beam output it is worth a few moves per hundred paths. On an
already min-merged community file it is usually ZERO -- tetraminx measured 0 rewrites out
to 9-move windows on its 28,821 merge, and cube444 got -8 over 1043 paths at reach 10.
A zero here is informative: it means the remaining slack is global, not local, and the
next lever is search, not rewriting.
"""
from __future__ import annotations

import time

import numpy as np

from .backend import get_backend
from .balls import Balls, Zobrist, build_balls, collide_savings, word_to


def reduce_path(puzzle, s0, word, radius, bk=None, zob=None, chunk=150_000,
                stats=None, min_saving=1):
    """One pass of ball-collision shortening over a single path. Returns a new word.

    Candidate splices are applied best-saving-first on disjoint intervals, and every one
    is replayed against the real state before it is accepted -- so a 64-bit hash phantom
    costs a rejected candidate, never a corrupt path. (At ~1e8 ball entries phantoms are
    real: a 64-bit probe false-positives at ~2e-11 and a full sweep runs enough probes to
    see a handful. They are deterministic, so re-hashing cannot catch them; only replay
    can.)
    """
    bk = bk or get_backend("cpu")
    zob = zob or Zobrist(puzzle.state_size, puzzle.n_labels, bk)
    L = len(word)
    if L < 3:
        return list(word)
    states = puzzle.path_states(s0, word)
    balls = build_balls(puzzle, states, radius, bk=bk, zob=zob, chunk=chunk)
    cands = collide_savings(balls, min_saving=min_saving)
    if not cands:
        bk.free()
        return list(word)
    _, src_np, dist_np, par_np, mv_np = balls.to_numpy()
    del balls
    bk.free()

    taken = []
    used = np.zeros(L + 1, dtype=bool)
    for sav, e_lo, e_hi in cands:
        i, d1 = int(src_np[e_lo]), int(dist_np[e_lo])
        j, d2 = int(src_np[e_hi]), int(dist_np[e_hi])
        if j - i < 1 or (j - i) - (d1 + d2) != sav:
            continue
        if used[i:j].any():                       # keep accepted windows disjoint
            continue
        w1 = word_to(par_np, mv_np, e_lo)
        w2 = word_to(par_np, mv_np, e_hi)
        u = w1 + puzzle.inv_word(w2)              # s_i -> x -> s_j
        if len(u) >= j - i:
            continue
        if not np.array_equal(puzzle.apply_word(states[i], u), states[j]):
            _bump(stats, "phantom")
            continue
        taken.append((i, j, u))
        used[i:j] = True
    if not taken:
        return list(word)
    taken.sort()
    out, prev = [], 0
    for i, j, u in taken:
        out.extend(word[prev:i])
        out.extend(u)
        prev = j
    out.extend(word[prev:])
    _bump(stats, "splices", len(taken))
    _bump(stats, "saved", L - len(out))
    return out


def reduce_path_to_fixpoint(puzzle, s0, word, radius, max_rounds=4, **kw):
    """Re-sweep until a pass finds nothing. A splice changes the states downstream of it,
    so a second pass genuinely sees different windows -- but returns fall off fast, and
    `max_rounds` bounds the worst case."""
    cur = list(word)
    for _ in range(max_rounds):
        nxt = reduce_path(puzzle, s0, cur, radius, **kw)
        if len(nxt) >= len(cur):
            return cur
        cur = nxt
    return cur


def sweep(puzzle, tests, paths, radius, device="cpu", chunk=150_000, fixpoint=False,
          min_length=0, progress=None, log=print):
    """Window-reduce a whole submission. Returns (new_paths, report).

    `min_length` skips short paths, which is the right way to spend a bigger radius: the
    cost per path is flat in the path length but the chance of slack rises with it.
    """
    bk = get_backend(device)
    zob = Zobrist(puzzle.state_size, puzzle.n_labels, bk)
    stats = {"splices": 0, "saved": 0, "phantom": 0}
    out = {}
    t0 = time.time()
    todo = [p for p in sorted(paths) if len(paths[p]) >= min_length]
    skipped = len(paths) - len(todo)
    for k, pid in enumerate(sorted(paths)):
        w = paths[pid]
        if len(w) < min_length:
            out[pid] = list(w)
            continue
        s0 = tests[pid]
        fn = reduce_path_to_fixpoint if fixpoint else reduce_path
        new = fn(puzzle, s0, w, radius, bk=bk, zob=zob, chunk=chunk, stats=stats)
        assert puzzle.solves(s0, new), "pid %s: reduced path does not solve" % pid
        out[pid] = new
        if progress and (k + 1) % progress == 0:
            log("  %d/%d pids, saved %d moves, %.1fs"
                % (k + 1, len(paths), stats["saved"], time.time() - t0))
    before = sum(len(v) for v in paths.values())
    after = sum(len(v) for v in out.values())
    report = {
        "pids": len(paths), "swept": len(todo), "skipped_short": skipped,
        "before": before, "after": after, "saved": before - after,
        "splices": stats["splices"], "phantom_rejects": stats["phantom"],
        "radius": radius, "certified_geodesic_upto": 2 * radius,
        "seconds": round(time.time() - t0, 1),
    }
    return out, report


def certify_report(report):
    """One-line summary that states the BAND, so a zero cannot be over-read."""
    r = report
    return ("window reduce r=%d: %s -> %s (%+d moves, %d splices) | proves every window "
            "of length <= %d geodesic; longer windows only certified d > %d"
            % (r["radius"], "{:,}".format(r["before"]), "{:,}".format(r["after"]),
               -r["saved"], r["splices"], r["certified_geodesic_upto"],
               r["certified_geodesic_upto"]))


def _bump(stats, key, by=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + by
