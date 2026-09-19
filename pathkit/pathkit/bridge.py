"""Pooled cross-trajectory bridging: the generalisation that contains the other methods.

Take EVERY path we own for one puzzle id, pool their waypoints, and expand a ball of
radius r around each. Two trajectories that merely pass NEAR each other can then be joined:

    new length = g(x) + d(x, z) + d(z, y) + h(y)

where g(x) is the cheapest cost-from-start over all pooled paths reaching waypoint x, and
h(y) the cheapest cost-to-solved from waypoint y. Per colliding state z that is one min
over (g + d) and one over (h + d) -- no pair enumeration.

Because the pool holds every path, ONE pass subsumes:
    * n-way per-pid min-merge          -- x = start, y = solved, same path
    * exact crossover splicing         -- d1 = d2 = 0, two paths sharing a state
    * within-path window shortening    -- both waypoints from the same path
    * genuine bridges BETWEEN paths    -- the part nothing else does

and it does REWRITE-ALL-THEN-MERGE rather than merge-then-rewrite. That order matters and
is easy to get wrong: the rewriter is path-dependent, so a shorter path can be geodesic
everywhere and immune while a longer one contains a window that collapses hard. Merging
first throws the longer path away before any rewriter sees it. Merging after is free.

KNOW THE BOUND BEFORE SPENDING GPU ON IT. A splice through waypoint x can only beat the
incumbent if g(x) + d(x, solved) < incumbent, and since
d(x^-1 y) + h(y) >= d(x^-1 y) + d(y) >= d(x), a pool of paths that are all dominated has
no slack to contribute at all. Measured on tetraminx: over 6,979 waypoints with an exactly
known d(x), ZERO had slack -- adding more trajectories could not have helped. Check that
before scaling the pool.
"""
from __future__ import annotations

import time

import numpy as np

from .backend import get_backend
from .balls import Zobrist, build_balls, group_min, word_to


def bridge_pid(puzzle, s0, paths, radius, bk=None, zob=None, chunk=150_000, stats=None):
    """Best word for one pid given a pool of valid solutions. Never returns worse."""
    bk = bk or get_backend("cpu")
    zob = zob or Zobrist(puzzle.state_size, puzzle.n_labels, bk)
    paths = [list(p) for p in paths if p]
    if not paths:
        return []
    best = min(paths, key=len)
    if len(best) < 3:
        return list(best)

    # Pool waypoints. Per distinct state keep the cheapest way IN and the cheapest way OUT
    # (they may come from different paths -- that is exactly the crossover case).
    pool = {}
    for pi, w in enumerate(paths):
        st = puzzle.path_states(s0, w)
        L = len(w)
        for k in range(L + 1):
            key = st[k].tobytes()
            rec = pool.get(key)
            if rec is None:
                pool[key] = [k, pi, k, L - k, pi, k, st[k]]
            else:
                if k < rec[0]:
                    rec[0], rec[1], rec[2] = k, pi, k
                if L - k < rec[3]:
                    rec[3], rec[4], rec[5] = L - k, pi, k
    recs = list(pool.values())
    _bump(stats, "waypoints", len(recs))

    sources = np.stack([r[6] for r in recs])
    balls = build_balls(puzzle, sources, radius, bk=bk, zob=zob, chunk=chunk)
    g_lab = bk.asarray(np.array([r[0] for r in recs], dtype=np.int32))
    h_lab = bk.asarray(np.array([r[3] for r in recs], dtype=np.int32))
    s_idx = balls.src
    idx = s_idx.astype(np.int64) if not bk.is_torch else s_idx.long()
    dist32 = balls.dist.astype(np.int32) if not bk.is_torch else balls.dist.to(bk.int32)
    a = g_lab[idx] + dist32
    b = h_lab[idx] + dist32
    cands = group_min(balls, a, b, limit=len(best))
    if not cands:
        bk.free()
        return list(best)

    _, src_np, dist_np, par_np, mv_np = balls.to_numpy()
    del balls
    bk.free()
    for total, e_lo, e_hi in cands:
        r_lo, r_hi = recs[int(src_np[e_lo])], recs[int(src_np[e_hi])]
        w1 = word_to(par_np, mv_np, e_lo)
        w2 = word_to(par_np, mv_np, e_hi)
        cand = (list(paths[r_lo[1]][:r_lo[2]]) + w1 + puzzle.inv_word(w2)
                + list(paths[r_hi[4]][r_hi[5]:]))
        if len(cand) >= len(best):
            continue
        if not puzzle.solves(s0, cand):          # hash phantom, or a stale pooled path
            _bump(stats, "phantom")
            continue
        _bump(stats, "bridged")
        _bump(stats, "saved", len(best) - len(cand))
        return cand
    return list(best)


def sweep(puzzle, tests, pools, radius, device="cpu", chunk=150_000, max_paths=8,
          slack=8, progress=None, log=print):
    """Bridge every pid. `pools` maps pid -> list of candidate words.

    Only paths within `slack` moves of the incumbent are pooled, capped at `max_paths`:
    a path far above the incumbent contributes waypoints that can never satisfy the bound
    above, so it costs ball budget and returns nothing.
    """
    bk = get_backend(device)
    zob = Zobrist(puzzle.state_size, puzzle.n_labels, bk)
    stats = {"waypoints": 0, "bridged": 0, "saved": 0, "phantom": 0}
    out, t0 = {}, time.time()
    for k, pid in enumerate(sorted(pools)):
        cands = sorted(pools[pid], key=len)
        if not cands:
            continue
        keep = [c for c in cands if len(c) <= len(cands[0]) + slack][:max_paths]
        out[pid] = bridge_pid(puzzle, tests[pid], keep, radius,
                              bk=bk, zob=zob, chunk=chunk, stats=stats)
        assert puzzle.solves(tests[pid], out[pid]), "pid %s: bridged path does not solve" % pid
        if progress and (k + 1) % progress == 0:
            log("  %d/%d pids, saved %d, %.1fs"
                % (k + 1, len(pools), stats["saved"], time.time() - t0))
    before = sum(min(len(c) for c in pools[p]) for p in out)
    after = sum(len(out[p]) for p in out)
    return out, {
        "pids": len(out), "before_merged": before, "after": after,
        "saved_beyond_merge": before - after, "bridged_pids": stats["bridged"],
        "waypoints": stats["waypoints"], "phantom_rejects": stats["phantom"],
        "radius": radius, "seconds": round(time.time() - t0, 1),
    }


def slack_audit(puzzle, s0, paths, table, log=print):
    """Price bridging BEFORE running it: how many pooled waypoints could possibly help?

    For every waypoint x whose exact distance to solved is known from `table`, the pool
    can only beat the incumbent if cost_from_start(x) + d(x) < incumbent. If that count is
    zero, no radius and no extra trajectory will produce a bridge -- stop here.
    """
    best = min(len(p) for p in paths)
    known = slackful = 0
    for w in paths:
        st = puzzle.path_states(s0, w)
        depths = table.lookup(st)
        for k, d in enumerate(depths.tolist()):
            if d < 0:
                continue
            known += 1
            if k + d < best:
                slackful += 1
    log("waypoints with exact d(x): %d, of which slackful: %d (incumbent %d)"
        % (known, slackful, best))
    return {"known": known, "slackful": slackful, "incumbent": best}


def _bump(stats, key, by=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + by
