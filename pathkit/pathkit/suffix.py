"""Suffix re-solve: the one sub-problem a distance model can actually score.

Interior windows have an ARBITRARY target state, and a value model trained to predict
distance-to-solved says nothing about distance-to-s_j. The suffix is the exception: solving
s_{L-k} means reaching the real solved state, so the existing scorer, beam, table and
symmetry machinery all apply unchanged. On colour puzzles, where no residual can be
extracted at all (no inverse state), this is the only neural post-processing available.

USE IT AS A DIAGNOSTIC FIRST. Running it at several k answers a question no amount of
rewriting can: is our search able to beat this file anywhere? On the best public cube444
file the answer was recorded as

    k=12: 6/6 suffixes returned exactly 12    (matched, never beat)
    k=16: 3 matched, 1 worse, 2 no solution
    k=20: 2 matched, 1 worse, 3 no solution   -> beat-k: 0 / 18

which says the file is already optimal at every depth our beam can search, and the misses
at k=16/20 are OUR width limit rather than slack in the file. That result is what redirected
the effort from post-processing to a better global search, and it cost minutes to obtain.
"""
from __future__ import annotations

import numpy as np


def resolve_suffix(puzzle, s0, word, k, solver, verify=True):
    """Re-solve the last k moves. Returns (new_word, outcome) with outcome in
    {"beat", "match", "worse", "nosolution"}."""
    L = len(word)
    if k <= 0 or k > L:
        return list(word), "nosolution"
    states = puzzle.path_states(s0, word)
    tail_start = L - k
    v = solver(states[tail_start])
    if not v:
        return list(word), "nosolution"
    if verify and not puzzle.solves(states[tail_start], v):
        return list(word), "nosolution"
    if len(v) < k:
        return list(word[:tail_start]) + list(v), "beat"
    return list(word), ("match" if len(v) == k else "worse")


def probe(puzzle, tests, paths, solver, ks=(12, 16, 20), pids=None, log=print):
    """Run the diagnostic over a sample of pids and report the outcome table.

    `beat` counts are the only ones that mean there is slack. A column of pure `match`
    means the file is at our search's optimum for that depth: stop rewriting, improve the
    search.
    """
    pids = list(pids or sorted(paths))
    table = {}
    for k in ks:
        counts = {"beat": 0, "match": 0, "worse": 0, "nosolution": 0}
        saved = 0
        for pid in pids:
            new, outcome = resolve_suffix(puzzle, tests[pid], paths[pid], k, solver)
            counts[outcome] += 1
            saved += len(paths[pid]) - len(new)
        table[k] = dict(counts, saved=saved)
        log("k=%-3d beat=%d match=%d worse=%d nosolution=%d  (saved %d)"
            % (k, counts["beat"], counts["match"], counts["worse"],
               counts["nosolution"], saved))
    total_beat = sum(t["beat"] for t in table.values())
    if total_beat == 0:
        log("beat-k: 0 -- the search never beats this file on a sub-problem it can score. "
            "Remaining slack is global; a better search is the lever, not more rewriting.")
    return table


def sweep(puzzle, tests, paths, solver, ks=(12, 16, 20), progress=None, log=print):
    """Apply suffix re-solve for each k in turn, keeping every improvement."""
    out = {pid: list(w) for pid, w in paths.items()}
    stats = {"beat": 0, "saved": 0}
    for k in ks:
        for n, pid in enumerate(sorted(out)):
            new, outcome = resolve_suffix(puzzle, tests[pid], out[pid], k, solver)
            if outcome == "beat":
                assert puzzle.solves(tests[pid], new), "pid %s: suffix splice broke path" % pid
                stats["saved"] += len(out[pid]) - len(new)
                stats["beat"] += 1
                out[pid] = new
            if progress and (n + 1) % progress == 0:
                log("  k=%d %d/%d saved %d" % (k, n + 1, len(out), stats["saved"]))
    before = sum(len(v) for v in paths.values())
    after = sum(len(v) for v in out.values())
    return out, {"before": before, "after": after, "saved": before - after,
                 "improved_pids": stats["beat"], "ks": list(ks)}


def make_table_solver(table, puzzle, extend=0):
    """A solver callable backed by an exact BFS table -- handy for testing the harness
    without wiring a neural beam. Returns None when the state is out of reach."""
    def solve(state):
        d = int(table.lookup(state[None, :])[0])
        if d >= 0:
            return table.descend(state, puzzle)
        if extend > 0:
            frontier = state[None, :]
            words = [[]]
            for depth in range(1, extend + 1):
                nxt = frontier[:, puzzle.gens].reshape(-1, state.shape[0])
                nwords = [w + [g] for w in words for g in range(puzzle.n_gen)]
                ds = table.lookup(nxt)
                hits = np.nonzero(ds >= 0)[0]
                if hits.size:
                    best = min(hits, key=lambda h: int(ds[h]) + depth)
                    return nwords[best] + table.descend(nxt[best], puzzle)
                frontier, words = nxt, nwords
        return None
    return solve
