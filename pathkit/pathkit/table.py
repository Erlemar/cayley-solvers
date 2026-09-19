"""Exact BFS-table window rewriting, and the --extend trick that prices a deeper table.

WHEN TO USE THIS INSTEAD OF THE BALL SWEEP (window.py). The ball sweep is anchored at the
path states, so it must be rebuilt per path. A BFS table is anchored at the IDENTITY, so
it is built once and reused for every window of every path forever. On a permutation
puzzle a window [i, j) equals the group element

    w = s_i^-1 . s_j          (with apply(s, g) = s[g])

which is a single table lookup. That makes table rewriting the cheap way to sweep a whole
submission, and the ball sweep the way to go deeper than any table you can afford.

Colour puzzles cannot use this: without distinct labels there is no s_i^-1, and the right
tool is the state-space ball sweep. `window_reduce` asserts on that.

THE --extend TRICK, AND WHY IT MATTERS MORE THAN THE TABLE DEPTH. A table of depth D only
knows distances up to D. But if some short word v takes w INTO the table at depth d, then
w = (w.v) . v^-1, giving a word of length d + |v|. So BFS `k` moves out from w, look every
frontier state up, keep the best (d + |v|): effective reach becomes D + k WITHOUT building
a deeper table. This is how a d8 table was priced at zero on tetraminx -- a d6 table with
extend=2 reaches exactly what d8 would have reached, found nothing, and saved the build.
Cost per window is about n_gen^k lookups, so k=2 is usually the practical limit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class BfsTable:
    """Hash -> depth for every state within `max_depth` of a root, sorted for searchsorted.

    Compatible with this repo's bfs_endgame*.npz (keys: hashes, depths, ztab, max_depth).
    """

    def __init__(self, hashes, depths, ztab, max_depth, root=None):
        self.hashes = np.asarray(hashes)
        self.depths = np.asarray(depths)
        self.ztab = np.asarray(ztab)
        self.max_depth = int(max_depth)
        self.root = root

    # ------------------------------------------------------------------ build
    @classmethod
    def load(cls, path):
        d = np.load(Path(path), allow_pickle=True)
        return cls(d["hashes"], d["depths"], d["ztab"], int(d["max_depth"]))

    @classmethod
    def build(cls, puzzle, max_depth, root=None, seed=20260820, log=None):
        """BFS from `root` (default: the identity permutation, which is what window
        rewriting needs) out to `max_depth`. Only for puzzles small enough to hold."""
        n = puzzle.state_size
        root = np.arange(n, dtype=np.int64) if root is None else np.asarray(root)
        rng = np.random.default_rng(seed)
        ztab = rng.integers(-(2 ** 62), 2 ** 62, size=(n, puzzle.n_labels), dtype=np.int64)

        def hsh(states):
            states = np.atleast_2d(states)
            h = np.zeros(states.shape[0], dtype=np.int64)
            for i in range(states.shape[1]):
                h ^= ztab[i][states[:, i]]
            return h

        frontier = root[None, :]
        all_h = [hsh(frontier)]
        all_d = [np.zeros(1, dtype=np.int8)]
        seen = np.sort(all_h[0])
        for d in range(1, max_depth + 1):
            kids = frontier[:, puzzle.gens].reshape(-1, n)
            kh = hsh(kids)
            o = np.argsort(kh, kind="stable")
            kh_s, kids_s = kh[o], kids[o]
            uniq = np.ones(kh_s.shape[0], dtype=bool)
            uniq[1:] = kh_s[1:] != kh_s[:-1]
            kh_s, kids_s = kh_s[uniq], kids_s[uniq]
            pos = np.clip(np.searchsorted(seen, kh_s), 0, len(seen) - 1)
            fresh = seen[pos] != kh_s
            kh_s, kids_s = kh_s[fresh], kids_s[fresh]
            if kh_s.size == 0:
                break
            all_h.append(kh_s)
            all_d.append(np.full(kh_s.shape[0], d, dtype=np.int8))
            seen = np.sort(np.concatenate([seen, kh_s]))
            frontier = kids_s
            if log:
                log("  depth %d: %s states" % (d, "{:,}".format(kh_s.shape[0])))
        h = np.concatenate(all_h)
        dep = np.concatenate(all_d)
        o = np.argsort(h, kind="stable")
        return cls(h[o], dep[o], ztab, max_depth, root)

    def save(self, path):
        np.savez_compressed(path, hashes=self.hashes, depths=self.depths,
                            ztab=self.ztab, max_depth=self.max_depth)

    # ----------------------------------------------------------------- lookup
    def hash(self, states):
        states = np.atleast_2d(states)
        h = np.zeros(states.shape[0], dtype=np.int64)
        for i in range(states.shape[1]):
            h ^= self.ztab[i][states[:, i]]
        return h

    def lookup(self, states):
        """Depth of each state, or -1 when absent."""
        h = self.hash(states)
        pos = np.clip(np.searchsorted(self.hashes, h), 0, len(self.hashes) - 1)
        out = np.full(h.shape[0], -1, dtype=np.int64)
        hit = self.hashes[pos] == h
        out[hit] = self.depths[pos[hit]]
        return out

    def descend(self, state, puzzle):
        """Move indices taking `state` to the root, one depth level at a time."""
        d = int(self.lookup(state[None, :])[0])
        assert d >= 0, "state is not in the table"
        out = []
        cur = state
        while d > 0:
            children = cur[puzzle.gens]
            cd = self.lookup(children)
            nxt = np.nonzero(cd == d - 1)[0]
            assert nxt.size, "table descent stalled -- probable 64-bit hash phantom"
            out.append(int(nxt[0]))
            cur = children[out[-1]]
            d -= 1
        return out

    def shortest_word(self, puzzle, w, extend=0):
        """Shortest word equal to the group element `w`; exact whenever
        dist(w) <= max_depth + extend. Returns None if nothing is reachable."""
        best = None
        d0 = int(self.lookup(w[None, :])[0])
        if d0 >= 0:
            best = (d0, [])
        frontier = w[None, :]
        words = [[]]
        for depth in range(1, extend + 1):
            if best is not None and best[0] <= depth:
                break                                # cannot beat what we already have
            nxt = frontier[:, puzzle.gens].reshape(-1, w.shape[0])
            nwords = [wd + [g] for wd in words for g in range(puzzle.n_gen)]
            ds = self.lookup(nxt)
            for hidx in np.nonzero(ds >= 0)[0]:
                total = int(ds[hidx]) + depth
                if best is None or total < best[0]:
                    best = (total, nwords[hidx])
            frontier, words = nxt, nwords
        if best is None:
            return None
        total, prefix = best
        cur = w
        for g in prefix:
            cur = cur[puzzle.gens[g]]
        # w . prefix = t  =>  w = t . prefix^-1, and the word for t is descend(t) inverted
        tail = puzzle.inv_word(self.descend(cur, puzzle))
        word = tail + puzzle.inv_word(prefix)
        assert len(word) == total
        return word


def window_reduce(puzzle, s0, word, table, max_window=14, extend=0, extend_min=8,
                  stats=None):
    """Shorten one path by replacing any window whose group element sits in the table.

    Restarts the scan after each accepted rewrite: a splice changes every state downstream
    of it, so continuing from the old trace would test windows that no longer exist.
    """
    assert puzzle.is_permutation, (
        "table rewriting needs distinct labels (no inverse state on a colour puzzle); "
        "use window.sweep, the state-space ball method, instead")
    n = puzzle.state_size
    path = list(word)
    improved = True
    while improved:
        improved = False
        states = puzzle.path_states(s0, path)
        L = len(path)
        for i in range(L):
            hi = min(L, i + max_window)
            inv_si = np.empty(n, dtype=np.int64)
            inv_si[states[i]] = np.arange(n)
            for j in range(i + 2, hi + 1):
                w = inv_si[states[j]]
                new = None
                d = int(table.lookup(w[None, :])[0])
                if 0 <= d < j - i:
                    new = puzzle.inv_word(table.descend(w, puzzle))
                elif extend > 0 and (j - i) >= extend_min:
                    cand = table.shortest_word(puzzle, w, extend)
                    if cand is not None and len(cand) < j - i:
                        new = cand
                if new is not None:
                    if not np.array_equal(puzzle.apply_word(states[i], new), states[j]):
                        if stats is not None:
                            stats["phantom"] = stats.get("phantom", 0) + 1
                        continue
                    path = path[:i] + new + path[j:]
                    if stats is not None:
                        stats["splices"] = stats.get("splices", 0) + 1
                        stats["saved"] = stats.get("saved", 0) + (j - i) - len(new)
                    improved = True
                    break
            if improved:
                break
    assert puzzle.solves(s0, path), "table window reduce broke the path"
    return path


def sweep(puzzle, tests, paths, table, max_window=14, extend=0, extend_min=8,
          progress=None, log=print):
    """Table window-reduce a whole submission. Returns (new_paths, report)."""
    stats = {"splices": 0, "saved": 0, "phantom": 0}
    out = {}
    for k, pid in enumerate(sorted(paths)):
        out[pid] = window_reduce(puzzle, tests[pid], paths[pid], table,
                                 max_window, extend, extend_min, stats)
        if progress and (k + 1) % progress == 0:
            log("  %d/%d pids, saved %d" % (k + 1, len(paths), stats["saved"]))
    before = sum(len(v) for v in paths.values())
    after = sum(len(v) for v in out.values())
    return out, {
        "before": before, "after": after, "saved": before - after,
        "splices": stats["splices"], "phantom_rejects": stats["phantom"],
        "table_depth": table.max_depth, "extend": extend,
        "effective_reach": table.max_depth + extend, "max_window": max_window,
    }
