"""One puzzle abstraction for every solver in this repo, and for the demo puzzles.

All four competition puzzles ship the same puzzle_info.json:

    {"central_state": [...], "generators": {"f0": [...], "-f0": [...]}}

so one loader covers IHES picture cube (72 labels, all distinct), tetraminx (88),
megaminx (120) and cube444 (96 facelets but only 6 COLOURS). That last difference is
not cosmetic -- it changes what a legal path rewrite is, and is_permutation is the
flag every method here branches on when it reports what it proved.

CONVENTION (identical to the repo scripts): a generator is an index array and
apply(s, g) = s[g]. Composing a then b is therefore the permutation a[b].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Puzzle:
    names: list                      # move names, index i maps to gens[i]
    gens: np.ndarray                 # (n_gen, N) int64
    solved: np.ndarray               # (N,) int64 -- the target ("central_state")
    inv_move: np.ndarray             # (n_gen,) int64, index of the inverse of each move
    label: str = "puzzle"

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_puzzle_info(cls, path, label=None):
        path = Path(path)
        info = json.loads(path.read_text(encoding="utf-8"))
        names = list(info["generators"].keys())
        gens = np.array([info["generators"][n] for n in names], dtype=np.int64)
        solved = np.array(info["central_state"], dtype=np.int64)
        return cls.build(names, gens, solved, label or info.get("name") or path.parent.name)

    @classmethod
    def build(cls, names, gens, solved, label="puzzle"):
        gens = np.asarray(gens, dtype=np.int64)
        solved = np.asarray(solved, dtype=np.int64)
        inv = _inverse_map(names, gens)
        return cls(list(names), gens, solved, inv, label)

    # ------------------------------------------------------------ demo puzzles
    @classmethod
    def demo_lrx(cls, n=12):
        """LRX on n positions: L = cyclic left, R = cyclic right, X = swap first two.

        Generates the full symmetric group, so it is a real puzzle with a real diameter,
        it needs no data files, and X is its own inverse -- which exercises the structural
        inverse detection that a sign-prefix convention would hide.
        """
        idx = np.arange(n)
        gl = (idx + 1) % n
        gr = (idx - 1) % n
        gx = idx.copy()
        gx[0], gx[1] = 1, 0
        return cls.build(["L", "R", "X"], np.stack([gl, gr, gx]), idx, "lrx%d" % n)

    @classmethod
    def demo_lrx_coloured(cls, n=12, k=3):
        """Same group, but the solved state is a k-COLOUR blocking of the n positions.

        This is the cube444 situation in miniature: many states share a colouring, so a
        window may be replaced by any word landing on the same COLOURING, which is
        strictly weaker than landing on the same permutation.
        """
        assert n % k == 0, "n must divide by k"
        p = cls.demo_lrx(n)
        return cls.build(p.names, p.gens, np.repeat(np.arange(k), n // k),
                         "lrx%dc%d" % (n, k))

    # ------------------------------------------------------------- properties
    @property
    def n_gen(self):
        return self.gens.shape[0]

    @property
    def state_size(self):
        return self.gens.shape[1]

    @property
    def n_labels(self):
        return int(self.solved.max()) + 1

    @property
    def is_permutation(self):
        """True when every label is distinct, i.e. solved is the identity up to relabelling.

        When True, "states equal" and "permutations equal" coincide and every window
        rewrite found here is also a permutation rewrite. When False (colour puzzle) the
        state test is STRICTLY weaker and finds rewrites a permutation MITM cannot see.
        """
        s = np.sort(self.solved)
        return (s.shape[0] == self.state_size
                and bool(np.array_equal(s, np.arange(self.state_size))))

    # ------------------------------------------------------------- operations
    def apply_move(self, state, m):
        return state[self.gens[m]]

    def apply_word(self, state, word):
        s = state
        for m in word:
            s = s[self.gens[m]]
        return s

    def path_states(self, s0, word):
        """(len(word)+1, N) -- every intermediate state, s_0 through s_L."""
        out = np.empty((len(word) + 1, self.state_size), dtype=s0.dtype)
        out[0] = s0
        s = s0
        for k, m in enumerate(word):
            s = s[self.gens[m]]
            out[k + 1] = s
        return out

    def inv_word(self, word):
        return [int(self.inv_move[m]) for m in reversed(list(word))]

    def solves(self, s0, word):
        return bool(np.array_equal(self.apply_word(s0, word), self.solved))

    # ------------------------------------------------------------------- text
    def parse(self, path_str):
        idx = {n: i for i, n in enumerate(self.names)}
        return [idx[m] for m in path_str.split(".") if m]

    def format(self, word):
        return ".".join(self.names[int(m)] for m in word)

    def in_alphabet(self, path_str, probe=20):
        """Cheap content test: do the first `probe` tokens name moves of THIS puzzle?

        This is what makes a content-scan merge safe to point at unrelated folders -- a
        file from another puzzle self-rejects here instead of corrupting the floor.
        """
        alpha = set(self.names)
        toks = [t for t in path_str.split(".") if t][:probe]
        return bool(toks) and all(t in alpha for t in toks)

    # -------------------------------------------------------------- subgroups
    def restricted(self, keep):
        """A view using only generators `keep` (indices).

        A subset generates a subgroup with a much smaller branching factor, so the same
        ball budget reaches several moves deeper. Any word the restricted puzzle returns
        is still legal to splice anywhere: restricting the SEARCH does not restrict where
        the answer may be used.
        """
        keep = list(keep)
        assert keep, "empty generator subset"
        pos = {g: i for i, g in enumerate(keep)}
        for g in keep:
            assert int(self.inv_move[g]) in pos, (
                "subset not closed under inverse: " + str(self.names[g]))
        return Puzzle.build([self.names[g] for g in keep], self.gens[keep],
                            self.solved, self.label + "-sub" + str(len(keep)))

    # ------------------------------------------------------------------ misc
    def random_scramble(self, depth, rng):
        """A state at distance <= depth, plus a word that solves it."""
        word = []
        s = self.solved.copy()
        last_inv = -1
        for _ in range(depth):
            m = int(rng.integers(self.n_gen))
            if m == last_inv:                       # skip the trivial immediate undo
                continue
            s = s[self.gens[m]]
            word.append(m)
            last_inv = int(self.inv_move[m])
        return s, self.inv_word(word)

    def describe(self):
        kind = "permutation" if self.is_permutation else "colour (%d labels)" % self.n_labels
        return "%s: N=%d, %d generators, %s" % (
            self.label, self.state_size, self.n_gen, kind)


def _inverse_map(names, gens):
    """Inverse of every generator, by name when the sign convention holds, else structural.

    The structural path is not a fallback nicety: LRX-style generator sets have
    self-inverse moves and no sign convention at all, and a name-only mapping would
    silently produce a wrong inv_word -- which corrupts every splice rather than
    raising. Whatever the source, the result is verified below.
    """
    n_gen, N = gens.shape
    ident = np.arange(N)
    idx = {n: i for i, n in enumerate(names)}
    inv = np.full(n_gen, -1, dtype=np.int64)
    for i, nm in enumerate(names):                       # try the naming convention
        cand = nm[1:] if nm.startswith("-") else "-" + nm
        j = idx.get(cand)
        if j is not None and np.array_equal(gens[i][gens[j]], ident):
            inv[i] = j
    for i in range(n_gen):                               # structural for whatever is left
        if inv[i] >= 0:
            continue
        for j in range(n_gen):
            if np.array_equal(gens[i][gens[j]], ident):
                inv[i] = j
                break
    missing = [names[i] for i in range(n_gen) if inv[i] < 0]
    if missing:
        raise ValueError("generator set is not closed under inverse; no inverse for: "
                         + ", ".join(missing[:8]))
    for i in range(n_gen):                               # verify, never assume
        assert np.array_equal(gens[i][gens[inv[i]]], ident), (
            "bad inverse for " + str(names[i]))
    return inv
