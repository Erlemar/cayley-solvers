"""Dedup + exact-label override for cube444 random-walk training data.

THE PROBLEM. Two label streams feed the trainer and they contradict each other where they
overlap: random-walk labels (target = walk index) and exact BFS anchors (target = true
distance). A non-backtracking walk forbids only the immediate inverse, so it is NOT
geodesic -- measured on a sibling puzzle, 37.1% of walk pivots land inside the exact ball
and 25.9% of those carry a label != the true distance (always p >= d, never below).

WHY IT IS WORSE ON THIS CUBE. The state is a COLOURING, so it pins the group element only
up to the 4!^6 = 191,102,976 stabiliser of the solved colouring. The true distance of a
colouring is a minimum over that whole coset:

    d(colouring) = min over g' in g.Stab of |g'|

A walk hands you one specific g and labels the state |walk|. The colouring's real distance
is a min over ~1.9e8 elements and can be far below it. So labels over-estimate for two
stacked reasons -- non-geodesic slack AND coset collapse -- and the second has no analogue
on the permutation puzzles this was first measured on. It also makes DUPLICATE states with
DIFFERENT labels common, since two walks of different lengths reach the same colouring.

THE FIX. Two passes, both cheap because the exact ball is already resident for anchors:
  A. dedup by state, keep the MINIMUM label
  B. where the exact table knows the state, replace the label with the true distance

Pass B is one-way by construction (p >= d), so it can only lower a label. That is asserted
-- a table claiming a LONGER distance than the walk means the hashing or the generator
convention is wrong, not the table.

    python cube444/scripts/20_label_fix.py --self-test
    # or import: from importlib import ...; dedup_min_label / exact_override / fix_batch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "cube444" / "src"))


def make_ztab(state_size: int, num_classes: int, seed: int = 0) -> np.ndarray:
    """Zobrist table. MUST match the one used to build the exact table."""
    rng = np.random.default_rng(seed)
    return rng.integers(-(2 ** 63), 2 ** 63 - 1, size=(state_size, num_classes),
                        dtype=np.int64)


def zhash(states: np.ndarray, ztab: np.ndarray) -> np.ndarray:
    """(N, S) uint8 colourings -> (N,) int64 Zobrist."""
    h = np.zeros(states.shape[0], dtype=np.int64)
    for i in range(states.shape[1]):
        h ^= ztab[i][states[:, i]]
    return h


def dedup_min_label(states: np.ndarray, labels: np.ndarray, ztab: np.ndarray):
    """Collapse duplicate colourings, keeping the MINIMUM label for each.

    Keeping both copies trains the net to output two different numbers for one input;
    keeping the larger trains it to the worse of two upper bounds.
    """
    h = zhash(states, ztab)
    order = np.lexsort((labels, h))                 # group by hash, smallest label first
    h_s, s_s, l_s = h[order], states[order], labels[order]
    keep = np.empty(h_s.size, dtype=bool)
    keep[0] = True
    np.not_equal(h_s[1:], h_s[:-1], out=keep[1:])
    return s_s[keep], l_s[keep]


def exact_override(states: np.ndarray, labels: np.ndarray, table_hashes: np.ndarray,
                   table_depths: np.ndarray, ztab: np.ndarray):
    """Replace walk labels with exact distances wherever the table knows the state.

    Returns (labels, n_in_table, n_lowered).
    """
    h = zhash(states, ztab)
    pos = np.searchsorted(table_hashes, h)
    np.clip(pos, 0, table_hashes.size - 1, out=pos)
    hit = table_hashes[pos] == h
    exact = table_depths[pos].astype(labels.dtype)
    if np.any(hit & (exact > labels)):
        bad = int(np.count_nonzero(hit & (exact > labels)))
        raise AssertionError(
            f"{bad} states where the exact table claims a LONGER distance than the walk. "
            "A walk length is an upper bound, so this cannot happen: your Zobrist table "
            "or your generator convention disagrees with the one used to build the table.")
    n_lowered = int(np.count_nonzero(hit & (exact < labels)))
    out = np.where(hit, exact, labels)
    return out, int(hit.sum()), n_lowered


def fix_batch(states, labels, ztab, table_hashes=None, table_depths=None, verbose=True):
    """Full pipeline: dedup then (optionally) exact-override. Returns (states, labels)."""
    n0 = states.shape[0]
    states, labels = dedup_min_label(states, labels, ztab)
    n1 = states.shape[0]
    n_in = n_low = 0
    if table_hashes is not None:
        labels, n_in, n_low = exact_override(states, labels, table_hashes,
                                             table_depths, ztab)
    if verbose:
        print(f"  label-fix: {n0:,} -> {n1:,} rows (dup {100*(1-n1/max(n0,1)):.1f}%), "
              f"{n_in:,} in table, {n_low:,} labels lowered", flush=True)
    return states, labels


def sparse_q_drop_bad_pairs(pivot_states, undo_child, next_child, ztab,
                            table_hashes, table_depths):
    """For the sparse-Q objective: keep only rows whose asserted ordering is real.

    The label asserts Q(s,undo) = p-1 and Q(s,next) = p+1, i.e. a gap of exactly 2. Where
    both children are in the exact table we can check it. Measured on a sibling puzzle:
    ordering is right 92.75% of the time, tie 6.14%, INVERTED 1.11% -- so ~7.25% of
    labelled pairs assert a gap that does not exist. Ties are the benign-alternative-optima
    trap; inversions are outright wrong.

    Returns a boolean keep-mask.
    """
    du = zhash(undo_child, ztab)
    dn = zhash(next_child, ztab)
    keep = np.ones(pivot_states.shape[0], dtype=bool)
    for h, side in ((du, "undo"), (dn, "next")):
        pos = np.searchsorted(table_hashes, h)
        np.clip(pos, 0, table_hashes.size - 1, out=pos)
        hit = table_hashes[pos] == h
        d = np.where(hit, table_depths[pos], -1).astype(np.int64)
        if side == "undo":
            d_undo, hit_undo = d, hit
        else:
            d_next, hit_next = d, hit
    both = hit_undo & hit_next
    keep[both] = d_next[both] > d_undo[both]        # drop ties AND inversions
    return keep


def _self_test() -> int:
    """Synthetic: duplicates collapse to the min, and the override only lowers."""
    rng = np.random.default_rng(0)
    S, C = 96, 6
    ztab = make_ztab(S, C, seed=0)

    base = rng.integers(0, C, size=(50, S), dtype=np.uint8)
    states = np.repeat(base, 3, axis=0)
    labels = rng.integers(5, 40, size=states.shape[0]).astype(np.int64)
    ds, dl = dedup_min_label(states, labels, ztab)
    assert ds.shape[0] == 50, f"expected 50 unique, got {ds.shape[0]}"
    for i in range(50):
        want = labels[3 * i:3 * i + 3].min()
        h_one = zhash(ds, ztab)
        j = int(np.nonzero(h_one == zhash(base[i:i + 1], ztab)[0])[0][0])
        assert dl[j] == want, f"row {i}: kept {dl[j]}, min was {want}"
    print("  dedup_min_label: OK (duplicates collapse, minimum label kept)")

    # exact override: build a fake table over half the states with strictly lower depths
    keep_h = zhash(ds[:25], ztab)
    order = np.argsort(keep_h)
    th = keep_h[order]
    td = (dl[:25][order] - 2).astype(np.int8)       # strictly lower => must be applied
    lab2, n_in, n_low = exact_override(ds[:25], dl[:25], th, td, ztab)
    assert n_in == 25 and n_low == 25, f"in={n_in} low={n_low}"
    assert np.all(lab2 == dl[:25] - 2)
    print("  exact_override: OK (all in-table labels lowered to the exact value)")

    try:
        exact_override(ds[:25], dl[:25], th, (td + 10).astype(np.int8), ztab)
    except AssertionError:
        print("  exact_override: OK (rejects a table claiming a LONGER distance)")
    else:
        print("  exact_override: FAIL -- did not reject an impossible table")
        return 1
    print("self-test: PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(_self_test())
    ap.print_help()
