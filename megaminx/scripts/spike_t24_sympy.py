"""Day-1 spike for T2.4: try sympy SGS construction + coset_factor on megaminx.

Tests whether sympy can:
  (a) build the megaminx PermutationGroup with our 24 generators
  (b) compute a strong generating set in reasonable time
  (c) factor a known scramble into a generator word
  (d) measure factorization length on a few stratified pids vs current beam paths

If sympy fails any of these, T2.4 escalates to GAP (gap_python) or we abandon.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

import numpy as np

from megaminx.puzzle import Megaminx


def main() -> int:
    print("loading sympy...", flush=True)
    t0 = time.time()
    from sympy.combinatorics import Permutation, PermutationGroup
    print(f"  sympy loaded in {time.time()-t0:.2f}s")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    print(f"\nbuilding sympy PermutationGroup with {len(move_names)} generators...", flush=True)
    t0 = time.time()
    sympy_gens = []
    for n in move_names:
        sympy_gens.append(Permutation(list(puzzle.generators[n])))
    G = PermutationGroup(sympy_gens)
    print(f"  group constructed in {time.time()-t0:.2f}s")

    print(f"\ncomputing order via Schreier-Sims...", flush=True)
    t0 = time.time()
    try:
        order = G.order()
        print(f"  |G| = {order:,} ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 2

    print(f"\ncomputing strong generating set...", flush=True)
    t0 = time.time()
    try:
        sgs = G.strong_gens
        base = G.base
        print(f"  |SGS| = {len(sgs)}, |base| = {len(base)} ({time.time()-t0:.2f}s)")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 3

    # Smoke factorization: factor a single R move (should be 1 generator).
    print(f"\nsmoke factor: a single R move...", flush=True)
    R_perm = Permutation(list(puzzle.generators["R"]))
    t0 = time.time()
    try:
        factored = G.coset_factor(R_perm)
        print(f"  coset_factor returned {len(factored)} pieces in {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        return 4

    # Real test: factor a few stratified test scrambles, measure length.
    print(f"\nfactoring stratified test scrambles...", flush=True)
    test_states: dict[int, np.ndarray] = {}
    with open(PROJECT / "data" / "test.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_states[int(row["initial_state_id"])] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int8
            )

    sample_pids = [0, 100, 200, 500, 800]
    solved = list(puzzle.solved_state)
    solved_inv_perm = ~Permutation(solved)

    for pid in sample_pids:
        scramble_state = list(test_states[pid])
        scramble_perm = Permutation(scramble_state) * solved_inv_perm
        t0 = time.time()
        try:
            factored = G.coset_factor(scramble_perm)
            n_pieces = len(factored)
            elapsed = time.time() - t0
            print(f"  pid={pid:4d}: {n_pieces} pieces in {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"  pid={pid:4d}: FAILED: {e}")

    print("\n--- DECISION HINTS ---")
    print("- If coset_factor produced reasonable lengths (50-200) per pid: sympy is sufficient.")
    print("- If lengths are 500+ or factorization timed out: need real Minkwitz (port from GAP).")
    print("- Compare to current best path lengths:")
    cur_best = PROJECT / "submissions" / "merge_relinked_plus_pi05_top200.csv"
    if cur_best.exists():
        with open(cur_best, encoding="utf-8") as f:
            sub = {int(r["initial_state_id"]): r["path"].split(".") if r["path"] else []
                   for r in csv.DictReader(f)}
        for pid in sample_pids:
            print(f"  pid={pid:4d}: current best path = {len(sub[pid])} moves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
