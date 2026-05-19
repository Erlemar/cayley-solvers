"""T2.4 spike v2: convert sympy's coset_factor output to actual generator words.

`G.coset_factor(perm)` returns a list of coset representatives (one per SGS level).
Each rep is a group element. To get a generator-word, we need to:
  1. Compute the product of all reps (verify it equals the input permutation)
  2. For each rep, decompose into generators (this is where length really matters)

sympy's `Permutation.is_strong_gen_set_member` and friends give us coset rep info.
The naive way: each rep can be expressed as a sequence of strong generators (which
are themselves products of original generators). We need the FULL expansion.

This spike: try sympy's `G.generator_product(perm)` which gives a generator-word.
Compare to current best path lengths.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def main() -> int:
    from sympy.combinatorics import Permutation, PermutationGroup

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    sympy_gens = [Permutation(list(puzzle.generators[n])) for n in move_names]
    print("building group...", flush=True)
    G = PermutationGroup(sympy_gens)

    # Smoke: factor a simple R move; should be 1 generator long.
    R_perm = Permutation(list(puzzle.generators["R"]))
    try:
        word = G.generator_product(R_perm, original=True)
        print(f"smoke: generator_product(R) -> length {len(word)}")
    except Exception as e:
        print(f"smoke FAILED: {e}")
        return 1

    # Load test states.
    test_states: dict[int, np.ndarray] = {}
    with open(PROJECT / "data" / "test.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_states[int(row["initial_state_id"])] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int8
            )

    # Load current best to compare.
    cur_best = PROJECT / "submissions" / "merge_relinked_plus_pi05_top200.csv"
    sub: dict[int, list[str]] = {}
    if cur_best.exists():
        with open(cur_best, encoding="utf-8") as f:
            sub = {int(r["initial_state_id"]): r["path"].split(".") if r["path"] else []
                   for r in csv.DictReader(f)}

    # solved_inv computed once.
    solved_inv = ~Permutation(list(puzzle.solved_state))

    sample_pids = [0, 100, 200, 500, 800, 950, 1000]
    print(f"\nfactoring {len(sample_pids)} test pids and counting generator-word length...", flush=True)
    print(f"\n{'pid':<6} {'word_len':<10} {'current_best':<14} {'verdict':<10}")
    n_wins = 0
    n_loss = 0
    for pid in sample_pids:
        scramble_perm = Permutation(list(test_states[pid])) * solved_inv
        t0 = time.time()
        try:
            word = G.generator_product(scramble_perm, original=True)
            elapsed = time.time() - t0
            word_len = len(word)
            cur_len = len(sub.get(pid, []))
            verdict = "WIN" if cur_len > 0 and word_len < cur_len else "loss"
            if verdict == "WIN":
                n_wins += 1
            else:
                n_loss += 1
            print(f"{pid:<6} {word_len:<10} {cur_len:<14} {verdict:<10} ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"{pid:<6} FAILED: {e}")

    print(f"\nT2.4 sympy result: {n_wins}/{len(sample_pids)} wins, {n_loss}/{len(sample_pids)} losses on sample.")
    print("\n--- DECISION ---")
    print("- If word_len << current_best on most pids: sympy alone is sufficient. Run on full 1001.")
    print("- If word_len ~50-150 on most pids: sympy + min-merge picks up some long-tail wins.")
    print("- If word_len 500+ on most pids: need real Minkwitz (port from GAP).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
