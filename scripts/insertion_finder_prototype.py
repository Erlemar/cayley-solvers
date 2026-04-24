"""Minimal insertion-finder prototype on picture cube directly.

Test whether inserting short commutators into community paths + pair-cancellation
can shorten them. If 0 gain on a sample, full IF port is unlikely to help.

Key assumption: for a complete path to STAY valid after insertion, the inserted
commutator must evaluate to IDENTITY. Only generator pairs A, B that commute
produce identity commutators.

Strategy:
1. Find all commuting-pair commutators from our 18 generators (length 4 each).
2. Extend to length-6 "conjugated" commutators: X·[A,B]·X^-1 where [A,B]=id.
3. For each community path, try inserting each identity word at each position;
   run cancel_adjacent_inverses; keep any strict improvement.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.post_process import cancel_adjacent_inverses
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_path


def apply_path(state, path, puzzle):
    for m in path:
        state = puzzle.apply_move(state, m)
    return state


def invert(path, puzzle):
    return [puzzle.inverse_name(m) for m in reversed(path)]


def find_identity_words(puzzle, max_len=6):
    """Enumerate all move sequences up to max_len that evaluate to identity on solved state."""
    solved = puzzle.solved_state
    gens = list(puzzle.move_names)
    identity_words = []

    # BFS over sequences; skip trivial adjacent inverses.
    def recurse(word, state, depth):
        if depth > 0 and state == solved:
            identity_words.append(list(word))
        if depth == max_len:
            return
        for m in gens:
            if word and m == puzzle.inverse_name(word[-1]):
                continue
            new_state = puzzle.apply_move(state, m)
            word.append(m)
            recurse(word, new_state, depth + 1)
            word.pop()

    recurse([], solved, 0)
    # Deduplicate
    seen = set()
    unique = []
    for w in identity_words:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            unique.append(w)
    return unique


def try_insert(path, identity_words, puzzle, max_iter=3):
    """Greedy: try every (position, identity_word) insertion, pair-cancel,
    keep first strict improvement. Repeat until no improvement."""
    best = list(path)
    for _ in range(max_iter):
        improved = False
        for pos in range(len(best) + 1):
            for w in identity_words:
                new_path = best[:pos] + w + best[pos:]
                new_path = cancel_adjacent_inverses(new_path, puzzle)
                if len(new_path) < len(best):
                    best = new_path
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def main():
    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    community = load_submission(PROJECT / "submissions" / "ens_plus_community.csv")

    print("enumerating identity words up to length 4...")
    t0 = time.time()
    id4 = find_identity_words(puzzle, max_len=4)
    print(f"  {len(id4)} identity words of length <=4 ({time.time()-t0:.1f}s)")

    # Length 6 would be ~18^6 = 34M, too slow. Length 4 covers commuting pairs.
    identity_words = id4

    # Sort by length desc so we try longer words first (more cancellation potential).
    identity_words = sorted(identity_words, key=lambda w: -len(w))

    # Test on 20 sampled community paths (mix of lengths).
    sample_pids = sorted(community.keys(), key=lambda p: -len(community[p]))[:10]
    sample_pids += sorted(community.keys(), key=lambda p: len(community[p]))[-10:]

    print(f"\ntesting insertion on {len(sample_pids)} community paths...")
    n_improved = 0
    total_before = 0
    total_after = 0
    for pid in sample_pids:
        orig = community[pid]
        t0 = time.time()
        new_path = try_insert(orig, identity_words, puzzle, max_iter=3)
        elapsed = time.time() - t0
        # Verify
        if not verify_path(puzzle, states[pid], new_path).ok:
            print(f"  pid={pid} INVALID after insertion; skipping")
            new_path = orig
        total_before += len(orig)
        total_after += len(new_path)
        delta = len(new_path) - len(orig)
        marker = " <-- IMPROVED" if delta < 0 else ""
        print(f"  pid={pid:4d} {len(orig):3d} -> {len(new_path):3d} ({delta:+d}) [{elapsed:.1f}s]{marker}")
        if delta < 0:
            n_improved += 1

    print(f"\n{n_improved}/{len(sample_pids)} paths improved")
    print(f"total: {total_before} -> {total_after} ({total_after - total_before:+d})")


if __name__ == "__main__":
    main()
