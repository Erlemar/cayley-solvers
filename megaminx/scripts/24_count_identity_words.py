"""T2.1 Step 0: count non-trivial identity words on megaminx up to depth K.

An "identity word" is a sequence of generators w such that apply(solved, w) == solved.
For T2.1 (insertion-finder), we want to know:
  - how many depth-4, depth-6 identity words exist on megaminx
  - how many remain after stripping trivial cancellation chains
  (X X^{-1} ... is trivial; we want those that can't be reduced via
   adjacent-inverse cancellation alone)

This determines whether identity-word insertion is mechanically interesting.

Run: .venv/Scripts/python.exe megaminx/scripts/24_count_identity_words.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.puzzle import Megaminx
from megaminx.post_process import cancel_adjacent_inverses, reduce_same_face_runs


def enumerate_identity_words(puzzle: Megaminx, max_depth: int) -> list[list[str]]:
    """BFS enumeration of words that return solved -> solved.

    Skips trivially backward steps (consecutive A . A^{-1}). Returns ALL identity
    words up to max_depth, including ones that simplify via face-rule.
    """
    solved = puzzle.solved_state
    move_names = list(puzzle.move_names)
    inv_of = {n: puzzle.inverse_name(n) for n in move_names}

    identity_words: list[list[str]] = []

    def recurse(word: list[str], state, depth: int):
        # Even depths only: a length-1 word can't return to solved.
        if depth > 0 and depth % 2 == 0 and state == solved:
            identity_words.append(list(word))
        if depth == max_depth:
            return
        last = word[-1] if word else None
        for m in move_names:
            if last is not None and m == inv_of[last]:
                continue  # skip trivial A.A^{-1} pair
            new_state = puzzle.apply_move(state, m)
            word.append(m)
            recurse(word, new_state, depth + 1)
            word.pop()

    recurse([], solved, 0)
    return identity_words


def is_reducible_by_face_rule(word: list[str]) -> bool:
    """Is `word` reducible to a strict prefix via same-face + adjacent-inverse?"""
    after = cancel_adjacent_inverses(reduce_same_face_runs(word))
    return len(after) < len(word)


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"loaded megaminx: {len(puzzle.move_names)} generators")
    print(f"  generators: {puzzle.move_names}")

    for max_d in (2, 4, 6):
        t0 = time.time()
        words = enumerate_identity_words(puzzle, max_d)
        elapsed = time.time() - t0
        print(f"\n--- depth <= {max_d} ---")
        print(f"  identity words found (incl reducible): {len(words)} ({elapsed:.2f}s)")
        # Group by length
        from collections import Counter
        by_len = Counter(len(w) for w in words)
        for L in sorted(by_len):
            print(f"    length {L}: {by_len[L]}")
        # How many are NON-trivial (don't reduce via face rule)?
        non_trivial = [w for w in words if not is_reducible_by_face_rule(w)]
        print(f"  non-trivial (don't collapse via face/adj-inv rules): {len(non_trivial)}")
        if non_trivial:
            print(f"  sample non-trivial words:")
            for w in non_trivial[:5]:
                print(f"    {' . '.join(w)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
