"""Inversion-pair defensive check on test.csv.

If any two test puzzles s_a, s_b are permutation INVERSES of each other (i.e.
s_a applied to identity equals the same permutation that s_b inverts), they
have identical true distance. Solving one then INVERTING the path solves the
other for free — without running beam search.

Megaminx state = 120-element permutation `s` (s[i] = where piece i moves to).
Inverse permutation: inv[s[i]] = i for all i.

Almost certainly returns 0 pairs (state space ~10^68, 1001 puzzles), but it's a
free defensive check.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    test_csv = PROJECT / "data" / "test.csv"
    print(f"loading {test_csv}...")
    states: dict[int, np.ndarray] = {}
    with open(test_csv) as f:
        for row in csv.DictReader(f):
            pid = int(row["initial_state_id"])
            states[pid] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int8
            )
    print(f"  loaded {len(states)} states")

    # Hash each state (just bytes), and hash each state's INVERSE.
    # If state[a] == inverse(state[b]), they are inverse pairs.
    forward_hashes: dict[bytes, int] = {}
    inverse_hashes: dict[bytes, int] = {}
    for pid, s in states.items():
        # Forward hash = bytes of s
        forward_hashes[s.tobytes()] = pid
        # Inverse: inv[s[i]] = i  ->  inv has shape (120,), values 0..119
        inv = np.empty_like(s)
        inv[s.astype(np.int64)] = np.arange(len(s), dtype=np.int8)
        inverse_hashes[inv.tobytes()] = pid

    print()
    print("checking for state == inverse(other_state) ...")
    pairs: list[tuple[int, int]] = []
    for pid_a, s_a in states.items():
        s_a_bytes = s_a.tobytes()
        if s_a_bytes in inverse_hashes:
            pid_b = inverse_hashes[s_a_bytes]
            if pid_a < pid_b:
                pairs.append((pid_a, pid_b))

    print(f"  inversion pairs found: {len(pairs)}")
    if pairs:
        print()
        for pid_a, pid_b in pairs[:20]:
            print(f"    pid {pid_a} ↔ pid {pid_b}")
        if len(pairs) > 20:
            print(f"    ... and {len(pairs) - 20} more")
    else:
        print()
        print("  No pairs found - as expected. This was a defensive check.")
        print("  No free moves available from this lever.")

    # Also check: any state == identity (which would already be 0-move)?
    n_pieces = len(next(iter(states.values())))
    identity = np.arange(n_pieces, dtype=np.int8)
    identity_pid = None
    for pid, s in states.items():
        if np.array_equal(s, identity):
            identity_pid = pid
            break
    print()
    print(f"identity-state in test.csv: " +
          (f"YES at pid {identity_pid} (already solved)" if identity_pid is not None
           else "no"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
