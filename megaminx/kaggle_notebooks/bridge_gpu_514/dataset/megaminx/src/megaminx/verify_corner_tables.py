"""Verify corner_tables.pkl against full-state simulation.

For random move sequences, we should get the same corner_perm/corner_ori from:
  (a) composing tables['corner_perm'] / tables['corner_ori'] for each move, AND
  (b) computing it once from the resulting full state.

Also dumps a table of (move, n_twisted_corners) so we can sanity-check that face
moves twist 0 corners and adjacent-face moves twist some.
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]  # cayley/megaminx
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def state_to_corner_state(state, corner_slots, home_lookup):
    """Convert a full state to (corner_perm, corner_ori) of length n_corners.

    Returns (perm, ori) where perm[slot_i] = home_corner_id, ori[slot_i] = cyclic offset.
    """
    n = len(corner_slots)
    perm = np.zeros(n, dtype=np.int8)
    ori = np.zeros(n, dtype=np.int8)
    for slot_i in range(n):
        slot_positions = corner_slots[slot_i]
        current_values = tuple(state[p] for p in slot_positions)
        sorted_curr = tuple(sorted(current_values))
        home_corner_idx = home_lookup[sorted_curr]
        home_canon = corner_slots[home_corner_idx]
        for r in range(3):
            rotated = tuple(home_canon[(r + k) % 3] for k in range(3))
            if rotated == current_values:
                ori = ori.copy()
                ori[slot_i] = r
                break
        perm[slot_i] = home_corner_idx
    return perm, ori


def apply_corner_move(perm, ori, move_perm, move_ori):
    """Apply a move's (move_perm, move_ori) to current corner state (perm, ori).

    move_perm[slot_i] = which corner now occupies slot_i (i.e. before-move slot index of
    the corner that's now at slot_i).
    move_ori[slot_i] = orientation delta added to that corner's existing orientation.

    new_perm[slot_i] = perm[move_perm[slot_i]]
    new_ori[slot_i] = (ori[move_perm[slot_i]] + move_ori[slot_i]) % 3
    """
    new_perm = perm[move_perm]
    new_ori = (ori[move_perm] + move_ori) % 3
    return new_perm, new_ori


def main():
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    with open(PROJECT / "data" / "corner_tables.pkl", "rb") as f:
        tables = pickle.load(f)

    move_names = tables["move_names"]
    n_corners = tables["n_corners"]
    corner_slots = tables["corner_slots"]
    corner_perm = tables["corner_perm"]
    corner_ori = tables["corner_ori"]
    home_lookup = {tuple(sorted(slot)): i for i, slot in enumerate(corner_slots)}

    print(f"loaded: {n_corners} corners, {len(move_names)} moves\n", flush=True)

    # Survey: which moves twist corners?
    print("orientation profile per move:")
    print(f"  {'move':>5} | {'n_perm':>6} {'n_twist':>7} | sample twist values")
    print(f"  {'-'*5} | {'-'*6} {'-'*7} | {'-'*30}")
    for m_idx, m_name in enumerate(move_names):
        n_perm = int((corner_perm[m_idx] != np.arange(n_corners)).sum())
        twisted = corner_ori[m_idx] != 0
        n_twist = int(twisted.sum())
        # Show non-zero ori values
        nonzero = corner_ori[m_idx][corner_ori[m_idx] != 0].tolist()
        print(f"  {m_name:>5} | {n_perm:>6} {n_twist:>7} | {nonzero}")

    # End-to-end test: random walk, compare table-tracked vs state-tracked
    print(f"\nrandom-walk verification:")
    state_size = len(puzzle.solved_state)
    random.seed(0)

    n_trials = 5
    walk_len = 30
    n_failures = 0
    for trial in range(n_trials):
        # Start from solved
        full_state = list(puzzle.solved_state)
        perm = np.arange(n_corners, dtype=np.int8)
        ori = np.zeros(n_corners, dtype=np.int8)

        moves_applied = []
        for step in range(walk_len):
            m_idx = random.randint(0, len(move_names) - 1)
            m_name = move_names[m_idx]
            moves_applied.append(m_name)
            # Apply to full state
            g = puzzle.generators[m_name]
            full_state = [full_state[gi] for gi in g]
            # Apply to corner state via tables
            perm, ori = apply_corner_move(perm, ori, corner_perm[m_idx], corner_ori[m_idx])

        # Compute corner state directly from final full state
        ref_perm, ref_ori = state_to_corner_state(full_state, corner_slots, home_lookup)

        perm_ok = np.array_equal(perm, ref_perm)
        ori_ok = np.array_equal(ori, ref_ori)
        status = "OK" if (perm_ok and ori_ok) else "FAIL"
        if status == "FAIL":
            n_failures += 1
        print(f"  trial {trial}: {status}  (perm match: {perm_ok}, ori match: {ori_ok})")
        if not perm_ok:
            print(f"    table perm: {perm.tolist()}")
            print(f"    ref   perm: {ref_perm.tolist()}")
        if not ori_ok:
            print(f"    table ori : {ori.tolist()}")
            print(f"    ref   ori : {ref_ori.tolist()}")

    print(f"\n{n_failures}/{n_trials} trials failed")
    return 0 if n_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
