"""Verify edge_tables.pkl against full-state simulation.

Mirrors verify_corner_tables.py for edges. Each edge has 2 stickers, ori ∈ {0, 1}.
"""
from __future__ import annotations

import pickle
import random
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def state_to_edge_state(state, edge_slots, home_lookup):
    n = len(edge_slots)
    perm = np.zeros(n, dtype=np.int8)
    ori = np.zeros(n, dtype=np.int8)
    for slot_i in range(n):
        slot_positions = edge_slots[slot_i]
        current_values = tuple(state[p] for p in slot_positions)
        sorted_curr = tuple(sorted(current_values))
        home_edge_idx = home_lookup[sorted_curr]
        home_canon = edge_slots[home_edge_idx]
        for r in range(2):
            rotated = tuple(home_canon[(r + k) % 2] for k in range(2))
            if rotated == current_values:
                ori[slot_i] = r
                break
        perm[slot_i] = home_edge_idx
    return perm, ori


def apply_edge_move(perm, ori, move_perm, move_ori):
    new_perm = perm[move_perm]
    new_ori = (ori[move_perm] + move_ori) % 2
    return new_perm, new_ori


def main():
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    with open(PROJECT / "data" / "edge_tables.pkl", "rb") as f:
        tables = pickle.load(f)

    move_names = tables["move_names"]
    n_edges = tables["n_edges"]
    edge_slots = tables["edge_slots"]
    edge_perm = tables["edge_perm"]
    edge_ori = tables["edge_ori"]
    home_lookup = {tuple(sorted(slot)): i for i, slot in enumerate(edge_slots)}

    print(f"loaded: {n_edges} edges, {len(move_names)} moves\n", flush=True)

    print("orientation profile per move:")
    print(f"  {'move':>5} | {'n_perm':>6} {'n_flip':>6} | sample flips")
    for m_idx, m_name in enumerate(move_names):
        n_perm = int((edge_perm[m_idx] != np.arange(n_edges)).sum())
        flipped = edge_ori[m_idx] != 0
        n_flip = int(flipped.sum())
        nonzero = edge_ori[m_idx][edge_ori[m_idx] != 0].tolist()
        print(f"  {m_name:>5} | {n_perm:>6} {n_flip:>6} | {nonzero[:8]}")

    # End-to-end test: random walk
    print(f"\nrandom-walk verification:")
    state_size = len(puzzle.solved_state)
    random.seed(0)

    n_trials = 5
    walk_len = 30
    n_failures = 0
    for trial in range(n_trials):
        full_state = list(puzzle.solved_state)
        perm = np.arange(n_edges, dtype=np.int8)
        ori = np.zeros(n_edges, dtype=np.int8)

        for step in range(walk_len):
            m_idx = random.randint(0, len(move_names) - 1)
            m_name = move_names[m_idx]
            g = puzzle.generators[m_name]
            full_state = [full_state[gi] for gi in g]
            perm, ori = apply_edge_move(perm, ori, edge_perm[m_idx], edge_ori[m_idx])

        ref_perm, ref_ori = state_to_edge_state(full_state, edge_slots, home_lookup)
        perm_ok = np.array_equal(perm, ref_perm)
        ori_ok = np.array_equal(ori, ref_ori)
        status = "OK" if (perm_ok and ori_ok) else "FAIL"
        if status == "FAIL":
            n_failures += 1
        print(f"  trial {trial}: {status}  (perm: {perm_ok}, ori: {ori_ok})")
        if not perm_ok:
            print(f"    table: {perm.tolist()}")
            print(f"    ref  : {ref_perm.tolist()}")
        if not ori_ok:
            print(f"    table: {ori.tolist()}")
            print(f"    ref  : {ref_ori.tolist()}")

    print(f"\n{n_failures}/{n_trials} trials failed")
    return 0 if n_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
