"""Edge coordinate calculations for megaminx PDB construction.

Megaminx has 30 edges, each with 2 stickers (one per adjacent face). For PDB:
  - Identification: which of 30 edges is at each slot (by sticker-position-set match)
  - Orientation: cyclic offset (0 or 1) — i.e., flipped or not.

Coord space for K-edge PDB:
  - P(30, K) × 2^K
  - K=5: 17,100,720 × 32 = 547,223,040  (~547 MB depth table)
  - K=6: 427,518,000 × 64 = 27,361,152,000  (way too big)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np


def build_edge_tables(puzzle_info_path: Path):
    puzzle = json.loads(puzzle_info_path.read_text())
    gens = puzzle["generators"]
    move_names = list(gens.keys())
    state_size = len(puzzle["central_state"])

    def face_set_of(i):
        return frozenset(name.lstrip("-") for name, g in gens.items() if g[i] != i)

    sticker_face_sets = {i: face_set_of(i) for i in range(state_size)}

    # Edges have face_set of size 2 (2 adjacent faces meet at the edge)
    edge_pieces = {}  # face_set → [sticker positions sorted]
    for i, fs in sticker_face_sets.items():
        if len(fs) == 2:
            edge_pieces.setdefault(fs, []).append(i)

    sorted_edge_face_sets = sorted(edge_pieces.keys(), key=lambda fs: tuple(sorted(fs)))
    n_edges = len(sorted_edge_face_sets)
    assert n_edges == 30, f"expected 30 edges, got {n_edges}"

    edge_slots = [tuple(sorted(edge_pieces[fs])) for fs in sorted_edge_face_sets]
    edge_face_set = [tuple(sorted(fs)) for fs in sorted_edge_face_sets]

    home_lookup = {tuple(sorted(slot)): idx for idx, slot in enumerate(edge_slots)}

    edge_perm = np.zeros((len(move_names), n_edges), dtype=np.int8)
    edge_ori = np.zeros((len(move_names), n_edges), dtype=np.int8)
    SOLVED = list(range(state_size))

    for m_idx, m_name in enumerate(move_names):
        g = gens[m_name]
        new_state = [SOLVED[g[i]] for i in range(state_size)]

        for slot_i in range(n_edges):
            slot_positions = edge_slots[slot_i]  # canonical (sorted) order
            current_values = tuple(new_state[p] for p in slot_positions)
            sorted_curr = tuple(sorted(current_values))
            home_edge_idx = home_lookup.get(sorted_curr)
            assert home_edge_idx is not None, \
                f"move {m_name}, slot {slot_i}: values {sorted_curr} not a home edge"
            home_canon = edge_slots[home_edge_idx]
            ori = None
            for r in range(2):
                rotated = tuple(home_canon[(r + k) % 2] for k in range(2))
                if rotated == current_values:
                    ori = r
                    break
            assert ori is not None, \
                f"move {m_name}, slot {slot_i}: no cyclic match for {current_values} vs {home_canon}"
            edge_perm[m_idx, slot_i] = home_edge_idx
            edge_ori[m_idx, slot_i] = ori

    return {
        "n_edges": n_edges,
        "move_names": move_names,
        "edge_slots": edge_slots,
        "edge_face_set": edge_face_set,
        "edge_perm": edge_perm,
        "edge_ori": edge_ori,
    }


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    puzzle_info = project / "data" / "puzzle_info.json"
    out_pkl = project / "data" / "edge_tables.pkl"

    print(f"building edge tables...", flush=True)
    tables = build_edge_tables(puzzle_info)

    print(f"\nn_edges: {tables['n_edges']}")
    print(f"\nfirst 3 edges:")
    for i in range(3):
        print(f"  slot {i}: faces={tables['edge_face_set'][i]}  stickers={tables['edge_slots'][i]}")

    print(f"\nedge_perm[U]: {tables['edge_perm'][tables['move_names'].index('U')].tolist()}")
    print(f"edge_ori[U]:  {tables['edge_ori'][tables['move_names'].index('U')].tolist()}")

    # Verify each move's perm is bijective
    for m_idx, m_name in enumerate(tables["move_names"]):
        perm = tables["edge_perm"][m_idx].tolist()
        assert sorted(perm) == list(range(30)), f"move {m_name} edge_perm not a perm"

    with open(out_pkl, "wb") as f:
        pickle.dump(tables, f)
    print(f"\nwrote {out_pkl}")
