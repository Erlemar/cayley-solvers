"""Corner coordinate calculations for megaminx PDB construction.

Each corner has 3 stickers (one per adjacent face). For PDB:
  - Identification: which of 20 corners is at each slot (by sticker-position-set match)
  - Orientation: cyclic rotation index (0/1/2)

Canonical sticker order per corner: numerical (sorted by position). Orientation is
the cyclic rotation that maps canonical → current. Doesn't need face labels.

Coord space for K-corner PDB:
  - P(20, K) × 3^K combinations
  - K=5: 1.86e6 × 243 = 452M entries
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def build_corner_tables(puzzle_info_path: Path):
    puzzle = json.loads(puzzle_info_path.read_text())
    gens = puzzle["generators"]
    move_names = list(gens.keys())
    state_size = len(puzzle["central_state"])

    def face_set_of(i):
        return frozenset(name.lstrip("-") for name, g in gens.items() if g[i] != i)

    sticker_face_sets = {i: face_set_of(i) for i in range(state_size)}

    # Group stickers into pieces (corners have 3 stickers, |face_set|=3)
    corner_pieces = {}  # face_set → list of sticker positions (sorted numerically)
    for i, fs in sticker_face_sets.items():
        if len(fs) == 3:
            corner_pieces.setdefault(fs, []).append(i)

    # Sort corners by face_set (lex order) for consistent indexing
    sorted_corner_face_sets = sorted(corner_pieces.keys(), key=lambda fs: tuple(sorted(fs)))
    n_corners = len(sorted_corner_face_sets)
    assert n_corners == 20, f"expected 20 corners, got {n_corners}"

    # corner_slots[i] = sticker positions for slot i, sorted numerically
    corner_slots = [tuple(sorted(corner_pieces[fs])) for fs in sorted_corner_face_sets]
    corner_face_set = [tuple(sorted(fs)) for fs in sorted_corner_face_sets]

    # For lookup: which corner has these stickers (sorted)? home_lookup[(s1,s2,s3)] = corner_id
    home_lookup = {tuple(sorted(slot)): idx for idx, slot in enumerate(corner_slots)}

    # Now compute corner_perm and corner_ori for each generator
    # by applying gen to SOLVED state and reading off corner movements.
    corner_perm = np.zeros((len(move_names), n_corners), dtype=np.int8)
    corner_ori = np.zeros((len(move_names), n_corners), dtype=np.int8)
    SOLVED = list(range(state_size))

    for m_idx, m_name in enumerate(move_names):
        g = gens[m_name]
        # Apply: new_state[i] = SOLVED[g[i]]
        new_state = [SOLVED[g[i]] for i in range(state_size)]

        for slot_i in range(n_corners):
            slot_positions = corner_slots[slot_i]  # canonical (sorted) order
            # Read values at slot's positions in CANONICAL order
            current_values = tuple(new_state[p] for p in slot_positions)
            # Identify which home corner is at this slot
            sorted_curr = tuple(sorted(current_values))
            home_corner_idx = home_lookup.get(sorted_curr)
            assert home_corner_idx is not None, \
                f"move {m_name}, slot {slot_i}: values {sorted_curr} not a home corner"
            # Determine orientation: how does current_values relate to home canonical order?
            # Home corner's stickers (sorted, the corner's "canonical" position values when at its
            # home slot) = corner_slots[home_corner_idx]. The values at home are these positions.
            # So when home corner is at home slot, current_values = corner_slots[home_corner_idx].
            # When at another slot, current_values is some cyclic rotation of the home's values.
            home_canon = corner_slots[home_corner_idx]
            # Find r in [0, 3) such that current_values is cyclic rotation of home_canon by r
            ori = None
            for r in range(3):
                rotated = tuple(home_canon[(r + k) % 3] for k in range(3))
                if rotated == current_values:
                    ori = r
                    break
            assert ori is not None, \
                f"move {m_name}, slot {slot_i}: no cyclic match for {current_values} vs {home_canon}"
            corner_perm[m_idx, slot_i] = home_corner_idx
            corner_ori[m_idx, slot_i] = ori

    return {
        "n_corners": n_corners,
        "move_names": move_names,
        "corner_slots": corner_slots,
        "corner_face_set": corner_face_set,
        "corner_perm": corner_perm,
        "corner_ori": corner_ori,
    }


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    puzzle_info = project / "data" / "puzzle_info.json"
    out_pkl = project / "data" / "corner_tables.pkl"

    print(f"building corner tables...", flush=True)
    tables = build_corner_tables(puzzle_info)

    print(f"\nn_corners: {tables['n_corners']}")
    print(f"\nfirst 3 corners:")
    for i in range(3):
        print(f"  slot {i}: faces={tables['corner_face_set'][i]}  stickers={tables['corner_slots'][i]}")

    print(f"\ncorner_perm[U]: {tables['corner_perm'][tables['move_names'].index('U')].tolist()}")
    print(f"corner_ori[U]:  {tables['corner_ori'][tables['move_names'].index('U')].tolist()}")

    # Verify perm + inverse perm gives identity
    u_idx = tables["move_names"].index("U")
    nu_idx = tables["move_names"].index("-U")
    pu = tables["corner_perm"][u_idx]
    pnu = tables["corner_perm"][nu_idx]
    composed = pu[pnu]  # apply U then -U: corners = pu[pnu[corners]]
    print(f"\nU * -U corner_perm: {composed.tolist()}")
    print(f"  is identity: {(composed == np.arange(20)).all()}")

    # Verify each move's perm is bijective
    for m_idx, m_name in enumerate(tables["move_names"]):
        perm = tables["corner_perm"][m_idx].tolist()
        if sorted(perm) != list(range(20)):
            print(f"  ERROR: move {m_name} corner_perm not a perm")

    with open(out_pkl, "wb") as f:
        pickle.dump(tables, f)
    print(f"\nwrote {out_pkl}")
