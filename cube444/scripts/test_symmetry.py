"""Self-test for the 24-rotation symmetry tables.

    python3 cube444/scripts/test_symmetry.py

Checks the two properties a sym-ensemble beam depends on:
  1. sym(s, R) sends solved -> solved and commutes with moves via move_relabel.
  2. A solution found in rotated frame k, mapped through move_relabel_inv[k],
     solves the ORIGINAL state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube444.puzzle import Cube444
from cube444.symmetry import (apply_symmetry, build_move_relabel, build_rotations,
                              invert_move_relabel)


def main() -> int:
    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    puz.verify_inverse_pairs()
    print("generator inverse pairs: OK")

    rot, cmap = build_rotations(puz)
    relabel = build_move_relabel(puz, rot)
    inv_relabel = invert_move_relabel(relabel)
    print(f"rotation group order: {rot.shape[0]}  (expect 24)")

    central = np.array(puz.solved_state, dtype=np.int64)
    names = list(puz.move_names)
    G = np.array([puz.generators[n] for n in names], dtype=np.int64)
    rng = np.random.default_rng(0)

    # solved -> solved under every rotation
    n_ok = sum(int(np.array_equal(apply_symmetry(central.reshape(1, -1), rot[k], cmap[k])[0],
                                  central)) for k in range(24))
    print(f"sym(solved, R_k) == solved: {n_ok}/24")
    assert n_ok == 24

    # property 1: commutation
    ok1 = 0
    for _ in range(500):
        word = rng.integers(0, 24, 6)
        s = central.copy()
        for g in word:
            s = s[G[g]]
        k = int(rng.integers(0, 24))
        lhs = apply_symmetry(s.reshape(1, -1), rot[k], cmap[k])[0]
        rhs = central.copy()
        for g in word:
            rhs = rhs[G[relabel[k, g]]]
        ok1 += int(np.array_equal(lhs, rhs))
    print(f"sym(apply(s,g),R) == apply(sym(s,R), relabel[k,g]): {ok1}/500")

    # property 2: round-trip -- solve the rotated frame, translate back
    ok2 = 0
    for _ in range(500):
        word = rng.integers(0, 24, 8)
        s = central.copy()
        for g in word:
            s = s[G[g]]
        k = int(rng.integers(0, 24))
        s_k = apply_symmetry(s.reshape(1, -1), rot[k], cmap[k])[0]
        # a solution for frame k: the relabelled inverse word
        sol_k = [int(relabel[k, g]) for g in word]
        sol_k = [names.index(puz.inverse_name(names[g])) for g in reversed(sol_k)]
        # sanity: it really solves the rotated state
        t = s_k.copy()
        for g in sol_k:
            t = t[G[g]]
        assert np.array_equal(t, central), "frame-k solution does not solve frame k"
        # translate back to the original frame
        sol_orig = [int(inv_relabel[k, g]) for g in sol_k]
        u = s.copy()
        for g in sol_orig:
            u = u[G[g]]
        ok2 += int(np.array_equal(u, central))
    print(f"round-trip (solve frame k -> translate -> solves original): {ok2}/500")

    if ok1 == 500 and ok2 == 500 and n_ok == 24:
        print("\nSYMMETRY TABLES OK")
        return 0
    print("\nSYMMETRY TABLES BROKEN")
    return 1


if __name__ == "__main__":
    sys.exit(main())
