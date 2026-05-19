"""Standalone correctness test for symmetry-ensemble path translation.

Verifies that for a non-trivial rotation R from rotations.npy:
  1. apply_rotation(solved, R) == solved (i.e., R preserves the solved state).
  2. R_inv * g_n * R is a generator for every g_n (R is a true symmetry).
  3. apply_path(state, path_orig) == solved iff
     apply_path(R*state*R_inv, path_rot) == solved
     where path_orig[i] = name(R_inv * g_{path_rot[i]} * R).

If all 3 hold for several rotations on several puzzles, the math is correct
and we can wire this into 03_solve.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.puzzle import Megaminx
from cayley.verify import load_test_states, verify_path


def apply_rotation_to_state(state, R, R_inv):
    """R * s * R_inv. Convention: new_state[i] = R[state[R_inv[i]]]."""
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))


def compute_conjugation_map(R, R_inv, generators_dict, gen_names):
    """rotated_name -> original_name = R_inv * g_rotated * R."""
    n = len(R)
    perm_to_name = {tuple(g): nm for nm, g in generators_dict.items()}
    conj_map = {}
    for nm in gen_names:
        g = generators_dict[nm]
        conj = tuple(R_inv[g[R[i]]] for i in range(n))
        if conj not in perm_to_name:
            raise ValueError(
                f"rotation is NOT a symmetry: R_inv * g_{nm} * R is not a generator"
            )
        conj_map[nm] = perm_to_name[conj]
    return conj_map


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    rotations = np.load(PROJECT / "data" / "rotations.npy")
    print(f"loaded {rotations.shape[0]} rotations")

    # 1. Identity preservation check (trivial — for completeness).
    solved = puzzle.solved_state
    n_size = len(solved)
    identity = tuple(range(n_size))
    n_id_ok = 0
    for r_arr in rotations:
        R = tuple(r_arr.tolist())
        R_inv = tuple(np.argsort(r_arr).tolist())
        rotated_solved = apply_rotation_to_state(solved, R, R_inv)
        if rotated_solved == solved:
            n_id_ok += 1
    print(f"  R*solved*R_inv == solved: {n_id_ok}/{rotations.shape[0]}")

    # 2. Conjugation map: every R must yield a valid name->name map.
    n_conj_ok = 0
    for r_arr in rotations:
        R = tuple(r_arr.tolist())
        R_inv = tuple(np.argsort(r_arr).tolist())
        try:
            cm = compute_conjugation_map(R, R_inv, puzzle.generators, puzzle.move_names)
            assert len(cm) == len(puzzle.move_names)
            assert len(set(cm.values())) == len(puzzle.move_names)
            n_conj_ok += 1
        except (ValueError, AssertionError) as e:
            print(f"  conj-map FAIL: {e}")
    print(f"  conjugation map valid: {n_conj_ok}/{rotations.shape[0]}")

    # 3. Path translation round-trip on real test puzzles.
    # Strategy: take a known fallback path for some pid, apply it forward in the
    # ROTATED frame, and verify it reaches solved (in rotated frame, which =
    # solved in original frame since solved is the identity permutation).
    # Then: take the ORIGINAL-frame path and translate it via inverse-conj-map
    # to get the rotated-frame path. Apply to s_rot. Confirm reaches solved.
    print()
    print("path translation round-trip test:")
    fallback_csv = PROJECT / "data" / "pp_bfs6_fallback.csv"
    import csv
    fb_paths: dict[int, list[str]] = {}
    with open(fallback_csv) as f:
        for row in csv.DictReader(f):
            fb_paths[int(row["initial_state_id"])] = row["path"].split(".")

    rng = np.random.default_rng(42)
    test_pids = list(rng.choice(sorted(states.keys()), size=4, replace=False))
    test_rot_idx = list(rng.choice(rotations.shape[0], size=4, replace=False))
    test_rot_idx = [int(i) for i in test_rot_idx]

    n_round_trip_ok = 0
    n_total = 0
    for pid in test_pids:
        pid = int(pid)
        state = states[pid]
        path_orig = fb_paths[pid]
        # Sanity: original path must already verify.
        assert verify_path(puzzle, state, path_orig).ok, \
            f"fallback path for pid {pid} doesn't verify"
        for r_idx in test_rot_idx:
            n_total += 1
            r_arr = rotations[r_idx]
            R = tuple(r_arr.tolist())
            R_inv = tuple(np.argsort(r_arr).tolist())
            cm = compute_conjugation_map(R, R_inv, puzzle.generators, puzzle.move_names)
            inv_cm = {v: k for k, v in cm.items()}
            assert len(inv_cm) == len(cm), "conj map not bijective"
            # Translate ORIGINAL path -> ROTATED path via inverse conj map.
            path_rot = [inv_cm[m] for m in path_orig]
            # Build rotated state.
            s_rot = apply_rotation_to_state(state, R, R_inv)
            # Apply path_rot to s_rot — should reach solved.
            cur = s_rot
            for m in path_rot:
                cur = puzzle.apply_move(cur, m)
            ok_rot = (tuple(cur) == solved)
            # Also: translate rotated path back to original via cm.
            path_back = [cm[m] for m in path_rot]
            ok_back = (path_back == path_orig)
            if ok_rot and ok_back:
                n_round_trip_ok += 1
            else:
                print(f"  pid={pid} rot={r_idx}: ok_rot={ok_rot} ok_back={ok_back}")
    print(f"  round-trip: {n_round_trip_ok}/{n_total}")

    # 4. Critical: solve in rotated frame, translate back, verify on ORIGINAL.
    # We don't run beam — just construct a rotated puzzle and solve it via the
    # known fallback path translated into the rotated frame. If the rotated
    # solve works AND translating its move sequence via cm reaches the same
    # original-frame moves, we're consistent.
    # Actually the above test #3 already proves this — both directions verified.

    if n_id_ok == rotations.shape[0] and n_conj_ok == rotations.shape[0] and n_round_trip_ok == n_total:
        print("\nALL CHECKS PASSED")
        return 0
    else:
        print("\nFAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
