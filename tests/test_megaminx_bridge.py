"""Unit tests for the residual composition used by bridge compression.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_megaminx_bridge.py -v

Or as a script:
    .venv/Scripts/python.exe tests/test_megaminx_bridge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "megaminx" / "src"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from megaminx.bridge import compute_prefix_states, make_residual
from megaminx.puzzle import Megaminx


PUZZLE = Megaminx.load(ROOT / "megaminx" / "data" / "puzzle_info.json")
SOLVED = PUZZLE.solved_state


def test_residual_of_zero_window_is_identity():
    """If S_i == S_j, the residual is the identity permutation (solved state)."""
    s = list(SOLVED)
    r = make_residual(s, s)
    assert tuple(r.tolist()) == SOLVED


def test_residual_of_single_move_is_that_move_perm():
    """For window [i, j=i+1] with move m, residual X = gen[m].

    Reasoning: S_j[k] = S_i[gen[m][k]], so inv_S_j[S_i[k]] = position-of-S_i[k] in
    S_j = position k such that S_i[gen[m][k]] = S_i[k] mapped through ... easier:
    in particular for S_i = solved (identity), S_j = solved[gen[m]] = gen[m].
    Then inv_S_j[k] = position-of-k in gen[m] = inv(gen[m])[k].
    X[k] = inv_S_j[S_i[k]] = inv_S_j[k] = inv(gen[m])[k].

    So X = inv(gen[m]), not gen[m] itself. Verify by applying the same single
    move m to X: apply(X, m)[k] = X[gen[m][k]] = inv(gen[m])[gen[m][k]] = k.
    """
    for move_name in PUZZLE.move_names:
        s_i = list(SOLVED)
        s_j = list(PUZZLE.apply_move(SOLVED, move_name))
        r = make_residual(s_i, s_j)
        # Apply move m to residual; should land on solved.
        after = list(PUZZLE.apply_move(r.tolist(), move_name))
        assert tuple(after) == SOLVED, (
            f"single-move residual: apply({move_name}) to residual didn't reach solved "
            f"for {move_name}"
        )


def test_random_window_original_path_solves_residual():
    """The bedrock property: for any valid window A = path[i:j] going from
    S_i to S_j, the original window path A itself solves the residual
    X = inv_S_j[S_i]. (A shorter bridge replacing A would also solve X.)
    """
    rng = np.random.default_rng(seed=0)
    # Build a random scramble of length 60.
    path = []
    cur = SOLVED
    for _ in range(60):
        m = PUZZLE.move_names[rng.integers(0, len(PUZZLE.move_names))]
        cur = PUZZLE.apply_move(cur, m)
        path.append(m)
    prefix_states = compute_prefix_states(SOLVED, path, PUZZLE)
    # Try many random windows.
    for _ in range(50):
        i = int(rng.integers(0, len(path) - 1))
        j = int(rng.integers(i + 1, len(path) + 1))
        s_i = prefix_states[i]
        s_j = prefix_states[j]
        r = make_residual(s_i, s_j)
        # The original window path itself must take residual to solved.
        end = PUZZLE.apply_path(r.tolist(), path[i:j])
        assert tuple(end) == SOLVED, (
            f"window [{i},{j}] (len {j-i}): original path didn't solve residual"
        )


def test_window_endpoints_match():
    """If you have S_i and S_j, the residual is non-trivial iff S_i != S_j."""
    rng = np.random.default_rng(seed=1)
    cur = SOLVED
    moves = []
    for _ in range(40):
        m = PUZZLE.move_names[rng.integers(0, len(PUZZLE.move_names))]
        cur = PUZZLE.apply_move(cur, m)
        moves.append(m)
    s_i = SOLVED
    s_j = cur
    r = make_residual(s_i, s_j)
    # r != solved (since cur != solved with high probability after 40 moves)
    assert tuple(r.tolist()) != SOLVED


def test_residual_under_inversion_symmetry():
    """make_residual(S_j, S_i) should be the inverse permutation of
    make_residual(S_i, S_j).
    """
    rng = np.random.default_rng(seed=7)
    cur = SOLVED
    moves = []
    for _ in range(25):
        m = PUZZLE.move_names[rng.integers(0, len(PUZZLE.move_names))]
        cur = PUZZLE.apply_move(cur, m)
        moves.append(m)
    s_i = SOLVED
    s_j = cur
    r_ij = make_residual(s_i, s_j)
    r_ji = make_residual(s_j, s_i)
    n = len(SOLVED)
    inv_r_ji = np.empty(n, dtype=np.int64)
    inv_r_ji[r_ji] = np.arange(n, dtype=np.int64)
    assert (r_ij == inv_r_ji).all()


if __name__ == "__main__":
    tests = [
        ("zero-window is identity", test_residual_of_zero_window_is_identity),
        ("single-move residual", test_residual_of_single_move_is_that_move_perm),
        ("random-window original-path solve", test_random_window_original_path_solves_residual),
        ("window endpoints", test_window_endpoints_match),
        ("inversion symmetry", test_residual_under_inversion_symmetry),
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
