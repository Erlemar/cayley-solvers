"""Unit tests for the 24 rotational symmetries."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube
from cayley.symmetry import apply_random_rotation, compute_rotations, invert_permutation_batch


DATA = PROJECT / "data" / "puzzle_info.json"


def test_count_is_24():
    p = PictureCube.load(DATA)
    R = compute_rotations(p)
    assert R.shape == (24, 72)


def test_rotations_are_permutations():
    p = PictureCube.load(DATA)
    R = compute_rotations(p)
    for i in range(24):
        assert sorted(R[i].tolist()) == list(range(72))


def test_identity_rotation_included():
    p = PictureCube.load(DATA)
    R = compute_rotations(p)
    identity = list(range(72))
    assert any(r.tolist() == identity for r in R)


def test_rotations_unique():
    p = PictureCube.load(DATA)
    R = compute_rotations(p)
    as_tuples = set(tuple(r.tolist()) for r in R)
    assert len(as_tuples) == 24


def test_solved_state_invariant_under_rotation():
    """R·solved·R⁻¹ = solved for every R."""
    import torch
    p = PictureCube.load(DATA)
    R_np = compute_rotations(p)
    R = torch.from_numpy(R_np)
    R_inv = invert_permutation_batch(R)
    solved = torch.tensor(p.solved_state, dtype=torch.int64).unsqueeze(0).repeat(24, 1)
    # apply each rotation to solved
    step1 = torch.gather(solved, 1, R_inv)
    rotated = torch.gather(R, 1, step1)
    for i in range(24):
        assert rotated[i].tolist() == list(p.solved_state), f"rotation {i} changed solved"


def test_rotated_state_reachable_in_same_depth():
    """A state reached by a k-move walk, rotated, is still reachable in k moves from solved.

    Concretely: the rotated state is a valid puzzle configuration (permutation of 0..71).
    """
    import torch
    p = PictureCube.load(DATA)
    R_np = compute_rotations(p)
    R = torch.from_numpy(R_np)
    R_inv = invert_permutation_batch(R)

    # Build a state via walk
    state = p.solved_state
    for m in ["f0", "r1", "-d2", "f2"]:
        state = p.apply_move(state, m)

    s = torch.tensor(state, dtype=torch.int64).unsqueeze(0).repeat(24, 1)
    step1 = torch.gather(s, 1, R_inv)
    rotated = torch.gather(R, 1, step1)
    for i in range(24):
        r = rotated[i].tolist()
        assert sorted(r) == list(range(72)), f"rotation {i} produced non-permutation"


def test_apply_random_rotation_preserves_shape():
    import torch
    p = PictureCube.load(DATA)
    R = torch.from_numpy(compute_rotations(p))
    R_inv = invert_permutation_batch(R)
    B, S = 100, 72
    states = torch.randint(0, 72, (B, S), dtype=torch.int64)
    # make states valid permutations
    perm = torch.arange(S, dtype=torch.int64)
    states = perm.unsqueeze(0).expand(B, S).clone()
    # shuffle each row independently
    for i in range(B):
        idx = torch.randperm(S)
        states[i] = states[i][idx]

    gen = torch.Generator().manual_seed(0)
    out = apply_random_rotation(states, R, R_inv, gen)
    assert out.shape == (B, S)
    # Each output row must still be a permutation of 0..71
    for i in range(B):
        assert sorted(out[i].tolist()) == list(range(72))
