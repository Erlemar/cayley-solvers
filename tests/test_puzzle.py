"""Unit tests for the puzzle module."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import N_GENERATORS, STATE_SIZE, PictureCube


DATA = PROJECT / "data" / "puzzle_info.json"


def test_load_shape():
    p = PictureCube.load(DATA)
    assert len(p.solved_state) == STATE_SIZE
    assert p.solved_state == tuple(range(STATE_SIZE))
    assert len(p.generators) == N_GENERATORS
    assert len(p.move_names) == N_GENERATORS


def test_every_generator_has_inverse():
    p = PictureCube.load(DATA)
    for name in p.move_names:
        assert p.inverse_name(name) in p.generators, f"no inverse for {name}"


def test_inverse_pair_roundtrip():
    p = PictureCube.load(DATA)
    p.verify_inverse_pairs()


def test_generators_are_permutations():
    p = PictureCube.load(DATA)
    for name, gen in p.generators.items():
        assert sorted(gen) == list(range(STATE_SIZE)), f"{name} is not a valid permutation"


def test_apply_move_then_inverse_is_identity_on_scrambled():
    p = PictureCube.load(DATA)
    scrambled = tuple(range(STATE_SIZE - 1, -1, -1))  # reverse order
    for name in p.move_names:
        if name.startswith("-"):
            continue
        inv = p.inverse_name(name)
        assert p.apply_move(p.apply_move(scrambled, name), inv) == scrambled


def test_is_solved():
    p = PictureCube.load(DATA)
    assert p.is_solved(p.solved_state)
    assert not p.is_solved((1, 0) + tuple(range(2, STATE_SIZE)))


def test_path_parse_and_format_roundtrip():
    p = PictureCube.load(DATA)
    raw = "d2.-f2.-d1.r2"
    parsed = p.parse_path(raw)
    assert parsed == ["d2", "-f2", "-d1", "r2"]
    assert p.format_path(parsed) == raw
    assert p.parse_path("") == []
    assert p.format_path([]) == ""


def test_invert_state_involution_on_solved():
    p = PictureCube.load(DATA)
    assert p.invert_state(p.solved_state) == p.solved_state


def test_invert_state_is_involution():
    """invert_state(invert_state(s)) == s for any state."""
    p = PictureCube.load(DATA)
    scrambled = p.apply_path(p.solved_state, ["f0", "r1", "-d2", "f2", "-r0"])
    assert p.invert_state(p.invert_state(scrambled)) == scrambled


def test_invert_state_matches_inverse_scramble():
    """If s = apply(solved, Q), then invert_state(s) = apply(solved, Q^-1)."""
    p = PictureCube.load(DATA)
    Q = ["f0", "r1", "-d2", "f2", "-r0", "d1"]
    s = p.apply_path(p.solved_state, Q)
    Q_inv = p.invert_path(Q)
    s_from_inv = p.apply_path(p.solved_state, Q_inv)
    assert p.invert_state(s) == s_from_inv


def test_invert_path_cancels_original():
    """apply_path(s, P + invert_path(P)) == s."""
    p = PictureCube.load(DATA)
    P = ["f0", "r1", "-d2", "f2"]
    s = p.apply_path(p.solved_state, ["d1", "-r2"])
    assert p.apply_path(s, P + p.invert_path(P)) == s


def test_niss_converted_path_solves_original():
    """Core NISS identity: if path P solves sigma^-1, then invert_path(P) solves sigma."""
    p = PictureCube.load(DATA)
    # Build a scrambled state, invert it, solve the inverse trivially (apply its own scramble
    # in reverse → solved), then convert back.
    Q = ["f0", "r1", "-d2", "f2", "-r0"]
    sigma = p.apply_path(p.solved_state, Q)
    sigma_inv = p.invert_state(sigma)

    # Any path that solves sigma_inv — here, one we can construct: sigma_inv is apply(solved, Q_inv),
    # so applying Q to sigma_inv gives solved.
    solve_inv = Q  # apply_path(sigma_inv, Q) == solved
    assert p.is_solved(p.apply_path(sigma_inv, solve_inv))

    # Convert: invert_path(solve_inv) should solve sigma.
    solve_sigma = p.invert_path(solve_inv)
    assert p.is_solved(p.apply_path(sigma, solve_sigma))
