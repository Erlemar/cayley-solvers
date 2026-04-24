"""Unit tests for piece decomposition."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.piece_features import (
    CENTERS,
    CORNERS,
    EDGES,
    N_CENTERS,
    N_CORNERS,
    N_EDGES,
    extract_features_numpy,
)
from cayley.puzzle import PictureCube


DATA = PROJECT / "data" / "puzzle_info.json"


def test_piece_indices_partition():
    """Corner + edge + center stickers cover 0..71 exactly, disjointly."""
    ci = {s for t in CORNERS for s in t}
    ei = {s for t in EDGES for s in t}
    mi = {s for t in CENTERS for s in t}
    assert len(ci) == 24 and len(ei) == 24 and len(mi) == 24
    assert not (ci & ei) and not (ci & mi) and not (ei & mi)
    assert ci | ei | mi == set(range(72))


def test_solved_state_features():
    """Solved state: every piece in its canonical slot with ori=0."""
    p = PictureCube.load(DATA)
    feats = extract_features_numpy(p.solved_state)
    corner_id = feats[: N_CORNERS]
    corner_ori = feats[N_CORNERS : 2 * N_CORNERS]
    edge_id = feats[2 * N_CORNERS : 2 * N_CORNERS + N_EDGES]
    edge_ori = feats[2 * N_CORNERS + N_EDGES : 2 * N_CORNERS + 2 * N_EDGES]
    center_id = feats[2 * N_CORNERS + 2 * N_EDGES : 2 * N_CORNERS + 2 * N_EDGES + N_CENTERS]
    center_ori = feats[2 * N_CORNERS + 2 * N_EDGES + N_CENTERS :]

    assert list(corner_id) == list(range(N_CORNERS))
    assert list(corner_ori) == [0] * N_CORNERS
    assert list(edge_id) == list(range(N_EDGES))
    assert list(edge_ori) == [0] * N_EDGES
    assert list(center_id) == list(range(N_CENTERS))
    assert list(center_ori) == [0] * N_CENTERS


def test_features_after_single_generator():
    """After applying any single generator to solved, piece ids remain a permutation of
    {0..N-1} for each class (no piece disappears)."""
    p = PictureCube.load(DATA)
    for name in p.move_names:
        state = p.apply_move(p.solved_state, name)
        feats = extract_features_numpy(state)
        corner_id = set(feats[:N_CORNERS].tolist())
        edge_id = set(feats[2 * N_CORNERS : 2 * N_CORNERS + N_EDGES].tolist())
        off = 2 * N_CORNERS + 2 * N_EDGES
        center_id = set(feats[off : off + N_CENTERS].tolist())
        assert corner_id == set(range(N_CORNERS)), f"corner bag broken by {name}: {corner_id}"
        assert edge_id == set(range(N_EDGES)), f"edge bag broken by {name}: {edge_id}"
        assert center_id == set(range(N_CENTERS)), f"center bag broken by {name}: {center_id}"


def test_features_after_random_walk():
    """Random walks preserve piece-id bags."""
    import random
    rng = random.Random(0)
    p = PictureCube.load(DATA)
    state = p.solved_state
    for _ in range(100):
        m = rng.choice(p.move_names)
        state = p.apply_move(state, m)
    feats = extract_features_numpy(state)
    corner_id = set(feats[:N_CORNERS].tolist())
    edge_id = set(feats[2 * N_CORNERS : 2 * N_CORNERS + N_EDGES].tolist())
    off = 2 * N_CORNERS + 2 * N_EDGES
    center_id = set(feats[off : off + N_CENTERS].tolist())
    assert corner_id == set(range(N_CORNERS))
    assert edge_id == set(range(N_EDGES))
    assert center_id == set(range(N_CENTERS))


def test_inverse_pairs_roundtrip_features():
    """move + inverse_move returns to solved → features return to canonical."""
    p = PictureCube.load(DATA)
    for name in p.move_names:
        if name.startswith("-"):
            continue
        inv = p.inverse_name(name)
        s = p.apply_move(p.apply_move(p.solved_state, name), inv)
        feats = extract_features_numpy(s)
        assert (feats == extract_features_numpy(p.solved_state)).all()


def test_orientation_sum_invariant():
    """After any scramble, total corner orientation sum mod 3 == 0 (conservation law)."""
    import random
    rng = random.Random(42)
    p = PictureCube.load(DATA)
    state = p.solved_state
    for _ in range(50):
        m = rng.choice(p.move_names)
        state = p.apply_move(state, m)
    feats = extract_features_numpy(state)
    corner_ori = feats[N_CORNERS : 2 * N_CORNERS]
    edge_ori = feats[2 * N_CORNERS + N_EDGES : 2 * N_CORNERS + 2 * N_EDGES]
    # These are standard cube invariants — test they hold for our encoding.
    # If they fail, our orientation encoding is inconsistent with the group's action.
    assert corner_ori.sum() % 3 == 0, f"corner ori sum mod 3 = {corner_ori.sum() % 3}"
    assert edge_ori.sum() % 2 == 0, f"edge ori sum mod 2 = {edge_ori.sum() % 2}"
