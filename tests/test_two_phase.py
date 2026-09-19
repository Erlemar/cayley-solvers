"""Unit tests for the two-phase (Kociemba-style) megaminx solver scaffolding.

Covers the pure-logic foundations that the trained models and beam solve rest on:
  - the movable/frozen sticker partition (35 frozen / 85 movable; frozen == LL),
  - the restricted Phase-2 puzzle (12 generators, inverse-closed),
  - frozen-invariance under TOP-only moves (the provable coset direction),
  - mask-by-value correctness + the frozen-placement sufficient-statistic property,
  - MaskedV config round-trip and forward,
  - the Phase-1 frozen goal-check predicate.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_two_phase.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "megaminx" / "src"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from cayley.model import ResMLPDistance
from megaminx.decomposition import (
    TOP_HALF_FACES,
    equator_sticker_positions,
    f2l_sticker_positions,
    ll_sticker_positions,
)
from megaminx.puzzle import Megaminx
from megaminx import two_phase as tp

PUZZLE_PATH = ROOT / "megaminx" / "data" / "puzzle_info.json"


def _puzzle() -> Megaminx:
    return Megaminx.load(PUZZLE_PATH)


def test_partition_counts_and_identity():
    pz = _puzzle()
    movable, frozen = tp.movable_frozen_positions(pz)
    assert len(movable) == 85
    assert len(frozen) == 35
    assert len(movable) + len(frozen) == len(pz.solved_state) == 120
    # no overlap
    assert set(movable).isdisjoint(set(frozen))
    # frozen are exactly the LL (bottom-only) stickers; movable are F2L + equator
    assert frozen == ll_sticker_positions(pz.generators)
    assert sorted(movable) == sorted(
        f2l_sticker_positions(pz.generators) + equator_sticker_positions(pz.generators)
    )


def test_restricted_puzzle_is_12_gens_inverse_closed():
    pz = _puzzle()
    p2 = tp.build_phase2_puzzle(pz)
    assert len(p2.move_names) == 12
    assert len(p2.generators) == 12
    # every generator's inverse is also present, and forward.inverse == identity
    for nm in p2.move_names:
        assert p2.inverse_name(nm) in p2.generators
    p2.verify_inverse_pairs()  # raises on mismatch
    # the 12 names are exactly the 6 TOP faces x {fwd, inv}
    faces = {nm.lstrip("-") for nm in p2.move_names}
    assert faces == set(TOP_HALF_FACES)


def test_top_moves_preserve_frozen():
    """The provable coset direction: Phase-2 (TOP) moves never touch frozen stickers."""
    pz = _puzzle()
    p2 = tp.build_phase2_puzzle(pz)
    movable, frozen = tp.movable_frozen_positions(pz)
    rng = np.random.default_rng(0)
    s = list(pz.solved_state)
    for _ in range(60):
        nm = p2.move_names[int(rng.integers(0, len(p2.move_names)))]
        s = list(p2.apply_move(s, nm))
    # frozen positions stay home through any TOP-only sequence
    assert all(s[p] == p for p in frozen)
    # and the movable region is genuinely exercised (sanity: not a no-op set)
    assert any(s[p] != p for p in movable)


def test_mask_by_value_keeps_frozen_placement():
    pz = _puzzle()
    movable, frozen = tp.movable_frozen_positions(pz)
    movable_set = set(movable)
    inner = ResMLPDistance(state_size=120, num_classes=121, hidden_dims=(32, 16),
                           num_res_blocks=1, encoding="embedding", embed_dim=8)
    model = tp.MaskedV(inner, movable)

    # On the identity, every movable slot holds a movable value -> masked; frozen kept.
    ident = torch.tensor([list(pz.solved_state)])
    m_ident = model.mask(ident)[0].tolist()
    assert all(m_ident[p] == tp.MASK_TOKEN for p in movable)
    assert all(m_ident[p] == p for p in frozen)

    # On an arbitrary full scramble: a slot is masked iff its VALUE is movable,
    # and the surviving (non-masked) entries are exactly the frozen-sticker placement.
    rng = np.random.default_rng(1)
    s = list(pz.solved_state)
    for _ in range(25):
        nm = pz.move_names[int(rng.integers(0, len(pz.move_names)))]
        s = list(pz.apply_move(s, nm))
    st = torch.tensor([s])
    masked = model.mask(st)[0].tolist()
    for p in range(120):
        if s[p] in movable_set:
            assert masked[p] == tp.MASK_TOKEN
        else:
            assert masked[p] == s[p]
    # sufficient statistic: surviving entries == {(slot, frozen-sticker) placements}
    survivors = {p: masked[p] for p in range(120) if masked[p] != tp.MASK_TOKEN}
    expected = {p: s[p] for p in range(120) if s[p] not in movable_set}
    assert survivors == expected


def test_maskedv_config_roundtrip():
    pz = _puzzle()
    movable, _frozen = tp.movable_frozen_positions(pz)
    inner = ResMLPDistance(state_size=120, num_classes=121, hidden_dims=(32, 16),
                           num_res_blocks=1, encoding="embedding", embed_dim=8)
    model = tp.MaskedV(inner, movable)
    cfg = model.get_model_config()
    assert cfg["model_class"] == "MaskedV"
    assert cfg["mask_token"] == tp.MASK_TOKEN
    assert cfg["inner_config"]["num_classes"] == 121

    # Rebuild from config + copy weights, assert identical masking + output.
    inner2 = ResMLPDistance(**{k: v for k, v in cfg["inner_config"].items()
                               if k != "model_class"})
    model2 = tp.MaskedV(inner2, cfg["movable_values"], cfg["mask_token"])
    model2.load_state_dict(model.state_dict())
    x = torch.tensor([list(pz.solved_state)])
    assert torch.equal(model.mask(x), model2.mask(x))
    with torch.no_grad():
        assert torch.allclose(model(x), model2(x))
    assert tuple(model(x).shape) == (1,)


def test_frozen_goal_check():
    pz = _puzzle()
    movable, frozen = tp.movable_frozen_positions(pz)
    goal = tp.make_frozen_goal_check(frozen, "cpu")

    solved = torch.tensor([list(pz.solved_state)])
    assert bool(goal(solved)[0]) is True

    # Frozen home but movable scrambled (a TOP walk) -> still a goal state.
    p2 = tp.build_phase2_puzzle(pz)
    s = list(pz.solved_state)
    rng = np.random.default_rng(2)
    for _ in range(30):
        s = list(p2.apply_move(s, p2.move_names[int(rng.integers(0, 12))]))
    assert bool(goal(torch.tensor([s]))[0]) is True

    # Disturb one frozen sticker -> not a goal state.
    bad = list(pz.solved_state)
    f0, f1 = frozen[0], frozen[1]
    bad[f0], bad[f1] = bad[f1], bad[f0]
    assert bool(goal(torch.tensor([bad]))[0]) is False


def test_phase_anchor_batches():
    pz = _puzzle()
    p2 = tp.build_phase2_puzzle(pz)
    _movable, frozen = tp.movable_frozen_positions(pz)

    p2_states, p2_depths = tp.phase2_v0_d1_anchors(
        p2, n_v0=2, n_d1_per_move=1, device="cpu")
    assert tuple(p2_states.shape) == (2 + len(p2.move_names), 120)
    assert torch.equal(p2_depths[:2], torch.zeros(2))
    assert torch.equal(p2_depths[2:], torch.ones(len(p2.move_names)))

    h_states, h_depths = tp.phase1_h_zero_anchors(
        p2, n_h_samples=20, h_k_max=5, seed=123, device="cpu", n_solved=2)
    assert tuple(h_states.shape) == (22, 120)
    assert torch.equal(h_depths, torch.zeros(22))
    for state in h_states.tolist():
        assert all(state[p] == p for p in frozen)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
