"""Focused tests for the standalone Tetraminx latent relational Q model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.models import LatentRelationalQ, _folded_piece_tokens, model_from_config


LAYOUT = PROJECT / "tetraminx" / "data" / "piece_layout.json"


def make_model(**overrides) -> LatentRelationalQ:
    cfg = {
        "layout_path": LAYOUT,
        "state_size": 88,
        "num_classes": 88,
        "n_actions": 24,
        "d_model": 32,
        "nhead": 4,
        "num_latents": 4,
        "num_layers": 1,
        "ff_dim": 64,
        "action_ff_dim": 64,
        "dropout": 0.0,
    }
    cfg.update(overrides)
    return LatentRelationalQ(**cfg)


def sample_states(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.stack([torch.randperm(88, generator=generator) for _ in range(batch)])


def test_forward_shape_finite_and_standalone_q_interface():
    model = make_model().eval()
    q = model(sample_states())
    assert q.shape == (3, 24)
    assert torch.isfinite(q).all()
    assert model.output_dim == 24
    assert not model.has_value_head
    assert model.return_value is False


def test_dueling_decomposition_is_identifiable():
    model = make_model().eval()
    states = sample_states()
    value, centred_advantage = model.q_components(states)
    q = model(states)
    torch.testing.assert_close(centred_advantage.mean(dim=1), torch.zeros(3),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(q.mean(dim=1), value, atol=1e-6, rtol=1e-6)


def test_folded_piece_encoder_matches_direct_concat_projection():
    model = make_model().eval()
    states = sample_states()
    b = states.size(0)
    vals = states.long().index_select(1, model.piece_positions.reshape(-1)).view(
        b, model.num_pieces, model.max_piece_size)
    offsets = torch.arange(model.max_piece_size) * model.num_classes
    local = model.local_value_embedding(vals + offsets.view(1, 1, -1))
    local = local * model.piece_mask.view(1, model.num_pieces, -1, 1)
    direct = model.piece_projection(local.flatten(start_dim=-2))
    direct = direct + model.piece_position_embedding(model.piece_indices).unsqueeze(0)
    direct = direct + model.piece_type_embedding(model.piece_types).unsqueeze(0)
    folded = _folded_piece_tokens(
        states, piece_positions=model.piece_positions, piece_mask=model.piece_mask,
        piece_types=model.piece_types, piece_indices=model.piece_indices,
        local_value_embedding=model.local_value_embedding,
        piece_projection=model.piece_projection,
        piece_position_embedding=model.piece_position_embedding,
        piece_type_embedding=model.piece_type_embedding,
        num_pieces=model.num_pieces, max_piece_size=model.max_piece_size,
        num_classes=model.num_classes, d_model=model.d_model,
    )
    torch.testing.assert_close(folded, direct, atol=1e-5, rtol=1e-5)


def test_gradients_reach_every_architectural_stage():
    model = make_model().train()
    loss = model(sample_states()).square().mean()
    loss.backward()
    required = (
        "local_value_embedding.weight",
        "piece_projection.weight",
        "latent_queries",
        "compressor.attn.q_proj.weight",
        "latent_blocks.0.attn.in_proj_weight",
        "action_queries",
        "action_decoder.attn.q_proj.weight",
        "advantage_head.weight",
        "state_value_head.weight",
    )
    params = dict(model.named_parameters())
    for name in required:
        grad = params[name].grad
        assert grad is not None, f"no gradient for {name}"
        assert torch.isfinite(grad).all(), f"non-finite gradient for {name}"


def test_config_and_state_dict_round_trip_reproduce_outputs():
    model = make_model().eval()
    states = sample_states()
    expected = model(states)
    cfg = model.get_model_config()
    restored = model_from_config(cfg).eval()
    restored.load_state_dict(model.state_dict())
    assert isinstance(restored, LatentRelationalQ)
    assert restored.get_model_config() == cfg
    torch.testing.assert_close(restored(states), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"d_model": 30, "nhead": 4}, "divisible"),
        ({"num_latents": 0}, "num_latents"),
        ({"num_layers": -1}, "num_layers"),
        ({"az_head": True}, "auxiliary AZ"),
    ],
)
def test_invalid_configurations_fail_loudly(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_model(**overrides)


def test_value_method_cannot_be_mistaken_for_auxiliary_az_head():
    model = make_model().eval()
    with pytest.raises(RuntimeError, match="no separately trained value head"):
        model.value(sample_states(1))
