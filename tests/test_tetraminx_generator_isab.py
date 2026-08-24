"""Focused tests for the residual Generator-ISAB Tetraminx Q model."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.models import GeneratorISABQ, ResMLPQ, model_from_config


LAYOUT = PROJECT / "tetraminx" / "data" / "piece_layout.json"


def make_model(**overrides) -> GeneratorISABQ:
    cfg = {
        "layout_path": LAYOUT,
        "state_size": 88,
        "num_classes": 88,
        "n_actions": 24,
        "hidden_dims": (64, 32),
        "num_res_blocks": 1,
        "embed_dim": 8,
        "d_model": 32,
        "nhead": 4,
        "num_latents": 4,
        "num_layers": 1,
        "ff_dim": 64,
        "action_ff_dim": 64,
        "dropout": 0.0,
    }
    cfg.update(overrides)
    return GeneratorISABQ(**cfg)


def sample_states(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    return torch.stack([torch.randperm(88, generator=generator) for _ in range(batch)])


def test_forward_shape_finite_and_standalone_interface():
    model = make_model().eval()
    q = model(sample_states())
    assert q.shape == (3, 24)
    assert torch.isfinite(q).all()
    assert model.output_dim == 24
    assert not model.has_value_head


def test_resmlp_warmstart_is_exact_at_zero_correction():
    torch.manual_seed(4)
    base = ResMLPQ(state_size=88, num_classes=88, hidden_dims=(64, 32),
                   num_res_blocks=1, embed_dim=8, n_actions=24).eval()
    model = make_model().eval()
    incompatible = model.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert not [key for key in incompatible.missing_keys
                if key.startswith(("embedding.", "input_stack.", "res_blocks.", "head."))]
    states = sample_states()
    torch.testing.assert_close(model(states), base(states), atol=0, rtol=0)
    assert torch.count_nonzero(model.correction(states)) == 0


def test_generator_geometry_is_exact_and_action_specific():
    model = make_model()
    assert model.generator_permutations.shape == (24, 88)
    assert model.action_source_pieces.shape == (24, 50)
    assert model.action_piece_mask.shape == (24, 50)
    assert model.generator_facelet_mask.shape == (24, 88)
    assert model.action_piece_mask.any(dim=1).all()
    assert model.generator_facelet_mask.any(dim=1).all()
    # All 24 generators must have distinct exact gather permutations.
    assert torch.unique(model.generator_permutations, dim=0).size(0) == 24


def test_frozen_base_and_gradients_reach_every_relational_stage():
    model = make_model().train()
    model.freeze_base()
    with torch.no_grad():
        model.correction_head.weight.normal_(std=0.02)
    loss = model(sample_states()).square().mean()
    loss.backward()

    params = dict(model.named_parameters())
    for name in (
        "local_value_embedding.weight",
        "piece_projection.weight",
        "induced_blocks.0.inducing_points",
        "induced_blocks.0.piece_to_latent.attn.q_proj.weight",
        "induced_blocks.0.latent_to_piece.attn.q_proj.weight",
        "generator_source_embedding.weight",
        "generator_destination_embedding.weight",
        "action_decoder.attn.q_proj.weight",
        "global_projection.weight",
        "correction_head.weight",
    ):
        grad = params[name].grad
        assert grad is not None, f"no gradient for {name}"
        assert torch.isfinite(grad).all(), f"non-finite gradient for {name}"
    for name, parameter in model.named_parameters():
        if name.startswith(("embedding.", "input_stack.", "res_blocks.", "head.")):
            assert not parameter.requires_grad
            assert parameter.grad is None


def test_config_and_state_dict_round_trip_reproduce_outputs():
    model = make_model().eval()
    with torch.no_grad():
        model.correction_head.weight.normal_(std=0.02)
    states = sample_states()
    expected = model(states)
    cfg = model.get_model_config()
    restored = model_from_config(cfg).eval()
    restored.load_state_dict(model.state_dict())
    assert isinstance(restored, GeneratorISABQ)
    assert restored.get_model_config() == cfg
    torch.testing.assert_close(restored(states), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"d_model": 30, "nhead": 4}, "divisible"),
        ({"num_latents": 0}, "num_latents"),
        ({"num_layers": 0}, "num_layers"),
        ({"az_head": True}, "auxiliary AZ"),
    ],
)
def test_invalid_configurations_fail_loudly(overrides, message):
    with pytest.raises(ValueError, match=message):
        make_model(**overrides)


def test_value_method_cannot_be_mistaken_for_auxiliary_head():
    with pytest.raises(RuntimeError, match="no separately trained value head"):
        make_model().value(sample_states(1))
