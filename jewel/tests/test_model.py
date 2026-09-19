import torch

from jewel.model import JewelTransformer, TransformerConfig


def test_transformer_shapes_and_gradients() -> None:
    model = JewelTransformer(TransformerConfig(d_model=64, n_heads=4, n_layers=2, dim_feedforward=128))
    ep = torch.arange(12).repeat(4, 1)
    eo = torch.zeros(4, 12, dtype=torch.long)
    ro = torch.zeros(4, 6, dtype=torch.long)
    outputs = model(ep, eo, ro)
    assert outputs["policy_logits"].shape == (4, 12)
    assert outputs["geodesic_logits"].shape == (4, 12)
    assert outputs["regret"].shape == (4, 12)
    assert outputs["distance"].shape == (4,)
    assert outputs["cdf_logits"].shape == (4, 33)
    sum(x.float().sum() for x in outputs.values()).backward()
