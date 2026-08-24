"""Structured action-query Transformer for Christopher's Jewel."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .puzzle import ACTION_EDGE_FLIP, ACTION_EDGE_SRC, ACTION_RING_DELTA


@dataclass(slots=True)
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.0
    cdf_bins: int = 33

    def to_dict(self) -> dict:
        return asdict(self)


def _action_features() -> torch.Tensor:
    features = []
    for action in range(12):
        src_onehot = np.eye(12, dtype=np.float32)[ACTION_EDGE_SRC[action]].reshape(-1)
        flips = ACTION_EDGE_FLIP[action].astype(np.float32)
        rings = ACTION_RING_DELTA[action].astype(np.float32)
        rings = np.where(rings == 3, -1.0, rings)
        features.append(np.concatenate((src_onehot, flips, rings)))
    return torch.from_numpy(np.stack(features))


class JewelTransformer(nn.Module):
    def __init__(self, config: TransformerConfig = TransformerConfig()):
        super().__init__()
        self.config = config
        d = config.d_model
        self.edge_piece = nn.Embedding(12, d)
        self.edge_orientation = nn.Embedding(2, d)
        self.edge_slot = nn.Embedding(12, d)
        self.ring_orientation = nn.Embedding(4, d)
        self.ring_slot = nn.Embedding(6, d)
        self.token_type = nn.Embedding(4, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.action_query = nn.Parameter(torch.zeros(1, 12, d))
        action_features = _action_features()
        self.register_buffer("action_features", action_features, persistent=True)
        self.action_feature_projection = nn.Linear(action_features.shape[1], d, bias=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d)
        self.policy_head = nn.Linear(d, 1)
        self.geodesic_head = nn.Linear(d, 1)
        self.regret_head = nn.Linear(d, 1)
        self.distance_head = nn.Linear(d, 1)
        self.cdf_head = nn.Linear(d, config.cdf_bins)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.action_query, std=0.02)

    def forward(self, edge_perm: torch.Tensor, edge_ori: torch.Tensor, ring_ori: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = edge_perm.shape[0]
        device = edge_perm.device
        edge_slots = torch.arange(12, device=device)
        ring_slots = torch.arange(6, device=device)

        edge_tokens = (
            self.edge_piece(edge_perm.long())
            + self.edge_orientation(edge_ori.long())
            + self.edge_slot(edge_slots)[None, :, :]
            + self.token_type.weight[1][None, None, :]
        )
        ring_tokens = (
            self.ring_orientation(ring_ori.long())
            + self.ring_slot(ring_slots)[None, :, :]
            + self.token_type.weight[2][None, None, :]
        )
        action_tokens = (
            self.action_query.expand(batch, -1, -1)
            + self.action_feature_projection(self.action_features)[None, :, :]
            + self.token_type.weight[3][None, None, :]
        )
        cls = self.cls.expand(batch, -1, -1) + self.token_type.weight[0][None, None, :]
        encoded = self.final_norm(self.encoder(torch.cat((cls, edge_tokens, ring_tokens, action_tokens), dim=1)))
        global_token = encoded[:, 0]
        actions = encoded[:, -12:]
        return {
            "policy_logits": self.policy_head(actions).squeeze(-1),
            "geodesic_logits": self.geodesic_head(actions).squeeze(-1),
            "regret": self.regret_head(actions).squeeze(-1),
            "distance": self.distance_head(global_token).squeeze(-1),
            "cdf_logits": self.cdf_head(global_token),
        }


def load_transformer(path: str, device: str | torch.device = "cpu") -> JewelTransformer:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = TransformerConfig(**checkpoint["config"])
    model = JewelTransformer(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model

