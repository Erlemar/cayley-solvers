"""Shared-cluster primitive-cost value model for cube666 macro search."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cube666.macro_policy import CLUSTER_COUNT, CLUSTER_SIZE, ResidualMLPBlock


@dataclass(frozen=True)
class MacroFactorizedValueConfig:
    local_dim: int = 192
    local_blocks: int = 3
    global_dim: int = 512
    global_blocks: int = 3
    total_scale: float = 256.0
    cluster_scale: float = 128.0
    dropout: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MacroFactorizedPrimitiveValueNet(nn.Module):
    """Share local permutation reasoning, then coordinate all six clusters."""

    def __init__(self, config: MacroFactorizedValueConfig) -> None:
        super().__init__()
        self.config = config
        local_input = CLUSTER_SIZE * CLUSTER_SIZE
        self.local_stem = nn.Sequential(
            nn.Linear(local_input, config.local_dim),
            nn.LayerNorm(config.local_dim),
            nn.SiLU(),
        )
        self.local_blocks = nn.ModuleList(
            ResidualMLPBlock(config.local_dim, config.dropout)
            for _ in range(config.local_blocks)
        )
        self.cluster_embedding = nn.Parameter(
            torch.zeros(CLUSTER_COUNT, config.local_dim)
        )
        self.local_norm = nn.LayerNorm(config.local_dim)
        self.cluster_head = nn.Linear(config.local_dim, 1)
        self.global_stem = nn.Sequential(
            nn.Linear(CLUSTER_COUNT * config.local_dim, config.global_dim),
            nn.LayerNorm(config.global_dim),
            nn.SiLU(),
        )
        self.global_blocks = nn.ModuleList(
            ResidualMLPBlock(config.global_dim, config.dropout)
            for _ in range(config.global_blocks)
        )
        self.global_norm = nn.LayerNorm(config.global_dim)
        self.total_head = nn.Sequential(
            nn.Linear(config.global_dim, config.global_dim // 2),
            nn.SiLU(),
            nn.Linear(config.global_dim // 2, 1),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 3 or tuple(states.shape[1:]) != (
            CLUSTER_COUNT,
            CLUSTER_SIZE,
        ):
            raise ValueError("states must have shape (batch, 6, 24)")
        encoded = F.one_hot(states.long(), num_classes=CLUSTER_SIZE)
        local = self.local_stem(encoded.flatten(start_dim=2).to(self.local_stem[0].weight.dtype))
        for block in self.local_blocks:
            local = block(local)
        local = self.local_norm(local + self.cluster_embedding[None])
        cluster_scaled = self.cluster_head(local).squeeze(-1)
        global_hidden = self.global_stem(local.flatten(start_dim=1))
        for block in self.global_blocks:
            global_hidden = block(global_hidden)
        total_scaled = self.total_head(self.global_norm(global_hidden)).squeeze(1)
        return total_scaled, cluster_scaled


def load_macro_factorized_value_checkpoint(
    path: str, device: torch.device | str = "cpu"
) -> tuple[MacroFactorizedPrimitiveValueNet, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = MacroFactorizedValueConfig(**payload["factorized_value_config"])
    model = MacroFactorizedPrimitiveValueNet(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
