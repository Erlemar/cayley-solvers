"""Geometry-aware dual encoder for retrieving cube666 macro effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cube666.macro_policy import CLUSTER_COUNT, CLUSTER_SIZE, ResidualMLPBlock


@dataclass(frozen=True)
class MacroDualPolicyConfig:
    state_local_dim: int = 192
    state_local_blocks: int = 3
    state_global_dim: int = 512
    state_global_blocks: int = 3
    action_local_dim: int = 128
    action_local_blocks: int = 2
    action_global_dim: int = 384
    action_global_blocks: int = 2
    embedding_dim: int = 256
    total_scale: float = 256.0
    cluster_scale: float = 128.0
    maximum_action_cost: float = 256.0
    dropout: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MacroDualPolicyValueNet(nn.Module):
    """Embed states and verified macro effects into a shared retrieval space."""

    def __init__(self, config: MacroDualPolicyConfig) -> None:
        super().__init__()
        self.config = config
        local_input = CLUSTER_SIZE * CLUSTER_SIZE
        self.local_stem = nn.Sequential(
            nn.Linear(local_input, config.state_local_dim),
            nn.LayerNorm(config.state_local_dim),
            nn.SiLU(),
        )
        self.local_blocks = nn.ModuleList(
            ResidualMLPBlock(config.state_local_dim, config.dropout)
            for _ in range(config.state_local_blocks)
        )
        self.cluster_embedding = nn.Parameter(
            torch.zeros(CLUSTER_COUNT, config.state_local_dim)
        )
        self.local_norm = nn.LayerNorm(config.state_local_dim)
        self.global_stem = nn.Sequential(
            nn.Linear(CLUSTER_COUNT * config.state_local_dim, config.state_global_dim),
            nn.LayerNorm(config.state_global_dim),
            nn.SiLU(),
        )
        self.global_blocks = nn.ModuleList(
            ResidualMLPBlock(config.state_global_dim, config.dropout)
            for _ in range(config.state_global_blocks)
        )
        self.global_norm = nn.LayerNorm(config.state_global_dim)
        self.state_projection = nn.Linear(config.state_global_dim, config.embedding_dim)
        self.total_head = nn.Sequential(
            nn.Linear(config.state_global_dim, config.state_global_dim // 2),
            nn.SiLU(),
            nn.Linear(config.state_global_dim // 2, 1),
        )
        self.cluster_head = nn.Linear(config.state_local_dim, 1)

        self.action_local_stem = nn.Sequential(
            nn.Linear(local_input, config.action_local_dim),
            nn.LayerNorm(config.action_local_dim),
            nn.SiLU(),
        )
        self.action_local_blocks = nn.ModuleList(
            ResidualMLPBlock(config.action_local_dim, config.dropout)
            for _ in range(config.action_local_blocks)
        )
        self.action_cluster_embedding = nn.Parameter(
            torch.zeros(CLUSTER_COUNT, config.action_local_dim)
        )
        self.action_local_norm = nn.LayerNorm(config.action_local_dim)
        self.action_global_stem = nn.Sequential(
            nn.Linear(
                CLUSTER_COUNT * config.action_local_dim + 1,
                config.action_global_dim,
            ),
            nn.LayerNorm(config.action_global_dim),
            nn.SiLU(),
        )
        self.action_global_blocks = nn.ModuleList(
            ResidualMLPBlock(config.action_global_dim, config.dropout)
            for _ in range(config.action_global_blocks)
        )
        self.action_global_norm = nn.LayerNorm(config.action_global_dim)
        self.action_projection = nn.Linear(config.action_global_dim, config.embedding_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def encode_state_hidden(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 3 or tuple(states.shape[1:]) != (
            CLUSTER_COUNT,
            CLUSTER_SIZE,
        ):
            raise ValueError("states must have shape (batch, 6, 24)")
        encoded = F.one_hot(states.long(), num_classes=CLUSTER_SIZE)
        local = self.local_stem(
            encoded.flatten(start_dim=2).to(self.local_stem[0].weight.dtype)
        )
        for block in self.local_blocks:
            local = block(local)
        local = self.local_norm(local + self.cluster_embedding[None])
        global_hidden = self.global_stem(local.flatten(start_dim=1))
        for block in self.global_blocks:
            global_hidden = block(global_hidden)
        return self.global_norm(global_hidden), local

    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        global_hidden, _ = self.encode_state_hidden(states)
        return F.normalize(self.state_projection(global_hidden).float(), dim=1)

    def value_only(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_hidden, local = self.encode_state_hidden(states)
        return self.total_head(global_hidden).squeeze(1), self.cluster_head(local).squeeze(2)

    def encode_actions(
        self, effects: torch.Tensor, costs: torch.Tensor
    ) -> torch.Tensor:
        if effects.ndim != 3 or tuple(effects.shape[1:]) != (
            CLUSTER_COUNT,
            CLUSTER_SIZE,
        ):
            raise ValueError("effects must have shape (actions, 6, 24)")
        encoded = F.one_hot(effects.long(), num_classes=CLUSTER_SIZE)
        local = self.action_local_stem(
            encoded.flatten(start_dim=2).to(self.action_local_stem[0].weight.dtype)
        )
        for block in self.action_local_blocks:
            local = block(local)
        local = self.action_local_norm(local + self.action_cluster_embedding[None])
        scaled_cost = costs.to(local.dtype).reshape(-1, 1) / self.config.maximum_action_cost
        hidden = self.action_global_stem(
            torch.cat((local.flatten(start_dim=1), scaled_cost), dim=1)
        )
        for block in self.action_global_blocks:
            hidden = block(hidden)
        hidden = self.action_global_norm(hidden)
        return F.normalize(self.action_projection(hidden).float(), dim=1)

    def score(
        self,
        states: torch.Tensor,
        effects: torch.Tensor,
        costs: torch.Tensor,
    ) -> torch.Tensor:
        queries = self.encode_states(states)
        keys = self.encode_actions(effects, costs)
        return queries @ keys.transpose(0, 1) * self.logit_scale.exp().clamp_max(100.0)


def load_macro_dual_policy_checkpoint(
    path: str | bytes, device: torch.device | str = "cpu"
) -> tuple[MacroDualPolicyValueNet, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = MacroDualPolicyConfig(**payload["dual_policy_config"])
    model = MacroDualPolicyValueNet(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
