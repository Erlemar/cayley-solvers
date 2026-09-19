"""History-conditioned primitive policy/value model for the 6x6x6 picture cube."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cube666.primitive_policy import PrimitivePolicyValueNetwork, ResidualBlock


@dataclass(frozen=True)
class PrimitiveHistoryPolicyConfig:
    orbit_count: int = 9
    orbit_size: int = 24
    action_count: int = 36
    history_length: int = 16
    history_embedding_dim: int = 32
    hidden_dim: int = 512
    residual_blocks: int = 3
    expansion: int = 2
    value_scale: float = 128.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class PrimitiveHistoryPolicyValueNetwork(nn.Module):
    """Encode the cube and recent primitive actions, retaining a state-only path."""

    def __init__(self, config: PrimitiveHistoryPolicyConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.orbit_count * config.orbit_size * config.orbit_size
        self.stem = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden_dim, config.expansion)
            for _ in range(config.residual_blocks)
        )
        self.history_embedding = nn.Embedding(
            config.action_count + 1,
            config.history_embedding_dim,
            padding_idx=config.action_count,
        )
        self.history_projection = nn.Linear(
            config.history_length * config.history_embedding_dim,
            config.hidden_dim,
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.policy_head = nn.Linear(config.hidden_dim, config.action_count)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        nn.init.zeros_(self.history_projection.weight)
        nn.init.zeros_(self.history_projection.bias)

    def initialize_state_path(self, source: PrimitivePolicyValueNetwork) -> None:
        if source.config.encoding != "orbit_one_hot":
            raise ValueError("source model must use orbit-one-hot encoding")
        if source.config.hidden_dim != self.config.hidden_dim:
            raise ValueError("source hidden dimension differs")
        if source.config.residual_blocks != self.config.residual_blocks:
            raise ValueError("source residual block count differs")
        self.stem.load_state_dict(source.stem.state_dict())
        self.blocks.load_state_dict(source.blocks.state_dict())
        self.final_norm.load_state_dict(source.final_norm.state_dict())
        self.policy_head.load_state_dict(source.policy_head.state_dict())
        self.value_head.load_state_dict(source.value_head.state_dict())

    def forward(
        self, states: torch.Tensor, histories: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.config.orbit_count, self.config.orbit_size)
        if states.ndim != 3 or tuple(states.shape[1:]) != expected:
            raise ValueError(f"states must have shape (batch, {expected[0]}, {expected[1]})")
        if histories.shape != (len(states), self.config.history_length):
            raise ValueError(
                f"histories must have shape (batch, {self.config.history_length})"
            )
        encoded = F.one_hot(
            states.long(), num_classes=self.config.orbit_size
        ).flatten(start_dim=1).to(self.stem[0].weight.dtype)
        hidden = self.stem(encoded)
        for block in self.blocks:
            hidden = block(hidden)
        history_hidden = self.history_projection(
            self.history_embedding(histories.long()).flatten(start_dim=1)
        )
        hidden = self.final_norm(hidden + history_hidden)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(1)


def load_primitive_history_checkpoint(
    path: str, device: torch.device | str = "cpu"
) -> tuple[PrimitiveHistoryPolicyValueNetwork, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = PrimitiveHistoryPolicyConfig(**payload["history_model_config"])
    model = PrimitiveHistoryPolicyValueNetwork(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
