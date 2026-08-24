"""Full-state primitive policy/value network for the 6x6x6 picture cube."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PrimitivePolicyConfig:
    """Serializable architecture and target-scaling configuration."""

    state_size: int = 216
    piece_count: int = 216
    action_count: int = 36
    embedding_dim: int = 16
    hidden_dim: int = 512
    residual_blocks: int = 3
    expansion: int = 2
    value_scale: float = 128.0
    encoding: str = "piece_embedding"
    orbit_count: int = 9
    orbit_size: int = 24

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ResidualBlock(nn.Module):
    def __init__(self, width: int, expansion: int) -> None:
        super().__init__()
        inner = width * expansion
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, inner)
        self.down = nn.Linear(inner, width)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.up(self.norm(inputs)))
        return inputs + self.down(hidden)


class PrimitivePolicyValueNetwork(nn.Module):
    """Embed a complete permutation and predict primitive policy plus distance."""

    def __init__(self, config: PrimitivePolicyConfig) -> None:
        super().__init__()
        self.config = config
        if config.encoding == "piece_embedding":
            self.piece_embedding: nn.Embedding | None = nn.Embedding(
                config.piece_count, config.embedding_dim
            )
            input_dim = config.state_size * config.embedding_dim + config.state_size
        elif config.encoding == "orbit_one_hot":
            self.piece_embedding = None
            input_dim = config.orbit_count * config.orbit_size * config.orbit_size
        else:
            raise ValueError(f"unknown primitive encoding: {config.encoding}")
        self.stem = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden_dim, config.expansion)
            for _ in range(config.residual_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.policy_head = nn.Linear(config.hidden_dim, config.action_count)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.register_buffer(
            "solved_positions",
            torch.arange(config.state_size, dtype=torch.long),
            persistent=False,
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.encoding == "piece_embedding":
            if states.ndim != 2 or states.shape[1] != self.config.state_size:
                raise ValueError("piece-embedding states must have shape (batch, 216)")
            states = states.long()
            if self.piece_embedding is None:
                raise AssertionError("piece embedding was not initialized")
            embedded = self.piece_embedding(states).flatten(start_dim=1)
            correct = states.eq(self.solved_positions).to(embedded.dtype)
            encoded = torch.cat((embedded, correct), dim=1)
        else:
            expected = (self.config.orbit_count, self.config.orbit_size)
            if states.ndim != 3 or tuple(states.shape[1:]) != expected:
                raise ValueError(
                    f"orbit states must have shape (batch, {expected[0]}, {expected[1]})"
                )
            encoded = F.one_hot(
                states.long(), num_classes=self.config.orbit_size
            ).flatten(start_dim=1).to(self.stem[0].weight.dtype)
        hidden = self.stem(encoded)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self.policy_head(hidden)
        scaled_value = self.value_head(hidden).squeeze(1)
        return logits, scaled_value


def load_primitive_checkpoint(
    path: str, device: torch.device | str = "cpu"
) -> tuple[PrimitivePolicyValueNetwork, dict[str, object]]:
    """Load a checkpoint and return an eval-mode model plus metadata."""

    payload = torch.load(path, map_location=device, weights_only=False)
    config = PrimitivePolicyConfig(**payload["model_config"])
    model = PrimitivePolicyValueNetwork(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
