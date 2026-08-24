"""Policy/value network for one exact 24-piece stabilizer projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StabilizerModelConfig:
    action_count: int
    embedding_dim: int = 32
    hidden_dim: int = 1024
    residual_blocks: int = 6


class StabilizerResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(F.silu(self.norm(inputs)))
        hidden = self.linear2(F.silu(hidden))
        return inputs + hidden


class StabilizerPolicyValue(nn.Module):
    def __init__(self, config: StabilizerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.piece_embedding = nn.Embedding(24, config.embedding_dim)
        self.position_embedding = nn.Parameter(torch.zeros(24, config.embedding_dim))
        self.input = nn.Linear(24 * config.embedding_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            StabilizerResidualBlock(config.hidden_dim)
            for _ in range(config.residual_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.policy = nn.Linear(config.hidden_dim, config.action_count)
        self.value = nn.Linear(config.hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.piece_embedding(states.long()) + self.position_embedding[None]
        hidden = self.input(embedded.flatten(1))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = F.silu(self.final_norm(hidden))
        return self.policy(hidden), self.value(hidden).squeeze(1)
