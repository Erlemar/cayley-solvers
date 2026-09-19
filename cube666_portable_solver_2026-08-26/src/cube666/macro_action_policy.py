"""Direct joint-action policy for the inverse-closed cube666 macro table."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cube666.macro_policy import CLUSTER_COUNT, CLUSTER_SIZE, ResidualMLPBlock


@dataclass(frozen=True)
class MacroActionPolicyConfig:
    action_count: int
    hidden_dim: int = 512
    residual_blocks: int = 4
    dropout: float = 0.0
    depth_conditioned: bool = False
    maximum_depth: int = 8

    def to_dict(self) -> dict[str, int | float | str]:
        return {"architecture": "direct_action", **asdict(self)}


class MacroActionPolicyNet(nn.Module):
    """Predict a joint macro ID without factorizing its sticker effect."""

    def __init__(self, config: MacroActionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Linear(CLUSTER_COUNT * CLUSTER_SIZE * CLUSTER_SIZE, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(config.hidden_dim, config.dropout)
            for _ in range(config.residual_blocks)
        )
        self.depth_embedding = (
            nn.Embedding(config.maximum_depth + 1, config.hidden_dim)
            if config.depth_conditioned
            else None
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.action_head = nn.Linear(config.hidden_dim, config.action_count)

    def forward(
        self,
        states: torch.Tensor,
        remaining_depth: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = F.one_hot(states.long(), num_classes=CLUSTER_SIZE)
        hidden = self.stem(
            encoded.flatten(start_dim=1).to(self.stem[0].weight.dtype)
        )
        if self.depth_embedding is not None:
            if remaining_depth is None:
                raise ValueError("depth-conditioned policy requires remaining_depth")
            if isinstance(remaining_depth, int):
                depth_ids = torch.full(
                    (len(states),),
                    remaining_depth,
                    dtype=torch.long,
                    device=states.device,
                )
            else:
                depth_ids = remaining_depth.to(device=states.device, dtype=torch.long)
                if depth_ids.ndim == 0:
                    depth_ids = depth_ids.expand(len(states))
            depth_ids = depth_ids.clamp(0, self.config.maximum_depth)
            hidden = hidden + self.depth_embedding(depth_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.action_head(self.final_norm(hidden))


def load_macro_action_policy_checkpoint(
    path: str, device: torch.device | str = "cpu"
) -> tuple[MacroActionPolicyNet, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    raw_config = dict(payload["model_config"])
    raw_config.pop("architecture", None)
    config = MacroActionPolicyConfig(**raw_config)
    model = MacroActionPolicyNet(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
