"""Autoregressive primitive-token decoder for verified short cube666 macros."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cube666.macro_policy import CLUSTER_COUNT, CLUSTER_SIZE, ResidualMLPBlock


@dataclass(frozen=True)
class MacroAutoregressiveConfig:
    primitive_action_count: int = 36
    maximum_tokens: int = 15
    hidden_dim: int = 512
    residual_blocks: int = 4
    token_embedding_dim: int = 64
    decoder_layers: int = 2
    value_scale: float = 256.0
    dropout: float = 0.0

    @property
    def eos_token(self) -> int:
        return self.primitive_action_count

    @property
    def bos_token(self) -> int:
        return self.primitive_action_count + 1

    @property
    def vocabulary_size(self) -> int:
        return self.primitive_action_count + 1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MacroAutoregressivePolicyValueNet(nn.Module):
    """Encode a six-cluster state and decode a short macro causally."""

    def __init__(self, config: MacroAutoregressiveConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = CLUSTER_COUNT * CLUSTER_SIZE * CLUSTER_SIZE
        self.stem = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(config.hidden_dim, config.dropout)
            for _ in range(config.residual_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.token_embedding = nn.Embedding(
            config.primitive_action_count + 2, config.token_embedding_dim
        )
        self.decoder = nn.GRU(
            config.token_embedding_dim,
            config.hidden_dim,
            num_layers=config.decoder_layers,
            batch_first=True,
        )
        self.initial_hidden = nn.Linear(
            config.hidden_dim, config.decoder_layers * config.hidden_dim
        )
        self.token_head = nn.Linear(config.hidden_dim, config.vocabulary_size)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or tuple(states.shape[1:]) != (
            CLUSTER_COUNT,
            CLUSTER_SIZE,
        ):
            raise ValueError("states must have shape (batch, 6, 24)")
        encoded = F.one_hot(states.long(), num_classes=CLUSTER_SIZE)
        hidden = self.stem(encoded.flatten(start_dim=1).to(self.stem[0].weight.dtype))
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def decoder_initial(self, state_hidden: torch.Tensor) -> torch.Tensor:
        return self.initial_hidden(state_hidden).reshape(
            len(state_hidden), self.config.decoder_layers, self.config.hidden_dim
        ).transpose(0, 1).contiguous()

    def decode_step(
        self, tokens: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_hidden = self.decoder(
            self.token_embedding(tokens.long())[:, None], hidden
        )
        return self.token_head(output[:, 0]), next_hidden

    def forward(
        self, states: torch.Tensor, prefixes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_hidden = self.encode(states)
        output, _ = self.decoder(
            self.token_embedding(prefixes.long()), self.decoder_initial(state_hidden)
        )
        return self.token_head(output), self.value_head(state_hidden).squeeze(1)


@dataclass(frozen=True)
class MacroFactorizedAutoregressiveConfig:
    primitive_action_count: int = 36
    maximum_tokens: int = 15
    local_dim: int = 192
    local_blocks: int = 3
    hidden_dim: int = 512
    global_blocks: int = 3
    token_embedding_dim: int = 64
    decoder_layers: int = 2
    value_scale: float = 256.0
    dropout: float = 0.0

    @property
    def eos_token(self) -> int:
        return self.primitive_action_count

    @property
    def bos_token(self) -> int:
        return self.primitive_action_count + 1

    @property
    def vocabulary_size(self) -> int:
        return self.primitive_action_count + 1

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MacroFactorizedAutoregressivePolicyValueNet(nn.Module):
    """Shared per-cluster encoder followed by a coordinated macro decoder."""

    def __init__(self, config: MacroFactorizedAutoregressiveConfig) -> None:
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
        self.global_stem = nn.Sequential(
            nn.Linear(CLUSTER_COUNT * config.local_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.global_blocks = nn.ModuleList(
            ResidualMLPBlock(config.hidden_dim, config.dropout)
            for _ in range(config.global_blocks)
        )
        self.global_norm = nn.LayerNorm(config.hidden_dim)
        self.token_embedding = nn.Embedding(
            config.primitive_action_count + 2, config.token_embedding_dim
        )
        self.decoder = nn.GRU(
            config.token_embedding_dim,
            config.hidden_dim,
            num_layers=config.decoder_layers,
            batch_first=True,
        )
        self.initial_hidden = nn.Linear(
            config.hidden_dim, config.decoder_layers * config.hidden_dim
        )
        self.token_head = nn.Linear(config.hidden_dim, config.vocabulary_size)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
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
        hidden = self.global_stem(local.flatten(start_dim=1))
        for block in self.global_blocks:
            hidden = block(hidden)
        return self.global_norm(hidden)

    def decoder_initial(self, state_hidden: torch.Tensor) -> torch.Tensor:
        return self.initial_hidden(state_hidden).reshape(
            len(state_hidden), self.config.decoder_layers, self.config.hidden_dim
        ).transpose(0, 1).contiguous()

    def decode_step(
        self, tokens: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, next_hidden = self.decoder(
            self.token_embedding(tokens.long())[:, None], hidden
        )
        return self.token_head(output[:, 0]), next_hidden

    def forward(
        self, states: torch.Tensor, prefixes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_hidden = self.encode(states)
        output, _ = self.decoder(
            self.token_embedding(prefixes.long()), self.decoder_initial(state_hidden)
        )
        return self.token_head(output), self.value_head(state_hidden).squeeze(1)


def load_macro_autoregressive_checkpoint(
    path: str, device: torch.device | str = "cpu"
) -> tuple[
    MacroAutoregressivePolicyValueNet | MacroFactorizedAutoregressivePolicyValueNet,
    dict[str, object],
]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "factorized_autoregressive_config" in payload:
        config = MacroFactorizedAutoregressiveConfig(
            **payload["factorized_autoregressive_config"]
        )
        model = MacroFactorizedAutoregressivePolicyValueNet(config).to(device)
    else:
        config = MacroAutoregressiveConfig(**payload["model_config"])
        model = MacroAutoregressivePolicyValueNet(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
