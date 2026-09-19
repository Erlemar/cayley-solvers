"""Neural shortlist policy for the exact 6x6x6 bulk-macro action table."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


# Keep the neural model module importable in lightweight inference/training
# bundles.  These dimensions are the representation contract; importing the
# full macro-data loader here unnecessarily pulls in the classical solver.
CLUSTER_COUNT = 6
CLUSTER_SIZE = 24


@dataclass(frozen=True)
class MacroPolicyConfig:
    action_count: int
    hidden_dim: int = 384
    residual_blocks: int = 3
    dropout: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(inputs)
        hidden = F.silu(self.linear_in(hidden))
        hidden = self.dropout(self.linear_out(hidden))
        return inputs + hidden


class MacroPolicyValueNet(nn.Module):
    """Predict macro logits plus exact-residual auxiliary targets.

    Input is a lossless one-hot representation of six 24-piece permutations.
    The policy output is never applied directly: search exact-scores its top-k.
    """

    def __init__(self, config: MacroPolicyConfig) -> None:
        super().__init__()
        if config.action_count <= 0:
            raise ValueError("action_count must be positive")
        if config.hidden_dim <= 0 or config.residual_blocks < 0:
            raise ValueError("invalid macro policy dimensions")
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
        self.policy_head = nn.Linear(config.hidden_dim, config.action_count)
        self.total_cost_head = nn.Linear(config.hidden_dim, 1)
        self.cluster_cost_head = nn.Linear(config.hidden_dim, CLUSTER_COUNT)

    @staticmethod
    def encode(states: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if states.ndim != 3 or states.shape[1:] != (CLUSTER_COUNT, CLUSTER_SIZE):
            raise ValueError(
                f"expected states shaped (batch, {CLUSTER_COUNT}, {CLUSTER_SIZE}); "
                f"got {tuple(states.shape)}"
            )
        encoded = F.one_hot(states.long(), num_classes=CLUSTER_SIZE)
        return encoded.flatten(start_dim=1).to(dtype=dtype)

    def encode_hidden(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(self.encode(states, self.stem[0].weight.dtype))
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def value_only(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(states)
        return (
            self.total_cost_head(hidden).squeeze(-1),
            self.cluster_cost_head(hidden),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(states)
        return (
            self.policy_head(hidden),
            self.total_cost_head(hidden).squeeze(-1),
            self.cluster_cost_head(hidden),
        )


@dataclass(frozen=True)
class FactorizedMacroPolicyConfig:
    action_count: int
    hidden_dim: int = 256
    residual_blocks: int = 3
    dropout: float = 0.0

    def as_dict(self) -> dict[str, int | float | str]:
        return {"architecture": "factorized", **asdict(self)}


class FactorizedMacroPolicyValueNet(nn.Module):
    """Shared policy/value solver for six independent 24-piece clusters.

    The isolated action library consists of six contiguous, identically ordered
    blocks of directed 3-cycles.  Sharing the local network across those blocks
    turns every cube state into six training examples and encodes the exact
    additive structure of the residual distance.
    """

    def __init__(self, config: FactorizedMacroPolicyConfig) -> None:
        super().__init__()
        if config.action_count <= 0 or config.action_count % CLUSTER_COUNT:
            raise ValueError("factorized action count must be positive and divisible by six")
        if config.hidden_dim <= 0 or config.residual_blocks < 0:
            raise ValueError("invalid factorized macro policy dimensions")
        self.config = config
        self.local_action_count = config.action_count // CLUSTER_COUNT
        input_dim = CLUSTER_SIZE * CLUSTER_SIZE
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
        self.policy_head = nn.Linear(config.hidden_dim, self.local_action_count)
        self.cluster_cost_head = nn.Linear(config.hidden_dim, 1)

    def encode_hidden(self, states: torch.Tensor) -> tuple[torch.Tensor, int]:
        if states.ndim != 3 or states.shape[1:] != (CLUSTER_COUNT, CLUSTER_SIZE):
            raise ValueError(
                f"expected states shaped (batch, {CLUSTER_COUNT}, {CLUSTER_SIZE}); "
                f"got {tuple(states.shape)}"
            )
        batch_size = states.shape[0]
        encoded = F.one_hot(states.long(), num_classes=CLUSTER_SIZE)
        encoded = encoded.reshape(batch_size * CLUSTER_COUNT, -1)
        hidden = self.stem(encoded.to(dtype=self.stem[0].weight.dtype))
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden), batch_size

    def value_only(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, batch_size = self.encode_hidden(states)
        cluster_values = self.cluster_cost_head(hidden).reshape(batch_size, CLUSTER_COUNT)
        total_values = cluster_values.sum(dim=1) / CLUSTER_COUNT
        return total_values, cluster_values

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, batch_size = self.encode_hidden(states)
        logits = self.policy_head(hidden).reshape(batch_size, -1)
        cluster_values = self.cluster_cost_head(hidden).reshape(batch_size, CLUSTER_COUNT)
        total_values = cluster_values.sum(dim=1) / CLUSTER_COUNT
        return logits, total_values, cluster_values


def build_macro_policy_model(
    raw_config: dict[str, int | float | str],
) -> MacroPolicyValueNet | FactorizedMacroPolicyValueNet | MacroEffectPolicyValueNet:
    """Build either checkpoint-compatible direct or factorized architecture."""

    config = dict(raw_config)
    architecture = str(config.pop("architecture", "direct"))
    if architecture == "direct":
        return MacroPolicyValueNet(MacroPolicyConfig(**config))
    if architecture == "factorized":
        return FactorizedMacroPolicyValueNet(FactorizedMacroPolicyConfig(**config))
    if architecture == "effect":
        return MacroEffectPolicyValueNet(MacroEffectPolicyConfig(**config))
    raise ValueError(f"unknown macro policy architecture: {architecture}")


@dataclass(frozen=True)
class MacroEffectPolicyConfig:
    action_count: int
    hidden_dim: int = 384
    residual_blocks: int = 3
    dropout: float = 0.0

    def as_dict(self) -> dict[str, int | float | str]:
        return {"architecture": "effect", **asdict(self)}


class MacroEffectPolicyValueNet(nn.Module):
    """Predict a useful macro effect instead of memorizing opaque action IDs."""

    def __init__(self, config: MacroEffectPolicyConfig) -> None:
        super().__init__()
        if config.action_count <= 0 or config.hidden_dim <= 0 or config.residual_blocks < 0:
            raise ValueError("invalid macro effect policy dimensions")
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
        self.effect_head = nn.Linear(
            config.hidden_dim,
            CLUSTER_COUNT * CLUSTER_SIZE * CLUSTER_SIZE,
        )
        self.total_cost_head = nn.Linear(config.hidden_dim, 1)
        self.cluster_cost_head = nn.Linear(config.hidden_dim, CLUSTER_COUNT)

    def encode_hidden(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(MacroPolicyValueNet.encode(states, self.stem[0].weight.dtype))
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)

    def value_only(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(states)
        return (
            self.total_cost_head(hidden).squeeze(-1),
            self.cluster_cost_head(hidden),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encode_hidden(states)
        effect_logits = self.effect_head(hidden).reshape(
            -1,
            CLUSTER_COUNT,
            CLUSTER_SIZE,
            CLUSTER_SIZE,
        )
        return (
            effect_logits,
            self.total_cost_head(hidden).squeeze(-1),
            self.cluster_cost_head(hidden),
        )


def score_macro_effect_candidates(
    effect_logits: torch.Tensor,
    action_effects: torch.Tensor,
) -> torch.Tensor:
    """Score matched per-sample candidate effects by log-odds over identity."""

    if effect_logits.ndim != 4 or effect_logits.shape[1:] != (
        CLUSTER_COUNT,
        CLUSTER_SIZE,
        CLUSTER_SIZE,
    ):
        raise ValueError("effect_logits has the wrong shape")
    if action_effects.ndim != 4 or action_effects.shape[0] != effect_logits.shape[0]:
        raise ValueError("action_effects must have shape (batch, candidates, 6, 24)")
    log_probs = F.log_softmax(effect_logits.float(), dim=-1)
    candidate_count = action_effects.shape[1]
    expanded = log_probs[:, None].expand(-1, candidate_count, -1, -1, -1)
    selected = expanded.gather(-1, action_effects.long().unsqueeze(-1)).squeeze(-1)
    identity = torch.arange(CLUSTER_SIZE, device=effect_logits.device)
    identity = identity.view(1, 1, 1, CLUSTER_SIZE, 1).expand(
        effect_logits.shape[0],
        candidate_count,
        CLUSTER_COUNT,
        -1,
        -1,
    )
    baseline = expanded.gather(-1, identity).squeeze(-1)
    active = action_effects.ne(
        torch.arange(CLUSTER_SIZE, device=effect_logits.device).view(1, 1, 1, -1)
    )
    active_count = active.sum(dim=(2, 3)).clamp_min(1)
    return ((selected - baseline) * active).sum(dim=(2, 3)) / active_count.sqrt()


def macro_effect_policy_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    teacher_actions: torch.Tensor,
    teacher_action_counts: torch.Tensor,
    cluster_cost_targets: torch.Tensor,
    action_table_effects: torch.Tensor,
    negative_actions: torch.Tensor,
    *,
    token_weight: float = 0.2,
    value_weight: float = 0.2,
    cluster_value_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Contrast exact teacher effects against sampled negatives."""

    effect_logits, total_cost_prediction, cluster_cost_prediction = outputs
    batch_size, positive_slots = teacher_actions.shape
    if negative_actions.ndim != 2 or negative_actions.shape[0] != batch_size:
        raise ValueError("negative_actions has the wrong shape")
    valid_positive = (
        torch.arange(positive_slots, device=teacher_actions.device)[None, :]
        < teacher_action_counts[:, None]
    )
    positive_ids = teacher_actions.clamp_min(0)
    candidate_ids = torch.cat((positive_ids, negative_actions), dim=1)
    candidate_effects = action_table_effects[candidate_ids]
    scores = score_macro_effect_candidates(effect_logits, candidate_effects)
    scores[:, :positive_slots] = scores[:, :positive_slots].masked_fill(
        ~valid_positive,
        -torch.inf,
    )
    negative_duplicates = negative_actions[:, :, None].eq(positive_ids[:, None, :])
    negative_duplicates &= valid_positive[:, None, :]
    scores[:, positive_slots:] = scores[:, positive_slots:].masked_fill(
        negative_duplicates.any(dim=2),
        -torch.inf,
    )
    numerator = torch.logsumexp(scores[:, :positive_slots], dim=1)
    denominator = torch.logsumexp(scores, dim=1)
    contrastive = (denominator - numerator).mean()

    target_effects = action_table_effects[positive_ids[:, 0]]
    flat_logits = effect_logits.reshape(-1, CLUSTER_SIZE)
    flat_targets = target_effects.reshape(-1).long()
    token_all = F.cross_entropy(flat_logits.float(), flat_targets)
    identity = torch.arange(CLUSTER_SIZE, device=effect_logits.device).view(1, 1, -1)
    active = target_effects.ne(identity).reshape(-1)
    token_active = F.cross_entropy(flat_logits[active].float(), flat_targets[active])
    token = token_all + 2.0 * token_active

    cluster_targets = cluster_cost_targets.float()
    total_targets = cluster_targets.sum(dim=1)
    value = F.smooth_l1_loss(total_cost_prediction.float(), total_targets / 72.0)
    cluster_value = F.smooth_l1_loss(cluster_cost_prediction.float(), cluster_targets / 12.0)
    total = (
        contrastive
        + token_weight * token
        + value_weight * value
        + cluster_value_weight * cluster_value
    )
    return total, {
        "loss": total.detach(),
        "contrastive_loss": contrastive.detach(),
        "token_loss": token.detach(),
        "value_loss": value.detach(),
        "cluster_value_loss": cluster_value.detach(),
    }


def macro_policy_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    teacher_actions: torch.Tensor,
    teacher_action_counts: torch.Tensor,
    cluster_cost_targets: torch.Tensor,
    *,
    value_weight: float = 0.2,
    cluster_value_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Multi-positive policy loss with normalized exact-cost auxiliaries."""

    logits, total_cost_prediction, cluster_cost_prediction = outputs
    if teacher_actions.ndim != 2 or teacher_actions.shape[0] != logits.shape[0]:
        raise ValueError("teacher action tensor has the wrong shape")
    label_slots = torch.arange(teacher_actions.shape[1], device=teacher_actions.device)
    valid = label_slots[None, :] < teacher_action_counts[:, None]
    safe_actions = teacher_actions.clamp_min(0)
    positive_log_probs = F.log_softmax(logits.float(), dim=-1).gather(1, safe_actions)
    positive_log_probs = positive_log_probs.masked_fill(~valid, -torch.inf)
    policy = -torch.logsumexp(positive_log_probs, dim=1).mean()

    cluster_targets = cluster_cost_targets.float()
    total_targets = cluster_targets.sum(dim=1)
    value = F.smooth_l1_loss(total_cost_prediction.float(), total_targets / 72.0)
    cluster_value = F.smooth_l1_loss(cluster_cost_prediction.float(), cluster_targets / 12.0)
    total = policy + value_weight * value + cluster_value_weight * cluster_value
    return total, {
        "loss": total.detach(),
        "policy_loss": policy.detach(),
        "value_loss": value.detach(),
        "cluster_value_loss": cluster_value.detach(),
    }


@torch.no_grad()
def policy_recall_at_k(
    logits: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_action_counts: torch.Tensor,
    ks: tuple[int, ...] = (1, 16, 64),
) -> dict[int, float]:
    max_k = min(max(ks), logits.shape[1])
    proposed = logits.topk(max_k, dim=1).indices
    slots = torch.arange(teacher_actions.shape[1], device=teacher_actions.device)
    valid = slots[None, :] < teacher_action_counts[:, None]
    matches = proposed[:, :, None].eq(teacher_actions[:, None, :]) & valid[:, None, :]
    return {
        k: float(matches[:, : min(k, max_k)].any(dim=(1, 2)).float().mean().item())
        for k in ks
    }
