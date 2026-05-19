"""GFlowNet-style model with policy head + value head + logZ scalar.

Used for Trajectory Balance training (Malkin et al. 2022, Pan et al. 2026).
Architecture mirrors ResMLPDistance trunk; adds:
  - policy head: (B, n_gen) action logits → softmax = P_F(a|s)
  - value head: (B,) scalar = log F(s) (state log-flow), proxy for distance
  - logZ: single learnable scalar (~ partition function)

For shortest-path application: trajectories from scrambled state to solved (R=1).
P_B is uniform 1/n_gen for bijective puzzles (megaminx, picture cube).

TB loss per trajectory τ = (s_0, a_0, s_1, ..., a_{n-1}, s_n=V0):
    L = (logZ + Σ log P_F(a_t|s_t) - Σ log P_B(s_t|s_{t+1}) - log R)²
With shortest-path regularizer + λ·logZ that pushes Z toward minimal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cayley.model import ResBlock


class ResMLPGFlowNet(nn.Module):
    """ResMLP trunk + dual heads (policy logits, value scalar) + logZ scalar."""

    def __init__(
        self,
        state_size: int = 120,
        num_classes: int = 120,
        hidden_dims: tuple[int, ...] = (2048, 512),
        num_res_blocks: int = 2,
        encoding: str = "embedding",
        embed_dim: int = 16,
        n_actions: int = 24,
        inference_chunk_size: int | None = 2048,
    ):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.encoding = encoding
        self.embed_dim = embed_dim
        self.n_actions = n_actions
        self.inference_chunk_size = inference_chunk_size

        if encoding == "onehot":
            in_dim = state_size * num_classes
            self.embedding = None
        elif encoding == "embedding":
            in_dim = state_size * embed_dim
            self.embedding = nn.Embedding(num_classes, embed_dim)
        else:
            raise ValueError(f"encoding must be 'onehot' or 'embedding', got {encoding!r}")

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(num_res_blocks)])

        # Dual heads
        self.policy_head = nn.Linear(prev, n_actions)
        self.value_head = nn.Linear(prev, 1)
        # Single learnable scalar for logZ
        self.log_Z = nn.Parameter(torch.zeros(1))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.input_stack[0].weight.dtype
        if self.encoding == "onehot":
            one_hot = F.one_hot(x.long(), num_classes=self.num_classes)
            return one_hot.to(target_dtype).flatten(start_dim=-2)
        return self.embedding(x.long()).to(target_dtype).flatten(start_dim=-2)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor):
        """Returns (policy_logits, value).

        policy_logits: (B, n_actions) — apply log_softmax for log P_F(a|s)
        value:         (B,) — log F(s), proxy for negative distance
        """
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def predict_value(self, x: torch.Tensor) -> torch.Tensor:
        """Eval-mode value-only forward (drop-in for V models in beam search).

        Negates the log-flow to give a positive distance estimate (smaller is closer).
        At optimum, log F(s) ~ const - dist(s, V0); so -value ~ dist + const.
        """
        if self.training or self.inference_chunk_size is None or x.shape[0] <= self.inference_chunk_size:
            return -self(x)[1]
        outs = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            outs.append(-self(x[i : i + self.inference_chunk_size])[1])
        return torch.cat(outs, dim=0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def trajectory_balance_loss(
    log_Z: torch.Tensor,
    log_pf_per_step: torch.Tensor,  # (B, T) — log P_F(a_t|s_t) per step
    traj_lengths: torch.Tensor,     # (B,) — actual trajectory length n
    log_pb_uniform: float,          # = -log(n_gen), uniform backward
    log_R_terminal: torch.Tensor,   # (B,) — log R(x), typically 0 if R=1 at solved
):
    """Compute TB loss per trajectory and return scalar mean.

    L_TB(τ) = (logZ + Σ log P_F(a_t|s_t) - Σ log P_B(s_t|s_{t+1}) - log R)²

    For uniform backward P_B = 1/n_gen: Σ log P_B = -n * log(n_gen).
    Per-step log_pf is masked by trajectory length (padded steps contribute 0).
    """
    B, T = log_pf_per_step.shape
    step_mask = (torch.arange(T, device=log_pf_per_step.device).unsqueeze(0)
                 < traj_lengths.unsqueeze(1))  # (B, T) bool
    log_pf_sum = (log_pf_per_step * step_mask).sum(dim=1)  # (B,)
    log_pb_sum = log_pb_uniform * traj_lengths.float()      # (B,)
    residual = log_Z + log_pf_sum - log_pb_sum - log_R_terminal
    return (residual ** 2).mean()
