"""Distributional V head utilities (QR-DQN-style quantile regression).

Tested for Megaminx m42 experiment: predict N quantiles instead of scalar V,
allowing beam search to select on lower quantile (optimistic preference for
confident-close states). Reference: Dabney et al. 2018 (QR-DQN).

We treat the V regression as a distributional task where each prediction is a
distribution over plausible distances. Aleatoric uncertainty in the labels
(walk depth is an upper bound on true d) maps naturally to wide-quantile
distributions far from solved; epistemic uncertainty (model error) shrinks
quantile spread on confidently-known states.

The distributional Bellman backup is:

    For parent s:
        children = apply_all_generators(s)
        a* = argmin_a median(target_model(children[a]))
        target_distribution = clip(target_model(children[a*]) + 1, 0, walk_depth)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_quantile_taus(n_quantiles: int, device: str | torch.device) -> torch.Tensor:
    """Midpoint quantile fractions: (i + 0.5) / N for i in [0, N)."""
    return (torch.arange(n_quantiles, dtype=torch.float32, device=device) + 0.5) / n_quantiles


def quantile_huber_loss(
    pred: torch.Tensor,                  # (B, N) predicted quantiles
    target: torch.Tensor,                # (B, M) target distribution (typically M = N)
    taus: torch.Tensor,                  # (N,) quantile midpoints
    kappa: float = 1.0,
) -> torch.Tensor:
    """Asymmetric Huber loss between predicted and target quantile distributions.

    The N predicted quantiles are penalized for under/over-estimating each of
    the M target quantiles, weighted by `|tau_i - 1{delta < 0}|`. Huber-smoothing
    around |delta| <= kappa makes gradients well-behaved.

    Standard formulation from Dabney 2018 §3.3.
    """
    B, N = pred.shape
    M = target.shape[1]
    pred_e = pred.unsqueeze(2)           # (B, N, 1)
    target_e = target.unsqueeze(1)       # (B, 1, M)
    delta = target_e - pred_e            # (B, N, M)

    abs_d = delta.abs()
    huber = torch.where(
        abs_d <= kappa,
        0.5 * delta * delta,
        kappa * (abs_d - 0.5 * kappa),
    )

    indicator = (delta < 0).to(pred.dtype)               # (B, N, M)
    weight = (taus.view(1, N, 1) - indicator).abs()       # (B, N, M)
    loss = (weight * huber / max(kappa, 1e-6)).sum(dim=2).mean(dim=1).mean()
    return loss


@torch.no_grad()
def distributional_bellman_targets(
    target_model: torch.nn.Module,
    states: torch.Tensor,                # (B, S) int64
    walk_depths: torch.Tensor,           # (B,) float
    generators: torch.Tensor,            # (n_gen, S) int64
    solved_state: torch.Tensor,          # (S,) int64
    n_quantiles: int,
    chunk_size: int = 4096,
    clip_upper: bool = True,
    clip_lower: bool = True,
) -> torch.Tensor:
    """Compute target distributions for distributional Bellman backup.

    For each parent:
      a* = argmin_a median(target_model(child[a]))      # action selection on median
      target = clip(target_model(child[a*]) + 1, 0, walk_depth)   # quantile distribution shifted by 1

    Returns (B, n_quantiles) target distributions.
    """
    B, S = states.shape
    n_gen = generators.shape[0]

    children = torch.gather(
        states.unsqueeze(1).expand(B, n_gen, S), 2,
        generators.unsqueeze(0).expand(B, n_gen, S),
    )  # (B, n_gen, S)
    children_flat = children.reshape(B * n_gen, S)
    is_solved = (children_flat == solved_state).all(dim=1)  # (B*n_gen,)

    target_model.eval()
    child_q = torch.empty(B * n_gen, n_quantiles, dtype=torch.float32, device=states.device)
    for i in range(0, B * n_gen, chunk_size):
        chunk = children_flat[i:i + chunk_size]
        out = target_model(chunk).to(torch.float32)        # (chunk, n_quantiles)
        child_q[i:i + chunk_size] = out

    # Solved children get a degenerate distribution at 0 (boundary condition).
    child_q = torch.where(
        is_solved.unsqueeze(1), torch.zeros_like(child_q), child_q,
    )
    child_q = child_q.view(B, n_gen, n_quantiles)

    # Action selection: pick argmin over MEDIAN quantile.
    median_idx = n_quantiles // 2
    child_median = child_q[:, :, median_idx]               # (B, n_gen)
    a_star = child_median.argmin(dim=1)                    # (B,)

    # Target distribution = chosen child's quantiles + 1.
    chosen_q = child_q[torch.arange(B, device=states.device), a_star]  # (B, n_quantiles)
    target_q = chosen_q + 1.0

    if clip_upper:
        target_q = torch.minimum(target_q, walk_depths.unsqueeze(1))
    if clip_lower:
        target_q = target_q.clamp(min=0.0)

    return target_q
