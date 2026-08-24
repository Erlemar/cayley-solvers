"""Listwise rank loss for V/Q model training (m38).

For each batch state s with 24 children, the model produces predicted V values
for the children. We have target distances (from Bellman bootstrap or BFS).
Listwise loss encourages the model's RANKING of children to match the target's
ranking — independently of absolute scale.

Two formulations supported:
  - ListNet: cross-entropy between softmax(-pred/T) and softmax(-target/T)
    over the 24 children. Smooth, well-behaved gradient.
  - PairwiseHinge: for each pair (i, j) where target_i < target_j (i.e.,
    child i is closer to solved), require pred_i + margin <= pred_j else hinge.

Mix into Bellman trainer as auxiliary term: total_loss = MSE + lambda * listwise.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def listnet_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """ListNet (Cao 2007) loss over (B, K) children.

    Encourages softmax-distribution of preds to match softmax-distribution of
    targets. Lower distance = higher probability of being chosen as "best."
    We negate so that smaller V gets larger weight (smaller V = closer to solved).

    Args:
        pred:   (B, K) predicted V over K children
        target: (B, K) target distance over K children
        temperature: softmax temperature; smaller T = sharper distribution

    Returns:
        scalar mean cross-entropy over the batch
    """
    p_pred = F.softmax(-pred / temperature, dim=1)
    p_tgt = F.softmax(-target / temperature, dim=1)
    # Cross-entropy: -sum p_tgt * log(p_pred)
    eps = 1e-9
    ce = -(p_tgt * torch.log(p_pred + eps)).sum(dim=1)
    return ce.mean()


def pairwise_hinge_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Pairwise rank hinge loss over (B, K) children.

    For each pair (i, j) where target_i < target_j, require pred_i < pred_j by
    at least `margin`; else accumulate hinge violation.

    Args:
        pred:   (B, K) predicted V
        target: (B, K) target distance
        margin: minimum required gap

    Returns:
        scalar mean hinge violation
    """
    B, K = pred.shape
    # Pairwise differences: (B, K, K)
    pred_diff = pred.unsqueeze(2) - pred.unsqueeze(1)        # pred_i - pred_j
    target_diff = target.unsqueeze(2) - target.unsqueeze(1)  # target_i - target_j
    # We want pred_i < pred_j when target_i < target_j (i.e., target_diff < 0).
    # Mask: only count pairs where target_i < target_j (strict).
    mask = (target_diff < 0).float()
    # Hinge: max(0, pred_i - pred_j + margin) when we want pred_i < pred_j
    hinge = torch.clamp(pred_diff + margin, min=0)
    n = mask.sum().clamp(min=1)
    return (hinge * mask).sum() / n
