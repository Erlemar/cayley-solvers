"""Optimizer builders for the cayley training loop.

Supports vanilla Adam/AdamW (existing default) and a Muon+AdamW split.

Muon (Keller Jordan et al., Dec 2024): Newton-Schulz-orthogonalized momentum update for
2D hidden-layer weight matrices. Stronger conditioning on plateaus → potentially breaks
the MSE floor we've been stuck at.

Param split convention (standard Muon recipe):
 - Muon: weight tensors of `nn.Linear` in the hidden trunk (input_stack + res_blocks)
 - AdamW: everything else — nn.Embedding, nn.LayerNorm, all biases, the output `head` Linear
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


class MultiOptimizer:
    """Forwards step / zero_grad to multiple underlying optimizers.

    Matches the subset of the `torch.optim.Optimizer` API that our training loop uses.
    Lets us treat "Muon on 2D + AdamW on everything else" as a single object in the loop.
    """

    def __init__(self, *optimizers):
        self.optimizers = list(optimizers)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.optimizers:
            o.step()

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def state_dict(self):
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["optimizers"]):
            o.load_state_dict(s)


class MultiScheduler:
    """Steps multiple schedulers in lockstep. `get_last_lr()` returns the first scheduler's
    LR (for logging purposes)."""

    def __init__(self, *schedulers: LRScheduler):
        self.schedulers = list(schedulers)

    def step(self) -> None:
        for s in self.schedulers:
            s.step()

    def get_last_lr(self):
        # Concatenate all schedulers' LRs for logging; callers that only want "the" LR
        # can index [0] and will get the first (primary) optimizer's LR.
        out = []
        for s in self.schedulers:
            out.extend(s.get_last_lr())
        return out


def split_params_muon_adamw(
    model: nn.Module,
    head_module_name: str = "head",
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split model parameters into (muon_params, adamw_params) following the standard
    Muon recipe: Linear weights in hidden trunks go to Muon; embeddings, LayerNorms, all
    biases, and the output head (named `head_module_name`) go to AdamW.
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    seen = set()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            is_head = name == head_module_name
            if is_head:
                # Output head stays on AdamW per Muon's own recommendation.
                if module.weight.requires_grad and id(module.weight) not in seen:
                    adamw_params.append(module.weight); seen.add(id(module.weight))
                if module.bias is not None and module.bias.requires_grad and id(module.bias) not in seen:
                    adamw_params.append(module.bias); seen.add(id(module.bias))
            else:
                if module.weight.requires_grad and id(module.weight) not in seen:
                    muon_params.append(module.weight); seen.add(id(module.weight))
                if module.bias is not None and module.bias.requires_grad and id(module.bias) not in seen:
                    adamw_params.append(module.bias); seen.add(id(module.bias))
        elif isinstance(module, (nn.Embedding, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            for p in module.parameters(recurse=False):
                if p.requires_grad and id(p) not in seen:
                    adamw_params.append(p); seen.add(id(p))

    # Catch anything the walker missed (e.g., directly-registered parameters).
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            # Conservative default: everything unseen goes to AdamW (safer than Muon).
            adamw_params.append(p); seen.add(id(p))

    return muon_params, adamw_params


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    n_epochs: int,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    fused: bool = True,
    muon_lr: float = 2e-2,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.01,
    muon_ns_steps: int = 5,
    muon_adjust_lr_fn: str | None = None,
):
    """Returns (optimizer_or_MultiOptimizer, scheduler_or_MultiScheduler).

    optimizer_name:
      - 'adam' / 'adamw': single AdamW on all params. (AdamW iff weight_decay > 0.)
      - 'muon': torch.optim.Muon on hidden Linear weights + AdamW on the rest.
    """
    device_ok = any(p.device.type == "cuda" for p in model.parameters())

    name = optimizer_name.lower()
    if name in ("adam", "adamw"):
        cls = AdamW if (weight_decay > 0 or name == "adamw") else Adam
        kw = dict(lr=lr, weight_decay=weight_decay)
        if fused and device_ok:
            kw["fused"] = True
        opt = cls(model.parameters(), **kw)
        sched = CosineAnnealingLR(opt, T_max=n_epochs)
        return opt, sched

    if name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "torch.optim.Muon not available (need PyTorch 2.9+). "
                "Local env is fine; Kaggle P100 kernels pin torch 2.4.1, so don't use optimizer=muon there."
            )
        muon_params, adamw_params = split_params_muon_adamw(model)
        n_muon = sum(p.numel() for p in muon_params)
        n_adamw = sum(p.numel() for p in adamw_params)
        print(f"[Muon split] Muon: {len(muon_params)} tensors, {n_muon:,} params; "
              f"AdamW: {len(adamw_params)} tensors, {n_adamw:,} params")
        muon_kw = dict(lr=muon_lr, momentum=muon_momentum, weight_decay=muon_weight_decay,
                       ns_steps=muon_ns_steps)
        if muon_adjust_lr_fn is not None:
            muon_kw["adjust_lr_fn"] = muon_adjust_lr_fn
        muon_opt = torch.optim.Muon(muon_params, **muon_kw)
        adamw_kw = dict(lr=lr, weight_decay=weight_decay)
        if fused and device_ok:
            adamw_kw["fused"] = True
        adamw_opt = AdamW(adamw_params, **adamw_kw)
        opt = MultiOptimizer(muon_opt, adamw_opt)
        sched = MultiScheduler(
            CosineAnnealingLR(muon_opt, T_max=n_epochs),
            CosineAnnealingLR(adamw_opt, T_max=n_epochs),
        )
        return opt, sched

    raise ValueError(f"unknown optimizer_name {optimizer_name!r}. "
                     f"Expected 'adam', 'adamw', or 'muon'.")
