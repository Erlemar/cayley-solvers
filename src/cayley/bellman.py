"""Bellman-style target refinement for the diffusion-distance model.

Standard training uses walk depth as the label. Walk depth is an UPPER BOUND on the true
distance — a state reached after k random steps is provably within k moves of solved, but
may be much closer. The model learns to predict this overestimate.

Bellman refinement corrects this by using self-bootstrapped targets:

    d(s) = 0                           if s is solved
    d(s) = 1 + min_a d(apply(s, a))    otherwise

We train `model` so `model(s) ≈ 1 + min_a model_target(apply(s, a))`, where `model_target`
is a frozen snapshot updated periodically (prevents oscillation). We also clip the target
at the walk depth (still a valid upper bound) and at 0 (can't be negative).

Warm-start from a checkpoint trained with walk-depth labels (e.g. our E5 model). Training
from scratch with Bellman targets rarely converges.

Reference: DeepCubeA (Agostinelli et al. 2019), which used this recipe to go from ~60%
solve rate to 100% on regular 3×3×3.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.model import ResMLPDistance
from cayley.puzzle import PictureCube
from cayley.training import _LOSS_FNS_ELEMENTWISE, EpochStats, TrainConfig, TrainResult, _NullCtx, _iterate_batches


@dataclass
class BellmanConfig:
    """Additional knobs specific to the Bellman loop. Combine with `TrainConfig` for the
    underlying fast-recipe settings (batch, lr, amp, compile, etc.)."""

    warmstart_path: str = ""                # required — path to an existing .pt checkpoint
    target_update_every_epochs: int = 10   # how often to copy `model` → `target_model`
    target_net_chunk: int = 8192           # chunk size for target-net forward (memory)
    # safety: clip Bellman targets at the walk depth (upper bound) and 0 (lower bound).
    # these are virtually always active; knobs exposed for debugging.
    clip_upper: bool = True
    clip_lower: bool = True


def _apply_all_generators(
    states: torch.Tensor, generators: torch.Tensor
) -> torch.Tensor:
    """Return (B, n_gen, state_size) with children[i, g] = apply(states[i], generator g)."""
    B, S = states.shape
    n_gen = generators.shape[0]
    return torch.gather(
        states.unsqueeze(1).expand(B, n_gen, S),
        2,
        generators.unsqueeze(0).expand(B, n_gen, S),
    )


@torch.no_grad()
def _bellman_targets(
    target_model: torch.nn.Module,
    states: torch.Tensor,
    walk_depths: torch.Tensor,
    generators: torch.Tensor,
    solved_state: torch.Tensor,
    chunk_size: int,
    clip_upper: bool,
    clip_lower: bool,
) -> torch.Tensor:
    """Compute `y = clip(1 + min_a target(apply(s, a)), 0, walk_depth)` for a batch.

    A child that equals the solved state gets value 0 (not whatever the model predicts)
    — this is the only boundary condition and we want it exact.
    """
    B, S = states.shape
    children = _apply_all_generators(states, generators)  # (B, n_gen, S)
    n_gen = children.shape[1]
    children_flat = children.reshape(B * n_gen, S)

    is_solved = (children_flat == solved_state).all(dim=1)  # (B*n_gen,)

    # Target-net forward in chunks.
    child_values = torch.empty(B * n_gen, dtype=torch.float32, device=states.device)
    target_model.eval()
    for i in range(0, B * n_gen, chunk_size):
        chunk = children_flat[i : i + chunk_size]
        vals = target_model(chunk).flatten().to(torch.float32)
        child_values[i : i + chunk_size] = vals

    child_values = torch.where(is_solved, torch.zeros_like(child_values), child_values)
    child_values = child_values.view(B, n_gen)
    min_child = child_values.min(dim=1).values  # (B,)
    target = 1.0 + min_child

    if clip_upper:
        target = torch.minimum(target, walk_depths)
    if clip_lower:
        target = torch.clamp(target, min=0.0)
    return target


def _load_warmstart(path: str | Path, model: ResMLPDistance, device: str) -> None:
    """Load weights into `model` from a checkpoint, stripping any `_orig_mod.` prefix."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)


def train_bellman(
    model: ResMLPDistance,
    puzzle: PictureCube,
    cfg: TrainConfig,
    bcfg: BellmanConfig,
    checkpoint_dir: str | Path,
    on_epoch_end: Callable[[EpochStats], None] | None = None,
) -> TrainResult:
    """Bellman refinement training loop. Warm-starts from `bcfg.warmstart_path`."""
    if not bcfg.warmstart_path:
        raise ValueError("BellmanConfig.warmstart_path is required.")
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _load_warmstart(bcfg.warmstart_path, model, cfg.device)
    model = model.to(cfg.device)

    # Target network: frozen snapshot of the model, updated every N epochs.
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    # Optionally compile the trainable model (not the target — avoids double compile cost).
    if cfg.compile_model and cfg.device == "cuda":
        model = torch.compile(model, dynamic=False)

    optim_cls = torch.optim.AdamW if cfg.weight_decay > 0 else torch.optim.Adam
    optim_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.fused_optimizer and cfg.device == "cuda":
        optim_kwargs["fused"] = True
    optimizer = optim_cls(model.parameters(), **optim_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.n_epochs)

    # Precompute generators tensor + solved state tensor for the Bellman target function.
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(cfg.device)  # (n_gen, state_size)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=cfg.device)

    elem_loss = _LOSS_FNS_ELEMENTWISE[cfg.loss]
    batch_gen = torch.Generator(device=cfg.device)
    batch_gen.manual_seed(cfg.seed)

    use_amp = cfg.amp and cfg.device == "cuda"
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()

    result = TrainResult(final_loss=float("inf"))

    for epoch in range(cfg.n_epochs):
        t0 = time.time()
        n_walks = max(1, cfg.samples_per_epoch // cfg.k_max)
        states, depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=cfg.k_max, seed=cfg.seed + epoch,
            device=cfg.device, n_back=cfg.n_back,
        )
        depths_f = depths.to(torch.float32)

        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch_idx in _iterate_batches(states, depths_f, cfg.batch_size, batch_gen):
            bs = states[batch_idx]
            bd = depths_f[batch_idx]
            target = _bellman_targets(
                target_model, bs, bd, generators, solved_state,
                chunk_size=bcfg.target_net_chunk,
                clip_upper=bcfg.clip_upper, clip_lower=bcfg.clip_lower,
            )
            with autocast_ctx:
                pred = model(bs)
                loss = elem_loss(pred, target).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        stats = EpochStats(epoch=epoch, loss=avg_loss, lr=float(scheduler.get_last_lr()[0]),
                           elapsed_s=time.time() - t0)
        result.epochs.append(stats)
        result.final_loss = avg_loss
        if on_epoch_end is not None:
            on_epoch_end(stats)

        # Refresh target net. Use the compiled module's original state dict.
        if (epoch + 1) % bcfg.target_update_every_epochs == 0:
            source_sd = model.state_dict()
            if any(k.startswith("_orig_mod.") for k in source_sd):
                source_sd = {k.removeprefix("_orig_mod."): v for k, v in source_sd.items()}
            target_model.load_state_dict(source_sd)

        # Checkpoint.
        if (epoch + 1) % cfg.checkpoint_every_epochs == 0 or epoch == cfg.n_epochs - 1:
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            base = getattr(model, "_orig_mod", model)
            model_cfg = {
                "state_size": base.state_size,
                "num_classes": base.num_classes,
                "hidden_dims": [layer.out_features for layer in base.input_stack
                                if isinstance(layer, torch.nn.Linear)],
                "num_res_blocks": len(base.res_blocks),
                "encoding": base.encoding,
                "embed_dim": base.embed_dim,
            }
            save_sd = model.state_dict()
            if any(k.startswith("_orig_mod.") for k in save_sd):
                save_sd = {k.removeprefix("_orig_mod."): v for k, v in save_sd.items()}
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": save_sd,
                    "loss": avg_loss,
                    "train_config": cfg.__dict__,
                    "model_config": model_cfg,
                    "bellman_config": bcfg.__dict__,
                },
                path,
            )
    return result
