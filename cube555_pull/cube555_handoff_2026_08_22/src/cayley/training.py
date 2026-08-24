"""Training loop for the ResMLP diffusion-distance predictor.

Each epoch generates fresh data (non-backtracking random walks from solved) and trains
the model to predict walk depth from state. This matches the CayleyPy paper's recipe.

Run on Kaggle GPU for serious training; the local Windows box should suffice for a
single-epoch smoke test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from cayley.data import generate_walks_torch, load_kociemba_walks, sample_from_bfs_table
from cayley.model import ResMLPDistance
from cayley.optimizers import build_optimizer_and_scheduler
from cayley.puzzle import PictureCube


@dataclass
class TrainConfig:
    n_epochs: int = 100
    samples_per_epoch: int = 1_000_000
    batch_size: int = 4096
    k_max: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda"
    seed: int = 0
    log_every_batches: int = 200
    checkpoint_every_epochs: int = 5
    n_back: int = 1  # non-backtracking depth for random walks
    loss: str = "mse"  # "mse", "l1", or "huber"
    curriculum: bool = False  # depth-weighted loss: weight ∝ 1/k (DeepCubeA recipe)
    # --- speed optimizations ---
    amp: bool = False  # bf16 autocast on forward + backward
    compile_model: bool = False  # torch.compile the forward (first epoch slow)
    fused_optimizer: bool = True  # use fused Adam/AdamW kernel
    # --- data-side supplements (stack with the above) ---
    bfs_table_path: str = ""  # path to BFS-d5 pkl; empty = disabled
    bfs_mix_fraction: float = (
        0.0  # share of each epoch's samples drawn from BFS table (exact labels)
    )
    kociemba_walks_path: str = ""  # path to kociemba_walks.pkl; empty = disabled
    kociemba_mix_fraction: float = (
        0.0  # share of each epoch's samples from Kociemba walks
    )
    augment_symmetry: bool = (
        False  # apply a random rotational symmetry per sample per epoch
    )
    # --- optimizer selection ---
    # --- walk-label fix (02_DATA.md) ---
    # Dedup the walk stream by colouring keeping the MIN label, then override the label
    # with the exact distance wherever the BFS ball knows the state. OBJECTIVE CHANGE:
    # only meaningful on a fresh run, and it needs a matched control. Off by default so
    # the control arm is the untouched historical recipe.
    label_fix: bool = False
    label_fix_table_path: str = ""  # bfs_anchors.pt; empty = dedup only, no override
    optimizer: str = "adam"  # 'adam' / 'adamw' / 'muon' (Muon needs torch 2.9+ locally)
    muon_lr: float = 2e-2  # Muon typically wants ~10x the AdamW lr
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str = (
        ""  # empty = None; other values passed as-is to torch.optim.Muon
    )


@dataclass
class EpochStats:
    epoch: int
    loss: float
    lr: float
    elapsed_s: float


@dataclass
class TrainResult:
    final_loss: float
    epochs: list[EpochStats] = field(default_factory=list)


def _iterate_batches(
    states: torch.Tensor,
    depths: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    n = states.shape[0]
    # permutation on GPU to avoid a host roundtrip
    idx = torch.randperm(n, generator=generator, device=states.device)
    batches: list[torch.Tensor] = []
    for i in range(0, n, batch_size):
        batches.append(idx[i : i + batch_size])
    return batches


_LOSS_FNS_ELEMENTWISE = {
    "mse": lambda pred, tgt: (pred - tgt) ** 2,
    "l1": lambda pred, tgt: (pred - tgt).abs(),
    "huber": lambda pred, tgt: F.huber_loss(pred, tgt, delta=1.0, reduction="none"),
}

# Populated by `train()` if `augment_symmetry` is enabled in the config; read by
# `train_one_epoch` to apply per-sample random rotations without changing the function
# signature.
_SYM_STATE: tuple | None = None


def train_one_epoch(
    model: ResMLPDistance,
    optimizer,
    puzzle: PictureCube,
    cfg: TrainConfig,
    data_seed: int,
    batch_gen: torch.Generator,
    bfs_table=None,
    kociemba_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    anchor_sampler: Callable[[int], tuple[torch.Tensor, torch.Tensor]] | None = None,
    label_fixer=None,  # cayley.label_fix.LabelFixer | None
) -> float:
    # Budget each source's share of the epoch. Random walks get whatever isn't claimed
    # by the supplementary sources. Fractions are a fraction of samples_per_epoch, not
    # of each batch — this keeps the per-epoch total bounded and simplifies shape math.
    bfs_frac = cfg.bfs_mix_fraction if bfs_table is not None else 0.0
    koc_frac = cfg.kociemba_mix_fraction if kociemba_data is not None else 0.0
    if bfs_frac + koc_frac > 1.0:
        raise ValueError(f"mix fractions sum > 1: bfs={bfs_frac} koc={koc_frac}")
    walk_frac = 1.0 - bfs_frac - koc_frac

    walk_samples = int(cfg.samples_per_epoch * walk_frac)
    bfs_samples = int(cfg.samples_per_epoch * bfs_frac)
    koc_samples = int(cfg.samples_per_epoch * koc_frac)

    # Random walks. Each walk emits k_max samples; n_walks = walk_samples / k_max.
    n_walks = max(1, walk_samples // cfg.k_max)
    w_states, w_depths = generate_walks_torch(
        puzzle,
        n_walks=n_walks,
        k_max=cfg.k_max,
        seed=data_seed,
        device=cfg.device,
        n_back=cfg.n_back,
    )

    # Walk-label fix (02_DATA.md). Applied to the WALK stream only -- the BFS, Kociemba
    # and anchor streams below already carry exact or independently-sourced labels, and
    # deduping across them could drop an anchor.
    if label_fixer is not None:
        w_states, w_depths, _fix_stats = label_fixer.apply(
            w_states, w_depths, round_to=cfg.batch_size if cfg.compile_model else None
        )
        print(f"  {_fix_stats.render()}", flush=True)

    # BFS exact-label samples (uniform draw from the table).
    if bfs_samples > 0:
        b_states, b_depths = sample_from_bfs_table(
            bfs_table,
            n_samples=bfs_samples,
            device=cfg.device,
            seed=data_seed,
        )
    else:
        b_states = torch.empty(
            (0, w_states.shape[1]), dtype=torch.int64, device=cfg.device
        )
        b_depths = torch.empty((0,), dtype=torch.int64, device=cfg.device)

    # Kociemba walks (sub-sample without replacement).
    if koc_samples > 0 and kociemba_data is not None:
        k_all_states, k_all_depths = kociemba_data
        n_k = k_all_states.shape[0]
        take = min(koc_samples, n_k)
        # Sample-with-replacement is fine — epoch-to-epoch order differs, and `take` can
        # exceed `n_k` if the user wants heavy weighting. Use explicit generator for reproducibility.
        rand_gen = torch.Generator(device=cfg.device).manual_seed(data_seed)
        if koc_samples <= n_k:
            idx = torch.randperm(n_k, generator=rand_gen, device=cfg.device)[:take]
        else:
            idx = torch.randint(
                0, n_k, (koc_samples,), generator=rand_gen, device=cfg.device
            )
        k_states = k_all_states[idx]
        k_depths = k_all_depths[idx]
    else:
        k_states = torch.empty(
            (0, w_states.shape[1]), dtype=torch.int64, device=cfg.device
        )
        k_depths = torch.empty((0,), dtype=torch.int64, device=cfg.device)

    if anchor_sampler is not None:
        a_states, a_depths = anchor_sampler(data_seed)
        if a_states.numel() > 0 and a_states.shape[1] != w_states.shape[1]:
            raise ValueError(
                f"anchor state size {a_states.shape[1]} != walk state size {w_states.shape[1]}"
            )
        a_states = a_states.to(device=cfg.device, dtype=torch.int64)
        a_depths = a_depths.to(device=cfg.device)
    else:
        a_states = torch.empty(
            (0, w_states.shape[1]), dtype=torch.int64, device=cfg.device
        )
        a_depths = torch.empty((0,), dtype=torch.float32, device=cfg.device)

    states = torch.cat([w_states, b_states, k_states, a_states], dim=0)
    depths = torch.cat([w_depths, b_depths, k_depths, a_depths], dim=0)

    if cfg.augment_symmetry and _SYM_STATE is not None:
        rotations, rot_inv = _SYM_STATE
        aug_gen = torch.Generator(device=cfg.device).manual_seed(data_seed ^ 0xA55A5A5A)
        from cayley.symmetry import apply_random_rotation

        states = apply_random_rotation(states, rotations, rot_inv, aug_gen)

    depths_f = depths.to(torch.float32)

    elem_loss = _LOSS_FNS_ELEMENTWISE[cfg.loss]
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = cfg.amp and cfg.device == "cuda"
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()
    )
    for batch_idx in _iterate_batches(states, depths_f, cfg.batch_size, batch_gen):
        batch_targets = depths_f[batch_idx]
        with autocast_ctx:
            pred = model(states[batch_idx])
            per_sample = elem_loss(pred, batch_targets)
            if cfg.curriculum:
                # Weight ∝ 1/k; normalize so mean weight = 1 for comparable LR scale.
                weights = 1.0 / batch_targets.clamp_min(1.0)
                weights = weights * (weights.numel() / weights.sum())
                loss = (weights * per_sample).mean()
            else:
                loss = per_sample.mean()
        optimizer.zero_grad(set_to_none=True)
        # bf16 grads are stable — no GradScaler needed.
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def model_state_size(model) -> int:
    """`state_size` of a model that may be wrapped by torch.compile."""
    return int(getattr(model, "_orig_mod", model).state_size)


def model_num_classes(model) -> int:
    """`num_classes` of a model that may be wrapped by torch.compile.

    On this puzzle that is 6, not state_size -- see 06_GOTCHAS.md #1. Reading it off the
    model rather than assuming keeps the Zobrist table the right shape either way.
    """
    return int(getattr(model, "_orig_mod", model).num_classes)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def train(
    model: ResMLPDistance,
    puzzle: PictureCube,
    cfg: TrainConfig,
    checkpoint_dir: str | Path,
    on_epoch_end: Callable[[EpochStats], None] | None = None,
    anchor_sampler: Callable[[int], tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> TrainResult:
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.to(cfg.device)

    # Build optimizer(s) + scheduler(s) via the helper (handles Adam/AdamW/Muon+AdamW split).
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        n_epochs=cfg.n_epochs,
        optimizer_name=cfg.optimizer,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=cfg.fused_optimizer and cfg.device == "cuda",
        muon_lr=cfg.muon_lr,
        muon_momentum=cfg.muon_momentum,
        muon_weight_decay=cfg.muon_weight_decay,
        muon_ns_steps=cfg.muon_ns_steps,
        muon_adjust_lr_fn=(cfg.muon_adjust_lr_fn or None),
    )

    # torch.compile wraps the forward in an optimized graph. First epoch pays compile cost;
    # subsequent epochs run faster. Skip if the user opted out or we're on CPU.
    # Note: Muon's Newton-Schulz loop sometimes trips compile; if you observe issues,
    # set compile_model=false.
    if cfg.compile_model and cfg.device == "cuda":
        model = torch.compile(model, dynamic=False)

    batch_gen = torch.Generator(device=cfg.device)
    batch_gen.manual_seed(cfg.seed)

    # Pre-load optional data supplements once (they don't change across epochs).
    bfs_table = None
    if cfg.bfs_table_path:
        from cayley.bfs_table import BfsTable

        bfs_table = BfsTable.load(cfg.bfs_table_path)
    kociemba_data = None
    if cfg.kociemba_walks_path:
        kociemba_data = load_kociemba_walks(cfg.kociemba_walks_path, device=cfg.device)

    # Preload rotational symmetry tensors if augmentation is enabled. Stored as a module
    # global so train_one_epoch can reach them without plumbing extra args through.
    global _SYM_STATE
    _SYM_STATE = None
    if cfg.augment_symmetry:
        from cayley.symmetry import compute_rotations, invert_permutation_batch

        R_np = compute_rotations(puzzle)
        R = torch.from_numpy(R_np).to(cfg.device)
        R_inv = invert_permutation_batch(R)
        _SYM_STATE = (R, R_inv)

    # Walk-label fix. Built once -- the sorted exact table does not change per epoch.
    label_fixer = None
    if cfg.label_fix:
        from cayley.label_fix import LabelFixer, make_ztab

        if cfg.label_fix_table_path:
            label_fixer = LabelFixer.from_anchor_file(
                cfg.label_fix_table_path,
                device=cfg.device,
                state_size=model_state_size(model),
                num_classes=model_num_classes(model),
            )
            print(
                f"[label-fix] ON: dedup + exact override against "
                f"{label_fixer.table_size:,} colourings from "
                f"{cfg.label_fix_table_path}",
                flush=True,
            )
        else:
            empty = torch.empty(0, dtype=torch.int64, device=cfg.device)
            label_fixer = LabelFixer(
                empty,
                empty,
                make_ztab(
                    model_state_size(model), model_num_classes(model), device=cfg.device
                ),
                do_dedup=True,
                do_override=False,
            )
            print(
                "[label-fix] ON: dedup only (no label_fix_table_path given)", flush=True
            )

    result = TrainResult(final_loss=float("inf"))

    for epoch in range(cfg.n_epochs):
        t0 = time.time()
        loss = train_one_epoch(
            model,
            optimizer,
            puzzle,
            cfg,
            data_seed=cfg.seed + epoch,
            batch_gen=batch_gen,
            bfs_table=bfs_table,
            kociemba_data=kociemba_data,
            anchor_sampler=anchor_sampler,
            label_fixer=label_fixer,
        )
        scheduler.step()
        stats = EpochStats(
            epoch=epoch,
            loss=loss,
            lr=float(scheduler.get_last_lr()[0]),
            elapsed_s=time.time() - t0,
        )
        result.epochs.append(stats)
        result.final_loss = loss
        if on_epoch_end is not None:
            on_epoch_end(stats)
        if (epoch + 1) % cfg.checkpoint_every_epochs == 0 or epoch == cfg.n_epochs - 1:
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            base = getattr(model, "_orig_mod", model)
            if hasattr(base, "get_model_config"):
                model_cfg = base.get_model_config()
            else:
                # Legacy path for older ResMLPDistance checkpoints without the hook.
                model_cfg = {
                    "state_size": base.state_size,
                    "num_classes": base.num_classes,
                    "hidden_dims": [
                        layer.out_features
                        for layer in base.input_stack
                        if isinstance(layer, torch.nn.Linear)
                    ],
                    "num_res_blocks": len(base.res_blocks),
                    "encoding": base.encoding,
                    "embed_dim": base.embed_dim,
                }
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": loss,
                    "train_config": cfg.__dict__,
                    "model_config": model_cfg,
                },
                path,
            )

    return result
