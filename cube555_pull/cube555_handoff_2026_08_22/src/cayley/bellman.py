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
from cayley.training import (
    _LOSS_FNS_ELEMENTWISE,
    EpochStats,
    TrainConfig,
    TrainResult,
    _NullCtx,
    _iterate_batches,
)


@dataclass
class BellmanConfig:
    """Additional knobs specific to the Bellman loop. Combine with `TrainConfig` for the
    underlying fast-recipe settings (batch, lr, amp, compile, etc.)."""

    warmstart_path: str = ""  # required — path to an existing .pt checkpoint
    target_update_every_epochs: int = 10  # how often to copy `model` → `target_model`
    target_net_chunk: int = 8192  # chunk size for target-net forward (memory)
    # safety: clip Bellman targets at the walk depth (upper bound) and 0 (lower bound).
    # these are virtually always active; knobs exposed for debugging.
    clip_upper: bool = True
    clip_lower: bool = True

    # BFS-d6 exact-target mixin (Option B). Build the dataset with
    # `megaminx/scripts/14_build_bfs_d6_dataset.py`. When `bfs_d6_path` is set
    # AND `bfs_d6_fraction > 0`, each training batch is split:
    #   (1 - bfs_d6_fraction) * batch_size  -> random-walk states, Bellman target
    #   bfs_d6_fraction * batch_size        -> BFS-d6 states, EXACT distance target
    # Anchors the model to ground-truth distances at the d<=6 boundary, removing
    # the bootstrap circularity. Default 0.0 = pure Bellman (current behavior).
    bfs_d6_path: str = ""
    bfs_d6_fraction: float = 0.0

    # Double Bellman (van Hasselt 2010 / Double DQN 2016). Maintain TWO target
    # snapshots, A and B, refreshed in alternation. The Bellman target uses one
    # net to SELECT the child (a* = argmin_a V_A(child)) and the OTHER to
    # ESTIMATE the value (target = 1 + V_B(child[a*])). Decorrelating selection
    # from estimation prevents the optimistic bias in `min_a V_target(...)` that
    # produces the m05 / m17 / m26 / m27 plateau ceiling. False = standard
    # single-target Bellman (current behavior).
    double_bellman: bool = False

    # Symmetry rotation augmentation. When `rotations_path` is set AND
    # `rotation_aug_prob > 0`, after computing each batch's Bellman target on
    # the unrotated state, replace each row of `bs` with a randomly-rotated
    # version with the given probability. The target is unchanged because
    # rotations preserve distance-to-solved (R*solved*R_inv = solved AND
    # R_inv*g_n*R is still a generator). This is a pure data-coverage
    # augmentation: the model sees more orbit-equivalent states with the
    # SAME label, learning rotational invariance implicitly.
    # Default 0.0 = no augmentation (current behavior).
    rotations_path: str = ""
    rotation_aug_prob: float = 0.0

    # Soft-Bellman temperature (DAY-1 #3, 2026-04-30). When > 0, replaces the
    # hard `min_a V_target(child)` with `-T * logsumexp(-V_target/T)`, which
    # is the soft minimum. As T -> 0 this recovers hard min exactly. As T
    # increases, the target reflects the value distribution across children
    # rather than only the best — a noise-reduction lever on the bootstrap
    # signal. Default 0.0 = hard min (current behavior; backward compat).
    # Only applied to the single-target path (not Double Bellman, where the
    # selection is by argmin and changing it would alter the algorithm).
    softmin_temperature: float = 0.0

    # Polyak/EMA target update (DAY-1 #4, 2026-04-30). When > 0, replaces
    # the discrete "refresh every N epochs" refresh with a smooth
    # `target := tau*model + (1-tau)*target` update applied per step
    # (per-batch). Standard DQN trick (Lillicrap 2016 DDPG). Default 0.0
    # disables Polyak and uses the discrete `target_update_every_epochs`
    # (current behavior). Typical values: 0.001 to 0.01.
    # Single-target Bellman only; Double Bellman keeps its alternating
    # discrete refresh schedule.
    target_polyak_tau: float = 0.0

    # L_upper one-sided overestimate penalty (m40, 2026-05-03). When > 0,
    # adds `lambda_upper * mean(max(0, pred - walk_depth)^2)` to the per-batch
    # loss. Walk depth is a provable upper bound on d(s); penalizing
    # predictions that exceed it is information-theoretically free signal.
    # `clip_upper=True` already clips the TARGET to walk_depth — this term
    # additionally pressures the PREDICTION to respect the bound. Applied
    # to the random-walk portion of each batch only (BFS-d6 mixin uses exact
    # distances and trivially satisfies pred <= walk_depth).
    # Default 0.0 = inactive (current behavior; backward compat).
    lambda_upper: float = 0.0

    # Solver-trace exact-target mixin (m43, 2026-05-04). Mirror of the BFS-d6
    # mixin but uses solver-trace pairs (states from successful prior solves +
    # their exact remaining-path-length labels). Build with
    # `megaminx/scripts/28_mine_solver_trace.py`. When `solver_trace_path` is set
    # AND `solver_trace_fraction > 0`, each training batch is split:
    #   (1 - bfs_d6_fraction - solver_trace_fraction) * batch_size  -> RW Bellman
    #   bfs_d6_fraction * batch_size                                 -> BFS-d6 exact
    #   solver_trace_fraction * batch_size                           -> solver-trace exact
    # Distinct from BFS-d6: covers full distance distribution (~1-100), anchoring
    # the hard-tail where BFS-d6 (d<=6 only) doesn't reach. Distinct from m37
    # (which used solver-trace as PRIMARY signal, OOD catastrophe) — here it's
    # a 25% mixin supplementing standard walk-depth Bellman, NOT a replacement.
    # Default 0.0 = pure Bellman (current behavior).
    solver_trace_path: str = ""
    solver_trace_fraction: float = 0.0

    # Frontier-replay (DAgger-style) state-distribution mixin (Idea 3 v0,
    # 2026-05-05). Build the dataset with `megaminx/scripts/42_log_frontier_states.py`
    # — beam-search frontier states from real solver runs, deduped, NO LABELS.
    # When `frontier_path` is set AND `frontier_fraction > 0`, each batch is
    # split:
    #   rw_fraction * batch_size            -> RW Bellman target (existing path)
    #   frontier_fraction * batch_size      -> frontier states, BELLMAN BOOTSTRAP
    #                                          target (NOT realized-suffix labels)
    # Critical distinction from m37/m43: those used realized-suffix-length as
    # exact labels on solver-trace states (OOD-catastrophic at 25% mixin and
    # 100% primary). Here labels are computed identically to RW Bellman targets
    # — `1 + min_a V_target(apply(s, a))` — so the lever is purely the
    # *state distribution*, not the label source.
    # Walk-depth upper-bound clipping is disabled for frontier states (we don't
    # know their walk-depth upper bound); a synthetic high walk_depth=200 is
    # used so clip_upper effectively no-ops, while clip_lower (target>=0)
    # still applies.
    # Default 0.0 = no frontier mixin (current behavior).
    frontier_path: str = ""
    frontier_fraction: float = 0.0
    # Synthetic walk_depth used for clip_upper on frontier states. 200 is well
    # above any realistic distance bound and effectively disables clip_upper
    # for the frontier portion of the batch.
    frontier_walk_depth_cap: float = 200.0

    # F2L (Group decomposition, 2026-05-10). Train V_F2L: predicts moves to
    # nearest F2L-correct state instead of V0. The Bellman boundary is now
    # "any state with these positions matching solved values" (set of states),
    # not just V0. Provides a heuristic for stage-1 of two-stage solving.
    # When set, must be a list of int positions (e.g., [0,1,2,...,104,105] for
    # the 35-position F2L set from `megaminx.decomposition`). Default empty list
    # = standard V0 boundary (current behavior).
    f2l_positions: tuple = ()

    # Admissibility-aware loss (m_adm, 2026-05-10). When > 0, adds
    # `lambda_pdb * mean(max(0, h_pdb(s) - V_pred(s))^2)` to the per-batch loss
    # using a caller-provided pdb_lookup_fn passed to train_bellman(). The PDB
    # heuristic h_pdb is a provable lower bound on d(s) — penalizing V_pred
    # below it is admissibility-improvement that addresses the
    # V_full(V0)≈0.91 undershoot bug observed in m_curr_v3.
    # Default 0.0 = no admissibility penalty (current behavior).
    lambda_pdb: float = 0.0

    # Anchor mixin (m_dd, 2026-05-10). Always include V0 (n_anchor_v0 copies)
    # and the 24 d=1 children (n_anchor_d1 copies of each, total 24*n_anchor_d1)
    # with EXACT targets (0 and 1 respectively) in every batch. Addresses the
    # bug where V(V0)≈2 instead of 0 — V0 is rarely sampled in the random BFS-d6
    # mixin, so its training signal is weak. Anchoring guarantees gradient signal
    # on these critical states every step. Default 0 = no anchoring.
    n_anchor_v0: int = 0
    n_anchor_d1: int = 0

    # Certified stagnation/near-solved anchors (own-research 1B, 2026-06-12).
    # Dataset built by `megaminx/scripts/89_harvest_stagnation_anchors.py`:
    # beam-frontier states the model claimed were near solved (V < ~6.5), each
    # CERTIFIED against the exact BFS-d6 table — label_type 0 = exact distance
    # (d <= 6), label_type 1 = proven lower bound d >= 7. Measured on AZ v4
    # (2026-06-12 harvest, 8 hard pids): V in [4,5) is 59 percent provably
    # d>=7, V in [5,6) is 99 percent — a systematic optimism band exactly where
    # the beam ranks endgame states. Two batch components, both default-off:
    #   n_anchor_stag        exact rows per batch -> main MSE (like BFS-d6 mixin
    #                        but BEAM-distributed, not shell-uniform)
    #   n_anchor_stag_lb     lower-bound rows per batch -> one-sided hinge
    #                        lambda_stag_lb * mean(relu(lb - V_pred)^2)
    #                        (only pushes V UP to the proven bound; never down —
    #                        an MSE-to-7 target would be WRONG for true d > 7)
    # Distinct from m27 (BFS-d6 mixin, shell-uniform states, REJECTED) by the
    # state distribution (beam-visited optimism band) and the LB label type.
    stag_anchor_path: str = ""
    n_anchor_stag: int = 0
    n_anchor_stag_lb: int = 0
    lambda_stag_lb: float = 1.0

    # Early stopping. When > 0, monitor smoothed training loss (mean over last
    # `early_stop_smooth_window` epochs) and stop if it doesn't improve by at
    # least `early_stop_min_delta` for `early_stop_patience` consecutive epochs.
    # Always saves `best.pt` separately. Default 0 = no early stopping.
    early_stop_patience: int = 0
    early_stop_min_delta: float = 1e-4
    early_stop_smooth_window: int = 5

    # ---- Representation-upgrade bundle (m_repr_v0, doc §3.1 + §4.6 + Rule 23) ----
    # Consistency loss (doc §4.6): L_sym = mean((V(s) - V(R s R^-1))^2). Sample one
    # rotation per row from `rotations_path`; forward V on the conjugated state; MSE
    # with V(s). Distinct from `rotation_aug_prob` (m31, REJECTED) which uses the
    # rotated state as a new sample with its own walk-depth target. Consistency loss
    # imposes invariance as a soft constraint — no signal dilution. Requires
    # `rotations_path` to be set. Default 0.0 = inactive.
    lambda_sym: float = 0.0

    # Saturation soft penalty (Rule 23 lesson): L_sat = mean(relu(V(s) - sat_ceiling)^2).
    # Pushes V predictions DOWN when they exceed the puzzle's known diameter. Working
    # megaminx baselines saturate at V@d=80 ≈ 29; sat_ceiling = 30 gives 1.0 of
    # headroom. Direct defense against the state_inv / GT V failure mode where extra
    # capacity drifts V past true diameter. Default 0.0 = inactive.
    lambda_sat: float = 0.0
    sat_ceiling: float = 30.0

    # Child-rank listwise CE (doc §4.2): trains relative ordering directly. For each
    # RW state s, V on children is computed via LIVE model (with grad), target
    # distribution = softmax(-target_child_V / rank_temperature), loss = listwise CE
    # against -log_softmax(live_child_V). Cost: extra n_gen × forward on children per
    # batch (~2× the existing Bellman child-forward cost). Default 0.0 = inactive.
    lambda_rank: float = 0.0
    rank_temperature: float = 1.0

    # Per-depth diagnostic logging: per epoch, log mean(pred - target) bucketed by
    # walk_depth (RW portion only) into bins [0,10), [10,20), [20,30), [30,40),
    # [40,60), [60,80), [80,∞). Tracks V drift at high depths (Rule 23 canary).
    # Default False.
    per_depth_diagnostic: bool = False


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
    softmin_temperature: float = 0.0,
    f2l_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute `y = clip(1 + reduce_a target(apply(s, a)), 0, walk_depth)` for a batch.

    `reduce_a` is hard min by default. With `softmin_temperature > 0`, uses the
    soft min `-T * logsumexp(-V/T)` instead, which approaches the hard min as
    T -> 0 and the mean as T -> inf.

    A child that equals the solved state gets value 0 (not whatever the model predicts)
    — this is the only boundary condition and we want it exact.

    If `f2l_positions` is provided (1D long tensor of sticker indices), the
    boundary is generalized to "any state where these positions match
    solved_state" — used for V_F2L training (group-theoretic decomposition).
    """
    B, S = states.shape
    children = _apply_all_generators(states, generators)  # (B, n_gen, S)
    n_gen = children.shape[1]
    children_flat = children.reshape(B * n_gen, S)

    if f2l_positions is not None and f2l_positions.numel() > 0:
        # F2L boundary: any state matching solved on f2l_positions
        is_solved = (
            children_flat[:, f2l_positions] == solved_state[f2l_positions]
        ).all(dim=1)
    else:
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
    if softmin_temperature > 0:
        # soft min: -T * logsumexp(-V/T). Equivalent to hard min as T -> 0.
        T = softmin_temperature
        reduced = -T * torch.logsumexp(-child_values / T, dim=1)
    else:
        reduced = child_values.min(dim=1).values  # (B,)
    target = 1.0 + reduced

    if clip_upper:
        target = torch.minimum(target, walk_depths)
    if clip_lower:
        target = torch.clamp(target, min=0.0)
    return target


@torch.no_grad()
def _polyak_update(
    target_model: torch.nn.Module, source_model: torch.nn.Module, tau: float
) -> None:
    """Smooth target update: target_params := tau * source_params + (1 - tau) * target_params.

    Strips `_orig_mod.` prefix from compiled source models so unwrapped target
    parameters can be matched. Updates parameters AND buffers (e.g., embedding
    norm running stats if any).
    """
    src_sd = source_model.state_dict()
    if any(k.startswith("_orig_mod.") for k in src_sd):
        src_sd = {k.removeprefix("_orig_mod."): v for k, v in src_sd.items()}
    tgt_sd = target_model.state_dict()
    for k, v in src_sd.items():
        if k in tgt_sd and tgt_sd[k].shape == v.shape and tgt_sd[k].dtype == v.dtype:
            tgt_sd[k].mul_(1.0 - tau).add_(v.detach(), alpha=tau)
        # Skip mismatched keys silently (e.g. compiled-only buffers).


@torch.no_grad()
def _double_bellman_targets(
    target_a: torch.nn.Module,
    target_b: torch.nn.Module,
    states: torch.Tensor,
    walk_depths: torch.Tensor,
    generators: torch.Tensor,
    solved_state: torch.Tensor,
    chunk_size: int,
    clip_upper: bool,
    clip_lower: bool,
) -> torch.Tensor:
    """Double Bellman target: A selects the best child, B estimates its value.

    `a* = argmin_a V_A(apply(s, a))`
    `y  = clip(1 + V_B(apply(s, a*)), 0, walk_depth)`

    Decorrelates child-selection (V_A) from value-estimation (V_B). A child
    accidentally underestimated by V_A is no longer guaranteed to also be
    underestimated by V_B, breaking the optimistic-min bias propagation cycle.
    """
    B, S = states.shape
    children = _apply_all_generators(states, generators)  # (B, n_gen, S)
    n_gen = children.shape[1]
    children_flat = children.reshape(B * n_gen, S)
    is_solved = (children_flat == solved_state).all(dim=1)

    # --- Step 1: V_A picks the best child (one fwd per child).
    child_v_a = torch.empty(B * n_gen, dtype=torch.float32, device=states.device)
    target_a.eval()
    for i in range(0, B * n_gen, chunk_size):
        vals = target_a(children_flat[i : i + chunk_size]).flatten().to(torch.float32)
        child_v_a[i : i + chunk_size] = vals
    child_v_a = torch.where(is_solved, torch.zeros_like(child_v_a), child_v_a)
    child_v_a = child_v_a.view(B, n_gen)
    a_star = child_v_a.argmin(dim=1)  # (B,)

    # --- Step 2: V_B estimates the value of the chosen child.
    chosen_child = children[torch.arange(B, device=states.device), a_star]  # (B, S)
    chosen_is_solved = (chosen_child == solved_state).all(dim=1)
    child_v_b = torch.empty(B, dtype=torch.float32, device=states.device)
    target_b.eval()
    for i in range(0, B, chunk_size):
        vals = target_b(chosen_child[i : i + chunk_size]).flatten().to(torch.float32)
        child_v_b[i : i + chunk_size] = vals
    child_v_b = torch.where(chosen_is_solved, torch.zeros_like(child_v_b), child_v_b)

    target = 1.0 + child_v_b
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
    pdb_lookup_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
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

    # Double Bellman: a second target net, refreshed in alternation with the first.
    target_model_b = None
    target_swap = 0  # 0 = next refresh updates A, 1 = next refresh updates B
    if bcfg.double_bellman:
        target_model_b = copy.deepcopy(model).eval()
        for p in target_model_b.parameters():
            p.requires_grad = False
        print("[bellman] Double Bellman active: two target nets, alternating refresh")

    # Optionally compile the trainable model (not the target — avoids double compile cost).
    if cfg.compile_model and cfg.device == "cuda":
        model = torch.compile(model, dynamic=False)

    optim_cls = torch.optim.AdamW if cfg.weight_decay > 0 else torch.optim.Adam
    optim_kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.fused_optimizer and cfg.device == "cuda":
        optim_kwargs["fused"] = True
    optimizer = optim_cls(model.parameters(), **optim_kwargs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_epochs
    )

    # Precompute generators tensor + solved state tensor for the Bellman target function.
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(
        cfg.device
    )  # (n_gen, state_size)
    solved_state = torch.tensor(
        puzzle.solved_state, dtype=torch.int64, device=cfg.device
    )

    # Anchor states: V0 (solved) and 24 d=1 children. Always-included exact-label
    # mixin to address V(V0)≈2 bug.
    state_dtype = torch.int8  # generate_walks_torch returns int8
    anchor_v0 = solved_state.to(state_dtype).unsqueeze(0)  # (1, S)
    anchor_d1 = (
        _apply_all_generators(anchor_v0, generators).squeeze(0).to(state_dtype)
    )  # (24, S)
    if bcfg.n_anchor_v0 > 0 or bcfg.n_anchor_d1 > 0:
        print(
            f"[bellman] anchor mixin: V0={bcfg.n_anchor_v0}, "
            f"d=1 children=24x{bcfg.n_anchor_d1}",
            flush=True,
        )

    # Certified stagnation anchors (own-research 1B): exact pool + LB pool.
    stag_exact_states = stag_exact_targets = None
    stag_lb_states = stag_lb_values = None
    if bcfg.stag_anchor_path and (bcfg.n_anchor_stag > 0 or bcfg.n_anchor_stag_lb > 0):
        _stag = torch.load(
            bcfg.stag_anchor_path, map_location="cpu", weights_only=False
        )
        _st = _stag["states"].to(state_dtype)
        _lt = _stag["label_type"]
        _lv = _stag["label_value"].to(torch.float32)
        _ex_m = _lt == 0
        _lb_m = _lt == 1
        if bcfg.n_anchor_stag > 0 and int(_ex_m.sum()) > 0:
            stag_exact_states = _st[_ex_m].to(cfg.device)
            stag_exact_targets = _lv[_ex_m].to(cfg.device)
        if bcfg.n_anchor_stag_lb > 0 and int(_lb_m.sum()) > 0:
            stag_lb_states = _st[_lb_m].to(cfg.device)
            stag_lb_values = _lv[_lb_m].to(cfg.device)
        print(
            f"[bellman] stagnation anchors: exact pool="
            f"{0 if stag_exact_states is None else stag_exact_states.size(0)} "
            f"(use {bcfg.n_anchor_stag}/batch), lb pool="
            f"{0 if stag_lb_states is None else stag_lb_states.size(0)} "
            f"(use {bcfg.n_anchor_stag_lb}/batch, lambda={bcfg.lambda_stag_lb})",
            flush=True,
        )

    # F2L positions: if set, switches Bellman boundary to "any state matching solved
    # on these positions". Used for V_F2L training (group decomposition).
    f2l_positions_dev = None
    if bcfg.f2l_positions:
        f2l_positions_dev = torch.tensor(
            list(bcfg.f2l_positions), dtype=torch.long, device=cfg.device
        )
        print(
            f"[bellman] F2L mode: boundary = {len(bcfg.f2l_positions)} matching positions",
            flush=True,
        )

    elem_loss = _LOSS_FNS_ELEMENTWISE[cfg.loss]
    batch_gen = torch.Generator(device=cfg.device)
    batch_gen.manual_seed(cfg.seed)

    use_amp = cfg.amp and cfg.device == "cuda"
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else _NullCtx()
    )

    # ---- Symmetry rotations: shared by rotation augmentation AND consistency loss ----
    rotations_dev = None
    rotations_inv_dev = None
    rot_gen = torch.Generator(device=cfg.device)
    rot_gen.manual_seed(cfg.seed + 54321)  # offset from batch_gen seed
    needs_rotations = (bcfg.rotations_path and bcfg.rotation_aug_prob > 0) or (
        bcfg.rotations_path and bcfg.lambda_sym > 0
    )
    if needs_rotations:
        import numpy as np

        rot_arr = np.load(bcfg.rotations_path)
        rotations_dev = torch.from_numpy(rot_arr).to(cfg.device)  # (N, S) int8
        # Precompute inverses: inv[i] = argsort(rot_arr[i])
        rot_inv_arr = np.argsort(rot_arr, axis=1).astype(np.int64)
        rotations_inv_dev = torch.from_numpy(rot_inv_arr).to(cfg.device)
        if bcfg.rotation_aug_prob > 0:
            print(
                f"[bellman] rotation augmentation: {rot_arr.shape[0]} rotations, "
                f"prob={bcfg.rotation_aug_prob:.2f}",
                flush=True,
            )
        if bcfg.lambda_sym > 0:
            print(
                f"[bellman] symmetry consistency loss: {rot_arr.shape[0]} rotations, "
                f"lambda_sym={bcfg.lambda_sym:.3f}",
                flush=True,
            )

    # ---- Option B: BFS-d6 exact-target mixin ----
    # Pre-load the BFS-d6 dataset once if requested. Each epoch we sample fresh
    # subsets from it for the per-batch mixin.
    bfs6_states_cpu = None
    bfs6_dists_cpu = None
    bfs6_count_per_batch = 0
    bfs6_gen_cpu = torch.Generator(device="cpu")
    bfs6_gen_cpu.manual_seed(cfg.seed + 12345)  # offset from batch_gen seed
    if bcfg.bfs_d6_path and bcfg.bfs_d6_fraction > 0:
        print(f"[bellman] loading BFS-d6 dataset: {bcfg.bfs_d6_path}", flush=True)
        d = torch.load(bcfg.bfs_d6_path, map_location="cpu", weights_only=False)
        bfs6_states_cpu = d["states"]  # (N, 120) int8
        bfs6_dists_cpu = d["distances"]  # (N,) int8
        bfs6_count_per_batch = max(1, int(round(cfg.batch_size * bcfg.bfs_d6_fraction)))
        print(
            f"[bellman]   {bfs6_states_cpu.size(0):,} BFS-d6 states; "
            f"per-batch mixin = {bfs6_count_per_batch}/{cfg.batch_size} "
            f"({bcfg.bfs_d6_fraction * 100:.0f}%)",
            flush=True,
        )

    # ---- Solver-trace exact-target mixin (m43) ----
    # Same pattern as BFS-d6 mixin but with solver-trace pairs (full d distribution).
    st_states_cpu = None
    st_dists_cpu = None
    st_count_per_batch = 0
    st_gen_cpu = torch.Generator(device="cpu")
    st_gen_cpu.manual_seed(cfg.seed + 67890)  # offset from other generators
    if bcfg.solver_trace_path and bcfg.solver_trace_fraction > 0:
        print(
            f"[bellman] loading solver-trace dataset: {bcfg.solver_trace_path}",
            flush=True,
        )
        d = torch.load(bcfg.solver_trace_path, map_location="cpu", weights_only=False)
        st_states_cpu = d["states"]  # (N, 120) int8
        st_dists_cpu = d["distances"]  # (N,) int8
        st_count_per_batch = max(
            1, int(round(cfg.batch_size * bcfg.solver_trace_fraction))
        )
        print(
            f"[bellman]   {st_states_cpu.size(0):,} solver-trace pairs; "
            f"per-batch mixin = {st_count_per_batch}/{cfg.batch_size} "
            f"({bcfg.solver_trace_fraction * 100:.0f}%); "
            f"d range [{int(st_dists_cpu.min())}, {int(st_dists_cpu.max())}]",
            flush=True,
        )

    # ---- Frontier-replay state-distribution mixin (Idea 3 v0) ----
    # States only, no labels. Bellman-bootstrap target computed at training
    # time using the frozen target net, identical to the RW path. The lever is
    # the visited-frontier state distribution; labels are unchanged.
    fr_states_cpu = None
    fr_count_per_batch = 0
    fr_gen_cpu = torch.Generator(device="cpu")
    fr_gen_cpu.manual_seed(cfg.seed + 24680)
    if bcfg.frontier_path and bcfg.frontier_fraction > 0:
        print(
            f"[bellman] loading frontier-state dataset: {bcfg.frontier_path}",
            flush=True,
        )
        d = torch.load(bcfg.frontier_path, map_location="cpu", weights_only=False)
        fr_states_cpu = d["states"]  # (N, 120) int8
        fr_count_per_batch = max(1, int(round(cfg.batch_size * bcfg.frontier_fraction)))
        print(
            f"[bellman]   {fr_states_cpu.size(0):,} frontier states; "
            f"per-batch mixin = {fr_count_per_batch}/{cfg.batch_size} "
            f"({bcfg.frontier_fraction * 100:.0f}%); "
            f"target = Bellman bootstrap (NOT realized-suffix); "
            f"clip_upper synthetic cap = {bcfg.frontier_walk_depth_cap}",
            flush=True,
        )
        if bcfg.double_bellman:
            print(
                "[bellman]   WARNING: frontier mixin + double_bellman uses single-target Bellman "
                "for frontier states (target_a only); RW path keeps Double Bellman.",
                flush=True,
            )

    # ---- Walk-label fix (02_DATA.md) ----
    # Reuses the BFS anchor file already loaded for the mixin -- same exact labels, so
    # there is nothing extra to build or keep in sync. Driven by TrainConfig.label_fix so
    # the pretrain and Bellman legs of a run cannot disagree about the objective.
    label_fixer = None
    if cfg.label_fix:
        from cayley.label_fix import LabelFixer, make_ztab
        from cayley.training import model_num_classes, model_state_size

        table_path = cfg.label_fix_table_path or bcfg.bfs_d6_path
        if table_path:
            label_fixer = LabelFixer.from_anchor_file(
                table_path,
                device=cfg.device,
                state_size=model_state_size(model),
                num_classes=model_num_classes(model),
            )
            print(
                f"[label-fix] ON: dedup + exact override against "
                f"{label_fixer.table_size:,} colourings from {table_path}",
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
            print("[label-fix] ON: dedup only (no table path available)", flush=True)

    result = TrainResult(final_loss=float("inf"))

    # Early stopping state
    best_smoothed_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    recent_losses: list[float] = []

    for epoch in range(cfg.n_epochs):
        t0 = time.time()
        n_walks = max(1, cfg.samples_per_epoch // cfg.k_max)
        states, depths = generate_walks_torch(
            puzzle,
            n_walks=n_walks,
            k_max=cfg.k_max,
            seed=cfg.seed + epoch,
            device=cfg.device,
            n_back=cfg.n_back,
        )
        if label_fixer is not None:
            # Round to the RW-portion batch size, not cfg.batch_size: _iterate_batches
            # below slices this stream by `rw_per_batch` so the COMBINED batch (RW +
            # mixins) lands on cfg.batch_size. Same value, recomputed here because the
            # fix has to run before the stream is consumed. See LabelFixer.apply for why
            # a varying row count is a 27x torch.compile regression.
            _rw_per_batch = (
                cfg.batch_size
                - bfs6_count_per_batch
                - st_count_per_batch
                - fr_count_per_batch
            )
            states, depths, _fix_stats = label_fixer.apply(
                states, depths, round_to=_rw_per_batch if cfg.compile_model else None
            )
            print(f"  {_fix_stats.render()}", flush=True)
        depths_f = depths.to(torch.float32)

        # Pre-sample BFS-d6 states for this epoch (random per-epoch shuffle).
        if bfs6_states_cpu is not None:
            rw_per_batch_pre = max(
                1, cfg.batch_size - bfs6_count_per_batch - st_count_per_batch
            )
            n_per_epoch = bfs6_count_per_batch * (
                states.size(0) // rw_per_batch_pre + 1
            )
            n_per_epoch = min(n_per_epoch, bfs6_states_cpu.size(0))
            ep_perm = torch.randperm(bfs6_states_cpu.size(0), generator=bfs6_gen_cpu)[
                :n_per_epoch
            ]
            bfs6_ep_states = bfs6_states_cpu[ep_perm].to(cfg.device)
            bfs6_ep_dists = bfs6_dists_cpu[ep_perm].to(cfg.device).to(torch.float32)
        else:
            bfs6_ep_states = None
            bfs6_ep_dists = None

        # Pre-sample solver-trace states for this epoch.
        if st_states_cpu is not None:
            rw_per_batch_pre = max(
                1,
                cfg.batch_size
                - bfs6_count_per_batch
                - st_count_per_batch
                - fr_count_per_batch,
            )
            n_per_epoch_st = st_count_per_batch * (
                states.size(0) // rw_per_batch_pre + 1
            )
            n_per_epoch_st = min(n_per_epoch_st, st_states_cpu.size(0))
            ep_perm_st = torch.randperm(st_states_cpu.size(0), generator=st_gen_cpu)[
                :n_per_epoch_st
            ]
            st_ep_states = st_states_cpu[ep_perm_st].to(cfg.device)
            st_ep_dists = st_dists_cpu[ep_perm_st].to(cfg.device).to(torch.float32)
        else:
            st_ep_states = None
            st_ep_dists = None

        # Pre-sample frontier states for this epoch (states only, no labels).
        if fr_states_cpu is not None:
            rw_per_batch_pre = max(
                1,
                cfg.batch_size
                - bfs6_count_per_batch
                - st_count_per_batch
                - fr_count_per_batch,
            )
            n_per_epoch_fr = fr_count_per_batch * (
                states.size(0) // rw_per_batch_pre + 1
            )
            n_per_epoch_fr = min(n_per_epoch_fr, fr_states_cpu.size(0))
            ep_perm_fr = torch.randperm(fr_states_cpu.size(0), generator=fr_gen_cpu)[
                :n_per_epoch_fr
            ]
            fr_ep_states = fr_states_cpu[ep_perm_fr].to(cfg.device)
        else:
            fr_ep_states = None

        # RW portion is what _iterate_batches sees. Reduce its target batch by
        # bfs6 + st + fr counts so the COMBINED batch is exactly cfg.batch_size.
        rw_per_batch = (
            cfg.batch_size
            - bfs6_count_per_batch
            - st_count_per_batch
            - fr_count_per_batch
        )

        model.train()
        total_loss = 0.0
        n_batches = 0
        bfs6_cursor = 0
        st_cursor = 0
        fr_cursor = 0
        # Per-depth diagnostic accumulators (RW portion only).
        # Buckets: [0,10), [10,20), [20,30), [30,40), [40,60), [60,80), [80,∞).
        PD_BINS = (10.0, 20.0, 30.0, 40.0, 60.0, 80.0, float("inf"))
        PD_LABELS = (
            "[0,10)",
            "[10,20)",
            "[20,30)",
            "[30,40)",
            "[40,60)",
            "[60,80)",
            "[80+)",
        )
        pd_sum_diff = [0.0] * len(PD_BINS)
        pd_sum_pred = [0.0] * len(PD_BINS)
        pd_sum_pred_sq = [0.0] * len(
            PD_BINS
        )  # for per-bucket std (the binding beam canary)
        pd_count = [0] * len(PD_BINS)
        for batch_idx in _iterate_batches(states, depths_f, rw_per_batch, batch_gen):
            bs_rw = states[batch_idx]
            bd_rw = depths_f[batch_idx]
            if bcfg.double_bellman:
                target_rw = _double_bellman_targets(
                    target_model,
                    target_model_b,
                    bs_rw,
                    bd_rw,
                    generators,
                    solved_state,
                    chunk_size=bcfg.target_net_chunk,
                    clip_upper=bcfg.clip_upper,
                    clip_lower=bcfg.clip_lower,
                )
            else:
                target_rw = _bellman_targets(
                    target_model,
                    bs_rw,
                    bd_rw,
                    generators,
                    solved_state,
                    chunk_size=bcfg.target_net_chunk,
                    clip_upper=bcfg.clip_upper,
                    clip_lower=bcfg.clip_lower,
                    softmin_temperature=bcfg.softmin_temperature,
                    f2l_positions=f2l_positions_dev,
                )

            # After targets are computed on the UNROTATED state, optionally rotate
            # bs_rw with prob rotation_aug_prob. Target unchanged (distance-preserving).
            # Per-row independent rotation index; vectorized via gather.
            if rotations_dev is not None:
                B_rw, S_ = bs_rw.shape
                rot_idx = torch.randint(
                    0,
                    rotations_dev.size(0),
                    (B_rw,),
                    generator=rot_gen,
                    device=cfg.device,
                )
                R = rotations_dev[rot_idx].to(torch.long)  # (B_rw, S)
                R_inv = rotations_inv_dev[rot_idx]  # (B_rw, S) int64
                bs_long = bs_rw.to(torch.long)
                step1 = torch.gather(bs_long, 1, R_inv)
                rotated = torch.gather(R, 1, step1)
                aug_mask = (
                    torch.rand(B_rw, generator=rot_gen, device=cfg.device)
                    < bcfg.rotation_aug_prob
                ).unsqueeze(1)
                bs_rw = torch.where(aug_mask, rotated.to(bs_rw.dtype), bs_rw)

            bs_parts = [bs_rw]
            target_parts = [target_rw]
            # Anchor mixin: always include V0 + d=1 children with exact labels.
            if bcfg.n_anchor_v0 > 0:
                bs_parts.append(anchor_v0.expand(bcfg.n_anchor_v0, -1))
                target_parts.append(
                    torch.zeros(
                        bcfg.n_anchor_v0, dtype=torch.float32, device=cfg.device
                    )
                )
            if bcfg.n_anchor_d1 > 0:
                # anchor_d1: (24, S). Repeat n_anchor_d1 times along dim 0 → (24*n_anchor_d1, S)
                bs_parts.append(anchor_d1.repeat(bcfg.n_anchor_d1, 1))
                target_parts.append(
                    torch.ones(
                        24 * bcfg.n_anchor_d1, dtype=torch.float32, device=cfg.device
                    )
                )
            # Stagnation-anchor EXACT rows: beam-distributed d<=6 states with
            # exact distances — ride the main MSE like the BFS-d6 mixin.
            if stag_exact_states is not None:
                ridx = torch.randint(
                    0,
                    stag_exact_states.size(0),
                    (bcfg.n_anchor_stag,),
                    generator=batch_gen,
                    device=cfg.device,
                )
                bs_parts.append(stag_exact_states[ridx])
                target_parts.append(stag_exact_targets[ridx])
            if (
                bfs6_ep_states is not None
                and bfs6_cursor + bfs6_count_per_batch <= bfs6_ep_states.size(0)
            ):
                bs_parts.append(
                    bfs6_ep_states[bfs6_cursor : bfs6_cursor + bfs6_count_per_batch]
                )
                target_parts.append(
                    bfs6_ep_dists[bfs6_cursor : bfs6_cursor + bfs6_count_per_batch]
                )
                bfs6_cursor += bfs6_count_per_batch
            if (
                st_ep_states is not None
                and st_cursor + st_count_per_batch <= st_ep_states.size(0)
            ):
                bs_parts.append(
                    st_ep_states[st_cursor : st_cursor + st_count_per_batch]
                )
                target_parts.append(
                    st_ep_dists[st_cursor : st_cursor + st_count_per_batch]
                )
                st_cursor += st_count_per_batch
            # Frontier-replay slice: states with Bellman-bootstrap targets.
            # Use synthetic walk_depths (frontier_walk_depth_cap) so clip_upper
            # effectively no-ops; clip_lower still applies. Use single-target
            # Bellman even when double_bellman is enabled, since frontier states
            # don't have a stable selection/estimation duality (they're new
            # distribution states; the selection-vs-estimation trick assumes
            # the same distribution as the bootstrap propagates).
            if (
                fr_ep_states is not None
                and fr_cursor + fr_count_per_batch <= fr_ep_states.size(0)
            ):
                bs_fr = fr_ep_states[fr_cursor : fr_cursor + fr_count_per_batch]
                fr_cursor += fr_count_per_batch
                bd_fr = torch.full(
                    (bs_fr.size(0),),
                    float(bcfg.frontier_walk_depth_cap),
                    dtype=torch.float32,
                    device=cfg.device,
                )
                target_fr = _bellman_targets(
                    target_model,
                    bs_fr,
                    bd_fr,
                    generators,
                    solved_state,
                    chunk_size=bcfg.target_net_chunk,
                    clip_upper=bcfg.clip_upper,
                    clip_lower=bcfg.clip_lower,
                    softmin_temperature=bcfg.softmin_temperature,
                    f2l_positions=f2l_positions_dev,
                )
                bs_parts.append(bs_fr)
                target_parts.append(target_fr)
            if len(bs_parts) > 1:
                bs = torch.cat(bs_parts, dim=0)
                target = torch.cat(target_parts, dim=0)
            else:
                bs = bs_rw
                target = target_rw

            with autocast_ctx:
                pred = model(bs)
                loss = elem_loss(pred, target).mean()
                # L_upper one-sided overestimate penalty (m40). Penalize the RW
                # portion of pred for exceeding walk_depth (a provable upper
                # bound on d(s)). BFS-d6 portion uses exact targets so it is
                # trivially below — skip it.
                if bcfg.lambda_upper > 0:
                    pred_rw = pred[: bs_rw.size(0)]
                    overshoot = (pred_rw.float().flatten() - bd_rw).clamp(min=0.0)
                    upper_loss = (overshoot**2).mean()
                    loss = loss + bcfg.lambda_upper * upper_loss
                # Stagnation-anchor LOWER-BOUND hinge: states with certified
                # d >= lb (proved by BFS-d6 exclusion). One-sided: only pushes
                # V_pred UP to the bound; an MSE target at lb would be wrong
                # for states whose true d exceeds the bound.
                if stag_lb_states is not None:
                    ridx_lb = torch.randint(
                        0,
                        stag_lb_states.size(0),
                        (bcfg.n_anchor_stag_lb,),
                        generator=batch_gen,
                        device=cfg.device,
                    )
                    pred_lb = model(stag_lb_states[ridx_lb]).float().flatten()
                    undershoot_lb = (stag_lb_values[ridx_lb] - pred_lb).clamp(min=0.0)
                    loss = loss + bcfg.lambda_stag_lb * (undershoot_lb**2).mean()
                # Admissibility-aware penalty: pred should not undershoot
                # the admissible PDB heuristic. Applied to the FULL batch
                # (RW + mixins) since PDB is a global lower bound.
                if bcfg.lambda_pdb > 0 and pdb_lookup_fn is not None:
                    with torch.no_grad():
                        h_pdb = pdb_lookup_fn(bs).float()  # (B,)
                    pred_flat = pred.float().flatten()
                    undershoot = (h_pdb - pred_flat).clamp(min=0.0)
                    pdb_loss = (undershoot**2).mean()
                    loss = loss + bcfg.lambda_pdb * pdb_loss

                # ---- Representation-upgrade bundle (m_repr_v0) ----
                # Saturation soft penalty: pull V_pred down when exceeding sat_ceiling.
                # Direct defense against Rule 23 / state_inv high-depth drift.
                # Applied to all preds in the batch (global ceiling).
                if bcfg.lambda_sat > 0:
                    overshoot = (pred.float().flatten() - bcfg.sat_ceiling).clamp(
                        min=0.0
                    )
                    sat_loss_v = (overshoot**2).mean()
                    loss = loss + bcfg.lambda_sat * sat_loss_v

                # Symmetry consistency loss: V(s) ≈ V(R s R^-1). Applied to RW
                # portion only (anchor / bfs6 / st / frontier have exact targets).
                if bcfg.lambda_sym > 0 and rotations_dev is not None:
                    B_rw_now = bs_rw.size(0)
                    rot_idx_sym = torch.randint(
                        0,
                        rotations_dev.size(0),
                        (B_rw_now,),
                        generator=rot_gen,
                        device=cfg.device,
                    )
                    R_dyn = rotations_dev[rot_idx_sym].to(torch.long)  # (B_rw, S)
                    R_inv_dyn = rotations_inv_dev[rot_idx_sym]  # (B_rw, S) int64
                    bs_long_rw = bs_rw.to(torch.long)
                    step1 = torch.gather(bs_long_rw, 1, R_inv_dyn)
                    bs_rotated = torch.gather(R_dyn, 1, step1).to(bs_rw.dtype)
                    pred_rot = model(bs_rotated).float()
                    pred_rw_for_sym = pred[: bs_rw.size(0)].float()
                    sym_loss_v = F.mse_loss(pred_rw_for_sym, pred_rot)
                    loss = loss + bcfg.lambda_sym * sym_loss_v

                # Child-rank listwise CE on RW portion.
                if bcfg.lambda_rank > 0:
                    B_rw_now = bs_rw.size(0)
                    _, S_local = bs_rw.shape
                    n_gen = generators.shape[0]
                    children = _apply_all_generators(
                        bs_rw, generators
                    )  # (B_rw, n_gen, S)
                    children_flat = children.view(-1, S_local)
                    is_solved_child = (children_flat == solved_state).all(dim=-1)
                    with torch.no_grad():
                        target_child_v = torch.empty(
                            B_rw_now * n_gen,
                            dtype=torch.float32,
                            device=cfg.device,
                        )
                        for i in range(0, B_rw_now * n_gen, bcfg.target_net_chunk):
                            target_child_v[i : i + bcfg.target_net_chunk] = (
                                target_model(
                                    children_flat[i : i + bcfg.target_net_chunk]
                                )
                                .flatten()
                                .to(torch.float32)
                            )
                    target_child_v = torch.where(
                        is_solved_child,
                        torch.zeros_like(target_child_v),
                        target_child_v,
                    ).view(B_rw_now, n_gen)
                    live_child_v = model(children_flat).flatten().to(torch.float32)
                    live_child_v = torch.where(
                        is_solved_child,
                        torch.zeros_like(live_child_v),
                        live_child_v,
                    ).view(B_rw_now, n_gen)
                    target_dist = F.softmax(
                        -target_child_v / bcfg.rank_temperature, dim=1
                    )
                    log_q = F.log_softmax(-live_child_v, dim=1)
                    rank_loss_v = -(target_dist * log_q).sum(dim=1).mean()
                    loss = loss + bcfg.lambda_rank * rank_loss_v
            # Per-depth diagnostic: accumulate pred-target diffs and absolute preds
            # on RW portion, bucketed by walk depth.
            if bcfg.per_depth_diagnostic:
                with torch.no_grad():
                    pred_rw_diag = pred[: bs_rw.size(0)].float().flatten()
                    diff_rw = pred_rw_diag - target_rw.float()
                    bd_cpu = bd_rw.float().detach().cpu()
                    diff_cpu = diff_rw.detach().cpu()
                    pred_cpu = pred_rw_diag.detach().cpu()
                    lo = 0.0
                    for b_idx, hi in enumerate(PD_BINS):
                        mask = (bd_cpu >= lo) & (bd_cpu < hi)
                        n_in = int(mask.sum().item())
                        if n_in > 0:
                            pd_sum_diff[b_idx] += float(diff_cpu[mask].sum().item())
                            pd_sum_pred[b_idx] += float(pred_cpu[mask].sum().item())
                            pd_sum_pred_sq[b_idx] += float(
                                (pred_cpu[mask] ** 2).sum().item()
                            )
                            pd_count[b_idx] += n_in
                        lo = hi
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Polyak/EMA target update (DAY-1 #4). Per-step smooth update to
            # the target net. Only active in single-target Bellman; Double
            # Bellman uses the alternating discrete refresh.
            if bcfg.target_polyak_tau > 0 and not bcfg.double_bellman:
                _polyak_update(target_model, model, bcfg.target_polyak_tau)
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        stats = EpochStats(
            epoch=epoch,
            loss=avg_loss,
            lr=float(scheduler.get_last_lr()[0]),
            elapsed_s=time.time() - t0,
        )
        result.epochs.append(stats)
        result.final_loss = avg_loss
        if on_epoch_end is not None:
            on_epoch_end(stats)

        # Per-depth diagnostic readout (Rule 23 canary).
        if bcfg.per_depth_diagnostic:
            parts = []
            for b_idx, label in enumerate(PD_LABELS):
                n = pd_count[b_idx]
                if n > 0:
                    mean_diff = pd_sum_diff[b_idx] / n
                    mean_pred = pd_sum_pred[b_idx] / n
                    var = max(0.0, pd_sum_pred_sq[b_idx] / n - mean_pred * mean_pred)
                    std_pred = var**0.5
                    parts.append(
                        f"{label} V={mean_pred:+.2f} s={std_pred:.2f} d={mean_diff:+.3f} n={n}"
                    )
                else:
                    parts.append(f"{label} n=0")
            print(f"[bellman]   per-depth: {'  '.join(parts)}", flush=True)

        # Refresh target net. Use the compiled module's original state dict.
        # In Double Bellman, alternate between updating A and B.
        # If Polyak is active (single-target only), the per-step smooth update
        # already keeps target_model in sync — skip the discrete hard refresh.
        do_discrete_refresh = (epoch + 1) % bcfg.target_update_every_epochs == 0 and (
            bcfg.target_polyak_tau == 0.0 or bcfg.double_bellman
        )
        if do_discrete_refresh:
            source_sd = model.state_dict()
            if any(k.startswith("_orig_mod.") for k in source_sd):
                source_sd = {
                    k.removeprefix("_orig_mod."): v for k, v in source_sd.items()
                }
            if bcfg.double_bellman:
                if target_swap == 0:
                    target_model.load_state_dict(source_sd)
                else:
                    target_model_b.load_state_dict(source_sd)
                target_swap = 1 - target_swap
            else:
                target_model.load_state_dict(source_sd)

        # Snapshot helper: assemble checkpoint dict.
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
        save_sd = model.state_dict()
        if any(k.startswith("_orig_mod.") for k in save_sd):
            save_sd = {k.removeprefix("_orig_mod."): v for k, v in save_sd.items()}
        ckpt_dict = {
            "epoch": epoch,
            "state_dict": save_sd,
            "loss": avg_loss,
            "train_config": cfg.__dict__,
            "model_config": model_cfg,
            "bellman_config": bcfg.__dict__,
        }

        # Periodic checkpoint
        if (epoch + 1) % cfg.checkpoint_every_epochs == 0 or epoch == cfg.n_epochs - 1:
            torch.save(ckpt_dict, ckpt_dir / f"epoch_{epoch:04d}.pt")

        # Early stopping + best.pt tracking (smoothed loss)
        if bcfg.early_stop_patience > 0:
            recent_losses.append(avg_loss)
            window = bcfg.early_stop_smooth_window
            if len(recent_losses) > window:
                recent_losses = recent_losses[-window:]
            smoothed = sum(recent_losses) / len(recent_losses)
            improved = (best_smoothed_loss - smoothed) > bcfg.early_stop_min_delta
            if improved:
                best_smoothed_loss = smoothed
                best_epoch = epoch
                epochs_without_improvement = 0
                ckpt_dict_best = dict(ckpt_dict)
                ckpt_dict_best["best_smoothed_loss"] = smoothed
                torch.save(ckpt_dict_best, ckpt_dir / "best.pt")
                print(
                    f"[bellman]   new best smoothed loss {smoothed:.5f} -> best.pt",
                    flush=True,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= bcfg.early_stop_patience:
                    print(
                        f"[bellman] early stop at epoch {epoch}: no improvement for "
                        f"{epochs_without_improvement} epochs (best smoothed "
                        f"{best_smoothed_loss:.5f} at epoch {best_epoch})",
                        flush=True,
                    )
                    # Save final epoch checkpoint
                    torch.save(ckpt_dict, ckpt_dir / f"epoch_{epoch:04d}.pt")
                    break
    return result
