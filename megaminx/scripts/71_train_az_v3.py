"""m_az_v3: Hybrid Bellman + policy CE training.

Strategy: train ResMLPGFlowNet such that:
  - Value head is trained EXACTLY like a Bellman V model (random-walk states +
    BFS-d6 mixin + V0/d=1 anchor + admissibility-aware loss). Same recipe as
    m_dd_v0 — should produce a beam-usable value head.
  - Policy head is trained via CE on solver path actions (78,029 dataset).
  - Both heads share the trunk → joint representation learning.

The value head's beam usability is the existence proof that the recipe works.
The policy head is a "free" addition that can later be used as Q-shortlister
or for AlphaZero-style PUCT integration.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/71_train_az_v3.py \\
        --warmstart-trunk megaminx/models/m_dd_v0/epoch_0049.pt \\
        --policy-dataset megaminx/data/az_dataset_78029.pt \\
        --bfs-d6-path megaminx/data/bfs_d6_train.pt \\
        --output megaminx/models/m_az_v3 \\
        --epochs 100
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.gflow_model import ResMLPGFlowNet
from cayley.search import load_model_checkpoint
from megaminx.puzzle import Megaminx


def warmstart_from_v_model(model: ResMLPGFlowNet, ckpt_path: Path,
                            init_value: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    trunk_sd = {k: v for k, v in sd.items()
                if k.startswith(("embedding.", "input_stack.", "res_blocks."))}
    model_sd = model.state_dict()
    compatible = {
        k: v for k, v in trunk_sd.items()
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
    }
    skipped = len(trunk_sd) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    print(f"  warm-start: loaded {len(compatible)} trunk tensors"
          f" ({skipped} shape-mismatched skipped)", flush=True)
    if init_value and "head.weight" in sd:
        if (tuple(model.value_head.weight.shape) == tuple(sd["head.weight"].shape)
                and tuple(model.value_head.bias.shape) == tuple(sd["head.bias"].shape)):
            with torch.no_grad():
                model.value_head.weight.copy_(sd["head.weight"])
                model.value_head.bias.copy_(sd["head.bias"])
            print(f"  initialized value_head = V_distance_head", flush=True)
        else:
            print("  skipped value_head init: shape mismatch", flush=True)
    elif init_value and "value_head.weight" in sd:
        if (tuple(model.value_head.weight.shape) == tuple(sd["value_head.weight"].shape)
                and tuple(model.value_head.bias.shape) == tuple(sd["value_head.bias"].shape)):
            with torch.no_grad():
                model.value_head.weight.copy_(sd["value_head.weight"])
                model.value_head.bias.copy_(sd["value_head.bias"])
            print("  initialized value_head = AZ value_head", flush=True)
        else:
            print("  skipped AZ value_head init: shape mismatch", flush=True)
    if "policy_head.weight" in sd:
        if (tuple(model.policy_head.weight.shape) == tuple(sd["policy_head.weight"].shape)
                and tuple(model.policy_head.bias.shape) == tuple(sd["policy_head.bias"].shape)):
            with torch.no_grad():
                model.policy_head.weight.copy_(sd["policy_head.weight"])
                model.policy_head.bias.copy_(sd["policy_head.bias"])
            print("  initialized policy_head = AZ policy_head", flush=True)
        else:
            print("  skipped AZ policy_head init: shape mismatch", flush=True)


def value_only_forward(model: ResMLPGFlowNet, states: torch.Tensor,
                       chunk_size: int = 4096) -> torch.Tensor:
    """Run only value head (skip policy logits) for target net forward."""
    outs = []
    for i in range(0, states.size(0), chunk_size):
        chunk = states[i : i + chunk_size]
        h = model.trunk(chunk)
        v = model.value_head(h).squeeze(-1)
        outs.append(v)
    return torch.cat(outs, dim=0)


def teacher_value_forward(model: torch.nn.Module, states: torch.Tensor,
                          chunk_size: int = 4096) -> torch.Tensor:
    """Value-only forward for either AZ dual-head or vanilla V checkpoints."""
    if hasattr(model, "value_head") and hasattr(model, "trunk"):
        return value_only_forward(model, states, chunk_size=chunk_size)
    outs = []
    for i in range(0, states.size(0), chunk_size):
        chunk = states[i : i + chunk_size]
        outs.append(model(chunk).squeeze(-1))
    return torch.cat(outs, dim=0)


def load_value_teacher_checkpoint(path: Path, device: str, dtype: torch.dtype) -> torch.nn.Module:
    """Load either a vanilla V checkpoint or an AZ dual-head checkpoint as a value teacher."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    if "value_head.weight" not in sd:
        return load_model_checkpoint(path, device=device, dtype=dtype)

    mc = ckpt["model_config"]
    model = ResMLPGFlowNet(
        state_size=mc["state_size"],
        num_classes=mc["num_classes"],
        hidden_dims=tuple(mc["hidden_dims"]),
        num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        n_actions=mc.get("n_actions", 24),
    )
    model.load_state_dict(sd)
    return model.to(device=device, dtype=dtype).eval()


def load_rotation_tables(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rotations_np = np.load(path).astype(np.int64)
    rotations = torch.from_numpy(rotations_np).to(device=device, dtype=torch.long)
    inverse = torch.empty_like(rotations)
    idx = torch.arange(rotations.size(1), device=device, dtype=torch.long).expand_as(rotations)
    inverse.scatter_(1, rotations, idx)
    return rotations, inverse


def apply_state_rotations(states: torch.Tensor, rotations: torch.Tensor,
                          inverse_rotations: torch.Tensor,
                          rotation_idx: torch.Tensor) -> torch.Tensor:
    """Apply whole-cube sticker relabeling: R[state[R_inv[i]]]."""
    r = rotations[rotation_idx]
    r_inv = inverse_rotations[rotation_idx]
    pulled = torch.gather(states.long(), 1, r_inv)
    return torch.gather(r, 1, pulled)


@torch.no_grad()
def orbit_teacher_targets(teacher: torch.nn.Module, states: torch.Tensor,
                          rotations: torch.Tensor, inverse_rotations: torch.Tensor,
                          n_samples: int, rng: torch.Generator,
                          chunk_size: int,
                          autocast_ctx) -> torch.Tensor:
    acc = torch.zeros(states.size(0), device=states.device, dtype=torch.float32)
    n_rot = rotations.size(0)
    for _ in range(n_samples):
        rot_idx = torch.randint(0, n_rot, (states.size(0),),
                                generator=rng, device=states.device)
        rotated = apply_state_rotations(states, rotations, inverse_rotations, rot_idx)
        if autocast_ctx is not None:
            with autocast_ctx:
                vals = teacher_value_forward(teacher, rotated, chunk_size=chunk_size)
        else:
            vals = teacher_value_forward(teacher, rotated, chunk_size=chunk_size)
        acc += vals.float()
    return acc / float(max(1, n_samples))


@torch.no_grad()
def bellman_targets(target_model, states, walk_depths, generators, solved_state,
                    chunk_size, clip_upper=True, clip_lower=True):
    """Bellman target = clip(1 + min_a target(child), 0, walk_depth).

    Same logic as cayley.bellman._bellman_targets but for value head only.
    """
    B, S = states.shape
    n_gen = generators.shape[0]
    children = states.unsqueeze(1).expand(B, n_gen, S).clone()
    children = torch.gather(children, 2,
                            generators.unsqueeze(0).expand(B, n_gen, S))  # (B, n_gen, S)
    children_flat = children.reshape(B * n_gen, S)
    is_solved = (children_flat == solved_state).all(dim=1)

    target_model.eval()
    child_v = value_only_forward(target_model, children_flat, chunk_size).float()
    child_v = torch.where(is_solved, torch.zeros_like(child_v), child_v)
    child_v = child_v.view(B, n_gen)
    reduced = child_v.min(dim=1).values
    target = 1.0 + reduced
    if clip_upper:
        target = torch.minimum(target, walk_depths)
    if clip_lower:
        target = torch.clamp(target, min=0.0)
    return target


def pairwise_path_rank_loss(pred: torch.Tensor, target: torch.Tensor,
                            min_gap: float, margin: float) -> torch.Tensor:
    """Rank random path states by remaining path length without trusting scale."""
    n = (pred.numel() // 2) * 2
    if n < 2:
        return pred.new_tensor(0.0, dtype=torch.float32)
    pred_a = pred[:n:2].float()
    pred_b = pred[1:n:2].float()
    target_a = target[:n:2].float()
    target_b = target[1:n:2].float()
    diff = target_a - target_b
    mask = diff.abs() >= min_gap
    if not bool(mask.any()):
        return pred.new_tensor(0.0, dtype=torch.float32)
    signed_pred_gap = (pred_a - pred_b)[mask] * diff[mask].sign()
    return F.relu(margin - signed_pred_gap).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--warmstart-trunk", type=Path, required=True)
    ap.add_argument("--policy-dataset", required=True, type=Path,
                    help="AZ dataset (states_p, actions_p) for policy CE")
    ap.add_argument("--bfs-d6-path", type=Path, default=None,
                    help="Optional BFS-d6 dataset for exact-target mixin")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000,
                    help="Random-walk samples per epoch for Bellman value training")
    ap.add_argument("--rw-batch-size", type=int, default=4096)
    ap.add_argument("--policy-batch-size", type=int, default=512,
                    help="Policy CE batch size (sampled from policy dataset)")
    ap.add_argument("--anchor-v0", type=int, default=32)
    ap.add_argument("--anchor-d1", type=int, default=4)
    ap.add_argument("--bfs-d6-fraction", type=float, default=0.10)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="policy CE weight")
    ap.add_argument("--beta", type=float, default=1.0, help="value MSE weight")
    ap.add_argument("--gamma-path-value", type=float, default=0.0,
                    help="Optional MSE weight for path-distance labels in AZ dataset")
    ap.add_argument("--path-value-loss", choices=("mse", "huber"), default="mse",
                    help="Loss type for optional path-distance value labels")
    ap.add_argument("--path-value-clip", type=float, default=0.0,
                    help="If >0, clamp path-distance targets to this value")
    ap.add_argument("--path-value-huber-beta", type=float, default=5.0,
                    help="SmoothL1 beta for --path-value-loss huber")
    ap.add_argument("--gamma-path-rank", type=float, default=0.0,
                    help="Weight for pairwise ranking loss on path-distance labels")
    ap.add_argument("--path-rank-min-gap", type=float, default=4.0,
                    help="Minimum path-label gap for a pair to contribute to rank loss")
    ap.add_argument("--path-rank-margin", type=float, default=1.0,
                    help="Required predicted V gap for path ranking pairs")
    ap.add_argument("--lambda-orbit", type=float, default=0.0,
                    help="Weight for optional orbit-mean teacher value loss")
    ap.add_argument("--lambda-preserve", type=float, default=0.0,
                    help="Weight for optional value-preservation teacher loss")
    ap.add_argument("--preserve-teacher", type=Path, default=None,
                    help="Frozen V checkpoint to preserve on the value batch")
    ap.add_argument("--preserve-teacher-chunk", type=int, default=4096)
    ap.add_argument("--orbit-teacher", type=Path, default=None,
                    help="Frozen V checkpoint used for orbit-mean targets")
    ap.add_argument("--orbit-rotations", type=Path,
                    default=PROJECT / "data" / "rotations.npy")
    ap.add_argument("--orbit-samples", type=int, default=4,
                    help="Random rotations sampled per value state")
    ap.add_argument("--orbit-batch-size", type=int, default=1024,
                    help="Value-batch states receiving orbit loss per step")
    ap.add_argument("--orbit-teacher-chunk", type=int, default=4096)
    ap.add_argument("--target-update-every-epochs", type=int, default=10)
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    n_gen = len(puzzle.move_names)

    # Build model
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPGFlowNet(
        state_size=state_size, num_classes=state_size,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16, n_actions=n_gen,
    ).to(args.device)
    print(f"model params: {model.num_parameters():,}", flush=True)
    warmstart_from_v_model(model, args.warmstart_trunk, init_value=True)
    model.train()

    # Target net for Bellman bootstrap
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    # Generators + solved state
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(args.device)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=args.device)
    anchor_v0 = solved_state.unsqueeze(0)  # (1, S)
    anchor_d1 = torch.gather(
        anchor_v0.expand(n_gen, state_size), 1, generators
    ).squeeze(0) if n_gen > 1 else anchor_v0
    # Recompute anchor_d1 = (n_gen, S)
    anchor_d1 = torch.gather(anchor_v0.expand(n_gen, state_size), 1, generators)

    # Policy dataset
    print(f"loading policy dataset: {args.policy_dataset}", flush=True)
    policy_data = torch.load(args.policy_dataset, map_location=args.device, weights_only=False)
    states_p = policy_data["states"].to(args.device).long()
    actions_p = policy_data["actions"].to(args.device).long()
    values_p = None
    if "values" in policy_data:
        values_p = policy_data["values"].to(args.device).float()
    Np = states_p.size(0)
    print(f"  {Np:,} (state, action) pairs", flush=True)
    if args.gamma_path_value > 0.0 and values_p is None:
        raise ValueError("--gamma-path-value > 0 requires 'values' in policy dataset")
    if args.gamma_path_rank > 0.0 and values_p is None:
        raise ValueError("--gamma-path-rank > 0 requires 'values' in policy dataset")
    if args.path_value_huber_beta <= 0.0:
        raise ValueError("--path-value-huber-beta must be positive")
    if args.path_rank_min_gap < 0.0:
        raise ValueError("--path-rank-min-gap must be non-negative")
    if args.path_rank_margin <= 0.0:
        raise ValueError("--path-rank-margin must be positive")

    # BFS-d6 mixin (optional)
    bfs6_states = None
    bfs6_dists = None
    bfs6_per_batch = 0
    if args.bfs_d6_path:
        print(f"loading BFS-d6: {args.bfs_d6_path}", flush=True)
        bfs6 = torch.load(args.bfs_d6_path, map_location="cpu", weights_only=False)
        bfs6_states = bfs6["states"]   # (N, 120) int8
        bfs6_dists = bfs6["distances"] # (N,) int8
        bfs6_per_batch = max(1, int(round(args.rw_batch_size * args.bfs_d6_fraction)))
        print(f"  BFS-d6 mixin: {bfs6_per_batch}/{args.rw_batch_size}", flush=True)

    orbit_teacher = None
    rotations = None
    inverse_rotations = None
    if args.lambda_orbit > 0.0:
        if args.orbit_teacher is None:
            raise ValueError("--lambda-orbit > 0 requires --orbit-teacher")
        if args.orbit_samples <= 0:
            raise ValueError("--orbit-samples must be positive when orbit loss is enabled")
        teacher_dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
        print(f"loading orbit teacher: {args.orbit_teacher}", flush=True)
        orbit_teacher = load_model_checkpoint(args.orbit_teacher, device=args.device,
                                              dtype=teacher_dtype)
        for p in orbit_teacher.parameters():
            p.requires_grad = False
        rotations, inverse_rotations = load_rotation_tables(args.orbit_rotations, args.device)
        print(f"  orbit loss: lambda={args.lambda_orbit} samples={args.orbit_samples} "
              f"batch={args.orbit_batch_size} rotations={rotations.size(0)}",
              flush=True)

    preserve_teacher = None
    if args.lambda_preserve > 0.0:
        if args.preserve_teacher is None:
            raise ValueError("--lambda-preserve > 0 requires --preserve-teacher")
        teacher_dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
        print(f"loading preserve teacher: {args.preserve_teacher}", flush=True)
        preserve_teacher = load_value_teacher_checkpoint(args.preserve_teacher, device=args.device,
                                                         dtype=teacher_dtype)
        preserve_teacher.eval()
        for p in preserve_teacher.parameters():
            p.requires_grad = False
        print(f"  preserve loss: lambda={args.lambda_preserve}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=args.device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else None

    rng_p = torch.Generator(device=args.device).manual_seed(args.seed)
    rng_bfs = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    rng_device = "cuda" if str(args.device).startswith("cuda") else "cpu"
    rng_orbit = torch.Generator(device=rng_device).manual_seed(args.seed + 2)

    print(f"epochs: {args.epochs}  rw_batch: {args.rw_batch_size}  "
          f"policy_batch: {args.policy_batch_size}  alpha={args.alpha}  beta={args.beta}  "
          f"gamma_path_value={args.gamma_path_value}  "
          f"path_value_loss={args.path_value_loss}  path_value_clip={args.path_value_clip}  "
          f"gamma_path_rank={args.gamma_path_rank}  "
          f"lambda_orbit={args.lambda_orbit}  "
          f"lambda_preserve={args.lambda_preserve}",
          flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        rw_states, rw_depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch * 1000, device=args.device, n_back=args.n_back,
        )
        rw_depths_f = rw_depths.to(torch.float32)
        N_rw = rw_states.size(0)

        # Pre-shuffle BFS-d6 for the epoch
        if bfs6_states is not None:
            rw_per_batch = args.rw_batch_size - bfs6_per_batch - args.anchor_v0 - 24 * args.anchor_d1
            n_per_epoch = bfs6_per_batch * (N_rw // rw_per_batch + 1)
            n_per_epoch = min(n_per_epoch, bfs6_states.size(0))
            ep_perm = torch.randperm(bfs6_states.size(0), generator=rng_bfs)[:n_per_epoch]
            ep_bfs_states = bfs6_states[ep_perm].to(args.device).long()
            ep_bfs_dists = bfs6_dists[ep_perm].to(args.device).float()
        else:
            rw_per_batch = args.rw_batch_size - args.anchor_v0 - 24 * args.anchor_d1
            ep_bfs_states = None
            ep_bfs_dists = None

        n_rw_batches = max(1, N_rw // rw_per_batch)
        bfs_cursor = 0
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_path_v_loss = 0.0
        total_path_rank_loss = 0.0
        total_orbit_loss = 0.0
        total_preserve_loss = 0.0
        total_acc = 0.0

        for b in range(n_rw_batches):
            # 1. Random-walk batch + anchors + bfs6 → Bellman value training
            start = b * rw_per_batch
            end = min(start + rw_per_batch, N_rw)
            bs_rw = rw_states[start:end]
            bd_rw = rw_depths_f[start:end]
            with torch.no_grad():
                target_rw = bellman_targets(
                    target_model, bs_rw, bd_rw, generators, solved_state,
                    chunk_size=4096,
                )

            # Build full value batch: rw + anchors + bfs6
            bs_parts = [bs_rw]
            target_parts = [target_rw]
            if args.anchor_v0 > 0:
                bs_parts.append(anchor_v0.expand(args.anchor_v0, -1))
                target_parts.append(torch.zeros(args.anchor_v0, dtype=torch.float32,
                                                device=args.device))
            if args.anchor_d1 > 0:
                # 24 d=1 children, each repeated args.anchor_d1 times
                bs_parts.append(anchor_d1.repeat(args.anchor_d1, 1))
                target_parts.append(torch.ones(24 * args.anchor_d1, dtype=torch.float32,
                                                device=args.device))
            if ep_bfs_states is not None and bfs_cursor + bfs6_per_batch <= ep_bfs_states.size(0):
                bs_parts.append(ep_bfs_states[bfs_cursor : bfs_cursor + bfs6_per_batch])
                target_parts.append(ep_bfs_dists[bfs_cursor : bfs_cursor + bfs6_per_batch])
                bfs_cursor += bfs6_per_batch

            bs_v = torch.cat(bs_parts, dim=0)
            target_v = torch.cat(target_parts, dim=0)

            # 2. Policy batch (random sample from policy dataset)
            idx_p = torch.randint(0, Np, (args.policy_batch_size,),
                                  generator=rng_p, device=args.device)
            bs_p = states_p[idx_p]
            ba_p = actions_p[idx_p]
            bv_p = values_p[idx_p] if values_p is not None else None

            # Forward + losses
            if autocast_ctx is not None:
                with autocast_ctx:
                    # Value forward (rw + anchor + bfs6)
                    h_v = model.trunk(bs_v)
                    pred_v = model.value_head(h_v).squeeze(-1)
                    v_loss = F.mse_loss(pred_v.float(), target_v)
                    # Policy forward
                    h_p = model.trunk(bs_p)
                    logits_p = model.policy_head(h_p)
                    p_loss = F.cross_entropy(logits_p, ba_p)
                    path_v_loss = pred_v.new_tensor(0.0, dtype=torch.float32)
                    path_rank_loss = pred_v.new_tensor(0.0, dtype=torch.float32)
                    pred_path_v = None
                    if args.gamma_path_value > 0.0 or args.gamma_path_rank > 0.0:
                        pred_path_v = model.value_head(h_p).squeeze(-1)
                    if args.gamma_path_value > 0.0:
                        target_path_v = bv_p
                        if args.path_value_clip > 0.0:
                            target_path_v = target_path_v.clamp(max=args.path_value_clip)
                        if args.path_value_loss == "huber":
                            path_v_loss = F.smooth_l1_loss(
                                pred_path_v.float(), target_path_v,
                                beta=args.path_value_huber_beta,
                            )
                        else:
                            path_v_loss = F.mse_loss(pred_path_v.float(), target_path_v)
                    if args.gamma_path_rank > 0.0:
                        path_rank_loss = pairwise_path_rank_loss(
                            pred_path_v, bv_p,
                            min_gap=args.path_rank_min_gap,
                            margin=args.path_rank_margin,
                        )
                    loss = args.alpha * p_loss + args.beta * v_loss
            else:
                h_v = model.trunk(bs_v)
                pred_v = model.value_head(h_v).squeeze(-1)
                v_loss = F.mse_loss(pred_v.float(), target_v)
                h_p = model.trunk(bs_p)
                logits_p = model.policy_head(h_p)
                p_loss = F.cross_entropy(logits_p, ba_p)
                path_v_loss = pred_v.new_tensor(0.0, dtype=torch.float32)
                path_rank_loss = pred_v.new_tensor(0.0, dtype=torch.float32)
                pred_path_v = None
                if args.gamma_path_value > 0.0 or args.gamma_path_rank > 0.0:
                    pred_path_v = model.value_head(h_p).squeeze(-1)
                if args.gamma_path_value > 0.0:
                    target_path_v = bv_p
                    if args.path_value_clip > 0.0:
                        target_path_v = target_path_v.clamp(max=args.path_value_clip)
                    if args.path_value_loss == "huber":
                        path_v_loss = F.smooth_l1_loss(
                            pred_path_v.float(), target_path_v,
                            beta=args.path_value_huber_beta,
                        )
                    else:
                        path_v_loss = F.mse_loss(pred_path_v.float(), target_path_v)
                if args.gamma_path_rank > 0.0:
                    path_rank_loss = pairwise_path_rank_loss(
                        pred_path_v, bv_p,
                        min_gap=args.path_rank_min_gap,
                        margin=args.path_rank_margin,
                    )
                loss = args.alpha * p_loss + args.beta * v_loss
            if args.gamma_path_value > 0.0:
                loss = loss + args.gamma_path_value * path_v_loss
            if args.gamma_path_rank > 0.0:
                loss = loss + args.gamma_path_rank * path_rank_loss

            preserve_loss = pred_v.new_tensor(0.0, dtype=torch.float32)
            if preserve_teacher is not None:
                with torch.no_grad():
                    if autocast_ctx is not None:
                        with autocast_ctx:
                            target_preserve = teacher_value_forward(
                                preserve_teacher, bs_v,
                                chunk_size=args.preserve_teacher_chunk,
                            )
                    else:
                        target_preserve = teacher_value_forward(
                            preserve_teacher, bs_v,
                            chunk_size=args.preserve_teacher_chunk,
                        )
                preserve_loss = F.mse_loss(pred_v.float(), target_preserve.float())
                loss = loss + args.lambda_preserve * preserve_loss

            orbit_loss = pred_v.new_tensor(0.0, dtype=torch.float32)
            if orbit_teacher is not None:
                n_orbit = min(args.orbit_batch_size, bs_v.size(0))
                idx_o = torch.randint(0, bs_v.size(0), (n_orbit,),
                                      generator=rng_orbit, device=args.device)
                bs_o = bs_v[idx_o]
                target_o = orbit_teacher_targets(
                    orbit_teacher, bs_o, rotations, inverse_rotations,
                    n_samples=args.orbit_samples, rng=rng_orbit,
                    chunk_size=args.orbit_teacher_chunk,
                    autocast_ctx=autocast_ctx,
                )
                orbit_loss = F.mse_loss(pred_v[idx_o].float(), target_o)
                loss = loss + args.lambda_orbit * orbit_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_p_loss += float(p_loss.item())
            total_v_loss += float(v_loss.item())
            total_path_v_loss += float(path_v_loss.item())
            total_path_rank_loss += float(path_rank_loss.item())
            total_orbit_loss += float(orbit_loss.item())
            total_preserve_loss += float(preserve_loss.item())
            with torch.no_grad():
                total_acc += float((logits_p.argmax(dim=1) == ba_p).float().mean().item())
        sched.step()

        avg_p = total_p_loss / max(1, n_rw_batches)
        avg_v = total_v_loss / max(1, n_rw_batches)
        avg_path_v = total_path_v_loss / max(1, n_rw_batches)
        avg_path_rank = total_path_rank_loss / max(1, n_rw_batches)
        avg_orbit = total_orbit_loss / max(1, n_rw_batches)
        avg_preserve = total_preserve_loss / max(1, n_rw_batches)
        avg_acc = total_acc / max(1, n_rw_batches)
        wall = time.time() - t0
        msg = (f"epoch {epoch:4d} | p_loss {avg_p:.4f} | v_loss {avg_v:.4f} | "
               f"top-1 acc {avg_acc:.4f}")
        if args.gamma_path_value > 0.0:
            msg += f" | path_v_loss {avg_path_v:.4f}"
        if args.gamma_path_rank > 0.0:
            msg += f" | path_rank_loss {avg_path_rank:.4f}"
        if orbit_teacher is not None:
            msg += f" | orbit_loss {avg_orbit:.4f}"
        if preserve_teacher is not None:
            msg += f" | preserve_loss {avg_preserve:.4f}"
        msg += f" | lr {sched.get_last_lr()[0]:.2e} | {wall:.1f}s"
        print(msg, flush=True)

        # Refresh target net (for Bellman bootstrap)
        if (epoch + 1) % args.target_update_every_epochs == 0:
            sd_src = model.state_dict()
            target_model.load_state_dict(sd_src)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            sd = model.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": sd,
                "model_config": {
                    "state_size": state_size, "num_classes": state_size,
                    "hidden_dims": list(hidden_dims),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "embedding", "embed_dim": 16, "n_actions": n_gen,
                },
                "p_loss": avg_p, "v_loss": avg_v, "top1_acc": avg_acc,
                "path_v_loss": avg_path_v,
                "gamma_path_value": args.gamma_path_value,
                "path_value_loss": args.path_value_loss,
                "path_value_clip": args.path_value_clip,
                "path_value_huber_beta": args.path_value_huber_beta,
                "path_rank_loss": avg_path_rank,
                "gamma_path_rank": args.gamma_path_rank,
                "path_rank_min_gap": args.path_rank_min_gap,
                "path_rank_margin": args.path_rank_margin,
                "orbit_loss": avg_orbit,
                "lambda_orbit": args.lambda_orbit,
                "orbit_samples": args.orbit_samples,
                "orbit_teacher": str(args.orbit_teacher) if args.orbit_teacher else None,
                "preserve_loss": avg_preserve,
                "lambda_preserve": args.lambda_preserve,
                "preserve_teacher": str(args.preserve_teacher) if args.preserve_teacher else None,
            }, args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
