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

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.gflow_model import ResMLPGFlowNet
from megaminx.puzzle import Megaminx


def warmstart_from_v_model(model: ResMLPGFlowNet, ckpt_path: Path,
                            init_value: bool = True):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    trunk_sd = {k: v for k, v in sd.items()
                if k.startswith(("embedding.", "input_stack.", "res_blocks."))}
    model.load_state_dict(trunk_sd, strict=False)
    print(f"  warm-start: loaded {len(trunk_sd)} trunk tensors", flush=True)
    if init_value and "head.weight" in sd:
        with torch.no_grad():
            model.value_head.weight.copy_(sd["head.weight"])
            model.value_head.bias.copy_(sd["head.bias"])
        print(f"  initialized value_head = V_distance_head", flush=True)


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
    Np = states_p.size(0)
    print(f"  {Np:,} (state, action) pairs", flush=True)

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

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=args.device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else None

    rng_p = torch.Generator(device=args.device).manual_seed(args.seed)
    rng_bfs = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    print(f"epochs: {args.epochs}  rw_batch: {args.rw_batch_size}  "
          f"policy_batch: {args.policy_batch_size}  alpha={args.alpha}  beta={args.beta}",
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
                    loss = args.alpha * p_loss + args.beta * v_loss
            else:
                h_v = model.trunk(bs_v)
                pred_v = model.value_head(h_v).squeeze(-1)
                v_loss = F.mse_loss(pred_v.float(), target_v)
                h_p = model.trunk(bs_p)
                logits_p = model.policy_head(h_p)
                p_loss = F.cross_entropy(logits_p, ba_p)
                loss = args.alpha * p_loss + args.beta * v_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_p_loss += float(p_loss.item())
            total_v_loss += float(v_loss.item())
            with torch.no_grad():
                total_acc += float((logits_p.argmax(dim=1) == ba_p).float().mean().item())
        sched.step()

        avg_p = total_p_loss / max(1, n_rw_batches)
        avg_v = total_v_loss / max(1, n_rw_batches)
        avg_acc = total_acc / max(1, n_rw_batches)
        wall = time.time() - t0
        print(f"epoch {epoch:4d} | p_loss {avg_p:.4f} | v_loss {avg_v:.4f} | "
              f"top-1 acc {avg_acc:.4f} | lr {sched.get_last_lr()[0]:.2e} | {wall:.1f}s",
              flush=True)

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
            }, args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
