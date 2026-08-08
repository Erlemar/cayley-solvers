"""AZ-tetraminx: hybrid Bellman + policy CE training for the Professor Tetraminx.

Port of scripts/11_train_az_cube.py (itself the megaminx AZ v4 recipe -- the first
model to break the megaminx 6M cluster ceiling):
  - Value head trained EXACTLY like a Bellman V model: random-walk states
    (k_max=32) + BFS exact-distance mixin + V0/d=1 anchors, bootstrapped against
    a target net refreshed every N epochs.
  - Policy head trained via CE on the floor-path actions (29,622 floor moves,
    x48 sym/antisym augmented = 1.42M pairs; tetraminx/scripts/20_build_az_dataset.py).
  - Optional path-distance value loss on the same states via --gamma-path-value.
  - Both heads share the trunk, warm-started from the Stage B Bellman V.

Defaults follow the AZ v4 launch config (rw_batch 8192 / policy_batch 1024 --
NOT the old script defaults; megaminx CLAUDE.md Rule 13).

Usage:
    python3 tetraminx/scripts/21_train_az.py \
        --output tetraminx/models/taz_v1 \
        --warmstart-trunk tetraminx/models/tv0_bellman/best.pt \
        --policy-dataset tetraminx/data/az_dataset.pt \
        --bfs-d6-path tetraminx/data/bfs_d6_train.pt \
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.gflow_model import ResMLPGFlowNet
from tetraminx.puzzle import Tetraminx


def warmstart_from_v_model(model: ResMLPGFlowNet, ckpt_path: Path,
                           init_value: bool = True) -> None:
    """Load trunk (embedding/input_stack/res_blocks) + optionally value head
    from a ResMLPDistance or AZ checkpoint. Shape-mismatched tensors are
    skipped LOUDLY (megaminx Rule 18: a silent skip means training from
    random init)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    trunk_sd = {k: v for k, v in sd.items()
                if k.startswith(("embedding.", "input_stack.", "res_blocks."))}
    model_sd = model.state_dict()
    compatible = {k: v for k, v in trunk_sd.items()
                  if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)}
    skipped = len(trunk_sd) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    print(f"  warm-start: loaded {len(compatible)} trunk tensors "
          f"({skipped} shape-mismatched skipped)", flush=True)
    if skipped:
        raise SystemExit("warm-start shape mismatch -- refusing to train from "
                         "partially random init (pass a matching checkpoint)")
    if init_value and "head.weight" in sd:
        assert tuple(model.value_head.weight.shape) == tuple(sd["head.weight"].shape)
        with torch.no_grad():
            model.value_head.weight.copy_(sd["head.weight"])
            model.value_head.bias.copy_(sd["head.bias"])
        print("  initialized value_head from V distance head", flush=True)


def value_only_forward(model, states, chunk_size=4096):
    outs = []
    for i in range(0, states.size(0), chunk_size):
        h = model.trunk(states[i:i + chunk_size])
        outs.append(model.value_head(h).squeeze(-1))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def bellman_targets(target_model, states, walk_depths, generators, solved_state,
                    chunk_size, clip_upper=True, clip_lower=True):
    """Bellman target = clip(1 + min_a target(child), 0, walk_depth)."""
    B, S = states.shape
    n_gen = generators.shape[0]
    children = states.unsqueeze(1).expand(B, n_gen, S).clone()
    children = torch.gather(children, 2,
                            generators.unsqueeze(0).expand(B, n_gen, S))
    children_flat = children.reshape(B * n_gen, S)
    is_solved = (children_flat == solved_state).all(dim=1)

    target_model.eval()
    child_v = value_only_forward(target_model, children_flat, chunk_size).float()
    child_v = torch.where(is_solved, torch.zeros_like(child_v), child_v)
    child_v = child_v.view(B, n_gen)
    target = 1.0 + child_v.min(dim=1).values
    if clip_upper:
        target = torch.minimum(target, walk_depths)
    if clip_lower:
        target = torch.clamp(target, min=0.0)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--warmstart-trunk", type=Path, required=True)
    ap.add_argument("--policy-dataset", required=True, type=Path)
    ap.add_argument("--bfs-d6-path", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--rw-batch-size", type=int, default=8192)     # AZ v4 launch value
    ap.add_argument("--policy-batch-size", type=int, default=1024)  # AZ v4 launch value
    ap.add_argument("--anchor-v0", type=int, default=32)
    ap.add_argument("--anchor-d1", type=int, default=4)
    ap.add_argument("--bfs-d6-fraction", type=float, default=0.10)
    ap.add_argument("--k-max", type=int, default=32)               # tetraminx diameter ~27
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="policy CE weight")
    ap.add_argument("--beta", type=float, default=1.0, help="Bellman value MSE weight")
    ap.add_argument("--gamma-path-value", type=float, default=0.0,
                    help="MSE weight for near-exact path-distance labels")
    ap.add_argument("--path-value-clip", type=float, default=0.0)
    ap.add_argument("--target-update-every-epochs", type=int, default=10)
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    puzzle = Tetraminx.load(PROJECT / "tetraminx" / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    n_gen = len(puzzle.move_names)
    assert (state_size, n_gen) == (88, 24)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPGFlowNet(
        state_size=state_size, num_classes=state_size,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16, n_actions=n_gen,
    ).to(args.device)
    print(f"model params: {model.num_parameters():,}", flush=True)
    warmstart_from_v_model(model, args.warmstart_trunk, init_value=True)
    model.train()

    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(args.device)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=args.device)
    anchor_v0 = solved_state.unsqueeze(0)                                   # (1, S)
    anchor_d1 = torch.gather(anchor_v0.expand(n_gen, state_size), 1, generators)  # (n_gen, S)

    print(f"loading policy dataset: {args.policy_dataset}", flush=True)
    policy_data = torch.load(args.policy_dataset, map_location="cpu", weights_only=False)
    states_p = policy_data["states"].to(args.device).long()
    actions_p = policy_data["actions"].to(args.device).long()
    values_p = None
    if "dists" in policy_data:
        values_p = policy_data["dists"].to(args.device).float()
    elif "values" in policy_data:
        values_p = policy_data["values"].to(args.device).float()
    Np = states_p.size(0)
    print(f"  {Np:,} (state, action) pairs"
          + (f", dist labels {values_p.min():.0f}..{values_p.max():.0f}" if values_p is not None else ""),
          flush=True)
    if args.gamma_path_value > 0.0 and values_p is None:
        raise ValueError("--gamma-path-value > 0 requires dist labels in the dataset")

    bfs6_states = bfs6_dists = None
    bfs6_per_batch = 0
    if args.bfs_d6_path:
        print(f"loading BFS-d6: {args.bfs_d6_path}", flush=True)
        bfs6 = torch.load(args.bfs_d6_path, map_location="cpu", weights_only=False)
        bfs6_states = bfs6["states"]
        bfs6_dists = bfs6.get("depths", bfs6.get("distances"))
        bfs6_per_batch = max(1, int(round(args.rw_batch_size * args.bfs_d6_fraction)))
        print(f"  BFS-d6 mixin: {bfs6_per_batch}/{args.rw_batch_size} "
              f"({bfs6_states.size(0):,} states)", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=args.device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if args.device == "cuda" else None)

    rng_p = torch.Generator(device=args.device).manual_seed(args.seed)
    rng_bfs = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    # Config echo (megaminx Rule 13: the resolved config MUST be in the log).
    print("training config: " + " ".join(
        f"{k}={v}" for k, v in sorted(vars(args).items())), flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        rw_states, rw_depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch * 1000, device=args.device, n_back=args.n_back,
        )
        rw_depths_f = rw_depths.to(torch.float32)
        N_rw = rw_states.size(0)

        rw_per_batch = (args.rw_batch_size - args.anchor_v0
                        - n_gen * args.anchor_d1 - bfs6_per_batch)
        if bfs6_states is not None:
            n_per_epoch = min(bfs6_per_batch * (N_rw // rw_per_batch + 1),
                              bfs6_states.size(0))
            ep_perm = torch.randperm(bfs6_states.size(0), generator=rng_bfs)[:n_per_epoch]
            ep_bfs_states = bfs6_states[ep_perm].to(args.device).long()
            ep_bfs_dists = bfs6_dists[ep_perm].to(args.device).float()
        else:
            ep_bfs_states = ep_bfs_dists = None

        n_rw_batches = max(1, N_rw // rw_per_batch)
        bfs_cursor = 0
        total_p = total_v = total_path_v = total_acc = 0.0

        for b in range(n_rw_batches):
            start = b * rw_per_batch
            end = min(start + rw_per_batch, N_rw)
            bs_rw = rw_states[start:end]
            bd_rw = rw_depths_f[start:end]
            with torch.no_grad():
                target_rw = bellman_targets(
                    target_model, bs_rw, bd_rw, generators, solved_state,
                    chunk_size=8192,
                )

            bs_parts = [bs_rw]
            target_parts = [target_rw]
            if args.anchor_v0 > 0:
                bs_parts.append(anchor_v0.expand(args.anchor_v0, -1))
                target_parts.append(torch.zeros(args.anchor_v0, dtype=torch.float32,
                                                device=args.device))
            if args.anchor_d1 > 0:
                bs_parts.append(anchor_d1.repeat(args.anchor_d1, 1))
                target_parts.append(torch.ones(n_gen * args.anchor_d1, dtype=torch.float32,
                                               device=args.device))
            if ep_bfs_states is not None and bfs_cursor + bfs6_per_batch <= ep_bfs_states.size(0):
                bs_parts.append(ep_bfs_states[bfs_cursor:bfs_cursor + bfs6_per_batch])
                target_parts.append(ep_bfs_dists[bfs_cursor:bfs_cursor + bfs6_per_batch])
                bfs_cursor += bfs6_per_batch

            bs_v = torch.cat(bs_parts, dim=0)
            target_v = torch.cat(target_parts, dim=0)

            idx_p = torch.randint(0, Np, (args.policy_batch_size,),
                                  generator=rng_p, device=args.device)
            bs_p = states_p[idx_p]
            ba_p = actions_p[idx_p]
            bv_p = values_p[idx_p] if values_p is not None else None

            def compute_losses():
                h_v = model.trunk(bs_v)
                pred_v = model.value_head(h_v).squeeze(-1)
                v_loss = F.mse_loss(pred_v.float(), target_v)
                h_p = model.trunk(bs_p)
                logits_p = model.policy_head(h_p)
                p_loss = F.cross_entropy(logits_p, ba_p)
                path_v_loss = pred_v.new_tensor(0.0, dtype=torch.float32)
                if args.gamma_path_value > 0.0:
                    pred_path_v = model.value_head(h_p).squeeze(-1)
                    tgt = bv_p if args.path_value_clip <= 0.0 else bv_p.clamp(max=args.path_value_clip)
                    path_v_loss = F.mse_loss(pred_path_v.float(), tgt)
                return p_loss, v_loss, path_v_loss, logits_p

            if autocast_ctx is not None:
                with autocast_ctx:
                    p_loss, v_loss, path_v_loss, logits_p = compute_losses()
            else:
                p_loss, v_loss, path_v_loss, logits_p = compute_losses()
            loss = args.alpha * p_loss + args.beta * v_loss
            if args.gamma_path_value > 0.0:
                loss = loss + args.gamma_path_value * path_v_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_p += float(p_loss.item())
            total_v += float(v_loss.item())
            total_path_v += float(path_v_loss.item())
            with torch.no_grad():
                total_acc += float((logits_p.argmax(dim=1) == ba_p).float().mean().item())
        sched.step()

        nb = max(1, n_rw_batches)
        msg = (f"epoch {epoch:4d} | p_loss {total_p/nb:.4f} | v_loss {total_v/nb:.4f} | "
               f"top-1 acc {total_acc/nb:.4f}")
        if args.gamma_path_value > 0.0:
            msg += f" | path_v_loss {total_path_v/nb:.4f}"
        msg += f" | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s"
        print(msg, flush=True)

        if (epoch + 1) % args.target_update_every_epochs == 0:
            target_model.load_state_dict(model.state_dict())

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "model_config": {
                    "state_size": state_size, "num_classes": state_size,
                    "hidden_dims": list(hidden_dims),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "embedding", "embed_dim": 16, "n_actions": n_gen,
                },
                "p_loss": total_p/nb, "v_loss": total_v/nb, "top1_acc": total_acc/nb,
                "train_args": {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in vars(args).items()},
            }, args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
