"""m22: train value model with K-step lookahead Bellman targets.

Standard Bellman: target(s) = clip(1 + min_a V_target(apply(s, a)), 0, walk_depth)

K-step lookahead: target(s) = min over K-step paths from s of
    (path_cost + V_target(end_of_path))

When K=1, this reduces to standard Bellman. K>1 reduces "heuristic depression
regions" — local minima in V where beam wastes time before climbing back out.

Reference: arXiv 2511.10264 (limited-horizon updates reduce depression regions).

Implementation: at each training batch, expand K layers of children for each
state s (K-step BFS without dedup). Score all leaves with V_target. Take min
of (k-steps + leaf-V) over all leaves; clamp at 0 and walk_depth.

Cost vs standard Bellman: K=2 is ~24× more leaves to score per batch (24^2 = 576
vs 24); K=3 is 24^3 = 13,824. So K=2 is the tractable choice on a 4090.

Warmstart from m05. ~90 min training at K=2.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/10_train_m22_horizon.py \
        --warmstart megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --K 2 \
        --out-dir megaminx/models/m22_horizon_K2
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
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


@torch.no_grad()
def k_step_lookahead_targets(
    target_model: ResMLPDistance,
    states: torch.Tensor,           # (B, S)
    walk_depths: torch.Tensor,      # (B,)
    generators: torch.Tensor,       # (n_gen, S)
    solved_state: torch.Tensor,     # (S,)
    K: int,
    chunk_size: int,
) -> torch.Tensor:
    """Compute target = min over K-step paths of (k-steps + V_target(leaf)).

    k_steps is the number of moves in the path (1..K). For each state s and
    each path of length k, we get a leaf state. The target is min over all
    such (k, leaf) pairs.

    Edge cases:
      - Leaf == solved: contributes (k, 0). Final target = k.
      - K=1: equivalent to standard Bellman.
    """
    B, S = states.shape
    n_gen = generators.shape[0]

    # Build all paths up to length K. At step k, we have B * n_gen^k leaves.
    # For B=4096 and K=2, leaves = 4096 * 24^2 = 2.4M. Manageable.
    # For K=3, leaves = 4096 * 24^3 = 56M. Too much; chunk by states.
    target_model.eval()

    # Track best (smallest) value over k=1..K paths for each input state.
    best = torch.full((B,), float("inf"), dtype=torch.float32, device=states.device)

    cur_leaves = states  # (B, S)
    leaf_to_input = torch.arange(B, device=states.device)  # which input each leaf descends from
    for k in range(1, K + 1):
        # Expand all children of current leaves: (cur, n_gen, S)
        Bcur = cur_leaves.size(0)
        children = torch.gather(
            cur_leaves.unsqueeze(1).expand(Bcur, n_gen, S),
            2,
            generators.unsqueeze(0).expand(Bcur, n_gen, S),
        )  # (Bcur, n_gen, S)
        children_flat = children.reshape(Bcur * n_gen, S)
        leaf_input_flat = leaf_to_input.repeat_interleave(n_gen)  # (Bcur*n_gen,)

        # Score each leaf with target model
        leaf_v = torch.empty(Bcur * n_gen, dtype=torch.float32, device=states.device)
        for i in range(0, Bcur * n_gen, chunk_size):
            leaf_v[i : i + chunk_size] = target_model(
                children_flat[i : i + chunk_size]
            ).flatten().to(torch.float32)
        # Solved leaves get value 0 (boundary condition).
        is_solved = (children_flat == solved_state).all(dim=1)
        leaf_v = torch.where(is_solved, torch.zeros_like(leaf_v), leaf_v)

        # Path value at depth k = k + leaf_v
        path_v = float(k) + leaf_v

        # Group by input: for each input b, take min over its paths.
        # Use scatter_reduce_ for the min.
        new_best = torch.full((B,), float("inf"), dtype=torch.float32, device=states.device)
        new_best.scatter_reduce_(0, leaf_input_flat, path_v, reduce="amin", include_self=True)
        best = torch.minimum(best, new_best)

        # Prepare for next layer (only matters if k < K).
        cur_leaves = children_flat
        leaf_to_input = leaf_input_flat

    # Clip at walk depth (upper bound) and 0.
    target = torch.minimum(best, walk_depths)
    target = torch.clamp(target, min=0.0)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--K", type=int, default=2,
                    help="lookahead horizon. K=1 is standard Bellman; K=2 is the "
                         "sensible default; K=3 is feasible only with smaller batch.")
    ap.add_argument("--n-epochs", type=int, default=300)
    ap.add_argument("--samples-per-epoch", type=int, default=200_000,
                    help="smaller than m05's 500k because K-step lookahead per batch "
                         "is O(K * 24^K) more expensive.")
    ap.add_argument("--batch-size", type=int, default=2048,
                    help="K=2 expands 24^2 = 576 leaves per state; smaller batch keeps memory bounded.")
    ap.add_argument("--k-max", type=int, default=80,
                    help="random-walk depth used to make training data (NOT lookahead K).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-update-every-epochs", type=int, default=10)
    ap.add_argument("--target-net-chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=220)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=25)
    ap.add_argument("--n-back", type=int, default=1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}, K-step lookahead = {args.K}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # Load model + warmstart
    model = ResMLPDistance(
        state_size=120, num_classes=120,
        hidden_dims=(2048, 512), num_res_blocks=2,
        encoding="embedding", embed_dim=16,
    ).to(device)
    ckpt = torch.load(args.warmstart, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"warmstarted from {args.warmstart} ({sum(p.numel() for p in model.parameters()):,} params)")

    # Target net (frozen snapshot, updated periodically)
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0,
        fused=device == "cuda",
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(device)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  batch: {args.batch_size}")
    print(f"target update every {args.target_update_every_epochs} epochs")

    for epoch in range(args.n_epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        states, depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch, device=device, n_back=args.n_back,
        )
        depths_f = depths.to(torch.float32)
        N = states.shape[0]

        model.train()
        total_loss, n_batches = 0.0, 0
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            batch_states = states[idx]
            batch_walk_depths = depths_f[idx]

            target = k_step_lookahead_targets(
                target_model, batch_states, batch_walk_depths,
                generators, solved_state,
                K=args.K, chunk_size=args.target_net_chunk,
            )

            if autocast_ctx is not None:
                with autocast_ctx:
                    pred = model(batch_states)
                    loss = F.mse_loss(pred, target)
            else:
                pred = model(batch_states)
                loss = F.mse_loss(pred, target)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        sched.step()

        avg = total_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | loss {avg:.4f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time() - t0:.1f}s",
                  flush=True)

        # Target net refresh
        if (epoch + 1) % args.target_update_every_epochs == 0:
            sd_now = model.state_dict()
            if any(k.startswith("_orig_mod.") for k in sd_now):
                sd_now = {k.removeprefix("_orig_mod."): v for k, v in sd_now.items()}
            target_model.load_state_dict(sd_now)

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.n_epochs - 1:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "loss": avg,
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": [2048, 512], "num_res_blocks": 2,
                    "encoding": "embedding", "embed_dim": 16,
                },
                "horizon_K": args.K,
                "warmstart_from": str(args.warmstart),
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
