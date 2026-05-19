"""m38 — Train V with listwise rank loss as primary objective.

Beam search at every step uses ONLY the relative ordering of children. We've
been training V to predict absolute distances (noisy upper bound from random
walks). m38 shifts the objective to ORDERING: model is trained such that for
each state s, its 24 children's V values are correctly ranked by true distance.

This decouples the absolute-fit problem (noisy walk-depth labels, label scale
bias) from the ranking problem (what beam actually uses).

Recipe:
  - Warmstart from m07 (V baseline)
  - Bellman target as the "true distance" reference for child ordering
  - Total loss = MSE + lambda * listwise_loss (both on Bellman targets)
  - lambda annealed: starts low (relies on MSE for stability), grows over training

Two ablation knobs:
  - --listwise-mode {listnet, pairwise} — different rank-loss formulations
  - --listwise-lambda — weighting

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/36_train_v_listwise.py \\
      --warmstart megaminx/models/m07_big_k80/epoch_3999.pt \\
      --out megaminx/models/m38_listwise --epochs 500 \\
      --listwise-lambda 0.5 --listwise-mode listnet
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
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.bellman import _apply_all_generators, _bellman_targets
from cayley.data import GeneratorTable, generate_walks_torch
from cayley.listwise_loss import listnet_loss, pairwise_hinge_loss
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path,
                    default=PROJECT / "models" / "m07_big_k80" / "epoch_3999.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden-dims", default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=4096,
                    help="Smaller than m05 since each batch needs 24 child forwards.")
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--target-net-chunk", type=int, default=4096)
    ap.add_argument("--target-update-every-epochs", type=int, default=10)
    ap.add_argument("--listwise-lambda", type=float, default=0.5,
                    help="Weight on listwise loss term. 0 = pure MSE Bellman; >0 adds rank loss.")
    ap.add_argument("--listwise-mode", choices=["listnet", "pairwise"], default="listnet")
    ap.add_argument("--listwise-temperature", type=float, default=1.0,
                    help="ListNet temperature (smaller = sharper distribution).")
    ap.add_argument("--listwise-warmup-epochs", type=int, default=50,
                    help="Linear ramp lambda from 0 to listwise-lambda over this many epochs.")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=38)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"loaded {len(puzzle.move_names)} generators")

    hidden = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPDistance(
        state_size=120,
        num_classes=120,
        hidden_dims=hidden,
        num_res_blocks=args.num_res_blocks,
        encoding="embedding",
        embed_dim=args.embed_dim,
        output_dim=1,
    ).to(args.device)
    print(f"model params: {model.num_parameters():,}")

    # Warmstart
    if args.warmstart and args.warmstart.exists():
        print(f"warmstart from {args.warmstart}")
        ckpt = torch.load(args.warmstart, map_location=args.device, weights_only=False)
        sd = ckpt["state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        print(f"NO warmstart")

    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    if args.compile and args.device == "cuda":
        model_compiled = torch.compile(model, dynamic=False)
    else:
        model_compiled = model

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay,
                               fused=(args.device == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(args.device)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=args.device)

    use_amp = args.bf16 and args.device == "cuda"
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cuda", enabled=False)

    print(f"\ntraining: {args.epochs} ep, batch {args.batch_size}, "
          f"listwise lambda {args.listwise_lambda} ({args.listwise_mode})")

    n_walks = max(1, args.samples_per_epoch // args.k_max)

    for epoch in range(args.epochs):
        t0 = time.time()
        # Lambda warmup
        lam = args.listwise_lambda * min(1.0, epoch / max(args.listwise_warmup_epochs, 1))

        states, depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max, seed=args.seed + epoch,
            device=args.device, n_back=args.n_back,
        )
        depths_f = depths.to(torch.float32)

        model_compiled.train()
        perm = torch.randperm(states.size(0), device=args.device)
        total_mse = 0.0
        total_lw = 0.0
        n_batches = 0

        for b in range(0, states.size(0), args.batch_size):
            idx = perm[b : b + args.batch_size]
            bs = states[idx]                     # (B, 120)
            bd = depths_f[idx]                   # (B,)

            # Compute Bellman target for parent V
            bell_target = _bellman_targets(
                target_model, bs, bd, generators, solved_state,
                chunk_size=args.target_net_chunk,
                clip_upper=True, clip_lower=True,
                softmin_temperature=0.0,
            )

            # Compute target Q for each of 24 children (for listwise rank loss)
            B, S = bs.shape
            n_gen = generators.shape[0]
            children = _apply_all_generators(bs, generators)  # (B, n_gen, S)
            children_flat = children.reshape(B * n_gen, S)

            # Target net forward on children -> child V values (target Q for each action)
            with torch.no_grad():
                target_model.eval()
                child_q = torch.empty(B * n_gen, dtype=torch.float32, device=bs.device)
                for i in range(0, B * n_gen, args.target_net_chunk):
                    vals = target_model(children_flat[i : i + args.target_net_chunk]).flatten().to(torch.float32)
                    child_q[i : i + args.target_net_chunk] = vals
                child_q = child_q.view(B, n_gen)
                # Solved children get value 0
                is_solved = (children_flat == solved_state).all(dim=1).view(B, n_gen)
                child_q = torch.where(is_solved, torch.zeros_like(child_q), child_q)

            with autocast_ctx:
                # Parent V prediction (MSE on Bellman target)
                pred = model_compiled(bs).flatten()
                mse_loss = F.mse_loss(pred, bell_target)

                # Children V predictions (for listwise loss)
                if lam > 0:
                    pred_children = model_compiled(children_flat).view(B, n_gen).float()
                    if args.listwise_mode == "listnet":
                        lw_loss = listnet_loss(pred_children, child_q,
                                                temperature=args.listwise_temperature)
                    else:
                        lw_loss = pairwise_hinge_loss(pred_children, child_q, margin=0.5)
                    total_loss = mse_loss + lam * lw_loss
                else:
                    lw_loss = torch.tensor(0.0, device=bs.device)
                    total_loss = mse_loss

            optim.zero_grad(set_to_none=True)
            total_loss.backward()
            optim.step()
            total_mse += float(mse_loss.item())
            total_lw += float(lw_loss.item()) if lam > 0 else 0
            n_batches += 1

        scheduler.step()
        avg_mse = total_mse / max(n_batches, 1)
        avg_lw = total_lw / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch:4d} | mse {avg_mse:.4f} | listwise {avg_lw:.4f} | "
                  f"lambda {lam:.3f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{time.time()-t0:.1f}s", flush=True)

        # Refresh target net every N epochs
        if (epoch + 1) % args.target_update_every_epochs == 0:
            src_sd = model.state_dict()
            if any(k.startswith("_orig_mod.") for k in src_sd):
                src_sd = {k.removeprefix("_orig_mod."): v for k, v in src_sd.items()}
            target_model.load_state_dict(src_sd)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = args.out / f"epoch_{epoch:04d}.pt"
            sd = model.state_dict()
            if any(k.startswith("_orig_mod.") for k in sd):
                sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            torch.save({
                "epoch": epoch,
                "state_dict": sd,
                "loss": avg_mse,
                "listwise_loss": avg_lw,
                "model_config": {
                    "state_size": 120,
                    "num_classes": 120,
                    "hidden_dims": list(hidden),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "embedding",
                    "embed_dim": args.embed_dim,
                    "output_dim": 1,
                },
                "training_config": {
                    "listwise_mode": args.listwise_mode,
                    "listwise_lambda": args.listwise_lambda,
                    "listwise_warmup_epochs": args.listwise_warmup_epochs,
                    "warmstart": str(args.warmstart) if args.warmstart else "none",
                    "epochs": args.epochs,
                    "seed": args.seed,
                },
            }, ckpt_path)

    print(f"\nfinal mse: {avg_mse:.5f}, final listwise: {avg_lw:.5f}")
    print(f"checkpoints in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
