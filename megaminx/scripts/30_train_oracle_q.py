"""T3.3 — Train Q-head on the BFS-d6 oracle Q-target dataset.

The dataset (built by 29_build_oracle_q_dataset.py) provides EXACT Q-values
for 1.38M states at d<=5 in the BFS-d6 shell. Training is pure supervised
regression: no Bellman bootstrap, no walk-depth noise, no min-bias.

Architecture: same as m23 (hidden_dims=(2048, 1024), 3 res blocks, ~12M params,
output_dim=24).

Two phases:
  Phase 1 (oracle-only): train on d<=5 states with EXACT labels.
    Tests whether the cluster ceiling is mechanism-driven (binding) or
    capacity-driven (not).
  Phase 2 (oracle + walk-state distillation, OPTIONAL): mix in random-walk
    states with m05's Q-values as soft targets, to extend coverage to d>5.
    This is closer to a deployable Q-shortlister.

This script implements Phase 1. Phase 2 is a follow-up.

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/30_train_oracle_q.py \\
      --out megaminx/models/m35_oracle_q --epochs 500
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=PROJECT / "data" / "oracle_q_d5.pt")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory for checkpoints")
    ap.add_argument("--hidden-dims", default="2048,1024")
    ap.add_argument("--num-res-blocks", type=int, default=3)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"loading dataset: {args.dataset}")
    d = torch.load(args.dataset, weights_only=False)
    states = d["states"]      # (N, 120) int8
    q_targets = d["q_targets"]  # (N, 24) int8
    depths = d["depths"]       # (N,) int8
    n_total = states.size(0)
    n_gen = q_targets.size(1)
    print(f"  states: {tuple(states.shape)}, q_targets: {tuple(q_targets.shape)}")
    print(f"  depth distribution: {torch.bincount(depths.long()).tolist()}")

    # Move to device
    print(f"loading data to {args.device} ...")
    states = states.to(args.device).long()        # int8 -> long for embedding
    q_targets = q_targets.to(args.device).float()  # int8 -> float for loss
    depths = depths.to(args.device)
    valid_mask = (q_targets >= 0)  # boolean (N, 24)
    print(f"  valid Q-targets: {valid_mask.sum().item():,} / {valid_mask.numel():,}")

    # Build model
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    hidden = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPDistance(
        state_size=120,
        num_classes=120,
        hidden_dims=hidden,
        num_res_blocks=args.num_res_blocks,
        encoding="embedding",
        embed_dim=args.embed_dim,
        output_dim=n_gen,  # 24-output Q-head
    ).to(args.device)
    print(f"model params: {model.num_parameters():,}")

    if args.compile and args.device == "cuda":
        model_compiled = torch.compile(model, dynamic=False)
    else:
        model_compiled = model

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay, fused=(args.device == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    use_amp = args.bf16 and args.device == "cuda"
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cuda", enabled=False)

    n_batches_per_epoch = (n_total + args.batch_size - 1) // args.batch_size
    print(f"\ntraining: {args.epochs} epochs × {n_batches_per_epoch} batches "
          f"({n_total:,} samples / {args.batch_size} batch_size)")

    best_loss = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        model_compiled.train()
        # Shuffle
        perm = torch.randperm(n_total, device=args.device)
        total_loss = 0.0
        n_loss_terms = 0

        for b in range(n_batches_per_epoch):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            bs = states[idx]              # (B, 120) long
            bq = q_targets[idx]            # (B, 24) float
            bm = valid_mask[idx]           # (B, 24) bool
            with autocast_ctx:
                pred = model_compiled(bs)  # (B, 24) float
                # MSE loss on valid Q-targets only
                err = (pred - bq) ** 2
                loss = (err * bm.float()).sum() / bm.float().sum().clamp(min=1)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * bm.sum().item()
            n_loss_terms += bm.sum().item()
        scheduler.step()
        avg_loss = total_loss / max(n_loss_terms, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch:4d} | loss {avg_loss:.5f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{time.time()-t0:.1f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = args.out / f"epoch_{epoch:04d}.pt"
            base = getattr(model_compiled, "_orig_mod", model_compiled)
            model_cfg = {
                "state_size": 120,
                "num_classes": 120,
                "hidden_dims": list(hidden),
                "num_res_blocks": args.num_res_blocks,
                "encoding": "embedding",
                "embed_dim": args.embed_dim,
                "output_dim": n_gen,
            }
            sd = base.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": sd,
                "loss": avg_loss,
                "model_config": model_cfg,
                "training_config": {
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "seed": args.seed,
                },
            }, ckpt_path)

    print(f"\nfinal loss: {best_loss:.5f}")
    print(f"checkpoints in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
