"""m37 — Train V on solver-trace as the PRIMARY signal (no Bellman, no walk-depth).

Tests: "is the training/inference distribution mismatch the binding constraint?"

Data: megaminx/data/solver_trace_train.pt — 128,266 (state, true_remaining_d)
pairs mined from successful submissions. Distribution mean d=45, range 1-98.
States are exactly the kind beam-search visits at solve time.

Recipe:
  - Warmstart from m07 (gives the model some V baseline to refine)
  - Pure MSE on solver-trace, no Bellman bootstrap, no walk-depth supplement
  - 500 epochs, lr 5e-4 (matches m05's Bellman LR), batch 8192

Hypothesis:
  - If m37 beats m05 on strat-5 → distribution alignment IS the lever, cluster
    ceiling is recipe-bound (we trained on wrong distribution).
  - If m37 ties or worse → distribution alignment alone isn't enough; the
    label-noise floor or other constraints bind.

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/35_train_v_solver_trace.py \\
      --warmstart megaminx/models/m07_big_k80/epoch_3999.pt \\
      --out megaminx/models/m37_solver_trace --epochs 500
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
                    default=PROJECT / "data" / "solver_trace_train.pt")
    ap.add_argument("--warmstart", type=Path,
                    default=PROJECT / "models" / "m07_big_k80" / "epoch_3999.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden-dims", default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--compile", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=37)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"loading dataset: {args.dataset}")
    d = torch.load(args.dataset, weights_only=False)
    states = d["states"]      # (N, 120) int8
    distances = d["distances"]  # (N,) int8
    n_total = states.size(0)
    print(f"  states: {tuple(states.shape)}, distances: {tuple(distances.shape)}")
    print(f"  distance: mean={distances.float().mean():.2f}, "
          f"min={distances.min().item()}, max={distances.max().item()}")

    print(f"loading data to {args.device} ...")
    states = states.to(args.device).long()        # for embedding
    distances = distances.to(args.device).float()  # for MSE

    # Build model
    hidden = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPDistance(
        state_size=120,
        num_classes=120,
        hidden_dims=hidden,
        num_res_blocks=args.num_res_blocks,
        encoding="embedding",
        embed_dim=args.embed_dim,
        output_dim=1,  # V model
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
        print(f"NO warmstart (training from scratch)")

    if args.compile and args.device == "cuda":
        model_compiled = torch.compile(model, dynamic=False)
    else:
        model_compiled = model

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay,
                               fused=(args.device == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    use_amp = args.bf16 and args.device == "cuda"
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp else torch.amp.autocast("cuda", enabled=False)

    n_batches = (n_total + args.batch_size - 1) // args.batch_size
    print(f"\ntraining: {args.epochs} epochs x {n_batches} batches "
          f"({n_total:,} samples / {args.batch_size} batch)")

    for epoch in range(args.epochs):
        t0 = time.time()
        model_compiled.train()
        perm = torch.randperm(n_total, device=args.device)
        total_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            bs = states[idx]
            bd = distances[idx]
            with autocast_ctx:
                pred = model_compiled(bs).flatten()
                loss = F.mse_loss(pred, bd)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch:4d} | loss {avg_loss:.5f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | "
                  f"{time.time()-t0:.1f}s", flush=True)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            ckpt_path = args.out / f"epoch_{epoch:04d}.pt"
            base = getattr(model_compiled, "_orig_mod", model_compiled)
            sd = base.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": sd,
                "loss": avg_loss,
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
                    "lr": args.lr,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "warmstart": str(args.warmstart) if args.warmstart else "none",
                },
            }, ckpt_path)

    print(f"\nfinal loss: {avg_loss:.5f}")
    print(f"checkpoints in {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
