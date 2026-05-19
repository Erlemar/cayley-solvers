"""m21: train a 'distance-to-shell' value model.

Standard m05 predicts distance to solved (walk_depth target, clamped via Bellman).
m21 instead predicts `max(walk_depth - SHELL_DEPTH, 0)` — distance to the BFS-d6
shell. Combined with MITM beam termination, the search only needs to reach the
shell; the BFS path covers the last d steps exactly.

Two wins:
  1. Smaller dynamic range to learn (distances mostly ≤ ~70 instead of ~80).
  2. Model doesn't waste capacity on the d≤6 region we already have exact answers for.

Recipe: warm-start from m05 (already strong), retrain with shifted target. Same
arch [2048, 512]×2, 500 epochs, lr 5e-4. ~30 min on a 4090.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/08_train_m21_shell.py \
        --warmstart megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --shell-depth 6 \
        --out-dir megaminx/models/m21_shell_d6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import generate_walks_torch
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path, required=True,
                    help="path to m05 (or other) checkpoint to warmstart from")
    ap.add_argument("--shell-depth", type=int, default=6,
                    help="subtract this from walk_depth in target (clamped at 0). "
                         "matches the BFS-d shell depth used at search time.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=500)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=210)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=50)
    ap.add_argument("--n-back", type=int, default=1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # Build model + load warmstart weights.
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

    # Optimizer + cosine schedule.
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0,
        fused=device == "cuda",
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"target = max(walk_depth - {args.shell_depth}, 0)")
    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  batch: {args.batch_size}")
    print(f"output dir: {args.out_dir}")

    for epoch in range(args.n_epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        states, depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch, device=device, n_back=args.n_back,
        )
        # KEY DIFFERENCE FROM m05: target is shifted toward shell, not solved.
        targets = torch.clamp(depths.to(torch.float32) - args.shell_depth, min=0.0)

        model.train()
        N = states.shape[0]
        perm = torch.randperm(N, generator=batch_gen, device=device)
        total_loss, n_batches = 0.0, 0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            if autocast_ctx is not None:
                with autocast_ctx:
                    pred = model(states[idx])
                    loss = F.mse_loss(pred, targets[idx])
            else:
                pred = model(states[idx])
                loss = F.mse_loss(pred, targets[idx])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        sched.step()

        avg = total_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | loss {avg:.4f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time() - t0:.1f}s",
                  flush=True)

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
                "shell_depth": args.shell_depth,
                "warmstart_from": str(args.warmstart),
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
