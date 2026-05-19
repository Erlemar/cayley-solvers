"""m_curr: Curriculum k_max trainer (colleague-style, Option C).

Recipe:
  - ResMLPDistance, 6M params (matches m07/m05 arch), warmstart from m07
  - Walk-depth MSE target
  - AdamW + cosine LR
  - EMA (exponential moving average, tau per step)
  - Curriculum k_max:
      * first WARMUP_EPOCHS at k_max=35 (warmup, near-solved focus)
      * after warmup: mixed sampling from {50, 70, 80, 100} per batch
  - Early stopping: stop if val_mse hasn't improved for PATIENCE epochs
  - Max n_epochs (default 50K) with periodic validation

EMA snapshot saved as the "production" checkpoint at end of training.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/48_train_curriculum.py \\
        --warmstart megaminx/models/m07_big_k80/epoch_3999.pt \\
        --out-dir megaminx/models/m_curr_v0 \\
        --n-epochs 50000 --warmup-epochs 5000 --patience 1000
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import generate_walks_torch
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


@torch.no_grad()
def ema_update(ema_model: nn.Module, source: nn.Module, tau: float) -> None:
    """In-place EMA update: ema_p = (1-tau) * ema_p + tau * source_p."""
    for ep, sp in zip(ema_model.parameters(), source.parameters()):
        ep.mul_(1.0 - tau).add_(sp.detach(), alpha=tau)
    for eb, sb in zip(ema_model.buffers(), source.buffers()):
        if eb.dtype.is_floating_point:
            eb.mul_(1.0 - tau).add_(sb.detach(), alpha=tau)
        else:
            eb.copy_(sb)


@torch.no_grad()
def eval_val_mse(model: nn.Module, val_states: torch.Tensor, val_depths: torch.Tensor,
                 batch_size: int = 8192) -> float:
    """MSE on a fixed validation set."""
    model.eval()
    n = val_states.size(0)
    total = 0.0
    seen = 0
    for i in range(0, n, batch_size):
        bs = val_states[i : i + batch_size]
        bd = val_depths[i : i + batch_size]
        pred = model(bs)
        total += float(F.mse_loss(pred, bd, reduction="sum").item())
        seen += bs.size(0)
    return total / max(seen, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path, required=True,
                    help="m07-class V-model checkpoint to warmstart from")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=50_000,
                    help="max epochs (early stopping kicks in earlier)")
    ap.add_argument("--warmup-epochs", type=int, default=5_000,
                    help="number of warmup epochs at warmup k_max")
    ap.add_argument("--patience", type=int, default=1_000,
                    help="early stopping: stop if val_mse no improvement for this many epochs")
    ap.add_argument("--val-every-epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=480)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=500)
    ap.add_argument("--ema-tau", type=float, default=0.001,
                    help="EMA per-step decay; effective lag ≈ 1/tau steps")
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--warmup-k-max", type=int, default=35)
    ap.add_argument("--mix-k-list", default="50,70,80,100",
                    help="post-warmup k_max values to sample from each batch")
    ap.add_argument("--val-k-max", type=int, default=80,
                    help="k_max used to generate the held-out validation set")
    ap.add_argument("--val-n-walks", type=int, default=2000)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")

    # Build model: matches m07 arch exactly.
    model = ResMLPDistance(
        state_size=120, num_classes=120,
        hidden_dims=(2048, 512), num_res_blocks=2,
        encoding="embedding", embed_dim=16, output_dim=1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params")

    # Warmstart from m07.
    ckpt = torch.load(args.warmstart, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"warmstart loaded from {args.warmstart}")

    # EMA snapshot (initialized to copy of warmstart).
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else None)
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)
    k_gen = torch.Generator(device=device).manual_seed(args.seed + 7777)

    mix_k_list = [int(x) for x in args.mix_k_list.split(",") if x.strip()]
    print(f"curriculum: warmup k_max={args.warmup_k_max} for {args.warmup_epochs} epochs, "
          f"then mix from {mix_k_list}")

    # Build a FIXED validation set (drawn from k_max=val_k_max with fixed seed).
    print(f"building validation set ({args.val_n_walks} walks, k_max={args.val_k_max})...",
          flush=True)
    val_states, val_depths = generate_walks_torch(
        puzzle, n_walks=args.val_n_walks, k_max=args.val_k_max,
        seed=args.seed - 1, device=device, n_back=args.n_back,
    )
    val_depths = val_depths.to(torch.float32)
    print(f"  val: {val_states.size(0):,} states")

    print(f"epochs: {args.n_epochs}  patience: {args.patience}  "
          f"val_every: {args.val_every_epochs}  ema_tau: {args.ema_tau}")

    best_val = float("inf")
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(args.n_epochs):
        t0 = time.time()
        # Choose k_max for this epoch.
        if epoch < args.warmup_epochs:
            k_for_epoch = args.warmup_k_max
            k_label = f"k={k_for_epoch}"
        else:
            # Pick a k_max from the mix uniformly per epoch
            idx = int(torch.randint(0, len(mix_k_list), (1,), generator=k_gen, device=device).item())
            k_for_epoch = mix_k_list[idx]
            k_label = f"k={k_for_epoch}(mix)"

        n_walks = max(1, args.samples_per_epoch // k_for_epoch)
        states, depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=k_for_epoch,
            seed=args.seed + epoch, device=device, n_back=args.n_back,
        )
        depths_f = depths.to(torch.float32)
        N = states.shape[0]

        model.train()
        total_loss, n_batches = 0.0, 0
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs = states[idx]
            bd = depths_f[idx]
            if autocast_ctx is not None:
                with autocast_ctx:
                    pred = model(bs)
                    loss = F.mse_loss(pred, bd)
            else:
                pred = model(bs)
                loss = F.mse_loss(pred, bd)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            ema_update(ema_model, model, args.ema_tau)
            total_loss += float(loss.item())
            n_batches += 1
        sched.step()

        avg_loss = total_loss / max(n_batches, 1)

        # Periodic validation on EMA model.
        if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.n_epochs - 1:
            val_mse_train = eval_val_mse(model, val_states, val_depths)
            val_mse_ema = eval_val_mse(ema_model, val_states, val_depths)
            print(f"epoch {epoch:5d} | {k_label:<12} | train_mse {avg_loss:7.3f} | "
                  f"val_mse train {val_mse_train:7.3f} | val_mse ema {val_mse_ema:7.3f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time()-t0:.1f}s",
                  flush=True)
            if val_mse_ema < best_val:
                best_val = val_mse_ema
                best_epoch = epoch
                epochs_since_improve = 0
                # Save best EMA checkpoint.
                ckpt_path = args.out_dir / "best_ema.pt"
                torch.save({
                    "epoch": epoch,
                    "state_dict": ema_model.state_dict(),
                    "val_mse": val_mse_ema,
                    "model_config": {
                        "state_size": 120, "num_classes": 120,
                        "hidden_dims": [2048, 512], "num_res_blocks": 2,
                        "encoding": "embedding", "embed_dim": 16, "output_dim": 1,
                    },
                    "warmstart": str(args.warmstart),
                    "k_for_epoch": k_for_epoch,
                    "warmup_epochs": args.warmup_epochs,
                    "mix_k_list": mix_k_list,
                }, ckpt_path)
                print(f"  -> new best EMA: val_mse={val_mse_ema:.4f}, saved {ckpt_path.name}",
                      flush=True)
            else:
                epochs_since_improve = epoch - best_epoch
                # Don't early-stop during warmup or until at least a full patience window
                # past the warmup transition (the model needs time to adapt to mixed-k).
                early_stop_grace_epoch = args.warmup_epochs + args.patience
                if (epochs_since_improve >= args.patience
                        and epoch >= early_stop_grace_epoch):
                    print(f"\n*** early stopping at epoch {epoch}: no val_mse improvement "
                          f"for {args.patience} epochs (best at epoch {best_epoch}, "
                          f"val_mse={best_val:.4f})", flush=True)
                    break

        # Periodic raw checkpoint (non-EMA, for safety).
        if (epoch + 1) % args.checkpoint_every_epochs == 0:
            ckpt_path = args.out_dir / f"epoch_{epoch:05d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": [2048, 512], "num_res_blocks": 2,
                    "encoding": "embedding", "embed_dim": 16, "output_dim": 1,
                },
                "best_val": best_val,
                "best_epoch": best_epoch,
            }, ckpt_path)

    print(f"\nfinal: best val_mse={best_val:.4f} at epoch {best_epoch}; saved best_ema.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
