"""m_curr_v4_hardmine: Curriculum trainer with hard-state mining.

Same recipe as 48_train_curriculum.py except:
  - After each epoch, identify the top-K worst-predicted states (highest |V_pred - target|).
  - Maintain a circular FIFO hard-state buffer.
  - Each subsequent epoch: HARDMINE_FRACTION of the training batch is sampled
    from this buffer (instead of fresh walks).

Hypothesis: oversampling states where V is currently wrong should pull more
training capacity to those states. May improve V calibration on the long-tail
states beam search reaches at depth.

Honest expectation per the cluster-ceiling finding (megaminx_gotchas.md):
information-bound, not optimization-bound. Hard-mining is a sampling change;
likely also lands in cluster. But cheap to try.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/53_train_curriculum_hardmine.py \\
        --warmstart megaminx/models/m_curr_v0/best_ema.pt \\
        --out-dir megaminx/models/m_curr_v4_hardmine \\
        --n-epochs 5000 --warmup-epochs 0 \\
        --hardmine-fraction 0.15 --hardmine-buffer 200000 --hardmine-topk 10000
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
def ema_update(ema_model, source, tau):
    for ep, sp in zip(ema_model.parameters(), source.parameters()):
        ep.mul_(1.0 - tau).add_(sp.detach(), alpha=tau)
    for eb, sb in zip(ema_model.buffers(), source.buffers()):
        if eb.dtype.is_floating_point:
            eb.mul_(1.0 - tau).add_(sb.detach(), alpha=tau)
        else:
            eb.copy_(sb)


@torch.no_grad()
def eval_val_mse(model, val_states, val_depths, batch_size=8192):
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


class HardBuffer:
    """Circular FIFO buffer for hard (state, depth) pairs."""
    def __init__(self, capacity, state_size, device):
        self.capacity = capacity
        self.states = torch.zeros((capacity, state_size), dtype=torch.int8, device=device)
        self.depths = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.size = 0
        self.write_ptr = 0

    def add(self, new_states, new_depths):
        n = new_states.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            # Just keep last `capacity` entries
            self.states.copy_(new_states[-self.capacity:])
            self.depths.copy_(new_depths[-self.capacity:])
            self.size = self.capacity
            self.write_ptr = 0
            return
        end = self.write_ptr + n
        if end <= self.capacity:
            self.states[self.write_ptr : end] = new_states
            self.depths[self.write_ptr : end] = new_depths
        else:
            # Wrap around
            first_chunk = self.capacity - self.write_ptr
            self.states[self.write_ptr :] = new_states[:first_chunk]
            self.depths[self.write_ptr :] = new_depths[:first_chunk]
            second = n - first_chunk
            self.states[:second] = new_states[first_chunk:]
            self.depths[:second] = new_depths[first_chunk:]
        self.write_ptr = (self.write_ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, n, generator=None):
        if self.size == 0:
            return None, None
        idx = torch.randint(0, self.size, (n,), generator=generator, device=self.states.device)
        return self.states[idx], self.depths[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=5_000)
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="warmup at warmup-k-max; 0 = skip directly to mix-k")
    ap.add_argument("--patience", type=int, default=1_500)
    ap.add_argument("--val-every-epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=500)
    ap.add_argument("--ema-tau", type=float, default=0.001)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--warmup-k-max", type=int, default=35)
    ap.add_argument("--mix-k-list", default="50,70,80,100")
    ap.add_argument("--val-k-max", type=int, default=80)
    ap.add_argument("--val-n-walks", type=int, default=2000)
    # Hard-mining specific
    ap.add_argument("--hardmine-fraction", type=float, default=0.15,
                    help="fraction of each epoch's batch sampled from hard buffer")
    ap.add_argument("--hardmine-buffer", type=int, default=200_000,
                    help="capacity of hard buffer")
    ap.add_argument("--hardmine-topk", type=int, default=10_000,
                    help="top-K worst predicted states added to buffer per epoch")
    ap.add_argument("--hardmine-warmup-epochs", type=int, default=200,
                    help="don't start hard-mining until this many epochs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = 120

    model = ResMLPDistance(
        state_size=state_size, num_classes=120,
        hidden_dims=(2048, 512), num_res_blocks=2,
        encoding="embedding", embed_dim=16, output_dim=1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params", flush=True)

    ckpt = torch.load(args.warmstart, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"warmstart loaded from {args.warmstart}", flush=True)

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
    hard_gen = torch.Generator(device=device).manual_seed(args.seed + 13)

    mix_k_list = [int(x) for x in args.mix_k_list.split(",") if x.strip()]
    print(f"curriculum: warmup k_max={args.warmup_k_max} for {args.warmup_epochs} epochs, "
          f"then mix from {mix_k_list}", flush=True)
    print(f"hardmine: fraction={args.hardmine_fraction}, buffer={args.hardmine_buffer:,}, "
          f"topk={args.hardmine_topk:,}, start_at_epoch={args.hardmine_warmup_epochs}",
          flush=True)

    print(f"building validation set ({args.val_n_walks} walks, k_max={args.val_k_max})...",
          flush=True)
    val_states, val_depths = generate_walks_torch(
        puzzle, n_walks=args.val_n_walks, k_max=args.val_k_max,
        seed=args.seed - 1, device=device, n_back=args.n_back,
    )
    val_depths = val_depths.to(torch.float32)
    print(f"  val: {val_states.size(0):,} states", flush=True)

    hard_buffer = HardBuffer(args.hardmine_buffer, state_size, device)

    print(f"epochs: {args.n_epochs}  patience: {args.patience}  "
          f"val_every: {args.val_every_epochs}  ema_tau: {args.ema_tau}", flush=True)

    best_val = float("inf")
    best_epoch = 0

    for epoch in range(args.n_epochs):
        t0 = time.time()
        if epoch < args.warmup_epochs:
            k_for_epoch = args.warmup_k_max
            k_label = f"k={k_for_epoch}"
        else:
            idx = int(torch.randint(0, len(mix_k_list), (1,), generator=k_gen, device=device).item())
            k_for_epoch = mix_k_list[idx]
            k_label = f"k={k_for_epoch}(mix)"

        # Generate fresh walks
        n_walks = max(1, args.samples_per_epoch // k_for_epoch)
        states, depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=k_for_epoch,
            seed=args.seed + epoch, device=device, n_back=args.n_back,
        )
        depths_f = depths.to(torch.float32)
        N = states.shape[0]

        # If hard-mining active: replace a fraction of fresh states with hard samples
        if (epoch >= args.hardmine_warmup_epochs and
                hard_buffer.size > 0 and args.hardmine_fraction > 0):
            n_hard = min(int(args.hardmine_fraction * N), hard_buffer.size)
            if n_hard > 0:
                hard_states_sample, hard_depths_sample = hard_buffer.sample(n_hard, hard_gen)
                # Replace last n_hard slots
                states[-n_hard:] = hard_states_sample
                depths_f[-n_hard:] = hard_depths_sample

        model.train()
        total_loss, n_batches = 0.0, 0
        # Track per-sample loss for this epoch (in `perm` order)
        per_sample_losses = torch.empty(N, dtype=torch.float32, device=device)
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx_batch = perm[i : i + args.batch_size]
            bs = states[idx_batch]
            bd = depths_f[idx_batch]
            if autocast_ctx is not None:
                with autocast_ctx:
                    pred = model(bs)
                    per_sample_loss = (pred - bd) ** 2
                    loss = per_sample_loss.mean()
            else:
                pred = model(bs)
                per_sample_loss = (pred - bd) ** 2
                loss = per_sample_loss.mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            ema_update(ema_model, model, args.ema_tau)
            total_loss += float(loss.item())
            n_batches += 1
            # Record per-sample loss (in perm order)
            per_sample_losses[i : i + bs.size(0)] = per_sample_loss.detach().float()
        sched.step()

        avg_loss = total_loss / max(n_batches, 1)

        # End of epoch: identify top-K worst, add to buffer.
        # per_sample_losses[k] corresponds to states[perm[k]].
        if args.hardmine_topk > 0:
            k_top = min(args.hardmine_topk, N)
            top_in_perm = torch.topk(per_sample_losses, k_top).indices
            hard_state_idx = perm[top_in_perm]
            new_hard_states = states[hard_state_idx]
            new_hard_depths = depths_f[hard_state_idx]
            hard_buffer.add(new_hard_states, new_hard_depths)

        if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.n_epochs - 1:
            val_mse_train = eval_val_mse(model, val_states, val_depths)
            val_mse_ema = eval_val_mse(ema_model, val_states, val_depths)
            print(f"epoch {epoch:5d} | {k_label:<12} | train_mse {avg_loss:7.3f} | "
                  f"val_mse train {val_mse_train:7.3f} | val_mse ema {val_mse_ema:7.3f} | "
                  f"buf_size={hard_buffer.size:>7} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time()-t0:.1f}s",
                  flush=True)
            if val_mse_ema < best_val:
                best_val = val_mse_ema
                best_epoch = epoch
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
                    "hardmine_fraction": args.hardmine_fraction,
                    "hardmine_buffer": args.hardmine_buffer,
                    "hardmine_topk": args.hardmine_topk,
                }, ckpt_path)
                print(f"  -> new best EMA: val_mse={val_mse_ema:.4f}, saved {ckpt_path.name}",
                      flush=True)
            else:
                early_stop_grace = args.warmup_epochs + args.patience
                if (epoch - best_epoch >= args.patience and epoch >= early_stop_grace):
                    print(f"\n*** early stopping at epoch {epoch}: no val_mse improvement "
                          f"for {args.patience} epochs (best at epoch {best_epoch}, "
                          f"val_mse={best_val:.4f})", flush=True)
                    break

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

    print(f"\nfinal: best val_mse={best_val:.4f} at epoch {best_epoch}; saved best_ema.pt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
