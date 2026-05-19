"""m46 — Transformer-encoder distance predictor for Megaminx.

Different inductive bias vs the MLP family that's exhausted the cluster ceiling
(m05-m45, all 12+ variants landed at strat-5 mean 88-99). Transformer attention
operates over the 120 sticker tokens directly; if cluster is bounded by MLP's
feature-mixing pattern, transformer might break through.

Architecture: 8-layer encoder, d_model=512, n_heads=8, ffn=2048, ~30M params.

Phase 1 (this script): walk-depth pretrain. 1000 epochs, k_max=80, MSE loss,
warmup + cosine LR. After convergence, kick off Bellman warmstart separately
via 05_bellman_refine.py with this checkpoint as warmstart_path.

Wall estimate on 4090: 30-50s/epoch -> 8-14h for 1000 epochs.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/47_train_m46_transformer.py \\
        --out-dir megaminx/models/m46_transformer \\
        --n-epochs 1000
"""
from __future__ import annotations

import argparse
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
from megaminx.puzzle import Megaminx


class TransformerDistance(nn.Module):
    """Transformer encoder predicting scalar distance for a 120-element state.

    Built to plug into the same training+Bellman pipeline used for ResMLPDistance:
    same forward signature `(B, state_size) -> (B,)`. Exposes `output_dim=1` so
    the auto-detect in 03_solve.py picks the V-path correctly.
    """

    def __init__(
        self,
        state_size: int = 120,
        num_classes: int = 120,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        # Match ResMLPDistance config field names so 03_solve.py auto-loader works.
        self.encoding = "transformer"
        self.embed_dim = d_model
        self.output_dim = 1
        self.token_emb = nn.Embedding(num_classes, d_model)
        self.pos_emb = nn.Embedding(state_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="relu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("positions", torch.arange(state_size).unsqueeze(0))
        # Optional knob for callers that want to disable internal chunking.
        self.inference_chunk_size = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tok = self.token_emb(x.long())                                # (B, S, D)
        pos = self.pos_emb(self.positions.expand(B, self.state_size))  # (B, S, D)
        h = tok + pos
        h = self.encoder(h)                                            # (B, S, D)
        h = h.mean(dim=1)                                              # (B, D)
        return self.head(h).squeeze(-1)                                # (B,)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=1000)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=460)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=50)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=2048)
    ap.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")

    model = TransformerDistance(
        state_size=120, num_classes=120,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, ffn_dim=args.ffn_dim,
        dropout=0.0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"m46 transformer: {n_params:,} params "
          f"(d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}, "
          f"ffn={args.ffn_dim})")

    if args.compile_model and device == "cuda":
        model_compiled = torch.compile(model, dynamic=False)
    else:
        model_compiled = model

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        fused=device == "cuda",
    )
    # Linear warmup + cosine decay
    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        # cosine decay over remaining
        progress = (step - args.warmup_steps) / max(1, args.n_epochs - args.warmup_steps)
        progress = min(1.0, progress)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" and args.amp else None)
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  "
          f"batch: {args.batch_size}  lr: {args.lr}  warmup: {args.warmup_steps}")

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
            bs = states[idx]
            bd = depths_f[idx]
            if autocast_ctx is not None:
                with autocast_ctx:
                    pred = model_compiled(bs)
                    loss = F.mse_loss(pred, bd)
            else:
                pred = model_compiled(bs)
                loss = F.mse_loss(pred, bd)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batches += 1
        sched.step()

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | MSE {avg_loss:.4f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time()-t0:.1f}s",
                  flush=True)

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.n_epochs - 1:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            save_sd = model.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": save_sd,
                "loss": avg_loss,
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "encoding": "transformer",
                    "d_model": args.d_model, "n_heads": args.n_heads,
                    "n_layers": args.n_layers, "ffn_dim": args.ffn_dim,
                    "embed_dim": args.d_model, "output_dim": 1,
                },
                "n_params": n_params,
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    print(f"\nfinal loss: {avg_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
