"""m_gnn_v0: Minimal GNN distance predictor for megaminx (curiosity bet).

Different inductive bias from MLP/Transformer. Each of 120 stickers is a node;
edges connect stickers that share a face. K rounds of message passing aggregate
neighbor features; global mean pool + linear gives scalar V.

Adjacency derivation (lazy): for each sticker position i, neighbors are all
stickers that can be reached from i by applying a single generator. This
captures "stickers in the same orbit under rotation" — close to face-structure.

Walk-depth pretrain only (no Bellman). Quick experiment to see if message
passing on the sticker graph gives different signal than the MLP family.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/50_train_gnn.py \\
        --out-dir megaminx/models/m_gnn_v0 \\
        --n-epochs 100
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


def build_adjacency(puzzle: Megaminx) -> torch.Tensor:
    """For each sticker position i, find all positions reachable from i by one
    generator (= where stickers from position i can travel in one move).

    Returns a (state_size, max_deg) padded int64 tensor; -1 means "no neighbor"
    (used as sentinel; nodes with fewer real neighbors get padded).
    """
    state_size = len(puzzle.solved_state)
    nbr_set: list[set[int]] = [set() for _ in range(state_size)]
    for name in puzzle.move_names:
        gen = puzzle.generators[name]
        for i in range(state_size):
            j = gen[i]
            if j != i:
                nbr_set[i].add(j)
                nbr_set[j].add(i)  # symmetric
    max_deg = max(len(s) for s in nbr_set)
    adj = torch.full((state_size, max_deg), -1, dtype=torch.int64)
    for i, s in enumerate(nbr_set):
        for k, j in enumerate(sorted(s)):
            adj[i, k] = j
    return adj  # (state_size, max_deg)


class StickerGNN(nn.Module):
    """Minimal GIN-style GNN. Per-node features at each step:
        h_i' = MLP( h_i + sum_j h_j  for j in neighbors(i) )
    Final V = linear(global_mean_pool(h)).
    """

    def __init__(
        self,
        state_size: int,
        num_classes: int,
        embed_dim: int = 64,
        n_layers: int = 4,
        adjacency: torch.Tensor = None,
    ):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.output_dim = 1
        self.encoding = "gnn"
        self.inference_chunk_size = None

        self.token_emb = nn.Embedding(num_classes, embed_dim)
        self.pos_emb = nn.Embedding(state_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(embed_dim, 1)
        # Register adjacency as buffer.
        if adjacency is None:
            raise ValueError("adjacency required")
        self.register_buffer("adj", adjacency)         # (state_size, max_deg)
        self.register_buffer("adj_mask", adjacency >= 0)  # bool, valid neighbors
        # Cache positional indices.
        self.register_buffer("positions", torch.arange(state_size).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, state_size) int
        B = x.shape[0]
        h = self.token_emb(x.long()) + self.pos_emb(self.positions.expand(B, self.state_size))
        # h: (B, S, D)
        # Pre-gather neighbor indices once (constant across batches).
        # adj: (S, max_deg). For each (b, i): h_nbrs = h[b, adj[i]] for valid neighbors.
        adj = self.adj.clamp(min=0)              # (S, max_deg) int64
        mask = self.adj_mask.unsqueeze(0).unsqueeze(-1)  # (1, S, max_deg, 1) bool

        for layer in self.layers:
            # Gather neighbors: (B, S, max_deg, D)
            h_nbrs = h[:, adj]                   # (B, S, max_deg, D)
            h_nbrs = h_nbrs * mask               # zero out padding
            agg = h_nbrs.sum(dim=2)              # (B, S, D)
            h = h + agg                          # GIN-style sum aggregator
            h = layer(h)                         # MLP update
            h = F.relu(h)
        # Global mean pool over nodes
        h = h.mean(dim=1)                        # (B, D)
        return self.head(h).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=510)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=20)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--embed-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    adjacency = build_adjacency(puzzle).to(device)
    print(f"adjacency: state_size={adjacency.size(0)}, max_deg={adjacency.size(1)}")
    print(f"  avg deg: {(adjacency >= 0).sum().item() / adjacency.size(0):.1f}")

    model = StickerGNN(
        state_size=120, num_classes=120,
        embed_dim=args.embed_dim, n_layers=args.n_layers,
        adjacency=adjacency,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GNN: {n_params:,} params (embed_dim={args.embed_dim}, n_layers={args.n_layers})")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else None)
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  "
          f"batch: {args.batch_size}")

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
                    pred = model(bs)
                    loss = F.mse_loss(pred, bd)
            else:
                pred = model(bs)
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
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "loss": avg_loss,
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "embed_dim": args.embed_dim, "n_layers": args.n_layers,
                    "encoding": "gnn", "output_dim": 1,
                    "model_class": "StickerGNN",
                },
                "n_params": n_params,
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
