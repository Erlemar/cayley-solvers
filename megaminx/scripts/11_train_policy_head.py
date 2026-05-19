"""m24: train a policy head π(a|s) on solved-path triplets, for PHS-style beam scoring.

The policy head outputs a 24-dim logit per state: log-prior over moves. At beam
search time, candidates are scored as
    score(s, a) = V(apply(s, a)) - λ * log π(a|s_parent)

Lower score = preferred. The -log term penalizes moves the policy considers
unlikely; λ tunes the strength of the prior.

Training data: triplets (s, a) sampled from solved paths. Two sources:
  1. Inverse random walks from solved: at depth k, the "correct" move is the
     inverse of the action taken at step k+1. This is cheap to generate.
  2. Successful beam search outputs from our existing 95,682 submission CSV:
     each move along a solving path is a positive (s, a) example.

We use source #1 here for speed (1k random walks → ~80k labeled pairs in seconds);
optionally augment with #2 from `phase12_post.csv` for distribution match.

Architecture: same backbone as m05 with a 24-output policy head (cross-entropy
loss). Multi-task with V is possible but adds complexity; this script is policy-only.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/11_train_policy_head.py \
        --warmstart megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --out-dir megaminx/models/m24_policy

References:
- Levin Tree Search (IJCAI 2023)
- Policy-Guided Heuristic Search (PHS, arXiv 2103.11505)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def generate_policy_data(
    puzzle: Megaminx,
    n_walks: int,
    k_max: int,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inverse random walks from solved. At each step, the LABEL is the
    inverse of the move that produced the state — i.e., the move that would
    return toward solved.

    Returns (states, action_labels) where:
        states: (n_walks * k_max, S) — states visited along walks
        action_labels: (n_walks * k_max,) — action index that solves toward solved
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    gens = GeneratorTable.from_puzzle(puzzle)
    perms = torch.from_numpy(gens.perms).to(device)
    inv = torch.from_numpy(gens.inverse_idx).to(device)
    n_gen = perms.shape[0]
    S = perms.shape[1]

    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)
    states = solved.unsqueeze(0).expand(n_walks, S).clone()
    history = torch.full((n_walks, 1), -1, dtype=torch.int64, device=device)

    out_states = torch.empty((n_walks * k_max, S), dtype=torch.int64, device=device)
    out_labels = torch.empty(n_walks * k_max, dtype=torch.int64, device=device)

    for k in range(1, k_max + 1):
        action = torch.randint(0, n_gen, (n_walks,), generator=g, device=device)
        # Non-backtracking
        if k > 1:
            banned = inv[history.squeeze(-1)]
            for _ in range(4):
                bad = action == banned
                if not bool(bad.any()):
                    break
                n_bad = int(bad.sum().item())
                action = action.clone()
                action[bad] = torch.randint(0, n_gen, (n_bad,), generator=g, device=device)
        gen_rows = perms[action]
        states = torch.gather(states, 1, gen_rows)
        history = action.unsqueeze(-1)

        off = (k - 1) * n_walks
        out_states[off : off + n_walks] = states
        # Label: inverse of the action that produced this state == "go back toward solved"
        out_labels[off : off + n_walks] = inv[action]
    return out_states, out_labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path, required=True,
                    help="m05 (or other) checkpoint — body weights warmstarted, head reinit")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=240)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=25)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)

    # Build model with policy head (output_dim=n_gen).
    model = ResMLPDistance(
        state_size=120, num_classes=120,
        hidden_dims=(2048, 512), num_res_blocks=2,
        encoding="embedding", embed_dim=16,
        output_dim=n_gen,  # policy head: logits over moves
    ).to(device)

    # Warmstart body weights from m05 (V-model). Head is reinit (different output_dim).
    teacher_ckpt = torch.load(args.warmstart, map_location=device, weights_only=False)
    teacher_sd = teacher_ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in teacher_sd):
        teacher_sd = {k.removeprefix("_orig_mod."): v for k, v in teacher_sd.items()}
    # Drop teacher's head weights — different output_dim
    teacher_sd_no_head = {k: v for k, v in teacher_sd.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(teacher_sd_no_head, strict=False)
    print(f"loaded body from {args.warmstart}: missing={[k for k in missing if 'head' not in k][:3]}... "
          f"({len([k for k in missing if 'head' in k])} head keys reinit)")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0,
        fused=device == "cuda",
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  batch: {args.batch_size}")

    for epoch in range(args.n_epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        states, action_labels = generate_policy_data(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch, device=device,
        )
        N = states.shape[0]

        model.train()
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs = states[idx]
            bl = action_labels[idx]

            if autocast_ctx is not None:
                with autocast_ctx:
                    logits = model(bs)  # (batch, n_gen)
                    loss = F.cross_entropy(logits, bl)
            else:
                logits = model(bs)
                loss = F.cross_entropy(logits, bl)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            with torch.no_grad():
                total_acc += float((logits.argmax(dim=1) == bl).float().mean().item())
            n_batches += 1
        sched.step()

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | CE {avg_loss:.4f} | top1_acc {avg_acc:.3f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time() - t0:.1f}s",
                  flush=True)

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.n_epochs - 1:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "ce_loss": avg_loss, "top1_acc": avg_acc,
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": [2048, 512], "num_res_blocks": 2,
                    "encoding": "embedding", "embed_dim": 16,
                    "output_dim": n_gen,
                },
                "warmstart_from": str(args.warmstart),
                "head_kind": "policy",
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
