"""Train a local beam-cutoff retention head from eviction labels.

This is intentionally *not* a new value model. The head learns only a local
ordering signal: the verified teacher child should score lower than the
near-cutoff survivor states from the narrow beam. At inference, `03_solve.py`
can use it only inside an overgenerated V pool via `--cutoff-model`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def _sample_indices(pool: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    pos = torch.randint(0, pool.numel(), (k,), generator=gen, device=pool.device)
    return pool[pos]


@torch.no_grad()
def _predict(model: torch.nn.Module, states: torch.Tensor, chunk: int) -> torch.Tensor:
    outs = []
    model.eval()
    for i in range(0, states.size(0), chunk):
        outs.append(model(states[i : i + chunk]).float().flatten())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def _eval_split(
    model: torch.nn.Module,
    goods: torch.Tensor,
    boundaries: torch.Tensor,
    mask: torch.Tensor,
    label_idx: torch.Tensor,
    margin: float,
    chunk: int,
) -> dict[str, float]:
    if label_idx.numel() == 0:
        return {}
    g = goods[label_idx]
    b = boundaries[label_idx]
    m = mask[label_idx].bool()
    n_labels, n_boundary, state_size = b.shape
    good_v = _predict(model, g, chunk).view(n_labels, 1)
    bound_v = _predict(model, b.reshape(-1, state_size), chunk).view(n_labels, n_boundary)
    valid_counts = m.sum(dim=1).clamp(min=1)
    pairwise = ((good_v < bound_v) & m).sum(dim=1).float() / valid_counts.float()
    margin_ok = ((good_v + margin <= bound_v) & m).sum(dim=1).float() / valid_counts.float()
    cutoff = bound_v.masked_fill(~m, float("-inf")).max(dim=1).values
    cutoff_ok = good_v.flatten() < cutoff
    cutoff_margin = good_v.flatten() + margin <= cutoff
    rank = ((bound_v < good_v) & m).sum(dim=1) + 1
    model.train()
    return {
        "pair": float(pairwise.mean().item()),
        "margin": float(margin_ok.mean().item()),
        "cutoff": float(cutoff_ok.float().mean().item()),
        "cutoff_margin": float(cutoff_margin.float().mean().item()),
        "rank": float(rank.float().mean().item()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eviction-path", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--hidden-dims", default="512,256")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps-per-epoch", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--boundary-subsample", type=int, default=64,
                    help="Number of boundary states per label per step; 0 means all.")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--hard-topk", type=int, default=16,
                    help="Also penalize the hardest K boundary states per label. 0 disables.")
    ap.add_argument("--lambda-mean", type=float, default=1.0)
    ap.add_argument("--lambda-hard", type=float, default=1.0)
    ap.add_argument("--lambda-l2", type=float, default=1.0e-4)
    ap.add_argument("--lr", type=float, default=3.0e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--eval-chunk", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=117)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    data = torch.load(args.eviction_path, map_location="cpu", weights_only=False)
    goods = data["goods"].to(args.device)
    boundaries = data["boundaries"].to(args.device)
    mask = data["boundary_mask"].to(args.device).bool()
    n_labels = goods.size(0)
    n_boundary = boundaries.size(1)

    gen_cpu = torch.Generator(device="cpu").manual_seed(args.seed)
    perm = torch.randperm(n_labels, generator=gen_cpu)
    n_val = max(1, int(round(n_labels * args.val_frac)))
    val_idx = perm[:n_val].to(args.device)
    train_idx = perm[n_val:].to(args.device)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    model = ResMLPDistance(
        state_size=state_size,
        num_classes=state_size,
        hidden_dims=hidden_dims,
        num_res_blocks=args.num_res_blocks,
        encoding="embedding",
        embed_dim=16,
        output_dim=1,
        inference_chunk_size=2048,
    ).to(args.device)
    print(f"labels: train={train_idx.numel()} val={val_idx.numel()} boundary={n_boundary}", flush=True)
    print(f"model params: {model.num_parameters():,}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=str(args.device).startswith("cuda"),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    gen_dev = torch.Generator(device=args.device).manual_seed(args.seed)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        totals = {"loss": 0.0, "mean": 0.0, "hard": 0.0, "l2": 0.0}
        for _step in range(args.steps_per_epoch):
            idx = _sample_indices(train_idx, args.batch_size, gen_dev)
            g = goods[idx]
            b = boundaries[idx]
            m = mask[idx]
            if 0 < args.boundary_subsample < n_boundary:
                cols = torch.randint(0, n_boundary, (args.batch_size, args.boundary_subsample),
                                     generator=gen_dev, device=args.device)
                b = b[torch.arange(args.batch_size, device=args.device).unsqueeze(1), cols]
                m = m[torch.arange(args.batch_size, device=args.device).unsqueeze(1), cols]

            pred_g = model(g).float().view(-1, 1)
            pred_b = model(b.reshape(-1, b.size(-1))).float().view(b.size(0), b.size(1))
            violation = (pred_g + args.margin - pred_b).clamp(min=0.0)
            valid_violation = violation[m]
            loss_mean = (valid_violation ** 2).mean()
            losses = [args.lambda_mean * loss_mean]

            loss_hard = torch.zeros((), device=args.device)
            if args.hard_topk > 0:
                masked = violation.masked_fill(~m, 0.0)
                k = min(args.hard_topk, masked.size(1))
                loss_hard = (torch.topk(masked, k=k, dim=1).values ** 2).mean()
                losses.append(args.lambda_hard * loss_hard)

            loss_l2 = (pred_g ** 2).mean() + (pred_b[m] ** 2).mean()
            if args.lambda_l2 > 0:
                losses.append(args.lambda_l2 * loss_l2)

            loss = torch.stack(losses).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.item())
            totals["mean"] += float(loss_mean.item())
            totals["hard"] += float(loss_hard.item())
            totals["l2"] += float(loss_l2.item())

        sched.step()
        val = _eval_split(model, goods, boundaries, mask, val_idx, args.margin, args.eval_chunk)
        denom = max(1, args.steps_per_epoch)
        print(
            f"epoch {epoch:4d} | loss {totals['loss']/denom:.4f} "
            f"| mean {totals['mean']/denom:.4f} | hard {totals['hard']/denom:.4f} "
            f"| l2 {totals['l2']/denom:.4f} | val_pair {val.get('pair', 0):.4f} "
            f"| val_cutoff {val.get('cutoff', 0):.4f} | val_rank {val.get('rank', 0):.2f} "
            f"| lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s",
            flush=True,
        )
        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "model_config": model.get_model_config(),
                    "train_args": vars(args),
                },
                args.output / f"epoch_{epoch:04d}.pt",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
