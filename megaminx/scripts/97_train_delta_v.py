"""Train a function-preserving residual delta on top of a frozen V model.

The composed scorer is:

    V_new(s) = V_base(s) + scale * Delta(s)

Delta's final layer is zero-initialized, so epoch 0 is exactly V_base. The
training objective focuses on known mistake-like states:

  * path states with verified remaining path lengths from an AZ dataset;
  * certified stagnation anchors (exact d<=6 plus one-sided d>=7 lower bounds);
  * cross-width eviction labels where a verified teacher child was pruned by a
    narrow beam and must outrank near-cutoff boundary states;
  * random-walk preservation states where Delta is penalized toward zero.

This is intentionally conservative: the base landscape stays intact unless a
state source gives us evidence that a residual correction is useful.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.data import generate_walks_torch
from cayley.model import DeltaVModel, ResMLPDistance, zero_init_resmlp_head
from cayley.search import load_model_checkpoint
from megaminx.puzzle import Megaminx


def _sample_indices(n: int, k: int, device: str, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(0, n, (k,), generator=gen, device=device)


@torch.no_grad()
def _base_value(base: torch.nn.Module, states: torch.Tensor, chunk: int) -> torch.Tensor:
    outs = []
    base.eval()
    for i in range(0, states.size(0), chunk):
        outs.append(base(states[i : i + chunk]).float().flatten())
    return torch.cat(outs, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--path-dataset", type=Path, default=PROJECT / "data" / "az_dataset_73614.pt")
    ap.add_argument("--stag-anchor-path", type=Path, default=PROJECT / "data" / "stagnation_anchors_v0.pt")
    ap.add_argument("--eviction-path", type=Path, default=None,
                    help="Optional cross-width eviction dataset from "
                         "90_harvest_cross_width_evictions.py")
    ap.add_argument("--hidden-dims", type=str, default="2048,1024")
    ap.add_argument("--num-res-blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps-per-epoch", type=int, default=128)
    ap.add_argument("--path-batch-size", type=int, default=1024)
    ap.add_argument("--path-candidate-mult", type=int, default=4)
    ap.add_argument("--exact-batch-size", type=int, default=512)
    ap.add_argument("--lb-batch-size", type=int, default=1024)
    ap.add_argument("--eviction-batch-size", type=int, default=256)
    ap.add_argument("--preserve-batch-size", type=int, default=2048)
    ap.add_argument("--preserve-k-max", type=int, default=80)
    ap.add_argument("--preserve-n-back", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lambda-path", type=float, default=0.10)
    ap.add_argument("--lambda-exact", type=float, default=1.0)
    ap.add_argument("--lambda-lb", type=float, default=1.0)
    ap.add_argument("--lambda-eviction", type=float, default=0.0)
    ap.add_argument("--lambda-preserve", type=float, default=0.03)
    ap.add_argument("--eviction-margin", type=float, default=0.25,
                    help="Require V(good_child) + margin <= V(boundary_state).")
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--target-chunk", type=int, default=4096)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=117)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)

    base = load_model_checkpoint(args.base_checkpoint, device=args.device, dtype=torch.float32)
    base.eval()
    base_cfg = copy.deepcopy(base.get_model_config())
    if int(base_cfg.get("output_dim", 1)) != 1:
        raise ValueError("--base-checkpoint must be a scalar V model")

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    delta = ResMLPDistance(
        state_size=state_size,
        num_classes=state_size,
        hidden_dims=hidden_dims,
        num_res_blocks=args.num_res_blocks,
        encoding="embedding",
        embed_dim=16,
        output_dim=1,
        inference_chunk_size=2048,
    ).to(args.device)
    zero_init_resmlp_head(delta)
    model = DeltaVModel(
        base=base,
        delta=delta,
        delta_scale=args.delta_scale,
        freeze_base=True,
        inference_chunk_size=2048,
    ).to(args.device)
    model.base.eval()
    model.delta.train()

    print(f"base: {args.base_checkpoint}", flush=True)
    print(f"delta params: {delta.num_parameters():,}", flush=True)
    print(f"total params: {model.num_parameters():,} trainable={model.trainable_parameters():,}", flush=True)

    path_data = torch.load(args.path_dataset, map_location="cpu", weights_only=False)
    path_states = path_data["states"].to(args.device)
    path_values = path_data["values"].to(args.device).float()
    print(f"path dataset: {path_states.size(0):,} states from {args.path_dataset}", flush=True)

    stag_exact_states = stag_exact_values = None
    stag_lb_states = stag_lb_values = None
    if args.stag_anchor_path.exists():
        stag = torch.load(args.stag_anchor_path, map_location="cpu", weights_only=False)
        st = stag["states"].to(args.device)
        lt = stag["label_type"].to(args.device)
        lv = stag["label_value"].to(args.device).float()
        exact_mask = lt == 0
        lb_mask = lt == 1
        stag_exact_states = st[exact_mask]
        stag_exact_values = lv[exact_mask]
        stag_lb_states = st[lb_mask]
        stag_lb_values = lv[lb_mask]
        print(
            f"stag anchors: exact={stag_exact_states.size(0):,} lb={stag_lb_states.size(0):,}",
            flush=True,
        )
    else:
        print(f"stag anchors: missing {args.stag_anchor_path}; skipping", flush=True)

    ev_goods = ev_boundaries = ev_mask = None
    if args.eviction_path is not None and args.eviction_path.exists():
        ev = torch.load(args.eviction_path, map_location="cpu", weights_only=False)
        ev_goods = ev["goods"].to(args.device)
        ev_boundaries = ev["boundaries"].to(args.device)
        ev_mask = ev["boundary_mask"].to(args.device)
        print(
            f"eviction labels: {ev_goods.size(0):,} labels, "
            f"boundary={ev_boundaries.size(1)} from {args.eviction_path}",
            flush=True,
        )
    elif args.lambda_eviction > 0:
        raise FileNotFoundError(f"--lambda-eviction > 0 but missing --eviction-path {args.eviction_path}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=(args.device == "cuda"),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    gen_cpu = torch.Generator(device="cpu").manual_seed(args.seed)
    gen_dev = torch.Generator(device=args.device).manual_seed(args.seed)

    n_walks_preserve = max(1, args.preserve_batch_size // args.preserve_k_max)

    for epoch in range(args.epochs):
        t0 = time.time()
        totals = {"loss": 0.0, "path": 0.0, "exact": 0.0, "lb": 0.0,
                  "eviction": 0.0, "preserve": 0.0}
        for step in range(args.steps_per_epoch):
            losses: list[torch.Tensor] = []

            if args.path_batch_size > 0 and args.lambda_path > 0:
                cand_n = max(args.path_batch_size, args.path_batch_size * args.path_candidate_mult)
                idx_c = _sample_indices(path_states.size(0), cand_n, args.device, gen_dev)
                cand_s = path_states[idx_c]
                cand_y = path_values[idx_c]
                with torch.no_grad():
                    cand_base = _base_value(model.base, cand_s, args.target_chunk)
                    err = (cand_base - cand_y).abs()
                    top = torch.topk(err, k=min(args.path_batch_size, cand_n), largest=True).indices
                bs = cand_s[top]
                by = cand_y[top]
                pred = model(bs)
                loss_path = F.mse_loss(pred.float(), by)
                losses.append(args.lambda_path * loss_path)
                totals["path"] += float(loss_path.item())

            if stag_exact_states is not None and args.exact_batch_size > 0 and args.lambda_exact > 0:
                idx_e = _sample_indices(stag_exact_states.size(0), args.exact_batch_size, args.device, gen_dev)
                pred_e = model(stag_exact_states[idx_e])
                loss_exact = F.mse_loss(pred_e.float(), stag_exact_values[idx_e])
                losses.append(args.lambda_exact * loss_exact)
                totals["exact"] += float(loss_exact.item())

            if stag_lb_states is not None and args.lb_batch_size > 0 and args.lambda_lb > 0:
                idx_l = _sample_indices(stag_lb_states.size(0), args.lb_batch_size, args.device, gen_dev)
                pred_l = model(stag_lb_states[idx_l]).float()
                undershoot = (stag_lb_values[idx_l] - pred_l).clamp(min=0.0)
                loss_lb = (undershoot ** 2).mean()
                losses.append(args.lambda_lb * loss_lb)
                totals["lb"] += float(loss_lb.item())

            if ev_goods is not None and args.eviction_batch_size > 0 and args.lambda_eviction > 0:
                idx_v = _sample_indices(ev_goods.size(0), args.eviction_batch_size, args.device, gen_dev)
                goods = ev_goods[idx_v]
                bounds = ev_boundaries[idx_v]
                mask = ev_mask[idx_v]
                pred_good = model(goods).float().view(-1, 1)
                flat_bounds = bounds.reshape(-1, bounds.size(-1))
                pred_bounds = model(flat_bounds).float().view(bounds.size(0), bounds.size(1))
                margin_violation = (pred_good + args.eviction_margin - pred_bounds).clamp(min=0.0)
                loss_eviction = (margin_violation[mask] ** 2).mean()
                losses.append(args.lambda_eviction * loss_eviction)
                totals["eviction"] += float(loss_eviction.item())

            if args.preserve_batch_size > 0 and args.lambda_preserve > 0:
                rw_s, _rw_d = generate_walks_torch(
                    puzzle,
                    n_walks=n_walks_preserve,
                    k_max=args.preserve_k_max,
                    seed=args.seed + epoch * 100000 + step,
                    device=args.device,
                    n_back=args.preserve_n_back,
                )
                if rw_s.size(0) > args.preserve_batch_size:
                    idx_p = _sample_indices(rw_s.size(0), args.preserve_batch_size, args.device, gen_dev)
                    rw_s = rw_s[idx_p]
                delta_only = model.delta(rw_s).float().flatten()
                loss_preserve = (delta_only ** 2).mean()
                losses.append(args.lambda_preserve * loss_preserve)
                totals["preserve"] += float(loss_preserve.item())

            if not losses:
                raise RuntimeError("No loss terms active")
            loss = torch.stack(losses).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.item())

        sched.step()
        denom = max(1, args.steps_per_epoch)
        print(
            f"epoch {epoch:4d} | loss {totals['loss']/denom:.4f} "
            f"| path {totals['path']/denom:.4f} | exact {totals['exact']/denom:.4f} "
            f"| lb {totals['lb']/denom:.4f} | evict {totals['eviction']/denom:.4f} "
            f"| preserve {totals['preserve']/denom:.4f} "
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
