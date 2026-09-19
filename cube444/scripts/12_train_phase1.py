"""Train the phase-1 masked value function (distance-to-R) for the 4x4x4.

Two stages, same as the single-phase pipeline:

  pretrain  V(s) ~ walk depth away from R   (random walks starting from a random
                                             element of R, not from solved)
  bellman   V(s) <- 1 + min_a V_target(a)   with the Dirichlet boundary V(R)=0 and
                                             exact R / d=1 anchors mixed in

Both stages optionally add the sparse-Q margin term (idea 2):

    L_margin = mean( (V(next) - V(prev) - 2)^2 )

over random-walk pivots, whose label difference has zero conditional variance and so
cannot be minimised by flattening the child gap. Set `lambda_margin: 0.0` for the
control arm -- everything else is bit-identical, which is what makes the A/B clean.

    python cube444/scripts/12_train_phase1.py --config cube444/configs/p1_margin.yaml \
        --output cube444/models/p1_margin --stage pretrain
    python cube444/scripts/12_train_phase1.py --config cube444/configs/p1_margin.yaml \
        --output cube444/models/p1_margin --stage bellman
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.model import ResMLPDistance  # noqa: E402
from cube444.orbits import N_NONCORNER  # noqa: E402
from cube444.phase1 import (  # noqa: E402
    MaskedV, Phase1Sampler, ReductionTest, WalkSpec, check_reduction_test,
)
from cube444.puzzle import Cube444  # noqa: E402


def build_masked_model(mcfg: dict) -> MaskedV:
    inner = ResMLPDistance(
        state_size=N_NONCORNER,
        num_classes=mcfg.get("num_classes", 6),
        hidden_dims=tuple(mcfg.get("hidden_dims", (2048, 512))),
        num_res_blocks=mcfg.get("num_res_blocks", 2),
        encoding=mcfg.get("encoding", "onehot"),
        embed_dim=mcfg.get("embed_dim", 16),
        inference_chunk_size=mcfg.get("inference_chunk_size", 8192),
    )
    return MaskedV(inner)


def save_ckpt(model: MaskedV, mcfg: dict, path: Path, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k.removeprefix("_orig_mod."): v
                           for k, v in model.state_dict().items()},
            "model_config": {"model_class": "MaskedV",
                             "inner": model.inner.get_model_config()},
            "phase1_model_cfg": mcfg,
            **extra,
        },
        path,
    )


def load_masked(path: Path, device: str) -> MaskedV:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_masked_model(ck["phase1_model_cfg"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--stage", choices=("pretrain", "bellman"), required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--warmstart", type=Path, default=None,
                    help="override the config warmstart_path")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mcfg = cfg["model"]
    tcfg = dict(cfg["training"][args.stage])
    if args.epochs is not None:
        tcfg["n_epochs"] = args.epochs
    seed = int(cfg.get("seed", 444))
    torch.manual_seed(seed)

    dev = args.device
    assert check_reduction_test(device=dev), "torch/numpy reduction tests disagree"

    puzzle = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    sampler = Phase1Sampler(puzzle, device=dev, seed=seed)
    rtest = ReductionTest(device=dev)
    spec = WalkSpec(
        k_max=int(tcfg.get("k_max", 120)),
        h_walk_max=int(tcfg.get("h_walk_max", 40)),
        n_back=int(tcfg.get("n_back", 1)),
        label_clip=float(tcfg.get("label_clip", 30.0)),
        margin_k_max=int(tcfg.get("margin_k_max", 20)),
    )
    # Margin pivots are sampled from their own, much shallower walk -- see WalkSpec.
    margin_spec = WalkSpec(k_max=spec.margin_k_max, h_walk_max=spec.h_walk_max,
                           n_back=spec.n_back)

    model = build_masked_model(mcfg).to(dev)
    warm = str(args.warmstart) if args.warmstart else tcfg.get("warmstart_path")
    if warm:
        model = load_masked(Path(warm), dev).to(dev)
        print(f"warmstarted from {warm}")
    model.train()

    lam = float(tcfg.get("lambda_margin", 0.0))
    lr = float(tcfg.get("lr", 5e-4))
    n_epochs = int(tcfg["n_epochs"])
    steps = int(tcfg.get("steps_per_epoch", 30))
    batch = int(tcfg.get("batch_size", 8192))
    m_batch = int(tcfg.get("margin_batch_size", 2048))
    anchor_r = int(tcfg.get("n_anchor_r", 256))
    anchor_d1 = int(tcfg.get("n_anchor_d1", 64))
    clip_hi = float(tcfg.get("clip_upper", spec.k_max))
    tgt_every = int(tcfg.get("target_update_every_epochs", 10))
    amp = bool(tcfg.get("amp", True))

    print(f"params: {model.num_parameters():,}   input slots: {N_NONCORNER}")
    print(f"stage={args.stage}  lambda_margin={lam}  k_max={spec.k_max}  "
          f"label_clip={spec.label_clip}  margin_k_max={spec.margin_k_max}  "
          f"epochs={n_epochs} steps/ep={steps} batch={batch}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(tcfg.get("weight_decay", 0.0)),
                            fused=(dev == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * steps)
    scaler = torch.amp.GradScaler(dev, enabled=amp and dev == "cuda")

    target = None
    if args.stage == "bellman":
        target = copy.deepcopy(model).eval()
        for p in target.parameters():
            p.requires_grad_(False)

    args.output.mkdir(parents=True, exist_ok=True)
    pool_batch = int(tcfg.get("pool_batch", 1024))
    hist = []
    for ep in range(n_epochs):
        t0 = time.time()
        tot, tot_main, tot_margin, nb = 0.0, 0.0, 0.0, 0
        # One deep walk per epoch, snapshotted at every step, is the epoch's pool.
        pool_s, pool_d = sampler.walk_harvest(pool_batch, spec)
        for _ in range(steps):
            sel = torch.randint(pool_s.shape[0], (batch,), device=dev)
            s = pool_s[sel].long()
            depth = pool_d[sel]

            with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
                if args.stage == "pretrain":
                    y = depth.clamp(max=spec.label_clip)
                else:
                    with torch.no_grad():
                        kids = sampler.children(s)                # (B, G, 96)
                        B, G, S = kids.shape
                        flat = kids.reshape(B * G, S)
                        vt = target(flat).float().reshape(B, G)
                        vt = torch.where(rtest.is_reduced(kids), torch.zeros_like(vt), vt)
                        y = 1.0 + vt.min(dim=1).values
                        y = torch.where(rtest.is_reduced(s), torch.zeros_like(y), y)
                        y = y.clamp(0.0, clip_hi)

                pred = model(s).float()
                loss_main = F.mse_loss(pred, y)

                # exact anchors: states in R -> 0, their children -> <= 1
                if anchor_r > 0:
                    h, d1 = sampler.r_anchors(anchor_r, spec)
                    va = model(h).float()
                    loss_main = loss_main + F.mse_loss(va, torch.zeros_like(va))
                    if anchor_d1 > 0:
                        vd = model(d1[:anchor_d1]).float()
                        tgt_d1 = torch.where(
                            rtest.is_reduced(d1[:anchor_d1]),
                            torch.zeros_like(vd), torch.ones_like(vd),
                        )
                        loss_main = loss_main + F.mse_loss(vd, tgt_d1)

                # sparse-Q margin: the label difference is exactly 2 by construction
                if lam > 0.0:
                    _, _, prev, nxt = sampler.walk(m_batch, margin_spec)
                    vp = model(prev).float()
                    vn = model(nxt).float()
                    loss_margin = F.mse_loss(vn - vp, torch.full_like(vp, 2.0))
                else:
                    loss_margin = torch.zeros((), device=dev)

                loss = loss_main + lam * loss_margin

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())
            tot_main += float(loss_main.detach())
            tot_margin += float(loss_margin.detach())
            nb += 1

        if args.stage == "bellman" and (ep + 1) % tgt_every == 0:
            target.load_state_dict(model.state_dict())
            target.eval()

        rec = {"epoch": ep, "loss": tot / nb, "loss_main": tot_main / nb,
               "loss_margin": tot_margin / nb, "lr": sched.get_last_lr()[0],
               "sec": time.time() - t0}
        hist.append(rec)
        print(f"epoch {ep:4d} | loss {rec['loss']:.4f} | main {rec['loss_main']:.4f} | "
              f"margin {rec['loss_margin']:.4f} | {rec['sec']:.1f}s", flush=True)

        every = int(tcfg.get("checkpoint_every_epochs", 25))
        if (ep + 1) % every == 0 or ep == n_epochs - 1:
            save_ckpt(model, mcfg, args.output / f"epoch_{ep:04d}.pt",
                      {"stage": args.stage, "epoch": ep, "lambda_margin": lam,
                       "walk_spec": vars(spec)})

    with open(args.output / f"history_{args.stage}.json", "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    print(f"done -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
