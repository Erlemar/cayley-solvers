"""m42 — Train distributional V (32 quantiles) with QR-DQN-style Bellman backup.

Predicts N quantiles per state instead of scalar V. Beam-time selection can use
lower quantile (optimistic preference for confident-close states), median, or
mean. Reference: Dabney 2018 (QR-DQN).

Recipe: same as m05 Bellman warmstart but with:
  - Model output_dim = N_QUANTILES (32 default)
  - Loss = quantile_huber_loss instead of MSE
  - Bellman backup: distributional_bellman_targets (target net's quantile dist on chosen child + 1)
  - Action selection within Bellman target uses median.
  - Warmstart from m05 V (output_dim=1) by REPLICATING the V scalar across all
    32 output dims. Equivalent to starting from a degenerate quantile distribution
    where all quantiles equal V_m05.

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe -u megaminx/scripts/38_train_distributional_v.py \\
      --config megaminx/configs/m42_distributional.yaml \\
      --output megaminx/models/m42_distributional
"""
from __future__ import annotations

import argparse
import copy
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

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.distributional import (
    distributional_bellman_targets,
    make_quantile_taus,
    quantile_huber_loss,
)
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def warmstart_from_v(model: ResMLPDistance, scalar_ckpt_path: str, device: str,
                     n_quantiles: int) -> None:
    """Load a scalar V checkpoint into a quantile-output model.

    Strategy: copy all body weights as-is. For the final output linear layer,
    replicate the single output row across all N quantile output rows. This
    initializes the quantile distribution as `V_m05 across all quantiles`
    (degenerate at the start; widens as training proceeds).
    """
    src_ckpt = torch.load(scalar_ckpt_path, map_location=device, weights_only=False)
    src_sd = src_ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in src_sd):
        src_sd = {k.removeprefix("_orig_mod."): v for k, v in src_sd.items()}

    tgt_sd = model.state_dict()
    n_loaded = 0
    n_replicated = 0
    n_skipped = 0
    for k, v_src in src_sd.items():
        if k not in tgt_sd:
            n_skipped += 1
            continue
        v_tgt = tgt_sd[k]
        if v_src.shape == v_tgt.shape:
            tgt_sd[k].copy_(v_src)
            n_loaded += 1
        elif v_src.dim() == v_tgt.dim() and v_src.shape[1:] == v_tgt.shape[1:] \
                and v_src.shape[0] == 1 and v_tgt.shape[0] == n_quantiles:
            # Final linear weight or bias: replicate single row across N quantiles.
            tgt_sd[k].copy_(v_src.expand_as(v_tgt))
            n_replicated += 1
        else:
            print(f"  WARN: skip {k} due to shape mismatch: src {v_src.shape} vs tgt {v_tgt.shape}")
            n_skipped += 1
    model.load_state_dict(tgt_sd)
    print(f"  warmstart: {n_loaded} loaded as-is, {n_replicated} replicated to "
          f"({n_quantiles} quantiles), {n_skipped} skipped")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override n_epochs from config (useful for smoke tests)")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    bellman_cfg = cfg["bellman"]
    distrib_cfg = cfg.get("distributional", {})
    if args.epochs is not None:
        train_cfg["n_epochs"] = args.epochs

    n_quantiles = distrib_cfg.get("n_quantiles", 32)
    kappa = distrib_cfg.get("kappa", 1.0)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"model params (n_quantiles={n_quantiles}):")
    model = ResMLPDistance(
        state_size=model_cfg["state_size"],
        num_classes=model_cfg["num_classes"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        encoding=model_cfg.get("encoding", "embedding"),
        embed_dim=model_cfg.get("embed_dim", 16),
        output_dim=n_quantiles,
    ).to(args.device)
    print(f"  total: {model.num_parameters():,}")

    warmstart_path = bellman_cfg["warmstart_path"]
    print(f"warmstart from {warmstart_path}")
    warmstart_from_v(model, warmstart_path, args.device, n_quantiles)

    # Target net: frozen copy of model. Updated periodically (discrete refresh).
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
        fused=(args.device == "cuda" and train_cfg.get("fused_optimizer", False)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["n_epochs"],
    )

    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(args.device)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64,
                                 device=args.device)
    taus = make_quantile_taus(n_quantiles, args.device)

    use_amp = train_cfg.get("amp", True) and args.device == "cuda"
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_amp
                    else torch.amp.autocast("cuda", enabled=False))

    n_walks = max(1, train_cfg["samples_per_epoch"] // train_cfg["k_max"])
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\ntraining: {train_cfg['n_epochs']} epochs, batch {train_cfg['batch_size']}, "
          f"n_quantiles {n_quantiles}, kappa {kappa}", flush=True)
    t0 = time.time()
    for epoch in range(train_cfg["n_epochs"]):
        ep_t0 = time.time()
        states, depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=train_cfg["k_max"],
            seed=seed + epoch, device=args.device,
            n_back=train_cfg.get("n_back", 1),
        )
        depths_f = depths.to(torch.float32)

        model.train()
        total_loss = 0.0
        n_batches = 0
        perm = torch.randperm(states.size(0), device=args.device)
        bs_size = train_cfg["batch_size"]
        for b in range(0, states.size(0), bs_size):
            idx = perm[b:b + bs_size]
            bs = states[idx]
            bd = depths_f[idx]

            target_q = distributional_bellman_targets(
                target_model, bs, bd, generators, solved_state,
                n_quantiles=n_quantiles,
                chunk_size=bellman_cfg.get("target_net_chunk", 4096),
                clip_upper=bellman_cfg.get("clip_upper", True),
                clip_lower=bellman_cfg.get("clip_lower", True),
            )

            with autocast_ctx:
                pred = model(bs).float()                # (B, n_quantiles)
                loss = quantile_huber_loss(pred, target_q.float(), taus, kappa=kappa)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        ep_wall = time.time() - ep_t0
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch:4d} | loss {avg_loss:.4f} | "
                  f"lr {scheduler.get_last_lr()[0]:.2e} | {ep_wall:.1f}s", flush=True)

        # Discrete target refresh.
        if (epoch + 1) % bellman_cfg.get("target_update_every_epochs", 10) == 0:
            src_sd = model.state_dict()
            if any(k.startswith("_orig_mod.") for k in src_sd):
                src_sd = {k.removeprefix("_orig_mod."): v for k, v in src_sd.items()}
            target_model.load_state_dict(src_sd)

        if (epoch + 1) % train_cfg.get("checkpoint_every_epochs", 50) == 0 \
                or epoch == train_cfg["n_epochs"] - 1:
            sd = model.state_dict()
            if any(k.startswith("_orig_mod.") for k in sd):
                sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
            torch.save({
                "epoch": epoch,
                "state_dict": sd,
                "loss": avg_loss,
                "model_config": {
                    "state_size": model_cfg["state_size"],
                    "num_classes": model_cfg["num_classes"],
                    "hidden_dims": list(model_cfg["hidden_dims"]),
                    "num_res_blocks": model_cfg["num_res_blocks"],
                    "encoding": model_cfg.get("encoding", "embedding"),
                    "embed_dim": model_cfg.get("embed_dim", 16),
                    "output_dim": n_quantiles,
                },
                "training_config": train_cfg,
                "distributional_config": {
                    "n_quantiles": n_quantiles, "kappa": kappa,
                },
            }, args.output / f"epoch_{epoch:04d}.pt")

    print(f"\nfinal loss: {avg_loss:.5f}")
    print(f"total wall: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
