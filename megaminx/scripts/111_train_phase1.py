"""Train the Phase-1 V model for the two-phase (Kociemba-style) solver.

Phase 1 reduces an arbitrary scramble INTO the subgroup H: it must get every
frozen (LL) sticker home while treating the 85 TOP-movable stickers as don't-care.
We train on full 24-generator random walks from solved, but wrap the model in
MaskedV so the movable sticker VALUES are masked to the sentinel token before
scoring. Every epoch also mixes exact H/coset anchors: restricted TOP-only walks
are already in H, so their Phase-1 target is 0. Correctness of any solve is
enforced later by verify_path.

    .venv/Scripts/python.exe megaminx/scripts/111_train_phase1.py \
        --config megaminx/configs/m_phase1_pretrain.yaml \
        --output megaminx/models/m_phase1_v0

The config `model` block is the INNER ResMLP config; num_classes must be 121.
See megaminx/two_phase_plan.md for the full recipe and validation gates.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.training import TrainConfig, train
from megaminx.puzzle import Megaminx
from megaminx.two_phase import (
    MASK_TOKEN,
    build_phase2_puzzle,
    build_masked_phase1_model,
    movable_frozen_positions,
    phase1_h_zero_anchors,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path, help="directory for checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None, help="override config n_epochs")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg_dict = cfg["training"]
    anchor_cfg = cfg.get("anchors", {})
    if args.epochs is not None:
        train_cfg_dict["n_epochs"] = args.epochs
    train_cfg_dict["device"] = args.device
    train_cfg_dict["seed"] = cfg.get("seed", 0)

    # Full 24-generator puzzle: Phase 1 may use any move.
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    phase2_puzzle = build_phase2_puzzle(puzzle)
    movable, frozen = movable_frozen_positions(puzzle)

    model = build_masked_phase1_model(puzzle, model_cfg)
    print(f"PHASE 1 (reduce into H) -- full {len(puzzle.move_names)} generators; "
          f"mask token {MASK_TOKEN}", flush=True)
    print(f"movable (masked) stickers: {len(movable)}; frozen (goal) stickers: {len(frozen)}",
          flush=True)
    print(f"model params: {model.num_parameters():,} (inner num_classes "
          f"{model_cfg['num_classes']})", flush=True)
    print(f"device: {args.device}", flush=True)
    print(f"training: {train_cfg_dict}", flush=True)
    print(f"anchors: {anchor_cfg}", flush=True)

    tc = TrainConfig(**train_cfg_dict)

    n_h_samples = int(anchor_cfg.get("h_samples_per_epoch", 0))
    h_k_max = int(anchor_cfg.get("h_k_max", train_cfg_dict.get("k_max", 30)))
    n_h_solved = int(anchor_cfg.get("n_h_solved", 0))
    n_h_back = int(anchor_cfg.get("h_n_back", train_cfg_dict.get("n_back", 1)))
    anchor_sampler = None
    if n_h_samples > 0 or n_h_solved > 0:
        print(f"anchor batch: solved={n_h_solved:,}, H-walk samples={n_h_samples:,}, "
              f"H k_max={h_k_max}, H n_back={n_h_back}", flush=True)

        def anchor_sampler(seed):
            return phase1_h_zero_anchors(
                phase2_puzzle,
                n_h_samples=n_h_samples,
                h_k_max=h_k_max,
                seed=seed ^ 0x515E_A5E,
                device=args.device,
                n_back=n_h_back,
                n_solved=n_h_solved,
            )

    def log(stats):
        print(f"epoch {stats.epoch:4d} | loss {stats.loss:.4f} | lr {stats.lr:.2e} | "
              f"{stats.elapsed_s:.1f}s", flush=True)

    result = train(model, puzzle, tc, checkpoint_dir=args.output, on_epoch_end=log,
                   anchor_sampler=anchor_sampler)
    print(f"final loss: {result.final_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
