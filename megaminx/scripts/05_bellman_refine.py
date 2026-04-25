"""Bellman refinement on top of an existing Megaminx checkpoint.

    python megaminx/scripts/05_bellman_refine.py \
        --config megaminx/configs/m05_bellman_warm.yaml \
        --output megaminx/models/m05_bellman_warm

Same flow as the IHES script: load weights from the `bellman.warmstart_path` field in
the config, then train with self-bootstrapped targets
`y = 1 + min_a f_target(apply(s, a))`. Target network refreshed every
`bellman.target_update_every_epochs` epochs.
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

from cayley.bellman import BellmanConfig, train_bellman
from cayley.model import ResMLPDistance
from cayley.training import TrainConfig
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg_dict = cfg["training"]
    bellman_cfg_dict = cfg["bellman"]
    if args.epochs is not None:
        train_cfg_dict["n_epochs"] = args.epochs
    train_cfg_dict["device"] = args.device
    train_cfg_dict["seed"] = cfg.get("seed", 0)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    model = ResMLPDistance(
        state_size=model_cfg["state_size"],
        num_classes=model_cfg["num_classes"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        encoding=model_cfg.get("encoding", "onehot"),
        embed_dim=model_cfg.get("embed_dim", 16),
    )
    print(f"model params: {model.num_parameters():,}")
    print(f"device: {args.device}")
    print(f"training: {train_cfg_dict}")
    print(f"bellman: {bellman_cfg_dict}")

    tc = TrainConfig(**train_cfg_dict)
    bc = BellmanConfig(**bellman_cfg_dict)

    def log(stats):
        print(f"epoch {stats.epoch:4d} | loss {stats.loss:.4f} | "
              f"lr {stats.lr:.2e} | {stats.elapsed_s:.1f}s", flush=True)

    result = train_bellman(model, puzzle, tc, bc, checkpoint_dir=args.output, on_epoch_end=log)
    print(f"final loss: {result.final_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
