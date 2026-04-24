"""Train a ResMLP distance predictor on Megaminx.

    python megaminx/scripts/02_train.py --config megaminx/configs/fast.yaml \
        --output megaminx/models/fast_first

Uses the shared cayley.training loop — only the puzzle object changes.
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

from cayley.model import ResMLPDistance
from cayley.training import TrainConfig, train
from megaminx.puzzle import Megaminx


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
    print(f"puzzle: state_size={len(puzzle.solved_state)} moves={len(puzzle.move_names)}")
    print(f"training: {train_cfg_dict}")

    tc = TrainConfig(**train_cfg_dict)

    def log(stats):
        print(f"epoch {stats.epoch:4d} | loss {stats.loss:.4f} | lr {stats.lr:.2e} | {stats.elapsed_s:.1f}s",
              flush=True)

    result = train(model, puzzle, tc, checkpoint_dir=args.output, on_epoch_end=log)
    print(f"final loss: {result.final_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
