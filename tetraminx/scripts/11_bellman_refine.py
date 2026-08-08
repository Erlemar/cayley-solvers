"""Stage B: Bellman refinement of a Tetraminx V model.

    python3 tetraminx/scripts/11_bellman_refine.py \
        --config tetraminx/configs/tv0_bellman.yaml \
        --output tetraminx/models/tv0_bellman
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.bellman import BellmanConfig, train_bellman
from cayley.model import ResMLPDistance
from cayley.training import TrainConfig
from tetraminx.puzzle import Tetraminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path, help="checkpoint dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None, help="override config n_epochs")
    ap.add_argument("--warmstart", type=Path, default=None,
                    help="override bellman.warmstart_path")
    ap.add_argument("--bfs-anchors", type=Path, default=None,
                    help="override bellman.bfs_d6_path (exact-distance anchor .pt)")
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "puzzle_info.json")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg_dict = cfg["training"]
    bellman_cfg_dict = cfg["bellman"]
    if args.epochs is not None:
        train_cfg_dict["n_epochs"] = args.epochs
    if args.warmstart is not None:
        bellman_cfg_dict["warmstart_path"] = str(args.warmstart)
    if args.bfs_anchors is not None:
        bellman_cfg_dict["bfs_d6_path"] = str(args.bfs_anchors)
    train_cfg_dict["device"] = args.device
    train_cfg_dict["seed"] = cfg.get("seed", 0)

    puzzle = Tetraminx.load(args.puzzle_info)
    model = ResMLPDistance(
        state_size=model_cfg["state_size"],
        num_classes=model_cfg["num_classes"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        encoding=model_cfg.get("encoding", "embedding"),
        embed_dim=model_cfg.get("embed_dim", 16),
    )
    print(f"model params: {model.num_parameters():,}", flush=True)
    print(f"device: {args.device}", flush=True)
    print(f"training: {train_cfg_dict}", flush=True)
    print(f"bellman: {bellman_cfg_dict}", flush=True)

    tc = TrainConfig(**train_cfg_dict)
    bc = BellmanConfig(**bellman_cfg_dict)

    def log(stats):
        print(f"epoch {stats.epoch:4d} | loss {stats.loss:.4f} | lr {stats.lr:.2e} "
              f"| {stats.elapsed_s:.1f}s", flush=True)

    result = train_bellman(model, puzzle, tc, bc, checkpoint_dir=args.output, on_epoch_end=log)
    print(f"final loss: {result.final_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
