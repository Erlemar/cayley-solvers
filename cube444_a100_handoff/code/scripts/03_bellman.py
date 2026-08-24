"""Stage 3 -- Bellman refinement of the 4x4x4 V model. The load-bearing stage.

    python3 cube444/scripts/03_bellman.py --config cube444/configs/c_bellman.yaml \
        --output cube444/models/c_bellman

Target: V(s) <- 1 + min_a V_target(child_a), with exact-label anchors mixed into
every batch (solved -> 0, the 24 d=1 children -> 1) plus a BFS d<=6 mixin.
The anchors are what prevent the V(V0)~2 bootstrap bug.

No PDB term (lambda_pdb=0): we have no PDB for this puzzle yet, and max-combining
a corner PDB with a strong V was neutral on megaminx.

Run 04_eval_v.py on the output BEFORE handing any checkpoint to a beam.
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
from cayley.training import TrainConfig
from cube444.puzzle import Cube444

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

build_model = import_module("02_train").build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg, train_cfg_dict, bellman_cfg_dict = cfg["model"], cfg["training"], cfg["bellman"]
    if args.epochs is not None:
        train_cfg_dict["n_epochs"] = args.epochs
    train_cfg_dict["device"] = args.device
    train_cfg_dict["seed"] = cfg.get("seed", 0)

    puzzle = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    model = build_model(model_cfg)
    print(f"model params: {model.num_parameters():,}")
    print(f"device: {args.device}")
    print(f"training: {train_cfg_dict}")
    print(f"bellman: {bellman_cfg_dict}", flush=True)

    tc = TrainConfig(**train_cfg_dict)
    bc = BellmanConfig(**bellman_cfg_dict)

    def log(stats):
        print(f"epoch {stats.epoch:4d} | loss {stats.loss:.4f} | lr {stats.lr:.2e} | "
              f"{stats.elapsed_s:.1f}s", flush=True)

    result = train_bellman(model, puzzle, tc, bc, checkpoint_dir=args.output, on_epoch_end=log)
    print(f"final loss: {result.final_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
