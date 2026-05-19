"""Bellman refinement with admissibility-aware loss term.

`loss = bellman + lambda_pdb * mean(relu(h_PDB(s) - V_pred(s))^2)`

Where h_PDB is a max-of-N disjoint K=5 corner PDBs (admissible lower bound on d(s)).
Penalty pressures V to never undershoot the PDB heuristic — addresses the
V_full(V0)≈0.91 undershoot bug observed in m_curr_v3.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/60_train_admissible.py \\
        --config megaminx/configs/m_adm_v0.yaml \\
        --output megaminx/models/m_adm_v0_smoke \\
        --epochs 50
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
from megaminx.pdb_heuristic import CornerPDBHeuristic
from megaminx.puzzle import Megaminx


def _build_model(model_cfg: dict):
    """Dispatch on `model_class`. Defaults to ResMLPDistance for backward compat."""
    model_class = model_cfg.get("model_class", "ResMLPDistance")
    if model_class in ("GraphTransformerV", "GraphTransformerVPi"):
        from megaminx.graph_transformer import build_graph_transformer_from_config
        gf = torch.load(PROJECT / "data" / "graph_features.pt", map_location="cpu",
                        weights_only=False)
        return build_graph_transformer_from_config(model_cfg, gf)
    return ResMLPDistance(
        state_size=model_cfg["state_size"],
        num_classes=model_cfg["num_classes"],
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        num_res_blocks=model_cfg["num_res_blocks"],
        encoding=model_cfg.get("encoding", "onehot"),
        embed_dim=model_cfg.get("embed_dim", 16),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--pdb-paths", type=str, default=None,
                    help="comma-separated PDB paths. Default: all 4 K5 corner PDBs.")
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
    model = _build_model(model_cfg)
    print(f"model params: {model.num_parameters():,}  "
          f"class={model_cfg.get('model_class', 'ResMLPDistance')}")
    print(f"device: {args.device}")
    print(f"training: {train_cfg_dict}")
    print(f"bellman: {bellman_cfg_dict}")

    tc = TrainConfig(**train_cfg_dict)
    bc = BellmanConfig(**bellman_cfg_dict)

    # Build PDB heuristic if lambda_pdb > 0
    pdb_lookup_fn = None
    if bc.lambda_pdb > 0:
        if args.pdb_paths:
            pdb_paths = [Path(p) for p in args.pdb_paths.split(",")]
        else:
            pdb_paths = [
                PROJECT / "data" / f"pdb_corner_K5{suffix}.pkl"
                for suffix in ["", "_p1", "_p2", "_p3"]
            ]
        for p in pdb_paths:
            if not p.exists():
                raise FileNotFoundError(p)
        h = CornerPDBHeuristic(pdb_paths, PROJECT / "data" / "corner_tables.pkl",
                                device=args.device)
        def pdb_lookup_fn(states):
            # Pass int8 states; CornerPDBHeuristic handles dtype.
            return h.lookup(states)
        print(f"[adm] PDB heuristic loaded: {len(h.pdbs)} PDBs covering "
              f"{sorted(set(c for p in h.pdbs for c in p['target_corners']))}", flush=True)
        print(f"[adm] lambda_pdb = {bc.lambda_pdb}", flush=True)

    def log(stats):
        print(f"epoch {stats.epoch:4d} | loss {stats.loss:.4f} | "
              f"lr {stats.lr:.2e} | {stats.elapsed_s:.1f}s", flush=True)

    result = train_bellman(
        model, puzzle, tc, bc, checkpoint_dir=args.output, on_epoch_end=log,
        pdb_lookup_fn=pdb_lookup_fn,
    )
    print(f"final loss: {result.final_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
