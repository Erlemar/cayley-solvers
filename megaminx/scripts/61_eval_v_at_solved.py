"""Evaluate V(V0) and V on near-solved states for given checkpoints.

Compares V predictions across BFS-d6 known states. The key metric is V(V0),
which should equal 0 (V0 is solved) but m_curr_v3 reports ~0.91.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/61_eval_v_at_solved.py \\
        --checkpoints megaminx/models/m_curr_v3/epoch_0499.pt,megaminx/models/m_adm_v0_smoke/epoch_0004.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.search import load_model_checkpoint
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


def load_model(path, device, output_dim=1):
    """Polymorphic loader — handles ResMLP and GraphTransformer checkpoints."""
    return load_model_checkpoint(path, device=device, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, type=str)
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = [Path(p) for p in args.checkpoints.split(",")]
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    V0 = torch.tensor([puzzle.solved_state], dtype=torch.int8, device=args.device)

    bfs = BfsBytesTable.load(PROJECT / "data" / "bfs_bytes_d6.pkl")
    items = list(bfs.table.items())[:args.n_samples]
    states = np.empty((len(items), state_size), dtype=np.int8)
    depths = np.empty(len(items), dtype=np.int32)
    for i, (k, p) in enumerate(items):
        states[i] = np.frombuffer(k, dtype=np.int8)
        depths[i] = len(p)
    states_t = torch.from_numpy(states).to(args.device)

    print(f"\n{'checkpoint':>50} | V(V0) | mean V@d=1 | mean V@d=4 | mean V@d=6 | undershoot states (V<d)")
    print("-" * 130)
    for path in paths:
        model = load_model(path, args.device)
        with torch.no_grad():
            v0 = float(model(V0).flatten().cpu().item())
            preds = model(states_t).flatten().cpu().numpy().astype(np.float32)

        n_undershoot = int(np.sum(preds < depths.astype(np.float32) - 0.001))
        per_depth = []
        for d in [1, 4, 6]:
            mask = depths == d
            if mask.sum() > 0:
                per_depth.append(f"{preds[mask].mean():.2f}")
            else:
                per_depth.append("N/A")
        name = str(path)[-50:]
        print(f"{name:>50} | {v0:5.3f} | {per_depth[0]:>10} | {per_depth[1]:>10} | "
              f"{per_depth[2]:>10} | {n_undershoot}/{len(items)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
