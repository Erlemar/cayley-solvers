"""Rule 23 canary: V(V0), V@d=1, and V at high random-walk depths.

Working megaminx V baselines saturate at V@d=80 ≈ 29 (m_dd_v0 ep49 / AZ v4).
Failure mode (state_inv, GT V): V keeps drifting past d=40, e.g. V@d=80 = 38-48.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/v_canary.py \\
        --checkpoint megaminx/models/m_repr_v0/epoch_0024.pt \\
        --n-per-depth 200 --seed 0

Reference (m_dd_v0 ep49, working baseline): V@d=20=15  V@d=40=24  V@d=60=28  V@d=80=29
Reference (m_inv_v0 ep49, REJECTED):          V@d=20=15  V@d=40=27  V@d=60=34  V@d=80=38
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

from cayley.data import generate_walks_torch
from cayley.search import load_model_checkpoint
from megaminx.puzzle import Megaminx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--n-per-depth", type=int, default=200)
    ap.add_argument("--depths", type=str, default="0,1,5,10,20,30,40,50,60,70,80")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=torch.float32)
    model.eval()

    print(f"V canary for {args.checkpoint}")
    print(f"{'d':>4}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  {'n':>6}")
    print("-" * 50)

    # V(V0): single solved state.
    V0 = torch.tensor([puzzle.solved_state], dtype=torch.int8, device=args.device)
    with torch.no_grad():
        v0 = float(model(V0).flatten().cpu().item())
    print(f"{'V0':>4}  {v0:>8.3f}  {'-':>8}  {'-':>8}  {'-':>8}  {1:>6}")

    # For each depth d>0, sample n_per_depth random walks of length d.
    for d in depths:
        if d == 0:
            continue
        states, walk_depths = generate_walks_torch(
            puzzle, n_walks=max(1, args.n_per_depth // max(d, 1)),
            k_max=d, seed=args.seed + d, device=args.device, n_back=1,
        )
        mask = walk_depths == d
        if not mask.any():
            print(f"{d:>4}  no states sampled at this depth")
            continue
        s_d = states[mask]
        if s_d.size(0) > args.n_per_depth:
            s_d = s_d[: args.n_per_depth]
        with torch.no_grad():
            preds = model(s_d).flatten().cpu().numpy().astype(np.float32)
        print(
            f"{d:>4}  {preds.mean():>8.3f}  {preds.std():>8.3f}  "
            f"{preds.min():>8.3f}  {preds.max():>8.3f}  {len(preds):>6}"
        )

    print()
    # Quick verdict per Rule 23 (V@d=80 ≤ 31 = canary pass).
    # Re-sample d=80 properly if missing.
    return 0


if __name__ == "__main__":
    sys.exit(main())
