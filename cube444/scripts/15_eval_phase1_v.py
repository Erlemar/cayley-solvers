"""Calibration / discrimination gate for a phase-1 masked V.

The cube444 rule is that ABSOLUTE V scale is not the thing to gate on -- Bellman
drives it down and the more-collapsed checkpoint beams better. What matters is
DISCRIMINATION relative to noise. But there is a floor to that: if V collapses to a
*constant* near 0 there is no ordering left at all, and the phase-1 beam stalls with
defects still on the board.

So this reports, per checkpoint:
  V(k)            mean predicted distance for states at walk depth k from R
  spread          V(30) - V(5): is there any usable range left?
  SNR             discrimination / std at mid depth
  child gap       V(next) - V(undo) at a pivot -- the quantity a beam actually ranks
                  on, and the one the sparse-Q margin term is meant to protect
  frac V<0.5      fraction of deep states the model believes are already reduced
                  (the collapse signature that stalls the beam)

Run:
  .venv/Scripts/python.exe cube444/scripts/15_eval_phase1_v.py \
      --models cube444/models/p1_control/*.pt
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module  # noqa: E402

from cube444.phase1 import Phase1Sampler, ReductionTest, WalkSpec  # noqa: E402
from cube444.puzzle import Cube444  # noqa: E402

load_masked = import_module("12_train_phase1").load_masked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--depths", type=int, nargs="+",
                    default=[1, 3, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--gap-depth", type=int, default=20,
                    help="pivot depth at which the child gap is reported. The "
                         "gap-of-2 ideal only holds while the walk is roughly "
                         "geodesic, so report it where the margin term trains.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    sampler = Phase1Sampler(puz, device=args.device, seed=1234)
    rt = ReductionTest(args.device)

    # fixed evaluation states, shared across checkpoints
    depths = sorted(set(args.depths) | {args.gap_depth})
    banks = {}
    for k in depths:
        spec = WalkSpec(k_max=k, h_walk_max=40, n_back=1)
        s, _, prev, nxt = sampler.walk(args.batch, spec)
        banks[k] = (s, prev, nxt)
    h_bank, _ = sampler.r_anchors(args.batch, WalkSpec())

    paths = []
    for pat in args.models:
        paths.extend(sorted(glob.glob(pat)))
    print(f"{'checkpoint':44s} {'V(R)':>6s} " +
          " ".join(f"V{k:<3d}" for k in args.depths) +
          f" {'spread':>7s} {'SNR':>6s} {'gap':>6s} {'frac<.5':>8s}")

    for p in paths:
        model = load_masked(Path(p), args.device)
        with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16,
                                             enabled=args.device == "cuda"):
            vr = float(model(h_bank).float().mean())
            means, stds, gaps = {}, {}, {}
            for k in depths:
                s, prev, nxt = banks[k]
                v = model(s).float()
                means[k] = float(v.mean())
                stds[k] = float(v.std())
                gaps[k] = float((model(nxt).float() - model(prev).float()).mean())
            deep = banks[max(args.depths)][0]
            frac = float((model(deep).float() < 0.5).float().mean())

        mid = args.depths[len(args.depths) // 2]
        spread = means[max(args.depths)] - means[args.depths[2]]
        snr = (means[max(args.depths)] - means[args.depths[0]]) / max(stds[mid], 1e-6)
        name = str(Path(p).parent.name + "/" + Path(p).stem)
        print(f"{name:44s} {vr:6.2f} " +
              " ".join(f"{means[k]:4.1f}" for k in args.depths) +
              f" {spread:7.2f} {snr:6.2f} {gaps[args.gap_depth]:6.2f} {frac:8.3f}")

    print("\n  spread  = V(deepest) - V(5). Below ~2 there is no ordering left.")
    print(f"  gap     = V(next)-V(undo) at pivot depth {args.gap_depth}; ideal 2.0.")
    print("            Reported here, not at the deepest depth, because a fully")
    print("            mixed walk has a TRUE gap near 0 -- the ideal only holds")
    print("            while the walk is still roughly geodesic.")
    print("  frac<.5 = deep states the model calls 'already reduced' -> beam stalls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
