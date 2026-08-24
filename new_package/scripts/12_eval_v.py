"""Acceptance gates for a Tetraminx V model.

Reports, in one pass:

  1. V(solved) and mean V over the 24 depth-1 children.
     Gate: V(solved) ~ 0, V(d=1) ~ 1.  A V0 that reads ~2 is the classic Bellman
     bootstrap bug -- the whole landscape sits too high near the goal.
  2. Exact calibration against the BFS table: mean V per TRUE distance 0..6.
  3. V by random-walk depth up to --deep-k (default 80), with std.
     Gate (Rule 23 + the dodeca false-positive lesson): V must SATURATE near the
     puzzle diameter.  Two-sided check --
        drift    : V@d80 - V@d40 <= --max-drift  (else it is predicting walk
                   depth, not distance -- GT-V failure mode)
        absolute : V@d80 within [--diam-lo, --diam-hi] (else it collapsed low --
                   the dodecahedral-CNN false positive, which passed the drift
                   check while under-predicting everything)
  4. Mid-depth spread: std of V at walk depth ~20.  The repr-bundle failure had a
     fine mean and a tripled variance, and variance is what kills a beam.
     NOTE this number is puzzle-calibrated, not universal: at a FIXED WALK DEPTH
     the true distance itself varies, so part of the spread is real.  Measured
     tetraminx baseline (tv0 Stage A, RW regression): std(V@d20) = 3.71 with a
     std of only 0.45 at exact d=6.  The gate default (5.0) is headroom over that
     baseline -- what matters is a variant BLOWING IT UP, not the absolute value.

    python3 tetraminx/scripts/12_eval_v.py --checkpoint <ckpt.pt>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.model import ResMLPDistance
from tetraminx.puzzle import Tetraminx

# Counting bound: |G| = 1.54e32, non-backtracking branching ~15.8 -> a uniformly
# random state sits at ~27 moves.  A saturating V should land in this band.
DIAM_LO_DEFAULT = 22.0
DIAM_HI_DEFAULT = 34.0


def load_model(ckpt_path: Path, device: str) -> ResMLPDistance:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=cfg.get("state_size", 88),
        num_classes=cfg.get("num_classes", 88),
        hidden_dims=tuple(cfg.get("hidden_dims", (2048, 512))),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        encoding=cfg.get("encoding", "embedding"),
        embed_dim=cfg.get("embed_dim", 16),
    )
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def predict(model, states: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    out = []
    for i in range(0, states.size(0), chunk):
        out.append(model(states[i:i + chunk]).flatten().float())
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "puzzle_info.json")
    ap.add_argument("--bfs-anchors", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "bfs_d6_train.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deep-k", type=int, default=80)
    ap.add_argument("--n-walks", type=int, default=4000)
    ap.add_argument("--max-drift", type=float, default=3.0)
    ap.add_argument("--diam-lo", type=float, default=DIAM_LO_DEFAULT)
    ap.add_argument("--diam-hi", type=float, default=DIAM_HI_DEFAULT)
    ap.add_argument("--max-mid-std", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    puzzle = Tetraminx.load(args.puzzle_info)
    table = GeneratorTable.from_puzzle(puzzle)
    model = load_model(args.checkpoint, args.device)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=args.device)
    n = solved.numel()
    failures: list[str] = []

    # ---- 1. V(solved) and V(d=1) -----------------------------------------
    v0 = predict(model, solved[None, :]).item()
    perms = torch.tensor(table.perms, dtype=torch.int64, device=args.device)
    children = solved[perms]                       # (n_gen, S)
    v1 = predict(model, children)
    print(f"V(solved)      = {v0:+.3f}      [gate: |V| < 0.5]")
    print(f"V(d=1)  mean   = {v1.mean():+.3f}  std {v1.std():.3f}   [gate: 0.5 .. 1.5]")
    if abs(v0) >= 0.5:
        failures.append(f"V(solved)={v0:+.3f} not ~0")
    if not (0.5 <= v1.mean().item() <= 1.5):
        failures.append(f"V(d=1)={v1.mean():.3f} not ~1")

    # ---- 2. exact calibration on BFS states -------------------------------
    if args.bfs_anchors.exists():
        d = torch.load(args.bfs_anchors, map_location="cpu", weights_only=False)
        bs, bd = d["states"], d["distances"]
        rng = np.random.default_rng(args.seed)
        print("\nexact-distance calibration (BFS table):")
        for dist in range(0, int(bd.max()) + 1):
            idx = (bd == dist).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            sel = idx[torch.from_numpy(
                rng.choice(idx.numel(), size=min(4000, idx.numel()), replace=False))]
            v = predict(model, bs[sel].to(torch.int64).to(args.device))
            sd = v.std().item() if v.numel() > 1 else 0.0
            print(f"  true d={dist}: V = {v.mean():+.3f}  (std {sd:.3f}, n={sel.numel()})")
    else:
        print(f"\n[skip] no BFS anchors at {args.bfs_anchors}")

    # ---- 3/4. walk-depth profile, saturation, mid-depth spread ------------
    states, depths = generate_walks_torch(
        puzzle, n_walks=args.n_walks, k_max=args.deep_k, device=args.device,
        seed=args.seed, n_back=1)
    v = predict(model, states)
    print(f"\nrandom-walk depth profile (k_max={args.deep_k}):")
    probes = [1, 5, 10, 15, 20, 25, 30, 40, 60, args.deep_k]
    means: dict[int, float] = {}
    stds: dict[int, float] = {}
    for k in probes:
        if k > args.deep_k:
            continue
        m = depths == k
        if m.sum() == 0:
            continue
        vv = v[m]
        means[k], stds[k] = vv.mean().item(), vv.std().item()
        print(f"  d={k:3d}: V = {means[k]:+.3f}  std {stds[k]:.3f}  n={int(m.sum())}")

    print("\n--- gates ---")
    if 40 in means and args.deep_k in means:
        drift = means[args.deep_k] - means[40]
        print(f"saturation drift  V@d{args.deep_k} - V@d40 = {drift:+.2f}  "
              f"[gate: <= {args.max_drift}]")
        if drift > args.max_drift:
            failures.append(f"no saturation: drift {drift:+.2f}")
        absolute = means[args.deep_k]
        print(f"absolute level    V@d{args.deep_k} = {absolute:.2f}  "
              f"[gate: {args.diam_lo} .. {args.diam_hi}]")
        if not (args.diam_lo <= absolute <= args.diam_hi):
            failures.append(f"V@d{args.deep_k}={absolute:.2f} outside diameter band")
    mid = stds.get(20)
    if mid is not None:
        print(f"mid-depth spread  std(V@d20) = {mid:.2f}  [gate: <= {args.max_mid_std}]")
        if mid > args.max_mid_std:
            failures.append(f"mid-depth std {mid:.2f} too high")

    if failures:
        print("\nRESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
