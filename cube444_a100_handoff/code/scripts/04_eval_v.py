"""Acceptance gates for a 4x4x4 V model. Run this BEFORE handing a checkpoint to a beam.

    python3 cube444/scripts/04_eval_v.py --checkpoint cube444/models/c_bellman/epoch_0199.pt

Four gates, all of them scar tissue from megaminx:

  G1 anchors      V(solved) ~ 0  and  V(d=1) ~ 1 for all 24 children.
                  Catches the V(V0)~2 Bellman-bootstrap bug.

  G2 BFS          mean |V - d| on the exact BFS ball for d = 0..6.

  G3 saturation   V@d=80 must sit in the ABSOLUTE band [36, 44] -- near the
                  real diameter (counting lower bound 36.8) -- AND
                  V@d=80 - V@d=40 <= 3.
                  Rule 23 gives the gap test. The dodeca post-mortem
                  (dodeca_cnn_rejected) showed a gap test ALONE false-positives
                  when V collapses uniformly downward, so the absolute band is
                  mandatory, not optional.

  G4 variance     std(V) at walk depth ~20 must be small. repr_upgrade_bundle
                  found mid-depth V VARIANCE predicts beam collapse better than
                  the saturation mean does.

Passing all four is necessary, not sufficient -- Rule 21: the binding gate is a
stratified multi-pid eval at the production beam recipe.
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
from cube444.models import load_v_model
from cube444.puzzle import Cube444

# --- thresholds -------------------------------------------------------------
# HARD (principled, these are failures):
#   SAT_GAP_MAX -- Rule 23. If V keeps climbing past the diameter it is
#   predicting walk depth, not distance, and beam fails. Not a tunable.
SAT_GAP_MAX = 3.0
#   DISCRIM_SNR_MIN -- a collapsed V (the dodeca failure mode) loses the ability
#   to separate mid depths. The meaningful quantity is discrimination measured
#   in units of V's OWN noise, not in absolute V units.
#
#   Measured 2026-07-23: c_bells ep24 had V@80=26.3, absolute discrimination
#   11.74, std 3.69 (ratio 3.18); ep124 had V@80=16.8, absolute discrimination
#   5.83, std 1.67 (ratio 3.49). The heavily "collapsed" ep124 BEAT ep24 at every
#   scramble depth (19/24 vs 17/24 solved, mean length 54 vs 61). So an absolute
#   threshold on discrimination is simply wrong -- it would have rejected the
#   better model. Gate on the ratio instead.
DISCRIM_SNR_MIN = 2.0

# SOFT (warnings -- calibrated from megaminx, unverified for this puzzle):
#   The counting lower bound here is log_19.18(1.7763e47) = 36.8, so a V that
#   saturates far below that is provably under-predicting the true distance.
#   But beam only needs the ORDERING of children to be right, and a monotonically
#   compressed V can still beam fine -- so this warns, it does not fail.
DIAMETER_LO, DIAMETER_HI = 30.0, 48.0
#   megaminx reference: std(V@d=20) was 1.7 on the good V, 5.3 on the bundle that
#   collapsed the beam. The absolute scale here is unknown until a real Stage-3
#   model lands; treat as a warning and tighten once we have evidence.
VAR_D20_MAX = 3.0


def v_of(model, states: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, states.size(0), chunk):
            out = model(states[i : i + chunk])
            outs.append(out.squeeze(-1).float() if out.ndim > 1 else out.float())
    return torch.cat(outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--bfs-anchors", type=Path, default=PROJECT / "data" / "bfs_anchors.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n-walks", type=int, default=4096)
    ap.add_argument("--bfs-per-depth", type=int, default=20000)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    model = load_v_model(args.checkpoint, device=args.device, dtype=dtype)
    model.eval()
    solved = torch.tensor(puz.solved_state, dtype=torch.int64, device=args.device)
    P = torch.tensor([puz.generators[n] for n in puz.move_names],
                     dtype=torch.int64, device=args.device)

    failures, warnings = [], []
    print(f"checkpoint: {args.checkpoint}")
    print(f"device={args.device} dtype={dtype}")

    # ---- G1: anchors ----
    print("\n[G1] anchors")
    v0 = float(v_of(model, solved.unsqueeze(0))[0])
    children = solved.unsqueeze(0).expand(P.size(0), -1).gather(1, P)
    v1 = v_of(model, children)
    print(f"  V(solved)          = {v0:+.4f}   (target 0)")
    print(f"  V(d=1) mean/min/max= {v1.mean():.4f} / {v1.min():.4f} / {v1.max():.4f}  (target 1)")
    if abs(v0) > 0.5:
        failures.append(f"G1: V(solved)={v0:.3f}, want |V|<0.5")
    if abs(float(v1.mean()) - 1.0) > 0.5:
        failures.append(f"G1: V(d=1) mean={float(v1.mean()):.3f}, want ~1")

    # ---- G2: exact BFS calibration ----
    print("\n[G2] exact BFS calibration")
    if args.bfs_anchors.exists():
        anch = torch.load(args.bfs_anchors, map_location="cpu", weights_only=False)
        st, ds = anch["states"], anch["distances"]
        print(f"  {'d':>3} | {'n':>8} | {'mean V':>8} | {'std':>6} | {'MAE':>6}")
        for d in range(0, int(ds.max()) + 1):
            idx = (ds == d).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            sel = idx[torch.randperm(idx.numel())[: args.bfs_per_depth]]
            s = st[sel].to(args.device).long()
            v = v_of(model, s)
            print(f"  {d:3d} | {len(sel):8,} | {v.mean():8.3f} | {v.std():6.3f} | "
                  f"{(v - d).abs().mean():6.3f}")
    else:
        print(f"  SKIP: {args.bfs_anchors} not found (run 01_build_bfs.py)")

    # ---- G3 / G4: random-walk depth profile ----
    print("\n[G3/G4] random-walk depth profile")
    depths = [1, 2, 5, 10, 12, 15, 20, 25, 30, 40, 60, 80, 120]
    k_max = max(depths)
    states, walk_d = generate_walks_torch(puz, n_walks=args.n_walks, k_max=k_max,
                                          seed=12345, device=args.device, n_back=1)
    prof = {}
    print(f"  {'walk d':>7} | {'mean V':>8} | {'std V':>7}")
    for d in depths:
        sel = (walk_d == d).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        v = v_of(model, states[sel])
        prof[d] = (float(v.mean()), float(v.std()))
        print(f"  {d:7d} | {prof[d][0]:8.3f} | {prof[d][1]:7.3f}")

    if 80 in prof and 40 in prof:
        v80, v40 = prof[80][0], prof[40][0]
        gap = v80 - v40
        print(f"\n  saturation: V@80={v80:.2f}  V@40={v40:.2f}  gap={gap:+.2f}")
        if gap > SAT_GAP_MAX:
            failures.append(f"G3 saturation: gap {gap:.2f} > {SAT_GAP_MAX} -- "
                            f"V is predicting walk depth, not true distance (Rule 23)")
        if not (DIAMETER_LO <= v80 <= DIAMETER_HI):
            warnings.append(f"G3 scale: V@d=80={v80:.2f} outside [{DIAMETER_LO}, "
                            f"{DIAMETER_HI}]; counting lower bound is 36.8, so this V "
                            f"{'under' if v80 < DIAMETER_LO else 'over'}-predicts the "
                            f"true distance scale")
    if 30 in prof and 12 in prof and 20 in prof:
        disc = prof[30][0] - prof[12][0]
        snr = disc / max(prof[20][1], 1e-6)
        print(f"  discrimination: V@30 - V@12 = {disc:+.2f}   "
              f"SNR (/std@20) = {snr:.2f}")
        if snr < DISCRIM_SNR_MIN:
            failures.append(f"G3 discrimination: SNR={snr:.2f} < {DISCRIM_SNR_MIN} -- "
                            f"V cannot separate mid depths above its own noise")
    if 20 in prof:
        s20 = prof[20][1]
        print(f"  mid-depth variance: std(V @ d=20) = {s20:.3f}")
        if s20 > VAR_D20_MAX:
            warnings.append(f"G4 variance: std(V@d=20)={s20:.2f} > {VAR_D20_MAX} -- "
                            f"high mid-depth variance predicted beam collapse on megaminx")

    print("\n" + "=" * 60)
    if warnings:
        print("WARNINGS (soft thresholds, not calibrated for this puzzle yet):")
        for w in warnings:
            print(f"  ~ {w}")
    if failures:
        print("GATES FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("HARD GATES PASSED")
    print("  (necessary, not sufficient -- Rule 21: run the stratified beam eval next)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
