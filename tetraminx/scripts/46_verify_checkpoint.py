"""Sanity-check a sparse-Q checkpoint against EXACT BFS ground truth.

Speed means nothing if the weights are wrong, and file size only proves the
tensors are present. This scores a checkpoint on labels we know exactly:

  * the solved state -- INFORMATIONAL ONLY, never a gate. All 24 children sit
    at d=1 so the "right" answer is 1.0, but the converged A100 models predict
    3.8-3.9 there while a 5-epoch model predicts 0.48. The solved state is out
    of distribution for this objective: the sampler pivots at depth >= 1 and the
    anchor pool contains it exactly once in 1.77M. It measures nothing.
  * d=1 states      -- the undo move reaches d=0, the rest are at d=2
  * the d<=5 anchor pool -- exact 24-way Q from the BFS table

    python tetraminx/scripts/46_verify_checkpoint.py \
        --checkpoint tetraminx/models/mx_tf_az_tpu5/epoch_0005.pt
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

from tetraminx.models import build_model, model_from_config
from tetraminx.puzzle import Tetraminx


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a sparse-Q checkpoint.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--anchors", type=Path,
                    default=PROJECT / "tetraminx/data/baked_anchors_d5.npz")
    ap.add_argument("--puzzle", type=Path,
                    default=PROJECT / "tetraminx/data/puzzle_info.json")
    ap.add_argument("--n-anchor", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dev = args.device
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    mcfg = ck["model_config"]
    print(f"checkpoint: {args.checkpoint}")
    print(f"  epoch {ck['epoch']}  val_loss {ck.get('val_loss', float('nan')):.4f}"
          f"  attn_impl {mcfg.get('attn_impl', 'sdpa')}")

    lp = Path(mcfg["layout_path"])
    if not lp.is_absolute():
        lp = PROJECT / lp
    mcfg = {**mcfg, "layout_path": str(lp)}

    model = model_from_config(mcfg).to(dev).eval()
    incompat = model.load_state_dict(ck["state_dict"], strict=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"  loaded strict=True; {n:,} params")

    # The TPU-only attention path must be interchangeable with the default build.
    alt = build_model("transformer", az_head=mcfg.get("az_head", False),
                      layout_path=str(lp), state_size=mcfg["state_size"],
                      num_classes=mcfg["num_classes"], n_actions=mcfg["n_actions"],
                      d_model=mcfg["d_model"], nhead=mcfg["nhead"],
                      num_layers=mcfg["num_layers"], ff_dim=mcfg["ff_dim"],
                      dropout=0.0,
                      attn_impl="sdpa" if mcfg.get("attn_impl") == "einsum"
                      else "einsum").to(dev).eval()
    alt.load_state_dict(ck["state_dict"], strict=True)

    puzzle = Tetraminx.load(args.puzzle)
    names = list(puzzle.move_names)
    A = len(names)
    gen = torch.tensor([puzzle.generators[m] for m in names], dtype=torch.int64, device=dev)
    inv_idx = torch.tensor([names.index(puzzle.inverse_name(m)) for m in names],
                           dtype=torch.int64, device=dev)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=dev)

    with torch.no_grad():
        # --- 1. solved state: every child is at distance 1 -------------------
        q0 = model(solved.unsqueeze(0)).float().squeeze(0)
        print("\n1. solved state (INFORMATIONAL -- converged models score WORSE "
              "here; see the docstring)")
        print(f"   mean {q0.mean():.3f}  min {q0.min():.3f}  max {q0.max():.3f}"
              f"  MAE vs 1.0 {(q0 - 1.0).abs().mean():.3f}")

        # --- 2. d=1 states: undo -> 0, the other 23 -> 2 ---------------------
        d1 = torch.gather(solved.unsqueeze(0).expand(A, -1), 1, gen)
        q1 = model(d1).float()
        rows = torch.arange(A, device=dev)
        undo = q1[rows, inv_idx]
        keep = torch.ones_like(q1, dtype=torch.bool)
        keep[rows, inv_idx] = False
        om = (q1 * keep).sum(dim=1) / keep.sum(dim=1)
        # +inf (not NaN) so min ignores the undo column; torch has no nanmin.
        other_min = q1.masked_fill(~keep, float("inf")).min(dim=1).values
        print("\n2. d=1 states (undo should be ~0.0, the other 23 ~2.0)")
        print(f"   undo:  mean {undo.mean():.3f}  MAE vs 0.0 {undo.abs().mean():.3f}")
        print(f"   other: mean {om.mean():.3f}  MAE vs 2.0 {(om - 2.0).abs().mean():.3f}")
        rank_ok = (undo < other_min).float().mean()
        print(f"   undo ranked strictly best on {rank_ok:.1%} of the 24 states")

        # --- 3. exact anchors ------------------------------------------------
        z = np.load(args.anchors)
        st, qt, dp = z["states"], z["q_targets"], z["depths"]
        rng = np.random.default_rng(0)
        idx = rng.choice(st.shape[0], size=min(args.n_anchor, st.shape[0]),
                         replace=False)
        xs = torch.from_numpy(st[idx]).to(dev).long()
        tt = torch.from_numpy(qt[idx]).to(dev).float()
        dd = torch.from_numpy(dp[idx]).to(dev).long()
        pred = model(xs).float()
        mae = (pred - tt).abs().mean()
        print(f"\n3. exact d<=5 anchors ({len(idx):,} states, 24 actions each)")
        print(f"   MAE vs exact Q: {mae:.4f}")
        best_pred = pred.argmin(dim=1)
        is_opt = (tt.gather(1, best_pred.unsqueeze(1)).squeeze(1)
                  == tt.min(dim=1).values).float().mean()
        print(f"   argmin picks a TRUE optimal move on {is_opt:.1%} of anchors")
        for d in range(int(dd.max()) + 1):
            m = dd == d
            if m.any():
                print(f"     d={d}: MAE {(pred[m] - tt[m]).abs().mean():.4f}"
                      f"  n={int(m.sum()):,}")

        # --- 4. attention-path interchange ----------------------------------
        pa = model(xs[:512]).float()
        pb = alt(xs[:512]).float()
        rel = (pa - pb).abs().max() / pa.abs().max()
        print(f"\n4. attn_impl interchange (einsum vs sdpa, same weights)")
        print(f"   max rel diff {rel:.3e}  "
              f"{'EQUIVALENT' if rel < 1e-4 else 'DIFFERS'}")

    # Bounds calibrated against the A100 references rather than picked by feel:
    #   ep1500 -> undo-rank 100%, anchor MAE 0.069, argmin-optimal 99.7%
    #   ep200  -> undo-rank 100%, anchor MAE 0.110, argmin-optimal 98.9%
    # undo-ranking is the invariant a working model holds at ANY budget; MAE and
    # argmin are left loose enough for an early checkpoint to pass. The solved
    # state is deliberately NOT a gate -- it false-fails ep1500 (see docstring).
    ok = bool(rank_ok > 0.95 and mae < 1.0 and is_opt > 0.5)
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}  "
          f"(undo-rank {rank_ok:.1%} > 95%, anchor MAE {mae:.3f} < 1.0, "
          f"argmin-optimal {is_opt:.1%} > 50%)")
    print("  reference: A100 ep1500 = 100% / 0.069 / 99.7%; "
          "ep200 = 100% / 0.110 / 98.9%")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
