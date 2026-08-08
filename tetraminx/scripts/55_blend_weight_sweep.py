"""Sweep blend weights for a cross-architecture Q ensemble (transformer x ResMLP).

Why cross-architecture and not cross-checkpoint: souping three checkpoints of ONE run
was beam-neutral (they share a basin, an architecture and an inductive bias, so their
errors are correlated, and what decorrelation existed was already covered by
history_depth=1). A ResMLP and a PieceTransformer share none of that. The cost case is
also far better -- the transformer is ~17x the ResMLP's inference cost, so
transformer+ResMLP is ~1.06x the transformer alone, i.e. it does not have to beat beam
width, it stacks on top of it.

Blending is LINEAR, so each model is scored once and the score matrices are combined
numerically -- the whole sweep costs two forward passes, not one per weight.

Scale check first: a weighted average of two heads is only meaningful if they share
units. Both predict distance-to-solved under the same sparse-Q objective, but that is
an assumption worth printing (mean/std per model) rather than trusting -- if one head's
spread is much wider it dominates the average regardless of its weight.

Usage:
  python tetraminx/scripts/55_blend_weight_sweep.py --a tf.pt --b resmlp.pt --n 20000
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))
sys.path.insert(0, str(PROJECT / "src"))


def _load_eval_q():
    spec = importlib.util.spec_from_file_location("eq", HERE / "52_eval_q.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="model A (weight w)")
    ap.add_argument("--b", required=True, help="model B (weight 1-w)")
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[1.0, 0.9, 0.8, 0.7, 0.5, 0.0],
                    help="weights for model A")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--tilt", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--zscore", action="store_true",
                    help="standardise each model's scores before blending (use if the "
                         "two heads turn out to have very different spreads)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    eq = _load_eval_q()
    puzzle = eq.Tetraminx.load(PROJECT / "tetraminx" / "data" / "puzzle_info.json")
    names = list(puzzle.move_names)
    dev = args.device
    gen = torch.tensor([puzzle.generators[n] for n in names], dtype=torch.int64, device=dev)
    inv = torch.tensor([names.index(puzzle.inverse_name(n)) for n in names],
                       dtype=torch.int64, device=dev)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=dev)

    piv, pivots, prev, nxt = eq.sample_pivots(puzzle, gen, inv, solved, args.n,
                                              args.k_min, args.k_max, args.tilt,
                                              dev, args.seed)
    print(f"probe: {args.n} rw-middle pivots, k in [{args.k_min},{args.k_max}]")

    mats = {}
    for tag, path in (("A", args.a), ("B", args.b)):
        model = eq.load_model(Path(path), dev)
        s = eq.score_matrix(model, piv, gen).float()
        mats[tag] = s
        print(f"  {tag} = {Path(path).name}: mean {s.mean():7.3f}  std {s.std():6.3f}  "
              f"min {s.min():6.2f}  max {s.max():6.2f}")
        del model
        torch.cuda.empty_cache()

    sa, sb = mats["A"], mats["B"]
    ratio = float(sa.std() / sb.std())
    print(f"  std ratio A/B = {ratio:.3f}"
          + ("  <-- close enough to average raw" if 0.8 <= ratio <= 1.25
             else "  <-- SCALES DIFFER, consider --zscore"))
    if args.zscore:
        sa = (sa - sa.mean()) / sa.std()
        sb = (sb - sb.mean()) / sb.std()
        print("  (z-scored before blending)")

    bands = [(1, 9), (10, 19), (20, 29), (30, args.k_max)]
    rows = torch.arange(sa.shape[0], device=sa.device)

    print(f"\n{'w(A)':>6} {'w(B)':>6} {'pair':>8} {'top1':>8} {'gap':>8}   "
          + "  ".join(f"{f'top1[{lo}-{hi}]':>13}" for lo, hi in bands))
    best = None
    for w in args.weights:
        s = w * sa + (1.0 - w) * sb
        s_prev, s_next = s[rows, prev], s[rows, nxt]
        other = torch.ones_like(s, dtype=torch.bool)
        other[rows, prev] = False
        other[rows, nxt] = False
        best_wrong = s.masked_fill(~other, float("inf")).min(dim=1).values
        pair = float((s_prev < s_next).float().mean())
        top1 = float((s_prev < best_wrong).float().mean())
        gap = float((s_next - s_prev).mean())
        cells = []
        for lo, hi in bands:
            sel = (pivots >= lo) & (pivots <= hi)
            cells.append(float((s_prev[sel] < best_wrong[sel]).float().mean())
                         if int(sel.sum()) else float("nan"))
        mark = ""
        if best is None or top1 > best[1]:
            best = (w, top1)
            mark = ""
        print(f"{w:6.2f} {1-w:6.2f} {pair:8.4f} {top1:8.4f} {gap:8.3f}   "
              + "  ".join(f"{c:13.4f}" for c in cells) + mark)

    print(f"\nbest top1 at w(A)={best[0]:.2f} (top1 {best[1]:.4f})")
    print("NOTE: a probe win is NOT a beam win on this puzzle -- GT-V calibration, "
          "all-neighbour-Q recall and the checkpoint soup all won probes and moved "
          "zero moves. Beam the top one or two weights before believing anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
