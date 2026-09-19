"""Q(s,next) - Q(s,undo) on random-walk pivots, bucketed by pivot depth (IHES, ~30 s).

The forgetting check for Beam-AVI. A locally geodesic walk has a true gap of exactly 2; deep
walks stop being geodesic, so even a perfect model's mean gap shrinks with depth. What matters
is how much each checkpoint KEEPS relative to the incumbent in the mid band.

Measured 2026-09-17: one faithful AVI round cut v1b's walk-depth 9-12 gap from 1.78 to 1.11
(n_red 2.03 -> 4.66). Beam states plus d<=6 anchors never supervise that region.

Columns per band: mean gap, fraction of pivots with gap > 1, mean n_red (#Q < min Q + 1).

    python scripts/94_ihes_gap_by_depth.py --ckpt v1b=models/ihes_tf_b_e1300a_s3/step_02000.pt \
        --ckpt r000=runs/ihes_avi/A/models/r000.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.puzzle import PictureCube  # noqa: E402

BANDS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20), (21, 24), (25, 27)]


def _t90():
    path = PROJECT / "scripts" / "90_ihes_train_avi.py"
    spec = importlib.util.spec_from_file_location("_ihes_t90", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Walk-pivot gap by depth, per checkpoint.")
    ap.add_argument("--ckpt", action="append", required=True, help="name=path.pt")
    ap.add_argument("--n", type=int, default=131072)
    ap.add_argument("--k-max", type=int, default=28)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--chunk-size", type=int, default=8192)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    t90 = _t90()
    t51 = t90._t51()
    puzzle = PictureCube.load(args.data_dir / "puzzle_info.json")
    names = list(puzzle.move_names)
    gens = torch.tensor([puzzle.generators[n] for n in names], dtype=torch.int64, device=dev)
    inv_idx = torch.tensor([names.index(puzzle.inverse_name(n)) for n in names],
                           dtype=torch.int64, device=dev)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=dev)
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    smp = t51.SparseQSampler(gens, inv_idx, solved, 2, args.k_max, 0.0, g)
    parts = [smp.sample(min(16384, args.n - i)) for i in range(0, args.n, 16384)]
    S = torch.cat([p[0] for p in parts])
    piv = torch.cat([p[1] for p in parts])
    prev = torch.cat([p[2] for p in parts])
    nxt = torch.cat([p[3] for p in parts])
    r = torch.arange(S.size(0), device=dev)
    print(f"{S.size(0):,} walk pivots, depth 1-{args.k_max - 1}")
    print(f"{'ckpt':>10} | " + " | ".join(f"d{a:02d}-{b:02d} gap  >1  nred" for a, b in BANDS))
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        p = Path(path) if Path(path).is_absolute() else PROJECT / path
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = t90.build_model(ck, p, dev).eval()
        model.return_value = False
        qs = []
        with torch.no_grad():
            for i in range(0, S.size(0), args.chunk_size):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                    qs.append(model(S[i:i + args.chunk_size]).float())
        q = torch.cat(qs)
        gap = q[r, nxt] - q[r, prev]
        nred = (q < q.min(dim=1, keepdim=True).values + 1.0).sum(dim=1).float()
        cells = []
        for a, b in BANDS:
            k = (piv >= a) & (piv <= b)
            cells.append(f"{float(gap[k].mean()):5.2f} {float((gap[k] > 1).float().mean()):4.2f} "
                         f"{float(nred[k].mean()):5.2f}")
        print(f"{name:>10} | " + " | ".join(cells), flush=True)
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
