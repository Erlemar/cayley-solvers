"""Targeted beam bench + canaries for a V checkpoint (ResMLPDistance format).

Canaries (megaminx lessons):
  * calibration: V(V0) ~ 0, V(d=1) ~ 1
  * saturation:  V@walk-d80 - V@walk-d40 <= 10 AND V@d80 near the effective
    diameter (Rule 23 + the dodeca false-positive lesson: check the absolute
    level, not just the gap)
  * mid-depth variance: std of V at walk-d ~ 20 (repr_upgrade lesson: tripled
    variance there predicts beam collapse)

Then solves --pids with KhoruzhiiSolver and compares against the floor CSV.

    .venv/Scripts/python.exe scripts/13_bench_v.py \
        --checkpoint models/az_cube_v1/v_only_ep24.pt \
        --pids 106 592 680 772 27 31 \
        --beam 65536 --floor submissions/floor_merged_20260712.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.puzzle import PictureCube
from cayley.search import load_model_checkpoint
from cayley.verify import load_test_states, verify_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--pids", type=int, nargs="+", required=True)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--floor", type=Path, default=PROJECT / "submissions" / "floor_merged_20260712.csv")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-canaries", action="store_true")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    dtype = torch.bfloat16 if args.bf16 and args.device == "cuda" else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    model.eval()

    gens = {nm: np.array(g) for nm, g in puzzle.generators.items()}
    names = list(gens.keys())
    solved = np.arange(72)

    if not args.skip_canaries:
        rng = np.random.default_rng(0)

        def walk_states(depth, n):
            out = np.empty((n, 72), dtype=np.int64)
            for i in range(n):
                s = solved.copy()
                prev = None
                for _ in range(depth):
                    while True:
                        nm = names[rng.integers(len(names))]
                        inv = nm[1:] if nm.startswith("-") else "-" + nm
                        if inv != prev or len(names) == 1:
                            break
                    s = s[gens[nm]]
                    prev = nm
                out[i] = s
            return out

        @torch.no_grad()
        def v_of(arr):
            t = torch.from_numpy(arr).to(args.device)
            return model(t).float().reshape(-1).cpu().numpy()

        v0 = float(v_of(solved[None, :])[0])
        d1 = float(np.mean(v_of(np.stack([solved[gens[nm]] for nm in names]))))
        v20 = v_of(walk_states(20, 256))
        v40 = v_of(walk_states(40, 256))
        v80 = v_of(walk_states(80, 256))
        gap = float(v80.mean() - v40.mean())
        print(f"canaries: V(V0)={v0:.3f}  V(d1)={d1:.3f}  "
              f"V@d20={v20.mean():.2f}(std {v20.std():.2f})  "
              f"V@d40={v40.mean():.2f}  V@d80={v80.mean():.2f}  gap(d80-d40)={gap:.2f}")
        print(f"  saturation gate (gap<=10): {'PASS' if gap <= 10 else 'FAIL'}")

    floor = {}
    with open(args.floor, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            floor[int(row["initial_state_id"])] = len(row["path"].split(".")) if row["path"] else 0

    states = load_test_states(PROJECT / "data" / "test.csv")
    solver = KhoruzhiiSolver(puzzle, model, device=args.device)
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps)

    total_ours = total_floor = 0
    n_solved = 0
    for pid in args.pids:
        t0 = time.time()
        found, length, raw_path = solver.solve(states[pid], cfg)
        wall = time.time() - t0
        fl = floor.get(pid, -1)
        if found:
            ok = verify_path(puzzle, states[pid], raw_path).ok
            n_solved += 1
            total_ours += length
            total_floor += fl
            delta = length - fl
            print(f"pid {pid:>4}: found {length:>3} (verify={ok})  floor {fl:>3}  "
                  f"delta {delta:+d}  {wall:.0f}s")
        else:
            print(f"pid {pid:>4}: NOT FOUND (floor {fl})  {wall:.0f}s")
    print(f"\nsolved {n_solved}/{len(args.pids)}  ours {total_ours} vs floor {total_floor} "
          f"({total_ours - total_floor:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
