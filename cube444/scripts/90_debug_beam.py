"""Diagnose WHY the beam fails: solve-rate vs known scramble depth, plus the
per-step min-V trajectory for a single scramble.

    python3 cube444/scripts/90_debug_beam.py --checkpoint <ckpt> --beam 16384 \
        --depths 10,15,20,25,30,40 --n-per-depth 4 --bf16

The test.csv ladder gives only ONE puzzle per random-walk length, so a failure
there is one sample. This generates n scrambles at each depth from the solved
state, so the depth-vs-solve-rate curve is actually estimable.

The --trace mode runs one scramble and prints, per beam step, the minimum V in
the beam. That separates the three failure modes:
  * V decreases steadily then stalls  -> V landscape is flat past its saturation
  * V never decreases                 -> V is not informative at this depth
  * loop exits early                  -> stagnation break or empty beam (a bug)
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

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver, _state_hash
from cube444.models import load_v_model
from cube444.puzzle import Cube444


def scramble(puz, P, rng, depth):
    s = np.array(puz.solved_state, dtype=np.int64)
    prev = -1
    for _ in range(depth):
        while True:
            g = int(rng.integers(0, len(P)))
            if prev < 0 or g != prev ^ 1:   # crude non-backtrack (pairs are (m, -m))
                break
        s = s[P[g]]
        prev = g
    return s


def trace_one(solver, puz, state, beam, steps):
    """Run the beam manually and report min-V per step."""
    dev = solver.device
    st = torch.tensor(state.tolist(), dtype=solver.state_dtype, device=dev).unsqueeze(0)
    bad = torch.empty(0, dtype=torch.int64, device=dev)
    print(f"  {'step':>5} {'beam':>9} {'minV':>8} {'meanV':>8}   note")
    for j in range(steps):
        st, y, mv, idx = solver._do_greedy_step(st, bad, beam)
        if st.numel() == 0:
            print(f"  {j:5d}  EMPTY BEAM -- search died")
            return
        solved_hit = (st == solver.V0).all(dim=1).any().item()
        if j % 5 == 0 or solved_hit or j < 3:
            print(f"  {j:5d} {st.size(0):9,} {float(y.min()):8.3f} {float(y.float().mean()):8.3f}"
                  f"   {'*** SOLVED ***' if solved_hit else ''}")
        if solved_hit:
            return
    print("  (ran out of steps without reaching solved)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--beam", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--depths", type=str, default="10,15,20,25,30,40")
    ap.add_argument("--n-per-depth", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--trace-depth", type=int, default=None,
                    help="if set, trace one scramble at this depth instead of sweeping")
    ap.add_argument("--trace-pid", type=int, default=None,
                    help="if set, trace this REAL test.csv puzzle instead of a synthetic one")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    P = np.array([puz.generators[n] for n in puz.move_names], dtype=np.int64)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_v_model(args.checkpoint, device=args.device, dtype=dtype)
    solver = KhoruzhiiSolver(puz, model, device=args.device, internal_batch_size=2 ** 15)
    rng = np.random.default_rng(args.seed)
    print(f"checkpoint: {args.checkpoint}\nbeam={args.beam} max_steps={args.max_steps}")

    if args.trace_pid is not None:
        import csv
        with open(PROJECT / "data" / "test.csv", encoding="utf-8") as f:
            row = {int(r["initial_state_id"]): r for r in csv.DictReader(f)}[args.trace_pid]
        s = np.array([int(x) for x in row["initial_state"].split(",")], dtype=np.int64)
        print(f"\ntrace: REAL pid {args.trace_pid} ({row['comment']})")
        trace_one(solver, puz, s, args.beam, args.max_steps)
        return 0

    if args.trace_depth is not None:
        s = scramble(puz, P, rng, args.trace_depth)
        print(f"\ntrace: scramble depth {args.trace_depth}")
        trace_one(solver, puz, s, args.beam, args.max_steps)
        return 0

    print(f"\n{'depth':>6} {'solved':>8} {'rate':>7} {'mean len':>9} {'(scramble len)':>15}")
    for d in (int(x) for x in args.depths.split(",")):
        n_ok, lens = 0, []
        for _ in range(args.n_per_depth):
            s = scramble(puz, P, rng, d)
            cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps,
                                        num_attempts=1, internal_batch_size=2 ** 15)
            found, L, path = solver.solve(s, cfg)
            if found:
                n_ok += 1
                lens.append(len(path))
        rate = n_ok / args.n_per_depth
        ml = f"{np.mean(lens):.1f}" if lens else "-"
        print(f"{d:6d} {n_ok:5d}/{args.n_per_depth:<2d} {rate:7.2f} {ml:>9} {d:15d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
