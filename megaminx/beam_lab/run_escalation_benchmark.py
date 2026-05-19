"""Adaptive beam escalation: per-puzzle, try small beam first, escalate on miss.

Saves compute on easy puzzles (most solve at small beam). Compares to single-beam
fixed-width baseline.

Usage:
    python run_escalation_benchmark.py --checkpoint ...epoch_0499.pt \\
        --beams 16384,65536,131072 --max-steps 120 --bf16 --compile \\
        --out results/local_escalation.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from beam_search import (  # noqa: E402
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    setup_model_for_compile,
    setup_model_for_inference,
)
from model import load_checkpoint  # noqa: E402
from puzzle import Megaminx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--puzzle-info", type=Path, default=HERE / "data" / "puzzle_info.json")
    ap.add_argument("--samples", type=Path, default=HERE / "data" / "sample_puzzles.csv")
    ap.add_argument("--beams", type=str, default="16384,65536,131072",
                    help="comma-separated beam widths to escalate through")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(args.puzzle_info)
    samples = []
    with open(args.samples) as f:
        for row in csv.DictReader(f):
            samples.append((int(row["initial_state_id"]),
                           tuple(int(x) for x in row["initial_state"].split(","))))
    print(f"samples: {len(samples)} puzzles")
    print(f"escalation beams: {args.beams}")

    model = load_checkpoint(str(args.checkpoint), device=device)
    if args.bf16 and device == "cuda":
        model = model.to(torch.bfloat16)
    setup_model_for_inference(model)

    if args.compile:
        print(f"compiling model (pre-warm at batch={args.internal_batch_size})...")
        t0 = time.time()
        model = setup_model_for_compile(model, batch_size=args.internal_batch_size)
        print(f"compile + warmup: {time.time() - t0:.1f}s")

    beams = [int(x) for x in args.beams.split(",")]
    solver = KhoruzhiiSolver(
        puzzle, model, device=device,
        internal_batch_size=args.internal_batch_size,
        random_seed=0, state_dtype=torch.int8,
        pad_to_batch_size=args.compile,
    )

    rows = []
    print()
    print(f"{'pid':>4} {'B*':>7} {'try1':>5} {'try2':>5} {'try3':>5} "
          f"{'path':>5} {'wall':>6}")
    print("-" * 60)
    total_wall = 0.0
    total_path = 0
    for pid, state in samples:
        per_pid_wall = 0.0
        used_beam = -1
        plen = -1
        wall_per_attempt = []
        for B in beams:
            cfg = KhoruzhiiSearchConfig(
                beam_width=B,
                num_steps=args.max_steps,
                num_attempts=1,
                internal_batch_size=args.internal_batch_size,
            )
            t0 = time.time()
            ok, plen_local, names, prof = solver.solve(state, cfg)
            wall = time.time() - t0
            per_pid_wall += wall
            wall_per_attempt.append(wall)
            if ok:
                used_beam = B
                plen = plen_local
                # verify
                cur = tuple(state)
                for m in names:
                    cur = puzzle.apply_move(cur, m)
                valid = puzzle.is_solved(cur)
                if not valid:
                    print(f"  pid {pid} INVALID PATH at beam {B}", flush=True)
                break
        # pad wall_per_attempt to 3
        while len(wall_per_attempt) < 3:
            wall_per_attempt.append(0.0)
        rows.append({"pid": pid, "used_beam": used_beam, "path_len": plen,
                     "wall": per_pid_wall,
                     "try1": wall_per_attempt[0], "try2": wall_per_attempt[1],
                     "try3": wall_per_attempt[2] if len(wall_per_attempt) > 2 else 0.0})
        total_wall += per_pid_wall
        total_path += plen if plen > 0 else 0
        try1 = f"{wall_per_attempt[0]:.1f}" if wall_per_attempt[0] > 0 else "  -"
        try2 = f"{wall_per_attempt[1]:.1f}" if wall_per_attempt[1] > 0 else "  -"
        try3 = f"{wall_per_attempt[2]:.1f}" if wall_per_attempt[2] > 0 else "  -"
        print(f"{pid:>4d} {used_beam:>7d} {try1:>5} {try2:>5} {try3:>5} "
              f"{plen:>5d} {per_pid_wall:>6.1f}", flush=True)

    print("-" * 60)
    print(f"total wall: {total_wall:.1f}s")
    print(f"total path: {total_path}")
    n_first = sum(1 for r in rows if r['used_beam'] == beams[0])
    n_second = sum(1 for r in rows if r['used_beam'] == beams[1])
    n_third = sum(1 for r in rows if len(beams) > 2 and r['used_beam'] == beams[2])
    print(f"used beam {beams[0]}: {n_first}/{len(rows)}; beam {beams[1]}: {n_second}/{len(rows)}"
          + (f"; beam {beams[2]}: {n_third}/{len(rows)}" if len(beams) > 2 else ""))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        cols = ["pid", "used_beam", "path_len", "wall", "try1", "try2", "try3"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
