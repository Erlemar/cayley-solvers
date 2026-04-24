"""Sweep beam widths on a subset of puzzles to pick the best operating point.

    python scripts/beam_sweep.py --checkpoint models/embed/epoch_0199.pt \\
        --beams 2048 8192 16384 32768 --n-puzzles 50

For each beam width, solves `n_puzzles` (deterministic sample) and prints solve rate +
avg moves + wall time. Picks puzzles across the difficulty range (sorted by sample length).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.post_process import full_post_process
from cayley.puzzle import PictureCube
from cayley.search import SearchConfig, Solver, load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--beams", nargs="+", type=int, default=[2048, 8192, 16384])
    ap.add_argument("--n-puzzles", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    sample = load_submission(PROJECT / "data" / "sample_submission.csv")

    # Pick puzzles evenly across difficulty (sample_len strata).
    all_ids = sorted(states.keys(), key=lambda pid: len(sample[pid]))
    step = max(1, len(all_ids) // args.n_puzzles)
    sampled_ids = all_ids[::step][: args.n_puzzles]

    model = load_model_checkpoint(args.checkpoint, device=args.device)
    solver = Solver(puzzle, model, device=args.device)

    print(f"sweep on {len(sampled_ids)} puzzles, max_steps={args.max_steps}")
    print(f"{'beam':>6} {'solved':>7} {'total':>7} {'avg':>6} {'time_s':>7}")
    for beam in args.beams:
        cfg = SearchConfig(beam_width=beam, max_steps=args.max_steps, beam_mode="simple")
        solved = 0
        total_moves = 0
        t0 = time.time()
        for pid in sampled_ids:
            torch.cuda.empty_cache()
            res = solver.solve(states[pid], cfg)
            if res.found:
                p = full_post_process(res.path, states[pid], puzzle)
                if verify_path(puzzle, states[pid], p).ok:
                    solved += 1
                    total_moves += len(p)
        elapsed = time.time() - t0
        avg = total_moves / solved if solved else 0
        print(f"{beam:>6} {solved:>4}/{len(sampled_ids):<2} {total_moves:>7} {avg:>6.1f} {elapsed:>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
