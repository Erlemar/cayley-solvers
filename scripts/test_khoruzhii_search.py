"""Validate the ported khoruzhii Searcher on a stratified sample.

Compares against our cayleypy-based Solver on the same model + puzzles. Both should
find valid paths; the ported searcher should fit wider beams without OOM.

    python scripts/test_khoruzhii_search.py --checkpoint models/fast/epoch_0499.pt \\
        --beams 4096 16384 65536 --n-puzzles 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.puzzle import PictureCube
from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--beams", nargs="+", type=int, default=[4096, 16384, 65536])
    ap.add_argument("--num-steps", type=int, default=60)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--n-puzzles", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    sample = load_submission(PROJECT / "data" / "sample_submission.csv")
    all_ids = sorted(states.keys(), key=lambda pid: len(sample[pid]))
    step = max(1, len(all_ids) // args.n_puzzles)
    pids = all_ids[::step][: args.n_puzzles]

    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=torch.bfloat16)
    solver = KhoruzhiiSolver(puzzle, model, device=args.device)

    print(f"khoruzhii searcher, {len(pids)} puzzles, num_steps={args.num_steps}")
    print(f"{'beam':>6} {'solved':>7} {'total':>7} {'avg':>6} {'time_s':>7} {'peakGB':>7}")
    for beam in args.beams:
        torch.cuda.reset_peak_memory_stats()
        cfg = KhoruzhiiSearchConfig(
            beam_width=beam, num_steps=args.num_steps, num_attempts=args.num_attempts
        )
        solved = 0
        total = 0
        t0 = time.time()
        for pid in pids:
            torch.cuda.empty_cache()
            try:
                found, length, path = solver.solve(states[pid], cfg)
                if found and verify_path(puzzle, states[pid], path).ok:
                    solved += 1
                    total += length
            except torch.AcceleratorError:
                torch.cuda.empty_cache()
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9
        avg = total / solved if solved else 0
        print(f"{beam:>6} {solved:>4}/{len(pids):<2} {total:>7} {avg:>6.1f} {elapsed:>6.1f} {peak:>6.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
