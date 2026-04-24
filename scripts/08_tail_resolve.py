"""Tail re-solve post-processing: for each submission path, try re-solving the tail
from an intermediate state with a fresh beam search and splice if shorter.

Given path P = [m_1, ..., m_N] that solves state σ, at position i the cube is at
state σ · π_i (= applying P[:i] to σ). The remaining task from that state is solved
by P[i:]. If a fresh beam search from state σ · π_i finds a shorter path to solved
than P[i:], we splice.

Strategy:
  1. For each puzzle's baseline path, pick several split positions i (e.g., every 5 moves).
  2. Replay to state_at_i.
  3. Run beam search from state_at_i for up to (len(P[i:]) - 1) steps.
  4. If a valid shorter continuation is found, splice and keep.

Why it helps beyond a forward wider-beam re-solve:
  - The intermediate state has a different "difficulty profile" — near-solved states are
    easier to fully enumerate.
  - Beam from σ vs beam from σ·π_i may escape different local optima.

    python scripts/08_tail_resolve.py \\
        --baseline submissions/ens_plus_community_pp.csv \\
        --checkpoint models/e6/epoch_0499.pt \\
        --beam 32768 --tail-max 20 --bf16 \\
        --min-length 25 \\
        --out submissions/tail_resolved.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.post_process import full_post_process
from cayley.puzzle import PictureCube
from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beam", type=int, default=32768, help="beam width for tail solves")
    ap.add_argument("--tail-max", type=int, default=20,
                    help="only attempt tails of length <= this")
    ap.add_argument("--tail-min", type=int, default=6,
                    help="only attempt tails of length >= this")
    ap.add_argument("--stride", type=int, default=3,
                    help="try every `stride`th split position in the path")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-length", type=int, default=25,
                    help="only process paths at least this long")
    ap.add_argument("--num-attempts", type=int, default=1)
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    baseline = load_submission(args.baseline)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    solver = KhoruzhiiSolver(puzzle, model, device=args.device)

    new_paths: dict[int, list[str]] = {}
    stats = {"processed": 0, "improved": 0, "saved_moves": 0, "attempted_splits": 0}
    t0 = time.time()

    pids = sorted(pid for pid, p in baseline.items() if len(p) >= args.min_length)
    print(f"processing {len(pids)} puzzles (len>= {args.min_length})")

    for i, pid in enumerate(pids):
        original = baseline[pid]
        scramble = states[pid]
        best = list(original)
        stats["processed"] += 1

        # Try splicing at every `stride`th position where the tail has length in
        # [tail_min, tail_max].
        for split in range(args.stride, len(original), args.stride):
            tail_len = len(original) - split
            if tail_len < args.tail_min or tail_len > args.tail_max:
                continue
            # State at position `split`
            mid_state = puzzle.apply_path(scramble, original[:split])
            stats["attempted_splits"] += 1

            cfg = KhoruzhiiSearchConfig(
                beam_width=args.beam, num_steps=tail_len - 1,
                num_attempts=args.num_attempts,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            found, new_len, raw = solver.solve(mid_state, cfg)
            if found and new_len < tail_len:
                spliced = original[:split] + raw
                spliced = full_post_process(spliced, scramble, puzzle)
                if verify_path(puzzle, scramble, spliced).ok and len(spliced) < len(best):
                    best = spliced

        new_paths[pid] = best
        if len(best) < len(original):
            stats["improved"] += 1
            stats["saved_moves"] += len(original) - len(best)

        if (i + 1) % 10 == 0 or i == len(pids) - 1:
            elapsed = time.time() - t0
            print(
                f"  {i + 1}/{len(pids)} improved={stats['improved']} "
                f"saved={stats['saved_moves']} attempts={stats['attempted_splits']} "
                f"({elapsed:.1f}s)"
            )

    # Write full CSV: new path where attempted, else baseline.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(states):
            path = new_paths.get(pid, baseline[pid])
            w.writerow([pid, ".".join(path)])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"wrote {args.out}")
    print(f"  stats: {stats}")
    print(f"  verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
