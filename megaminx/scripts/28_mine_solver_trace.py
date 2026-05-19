"""T1.4 solver-trace mining → tensor for Bellman mixin training.

For each verified submission CSV, walk every solved path and emit
`(state, remaining_path_length)` pairs. These are TIGHTER labels than
random-walk-depth (every pair is a strict upper bound on the true distance,
typically much tighter than the walk-depth sample).

Output format (matches `bfs_d6_train.pt`):
  {"states": (N, 120) int8, "distances": (N,) int8}

Usable as a Bellman mixin via BellmanConfig.bfs_d6_path field (same loader).
For the m34 recipe we COMBINE with bfs_d6 if available.

Usage:
  .venv/Scripts/python.exe megaminx/scripts/28_mine_solver_trace.py \\
      --submissions \\
        megaminx/submissions/merge_plus_sym8_top20.csv \\
        megaminx/submissions/merge_plus_sym4_top80.csv \\
        megaminx/submissions/phase_b_plus198.csv \\
      --out megaminx/data/solver_trace_train.pt
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
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.verify import load_submission, load_test_states, verify_path
from megaminx.puzzle import Megaminx


def mine_path(puzzle: Megaminx, initial_state, path: list[str]) -> list[tuple[tuple[int, ...], int]]:
    """Walk path, emit (state, remaining_len) for every step including the start.

    The end state is solved with remaining_len == 0; we drop that one (the
    boundary case is handled exactly by the trainer).
    """
    pairs: list[tuple[tuple[int, ...], int]] = []
    cur = tuple(initial_state)
    for i, m in enumerate(path):
        remaining = len(path) - i
        pairs.append((cur, remaining))
        cur = puzzle.apply_move(cur, m)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", nargs="+", required=True,
                    help="Verified submission CSVs to mine")
    ap.add_argument("--out", required=True,
                    help="Output .pt with {'states', 'distances'} keys")
    ap.add_argument("--max-distance", type=int, default=120,
                    help="Drop pairs with remaining > this (caps label noise)")
    ap.add_argument("--dedupe", action="store_true",
                    help="Keep only the SHORTEST distance per unique state (lower bound preserved)")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states_by_pid = load_test_states(PROJECT / "data" / "test.csv")

    all_pairs: dict[tuple[int, ...], int] | list[tuple[tuple[int, ...], int]]
    if args.dedupe:
        all_pairs = {}
    else:
        all_pairs = []

    n_paths_total = 0
    n_paths_kept = 0
    n_pairs_emitted = 0

    for sub_path in args.submissions:
        sub = load_submission(sub_path)
        print(f"\nmining {sub_path} ({len(sub)} pids) ...")
        t0 = time.time()
        n_invalid = 0
        for pid, path in sub.items():
            if not path:
                continue
            n_paths_total += 1
            initial = states_by_pid.get(pid)
            if initial is None:
                continue
            res = verify_path(puzzle, initial, path)
            if not res.ok:
                n_invalid += 1
                continue
            n_paths_kept += 1
            pairs = mine_path(puzzle, initial, path)
            for s, d in pairs:
                if d > args.max_distance:
                    continue
                if args.dedupe:
                    existing = all_pairs.get(s)
                    if existing is None or d < existing:
                        all_pairs[s] = d
                else:
                    all_pairs.append((s, d))
                n_pairs_emitted += 1
        print(f"  kept {n_paths_kept - (n_paths_total - len(sub))}/{len(sub)} "
              f"valid paths from this CSV (invalid: {n_invalid}); "
              f"running pair count {n_pairs_emitted:,} ({time.time()-t0:.1f}s)")

    # Materialize tensor
    if args.dedupe:
        items = list(all_pairs.items())
    else:
        items = all_pairs

    print(f"\nfinal pair count: {len(items):,} "
          f"(deduped: {args.dedupe})")
    states_arr = np.array([list(s) for s, _ in items], dtype=np.int8)
    dists_arr = np.array([d for _, d in items], dtype=np.int8)
    out = {
        "states": torch.from_numpy(states_arr),
        "distances": torch.from_numpy(dists_arr),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    # Distribution summary
    from collections import Counter
    dist_counter = Counter(int(d) for _, d in items)
    print(f"\ndistance distribution:")
    for d in sorted(dist_counter):
        bar = "#" * min(50, dist_counter[d] // max(1, len(items) // 1000))
        print(f"  d={d:>3}: {dist_counter[d]:>7,d} {bar}")

    print(f"\nwrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
