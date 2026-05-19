"""T2.2 — Tail re-solve post-processing.

For each pid path M of length N: truncate the last K moves and re-solve the
resulting prefix-state with a fresh beam. If the new tail is shorter than K,
accept M' = M[:N-K] + new_tail. Iterate K in {5, 10, 20, 40, 80}; per pid
take the shortest valid replacement (or keep the original).

Strict additive post-processing: cannot regress (we never accept a longer
total path; verifier guards correctness).

Why this is distinct from BFS-d6 windows:
  - BFS-d6 replaces sub-paths with their shortest equivalent for any perm
    reachable in d ≤ 6 moves. Bounded by depth 6.
  - Tail re-solve specifically targets the last K-suffix where beam often
    converges suboptimally near solved (heuristic noise dominates). Not
    bounded by perm-table depth — can find arbitrarily-long shorter tails.

Usage:
  .venv/Scripts/python.exe megaminx/scripts/26_tail_resolve.py \\
      --base megaminx/submissions/merge_plus_sym8_top20.csv \\
      --out  megaminx/submissions/tail_resolve_top50.csv \\
      --top 50 --beam 65536 --bf16

  # Full-1001:
  .venv/Scripts/python.exe megaminx/scripts/26_tail_resolve.py \\
      --base megaminx/submissions/merge_plus_sym8_top20.csv \\
      --out  megaminx/submissions/tail_resolve_full.csv \\
      --beam 65536 --bf16
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


def _try_resolve_tail(
    puzzle: Megaminx,
    solver: KhoruzhiiSolver,
    initial_state: tuple[int, ...],
    path: list[str],
    K: int,
    beam: int,
    max_steps: int,
    bfs_table,
    bfs_max_window: int,
) -> tuple[list[str] | None, int]:
    """Try to shorten path by re-solving its last K moves.

    Returns (new_path, savings) — savings = original_len - new_len; positive if
    improved. new_path is None if no improvement.
    """
    N = len(path)
    if K >= N:
        return None, 0
    prefix = path[: N - K]
    # Compute state after applying prefix.
    S = puzzle.apply_path(initial_state, prefix)
    if puzzle.is_solved(S):
        # Whole prefix already solves it — drop the entire tail.
        new_path = list(prefix)
        if verify_path(puzzle, initial_state, new_path).ok:
            return new_path, K
        return None, 0
    cfg = KhoruzhiiSearchConfig(beam_width=beam, num_steps=max_steps, num_attempts=1)
    found, _, raw_tail = solver.solve(S, cfg)
    if not found:
        return None, 0
    new_tail = full_post_process(raw_tail, puzzle=puzzle, bfs_table=bfs_table, max_window=bfs_max_window)
    if len(new_tail) >= K:
        return None, 0
    new_path = list(prefix) + list(new_tail)
    res = verify_path(puzzle, initial_state, new_path)
    if not res.ok:
        return None, 0
    return new_path, N - len(new_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Current best submission CSV")
    ap.add_argument("--out", required=True, help="Output CSV (merged tail-resolved)")
    ap.add_argument("--checkpoint", default="megaminx/models/m05_bellman_warm/epoch_0499.pt")
    ap.add_argument("--bfs-table", default="megaminx/data/bfs_bytes_d6.pkl")
    ap.add_argument("--bfs-max-window", type=int, default=12)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--K-list", default="5,10,20,40,80",
                    help="Comma-separated tail lengths to try, in order")
    ap.add_argument("--max-extra-steps", type=int, default=20,
                    help="Solver max_steps = K + this (allow tail to be a bit longer mid-search)")
    ap.add_argument("--top", type=int, default=0,
                    help="Process only top-N longest-path pids (0 = all 1001)")
    ap.add_argument("--pids", default="",
                    help="Optional comma-separated explicit pid list (overrides --top)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--internal-batch-size", type=int, default=2**14)
    ap.add_argument("--state-dtype", default="int8", choices=["int8", "int16", "int32", "int64"])
    ap.add_argument("--quiet", action="store_true", help="Skip per-pid trace output")
    args = ap.parse_args()

    K_list = [int(k) for k in args.K_list.split(",") if k.strip()]
    state_dtype = {"int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64}[args.state_dtype]
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"loading puzzle + model ...")
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    base = load_submission(args.base)
    print(f"  test states: {len(states)}, base submission: {len(base)} pids, "
          f"total moves: {sum(len(p) for p in base.values()):,}")

    bfs_path = Path(args.bfs_table)
    if bfs_path.exists():
        print(f"  loading bfs table {bfs_path} ...")
        bfs_table = BfsBytesTable.load(bfs_path)
    else:
        print(f"  WARNING: no bfs table at {bfs_path}, post-proc will be cancel-only")
        bfs_table = None

    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=model_dtype)
    solver = KhoruzhiiSolver(
        puzzle, model, device=args.device,
        internal_batch_size=args.internal_batch_size,
        state_dtype=state_dtype,
    )

    # Determine pid order: longest first (most savings expected on hard tail).
    if args.pids.strip():
        pid_order = [int(p) for p in args.pids.split(",") if p.strip()]
    else:
        pid_order = sorted(base.keys(), key=lambda p: -len(base[p]))
        if args.top > 0:
            pid_order = pid_order[: args.top]

    print(f"\nprocessing {len(pid_order)} pids; K values {K_list}; beam {args.beam}\n")

    out_paths: dict[int, list[str]] = dict(base)  # start with base; replace where we improve
    n_improved = 0
    total_savings = 0
    t0 = time.time()
    for rank, pid in enumerate(pid_order):
        path = list(base[pid])
        N = len(path)
        best_new = None
        best_savings = 0
        # Cap K at N - 1 to keep at least one prefix move.
        for K in K_list:
            if K >= N:
                continue
            max_steps = K + args.max_extra_steps
            new_path, savings = _try_resolve_tail(
                puzzle, solver, states[pid], path, K,
                args.beam, max_steps, bfs_table, args.bfs_max_window,
            )
            if new_path is not None and savings > best_savings:
                best_new = new_path
                best_savings = savings
        if best_new is not None:
            out_paths[pid] = best_new
            n_improved += 1
            total_savings += best_savings
            if not args.quiet:
                print(f"  pid={pid:4d} {N:3d}->{len(best_new):3d} (-{best_savings:2d}) "
                      f"[rank {rank+1}/{len(pid_order)}, total -{total_savings}]")
        else:
            if not args.quiet and (rank < 10 or rank % 50 == 0):
                print(f"  pid={pid:4d} {N:3d}        no improvement [rank {rank+1}/{len(pid_order)}]")

    elapsed = time.time() - t0
    print(f"\n=== summary ===")
    print(f"  improved: {n_improved}/{len(pid_order)} pids")
    print(f"  total savings: {total_savings} moves")
    print(f"  walltime: {elapsed/60:.1f} min")

    # Write merged output (full 1001 rows so it's a valid submission).
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths.keys()):
            w.writerow([pid, ".".join(out_paths[pid])])

    final_total = sum(len(p) for p in out_paths.values())
    base_total = sum(len(p) for p in base.values())
    print(f"\nwrote {out_path} ({len(out_paths)} rows)")
    print(f"  base total: {base_total:,} moves")
    print(f"  new  total: {final_total:,} moves  ({final_total - base_total:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
