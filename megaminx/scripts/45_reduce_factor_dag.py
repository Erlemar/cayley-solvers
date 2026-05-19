"""ReduceFactor DAG path optimization.

Idea 6 from megaminx/novel_ideas.md.

Replaces the greedy left-to-right window post-processing with a
globally-optimal cover. For path P of length N, builds DAG with nodes 0..N:

  - Pass-through edge (i, i+1) of weight 1, carrying move P[i].
  - Replacement edge (i, j) for window length w = j - i in [2, max_window]:
    compute the window's net permutation; look up in BFS-d6 bytes table; if
    a strictly shorter word L < w exists, add edge of weight L carrying the
    BFS-shortest word.

Linear-time DP (DAG is acyclic, i < j always) finds the globally-shortest
0 -> N path. Iterate to fixpoint (the shortened path opens new windows).

Cannot regress per-pid: minimum 0->N cost is bounded above by the all-
pass-through path (= len(P)).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/45_reduce_factor_dag.py \
        --base megaminx/submissions/merge_v19b_plus_community.csv \
        --out  megaminx/submissions/merge_v19b_plus_community_dag.csv \
        --max-window 12
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.verify import verify_submission
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.post_process import reduce_same_face_runs, cancel_adjacent_inverses
from megaminx.puzzle import Megaminx


def reduce_factor_dag_once(
    path_idx: list[int],
    gens: np.ndarray,
    bfs_table: dict[bytes, bytes],
    max_window: int,
    state_size: int,
) -> list[int]:
    """One DAG pass. path_idx is a list of generator indices.

    gens: (n_gen, state_size) int8 array of permutations.
    bfs_table: bytes-keyed dict (perm bytes -> shortest word as bytes of gen indices).
    Returns a possibly-shortened list of generator indices.
    """
    N = len(path_idx)
    if N < 2:
        return list(path_idx)

    INF = 10**9
    dp = [INF] * (N + 1)
    dp[0] = 0
    # parent[j] = (predecessor_i, list_of_gen_indices_along_edge)
    parent: list[tuple[int, list[int]] | None] = [None] * (N + 1)
    parent[0] = (-1, [])

    identity = np.arange(state_size, dtype=np.int8)

    for i in range(N + 1):
        if dp[i] >= INF:
            continue
        # Pass-through edge i -> i+1 (always available, weight 1)
        if i < N:
            cand = dp[i] + 1
            if cand < dp[i + 1]:
                dp[i + 1] = cand
                parent[i + 1] = (i, [path_idx[i]])
        # Replacement edges from i: scan windows of length 2..max_window
        if i >= N:
            continue
        state = identity.copy()
        upper = min(i + max_window + 1, N + 1)
        for j in range(i + 1, upper):
            # extend window by one move: state := state[gens[m]]
            m = path_idx[j - 1]
            state = state[gens[m]]
            w = j - i
            if w >= 2:
                key = state.tobytes()
                replacement = bfs_table.get(key)
                if replacement is not None and len(replacement) < w:
                    cost = len(replacement)
                    cand = dp[i] + cost
                    if cand < dp[j]:
                        dp[j] = cand
                        parent[j] = (i, list(replacement))

    # Reconstruct
    out: list[int] = []
    j = N
    while j > 0:
        pred, moves = parent[j]
        out = list(moves) + out
        j = pred
    return out


def reduce_factor_dag(
    path_names: list[str],
    puzzle: Megaminx,
    bfs_table_dict: dict[bytes, bytes],
    move_names: tuple[str, ...],
    max_window: int = 12,
    max_iter: int = 8,
) -> list[str]:
    """Apply same-face / cancel / DAG-window passes to fixpoint."""
    name_to_idx = {n: i for i, n in enumerate(move_names)}
    gens = np.stack(
        [np.array(puzzle.generators[n], dtype=np.int8) for n in move_names], axis=0
    )
    state_size = len(puzzle.solved_state)

    cur = list(path_names)
    for _ in range(max_iter):
        prev_len = len(cur)
        cur = reduce_same_face_runs(cur)
        cur = cancel_adjacent_inverses(cur)
        if len(cur) < 2:
            break
        path_idx = [name_to_idx[n] for n in cur]
        new_idx = reduce_factor_dag_once(path_idx, gens, bfs_table_dict, max_window, state_size)
        cur = [move_names[i] for i in new_idx]
        cur = reduce_same_face_runs(cur)
        cur = cancel_adjacent_inverses(cur)
        if len(cur) == prev_len:
            break
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="full submission to apply DAG post-processing to")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bfs-table", type=Path,
                    default=PROJECT / "data" / "bfs_bytes_d6.pkl")
    ap.add_argument("--max-window", type=int, default=12,
                    help="largest window size to consider for replacement (default 12)")
    ap.add_argument("--pids", default="",
                    help="comma-separated pids to process (default: all)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the final verify_submission step (faster smoke tests)")
    args = ap.parse_args()

    print(f"loading puzzle...", flush=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"loading BFS table from {args.bfs_table}...", flush=True)
    t0 = time.time()
    bfs = BfsBytesTable.load(args.bfs_table)
    print(f"  loaded {len(bfs.table):,} states (max_depth={bfs.max_depth}) in {time.time()-t0:.1f}s",
          flush=True)
    move_names = bfs.move_names
    assert move_names == puzzle.move_names, (
        f"BFS move_names {move_names} != puzzle move_names {puzzle.move_names}"
    )

    # Load base submission.
    base: dict[int, str] = {}
    with open(args.base) as f:
        for row in csv.DictReader(f):
            base[int(row["initial_state_id"])] = row["path"]
    print(f"base: {len(base)} pids, total = "
          f"{sum(len(p.split('.')) for p in base.values()):,} moves", flush=True)

    pids_to_process = (
        sorted(int(p) for p in args.pids.split(",") if p)
        if args.pids
        else sorted(base)
    )

    out: dict[int, str] = dict(base)
    n_improved = 0
    total_saved = 0
    t_start = time.time()
    for k, pid in enumerate(pids_to_process):
        path_str = base[pid]
        if not path_str:
            continue
        path_names = path_str.split(".")
        N0 = len(path_names)
        new_names = reduce_factor_dag(
            path_names, puzzle, bfs.table, move_names, max_window=args.max_window
        )
        N1 = len(new_names)
        if N1 < N0:
            n_improved += 1
            total_saved += (N0 - N1)
            out[pid] = ".".join(new_names)
        if (k + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (k + 1) / elapsed if elapsed > 0 else 0
            eta = (len(pids_to_process) - (k + 1)) / rate if rate > 0 else 0
            print(f"  [{k+1}/{len(pids_to_process)}] improved={n_improved} "
                  f"saved={total_saved} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone. {n_improved}/{len(pids_to_process)} pids improved, "
          f"saved {total_saved} moves in {elapsed:.0f}s.", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
        w.writeheader()
        for pid in sorted(out):
            w.writerow({"initial_state_id": pid, "path": out[pid]})
    print(f"wrote {args.out}", flush=True)

    if not args.no_verify:
        print("verifying...", flush=True)
        report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
        print(f"verify: {report.n_valid}/{report.n_total} valid, "
              f"total {report.total_moves:,}", flush=True)
        if not report.all_valid:
            print(f"  INVALID pids: {report.invalid_pids[:10]}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
