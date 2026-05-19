"""Idea 7 -- Path relinking across solutions.

Cross-solution version of T1.6 SA. For each pid with multiple valid solutions
(across our many submission CSVs), try splicing prefix-of-A + bridge + suffix-of-B
where the bridge is a BFS-d6 lookup between the two prefix states.

Key formula for the spliced length:
    new_len = iA + L_b + (|B| - iB)

where L_b is the bridge path length from state_A[iA] to state_B[iB]. Accept
if new_len < min(|A|, |B|). The win comes from cases where A is short over
the early portion of the path and B is short over the late portion -- splicing
them via a short bridge extracts both advantages.

Mechanism:
- For each pid, collect all distinct valid solutions across submissions/*.csv
- For each pair (A, B), evaluate all (iA, iB) splice points
- BFS-d6 table provides bridge lengths up to 6 moves
- Verify final path; output merged-best CSV

Usage:
    .venv/Scripts/python.exe megaminx/scripts/46_path_relink.py \\
        --base megaminx/submissions/merge_v19b_plus_community.csv \\
        --out  megaminx/submissions/merge_v19b_plus_community_relinked.csv \\
        --submissions-glob 'megaminx/submissions/*.csv'
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.verify import verify_submission
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


def load_csv(path: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pid = int(row["initial_state_id"])
                p_str = row["path"]
                out[pid] = p_str.split(".") if p_str else []
    except Exception as e:
        print(f"  WARNING: failed to load {path.name}: {e}", file=sys.stderr)
    return out


def verify_path(initial_state: np.ndarray, path: list[str],
                gens: np.ndarray, name_to_idx: dict[str, int],
                solved_state: np.ndarray) -> bool:
    """Apply path to initial_state, check it lands on solved_state."""
    state = initial_state.copy()
    for m_name in path:
        if m_name not in name_to_idx:
            return False
        state = state[gens[name_to_idx[m_name]]]
    return np.array_equal(state, solved_state)


def relink_pid(
    initial_state: np.ndarray,
    paths: list[list[str]],
    gens: np.ndarray,
    name_to_idx: dict[str, int],
    move_names: tuple[str, ...],
    bfs_table: dict[bytes, bytes],
    solved_state: np.ndarray,
) -> tuple[list[str], int, int]:
    """Try splicing all pairs of paths. Returns (best_path, best_len, n_relink_wins).

    n_relink_wins counts how many splice candidates beat the original best length
    (most are unused because they don't beat the running best).
    """
    if not paths:
        return [], 0, 0
    best_path = min(paths, key=len)
    best_len = len(best_path)
    n_wins = 0

    # Precompute state sequences for each path.
    n_paths = len(paths)
    path_states: list[list[np.ndarray]] = []
    path_idx: list[list[int]] = []
    for p in paths:
        idxs = [name_to_idx[m] for m in p]
        states = [initial_state.copy()]
        cur = initial_state.copy()
        for gi in idxs:
            cur = cur[gens[gi]]
            states.append(cur.copy())
        path_states.append(states)
        path_idx.append(idxs)

    # Pairwise splicing.
    for ai in range(n_paths):
        A = paths[ai]
        sA = path_states[ai]
        for bi in range(n_paths):
            if ai == bi:
                continue
            B = paths[bi]
            sB = path_states[bi]
            len_A = len(A)
            len_B = len(B)
            # Pruning: if even with bridge length 0 the candidate can't beat best, skip.
            # min new_len at this pair = max(iA + 0 + (len_B - iB)) over (iA, iB).
            # Lower bound: iA = 0, iB = len_B -> len = 0. But that's just the empty sB suffix.
            # Easier prune: iterate iA outer, prune when iA >= best_len.
            for iA in range(len_A + 1):
                if iA >= best_len:
                    break
                # Compute state_A_inv for this iA.
                state_a_inv = np.argsort(sA[iA]).astype(np.int8)
                # For iB: candidate = iA + L_b + (len_B - iB). For improvement,
                # L_b <= best_len - iA - (len_B - iB) - 1, i.e., len_B - iB <= best_len - iA - 1.
                # So iB >= len_B - (best_len - iA - 1).
                iB_min = max(0, len_B - (best_len - iA - 1))
                for iB in range(iB_min, len_B + 1):
                    suffix_len = len_B - iB
                    # Bridge perm: state_a_inv o state_b -- the perm sigma such that sB[iB][i] = sA[iA][sigma[i]].
                    bridge_perm = state_a_inv[sB[iB]]
                    key = bridge_perm.tobytes()
                    word_bytes = bfs_table.get(key)
                    if word_bytes is None:
                        continue
                    L_b = len(word_bytes)
                    new_len = iA + L_b + suffix_len
                    if new_len < best_len:
                        bridge_names = [move_names[b] for b in word_bytes]
                        candidate = list(A[:iA]) + bridge_names + list(B[iB:])
                        # Verify (defensive -- bridge perm math should be correct).
                        if verify_path(initial_state, candidate, gens, name_to_idx, solved_state):
                            best_path = candidate
                            best_len = new_len
                            n_wins += 1
                        # If verify fails (shouldn't), don't accept; keep going.
    return best_path, best_len, n_wins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="full submission CSV -- relinking output won't go below the per-pid "
                         "min over all collected paths, so use the current best as the floor")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--submissions-glob", default=str(PROJECT / "submissions" / "*.csv"),
                    help="glob for submission CSVs to harvest paths from")
    ap.add_argument("--bfs-table", type=Path,
                    default=PROJECT / "data" / "bfs_bytes_d6.pkl")
    ap.add_argument("--max-paths-per-pid", type=int, default=12,
                    help="cap on paths per pid to keep relinking time bounded "
                         "(takes the shortest N distinct paths)")
    ap.add_argument("--pids", default="",
                    help="comma-separated subset of pids to process (default: all)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the final verify_submission step")
    args = ap.parse_args()

    print(f"loading puzzle...", flush=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    name_to_idx = {n: i for i, n in enumerate(puzzle.move_names)}
    move_names = puzzle.move_names
    gens = np.stack([puzzle.generators[n] for n in puzzle.move_names], axis=0).astype(np.int8)
    solved_state = np.array(puzzle.solved_state, dtype=np.int8)

    print(f"loading BFS-d6 from {args.bfs_table}...", flush=True)
    t0 = time.time()
    bfs = BfsBytesTable.load(args.bfs_table)
    print(f"  {len(bfs.table):,} states (d<={bfs.max_depth}) in {time.time()-t0:.1f}s",
          flush=True)
    assert bfs.move_names == puzzle.move_names

    # Load test states.
    test_states: dict[int, np.ndarray] = {}
    with open(PROJECT / "data" / "test.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_states[int(row["initial_state_id"])] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int8
            )

    # Phase A -- collect all paths per pid across submissions.
    import glob
    csv_paths = sorted(glob.glob(args.submissions_glob))
    print(f"\nphase A: harvesting paths from {len(csv_paths)} submission CSVs...", flush=True)
    pid_paths: dict[int, set[tuple[str, ...]]] = defaultdict(set)
    n_skip_invalid = 0
    n_loaded = 0
    for csv_p in csv_paths:
        csv_p = Path(csv_p)
        if csv_p.name.startswith("_"):
            continue  # skip smoke / temp files
        sub = load_csv(csv_p)
        if not sub:
            continue
        kept = 0
        for pid, path in sub.items():
            if pid not in test_states:
                continue
            if not path:
                continue
            # Defensive: verify the path solves; if not, skip (don't taint relinking).
            if not verify_path(test_states[pid], path, gens, name_to_idx, solved_state):
                n_skip_invalid += 1
                continue
            pid_paths[pid].add(tuple(path))
            kept += 1
            n_loaded += 1
        if kept:
            print(f"  {csv_p.name}: kept {kept} valid pids", flush=True)
    print(f"phase A done: {n_loaded} valid (pid, path) pairs across {len(pid_paths)} pids; "
          f"{n_skip_invalid} skipped invalid", flush=True)

    # Pid coverage stats.
    n_paths_distrib = [len(s) for s in pid_paths.values()]
    print(f"  paths-per-pid: min={min(n_paths_distrib)} median={int(np.median(n_paths_distrib))} "
          f"p90={int(np.percentile(n_paths_distrib, 90))} max={max(n_paths_distrib)}")

    # Load base submission to start from (per-pid floor).
    base = load_csv(args.base)
    print(f"\nbase submission: {len(base)} pids, total = "
          f"{sum(len(p) for p in base.values()):,} moves", flush=True)

    # Phase B -- relink per pid.
    pids_to_process = (
        sorted(int(p) for p in args.pids.split(",") if p.strip())
        if args.pids else sorted(pid_paths)
    )
    out: dict[int, list[str]] = {pid: list(p) for pid, p in base.items()}
    n_improved = 0
    total_saved = 0
    n_pids_with_relink_wins = 0
    n_pids_with_no_pair = 0
    t_start = time.time()
    for k, pid in enumerate(pids_to_process):
        path_set = pid_paths.get(pid, set())
        # Always include the base path (it might be shorter than any harvested copy).
        if pid in base and base[pid]:
            base_path = tuple(base[pid])
            if verify_path(test_states[pid], base[pid], gens, name_to_idx, solved_state):
                path_set.add(base_path)
        if len(path_set) < 2:
            n_pids_with_no_pair += 1
            continue  # nothing to relink
        # Take the shortest max_paths_per_pid distinct paths.
        sorted_paths = sorted(path_set, key=len)[: args.max_paths_per_pid]
        as_lists = [list(p) for p in sorted_paths]
        base_len = len(base[pid]) if pid in base and base[pid] else min(len(p) for p in as_lists)
        best_path, best_len, n_wins = relink_pid(
            test_states[pid], as_lists, gens, name_to_idx, move_names,
            bfs.table, solved_state,
        )
        if n_wins > 0:
            n_pids_with_relink_wins += 1
        if best_len < base_len:
            n_improved += 1
            total_saved += base_len - best_len
            out[pid] = best_path
        if (k + 1) % 50 == 0 or k == 0:
            elapsed = time.time() - t_start
            rate = (k + 1) / max(elapsed, 1e-3)
            eta = (len(pids_to_process) - (k + 1)) / max(rate, 1e-3)
            print(f"  [{k+1}/{len(pids_to_process)}] improved={n_improved} "
                  f"saved={total_saved} relink_wins={n_pids_with_relink_wins} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    elapsed = time.time() - t_start
    print(f"\nphase B done: {n_improved}/{len(pids_to_process)} pids improved, "
          f"saved {total_saved} moves; {n_pids_with_relink_wins} pids had >=1 splice win; "
          f"{n_pids_with_no_pair} pids had <2 distinct paths to relink. "
          f"({elapsed:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
        w.writeheader()
        for pid in sorted(out):
            w.writerow({"initial_state_id": pid, "path": ".".join(out[pid])})
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
