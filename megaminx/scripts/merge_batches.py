"""Merge multiple batch / partial submission CSVs into a full 1001-pid submission.

    python megaminx/scripts/merge_batches.py \
        --batches megaminx/submissions/full_m07_*.csv \
        --fallback megaminx/data/pp_bfs6_fallback.csv \
        --out megaminx/submissions/full_m07_merged.csv

Each batch CSV has `initial_state_id,path` for a subset of pids (output of
`03_solve.py --pid-from --pid-to`). This script:
 1. For every pid, picks the shortest valid path across all batches.
 2. For any pid still missing, uses the fallback CSV (validated separately).
 3. Writes a full 1001-row submission and verifies it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.verify import load_submission, load_test_states, verify_path, verify_submission
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="+", type=Path, required=True)
    ap.add_argument("--fallback", type=Path, default=PROJECT / "data" / "pp_bfs6_fallback.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    fallback = load_submission(args.fallback)

    # For each pid, collect all valid paths from batches (min wins).
    by_pid: dict[int, tuple[list[str], str]] = {}  # pid -> (path, source)
    n_batch_rows = 0
    n_rejected = 0
    for b in args.batches:
        if not b.exists():
            print(f"WARN: batch file missing: {b}")
            continue
        rows = load_submission(b)
        print(f"{b.name}: {len(rows)} rows")
        for pid, path in rows.items():
            n_batch_rows += 1
            if pid not in states:
                print(f"  WARN: pid {pid} in batch but not in test set; skip")
                continue
            if not verify_path(puzzle, states[pid], path).ok:
                n_rejected += 1
                if n_rejected <= 5:
                    print(f"  REJECTED pid {pid} from {b.name}: path invalid ({len(path)} moves)")
                continue
            cur = by_pid.get(pid)
            if cur is None or len(path) < len(cur[0]):
                by_pid[pid] = (path, b.name)

    n_model_covered = len(by_pid)
    n_fallback_needed = sum(1 for p in states if p not in by_pid)
    print(f"\nbatches contributed: {n_model_covered} pids ({n_batch_rows} total rows, "
          f"{n_rejected} rejected as invalid)")
    print(f"fallback-fill needed for: {n_fallback_needed} pids")

    # Fallback-fill.
    rows_out: list[tuple[int, str]] = []
    total = 0
    src_counts = {"batch": 0, "fallback": 0}
    for pid in sorted(states):
        if pid in by_pid:
            path, _src = by_pid[pid]
            src_counts["batch"] += 1
        else:
            path = fallback[pid]
            src_counts["fallback"] += 1
        rows_out.append((pid, ".".join(path)))
        total += len(path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid, p in rows_out:
            w.writerow([pid, p])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"\nwrote {args.out}")
    print(f"  src: {src_counts}")
    print(f"  total moves: {total:,}")
    print(f"  verify: {report.n_valid}/{report.n_total} valid, {report.total_moves:,}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
