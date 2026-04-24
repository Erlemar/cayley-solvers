"""Apply post-processing to every path in a submission CSV.

    python scripts/post_process_submission.py --in submissions/combined.csv \\
        --out submissions/combined_pp.csv

Uses pair cancellation + state-hash shortcutting (see src/cayley/post_process.py).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.bfs_table import BfsTable
from cayley.post_process import full_post_process
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bfs-table", type=Path, default=None, help="pickle from build_bfs_table.py")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    submission = load_submission(args.inp)

    bfs_table = BfsTable.load(args.bfs_table) if args.bfs_table else None
    if bfs_table:
        print(f"loaded BFS table: {len(bfs_table.table):,} states at depth <= {bfs_table.max_depth}")

    n_changed = 0
    total_before = 0
    total_after = 0
    t0 = time.time()
    rows: list[tuple[int, str]] = []
    for pid in sorted(states):
        path = submission[pid]
        total_before += len(path)
        pp = full_post_process(path, states[pid], puzzle, bfs_table=bfs_table)
        # Safety: only accept if verified and not longer.
        if len(pp) < len(path) and verify_path(puzzle, states[pid], pp).ok:
            path = pp
            n_changed += 1
        total_after += len(path)
        rows.append((pid, ".".join(path)))
    elapsed = time.time() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["initial_state_id", "path"])
        for pid, p in rows:
            writer.writerow([pid, p])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"shortened: {n_changed}/{len(rows)} paths")
    print(f"total moves: {total_before} -> {total_after} ({total_before - total_after:+d})")
    print(f"verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves}")
    print(f"elapsed: {elapsed:.1f}s")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
