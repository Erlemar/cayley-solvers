"""Post-process TPU beam outputs and min-merge them against the current floor.

    python scripts/merge_tpu_outputs.py \
        --tpu "kaggle_notebooks/tpu_beam_ihes_shareable/_kernel_output*/share_cube_submission_*.csv" \
        --floor submissions/floor_merged_20260712.csv \
        --out submissions/floor_plus_tpu.csv

Each TPU path goes through full_post_process (with the BFS-d6 table) and a
verify gate before it may replace a floor path. Prints a per-pid win report
so TPU contributions are attributed against the true floor (never claim a win
the community already had -- see EXPERIMENTS.md Rule 26 analog).
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.bfs_table import BfsTable
from cayley.post_process import full_post_process
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpu", required=True,
                    help="glob for TPU share_cube_submission_*.csv files")
    ap.add_argument("--floor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bfs-table", type=Path, default=PROJECT / "data" / "bfs_table_d6.pkl")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    floor = load_submission(args.floor)
    bfs_table = BfsTable.load(args.bfs_table) if args.bfs_table.exists() else None
    if bfs_table:
        print(f"BFS table: {len(bfs_table.table):,} states at depth <= {bfs_table.max_depth}")

    tpu_files = sorted(glob.glob(args.tpu))
    if not tpu_files:
        print(f"no files match {args.tpu}")
        return 1
    print(f"TPU files: {len(tpu_files)}")

    # Best TPU candidate per pid across all files (post-processed + verified).
    tpu_best: dict[int, tuple[list[str], str]] = {}
    for fp in tpu_files:
        with open(fp, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if not row["path"]:
                    continue
                pid = int(row["initial_state_id"])
                path = row["path"].split(".")
                pp = full_post_process(path, states[pid], puzzle, bfs_table=bfs_table)
                if len(pp) < len(path) and verify_path(puzzle, states[pid], pp).ok:
                    path = pp
                if not verify_path(puzzle, states[pid], path).ok:
                    print(f"  WARN: invalid TPU path for pid {pid} in {fp} - skipped")
                    continue
                if pid not in tpu_best or len(path) < len(tpu_best[pid][0]):
                    tpu_best[pid] = (path, Path(fp).name)

    wins, ties, losses = [], 0, 0
    merged = dict(floor)
    for pid, (path, src) in sorted(tpu_best.items()):
        fl = len(floor[pid])
        if len(path) < fl:
            wins.append((pid, fl, len(path), src))
            merged[pid] = path
        elif len(path) == fl:
            ties += 1
        else:
            losses += 1

    print(f"\nTPU pids covered: {len(tpu_best)}  wins: {len(wins)}  ties: {ties}  worse: {losses}")
    for pid, fl, tl, src in wins:
        print(f"  WIN pid {pid}: {fl} -> {tl}  ({src})")

    total_floor = sum(len(p) for p in floor.values())
    total_merged = sum(len(p) for p in merged.values())
    print(f"\nfloor total: {total_floor} -> merged total: {total_merged} ({total_merged - total_floor:+d})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(merged):
            w.writerow([pid, ".".join(merged[pid])])
    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"wrote {args.out}  verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
