"""Merge a rescue/partial submission CSV into a base submission, taking shorter
paths per pid. Verify the result and report total moves.

Usage:
    python megaminx/scripts/16_merge_rescue.py \
        --base megaminx/submissions/phase_b_rescued.csv \
        --rescue megaminx/submissions/qshort_524k_top148.csv \
        --out megaminx/submissions/phase_b_plus_148rescue.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.verify import verify_submission
from megaminx.puzzle import Megaminx


def load_csv(path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["initial_state_id"])] = row["path"]
    return out


def plen(path_str: str) -> int:
    return len(path_str.split(".")) if path_str else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="full 1001-row submission to merge into")
    ap.add_argument("--rescue", required=True, type=Path,
                    help="partial submission with rescued (potentially shorter) paths")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    base = load_csv(args.base)
    rescue = load_csv(args.rescue)
    print(f"base: {len(base)} pids, total = {sum(plen(p) for p in base.values()):,}")
    print(f"rescue: {len(rescue)} pids, total = {sum(plen(p) for p in rescue.values()):,}")

    n_replaced = 0
    saved = 0
    longer = 0
    out: dict[int, str] = dict(base)
    for pid, rescue_path in rescue.items():
        if pid not in base:
            print(f"  WARNING: rescue pid {pid} not in base; skipping", file=sys.stderr)
            continue
        base_path = base[pid]
        if plen(rescue_path) < plen(base_path):
            saved_here = plen(base_path) - plen(rescue_path)
            saved += saved_here
            n_replaced += 1
            out[pid] = rescue_path
        elif plen(rescue_path) > plen(base_path):
            longer += 1

    print()
    print(f"replaced: {n_replaced}/{len(rescue)} (saved {saved} moves)")
    print(f"rescue longer than base: {longer}")
    print(f"merged total: {sum(plen(p) for p in out.values()):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["initial_state_id", "path"])
        w.writeheader()
        for pid in sorted(out):
            w.writerow({"initial_state_id": pid, "path": out[pid]})
    print(f"wrote {args.out}")

    # Verify
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"\nverify: {report.n_valid}/{report.n_total} valid, total {report.total_moves:,}")
    if not report.all_valid:
        print(f"  INVALID pids: {report.invalid_pids[:10]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
