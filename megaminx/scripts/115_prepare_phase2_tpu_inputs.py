"""Prepare reproducible TPU inputs for two-phase Phase 2.

Input is the JSON emitted by `113_phase1_export.py`. Output is:
  * a restricted 12-generator `puzzle_info_phase2.json`;
  * a quoted `test_phase2.csv` whose `initial_state_id` is the original pid;
  * an optional manifest carrying Phase-1 paths for final splicing.

The quoted CSV matters: an unquoted comma-separated state is parsed by
`csv.DictReader` as many columns, leaving `initial_state` with only the first value.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx
from megaminx.two_phase import build_phase2_puzzle, movable_frozen_positions


def _want_pid(pid: int, pids: set[int] | None) -> bool:
    return pids is None or pid in pids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-json", required=True, type=Path,
                    help="JSON emitted by scripts/113_phase1_export.py")
    ap.add_argument("--pids", default="",
                    help="optional comma-separated original pids to include")
    ap.add_argument("--puzzle-info-out", type=Path,
                    default=PROJECT / "kaggle_notebooks" / "tpu_beam_spmd_jax" / "puzzle_info_phase2.json")
    ap.add_argument("--test-csv-out", type=Path,
                    default=PROJECT / "kaggle_notebooks" / "tpu_beam_spmd_jax" / "test_phase2.csv")
    ap.add_argument("--manifest-out", type=Path, default=None,
                    help="optional JSON manifest with pid,len1,path1,row_index")
    args = ap.parse_args()

    pids = None
    if args.pids.strip():
        pids = {int(x) for x in args.pids.split(",") if x.strip()}

    full = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    phase2 = build_phase2_puzzle(full)
    _movable, frozen = movable_frozen_positions(full)

    phase1_rows = json.load(open(args.phase1_json, encoding="utf-8"))
    selected = []
    for row in phase1_rows:
        pid = int(row["pid"])
        if not _want_pid(pid, pids):
            continue
        if not row.get("found"):
            raise ValueError(f"pid {pid}: Phase 1 did not find an H-state")
        h_state = [int(x) for x in row["h_state"]]
        if len(h_state) != len(full.solved_state):
            raise ValueError(f"pid {pid}: H-state has length {len(h_state)}, expected {len(full.solved_state)}")
        if not all(h_state[p] == p for p in frozen):
            raise ValueError(f"pid {pid}: H-state is not frozen-home; refusing to write Phase-2 input")
        selected.append({
            "pid": pid,
            "len1": int(row["len1"]),
            "path1": list(row["path1"]),
            "h_state": h_state,
        })

    if not selected:
        raise ValueError("no Phase-1 rows selected")

    args.puzzle_info_out.parent.mkdir(parents=True, exist_ok=True)
    pinfo = {
        "central_state": list(phase2.solved_state),
        "generators": {nm: list(phase2.generators[nm]) for nm in phase2.move_names},
    }
    json.dump(pinfo, open(args.puzzle_info_out, "w", encoding="utf-8"))

    args.test_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.test_csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["initial_state_id", "initial_state"])
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "initial_state_id": row["pid"],
                "initial_state": ",".join(str(x) for x in row["h_state"]),
            })

    manifest = []
    for row_idx, row in enumerate(selected):
        manifest.append({
            "row_index": row_idx,
            "pid": row["pid"],
            "len1": row["len1"],
            "path1": row["path1"],
        })
    if args.manifest_out is not None:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(manifest, open(args.manifest_out, "w", encoding="utf-8"), indent=0)

    print(f"wrote {len(selected)} Phase-2 rows -> {args.test_csv_out}", flush=True)
    print(f"wrote restricted puzzle ({len(phase2.move_names)} generators) -> {args.puzzle_info_out}", flush=True)
    if args.manifest_out is not None:
        print(f"wrote manifest -> {args.manifest_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
