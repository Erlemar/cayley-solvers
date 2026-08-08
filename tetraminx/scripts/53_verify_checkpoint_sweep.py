"""Replay-verify a Tetraminx checkpoint sweep and its summary."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.puzzle import Tetraminx


EXPECTED_PIDS = {0, 50, 100, 200, 300, 400, 500, 600, 700, 800,
                 900, 950, 990, 995, 999}


def verify_csv(path: Path, puzzle: Tetraminx,
               states: dict[int, tuple[int, ...]]) -> tuple[int, int]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    pids = [int(row["initial_state_id"]) for row in rows]
    assert len(pids) == len(set(pids)), f"{path.name}: duplicate pid"
    assert set(pids) <= EXPECTED_PIDS, f"{path.name}: unexpected pid"

    total = 0
    for row in rows:
        pid = int(row["initial_state_id"])
        moves = puzzle.parse_path(row["path"])
        reached = puzzle.apply_path(states[pid], moves)
        assert puzzle.is_solved(reached), f"{path.name}: pid {pid} does not solve"
        total += len(moves)
    return len(rows), total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("tetraminx/results/mx_latent_relational"),
        help="Checkpoint-sweep result directory (relative paths use the project root)",
    )
    parser.add_argument(
        "--four-frame-epochs",
        type=int,
        nargs="*",
        default=[300, 1000],
        help="Epochs whose q1m_f4_h1 CSVs must also be replay-verified",
    )
    args = parser.parse_args()

    data_dir = PROJECT / "tetraminx" / "data"
    results = args.results
    if not results.is_absolute():
        results = PROJECT / results
    puzzle = Tetraminx.load(data_dir / "puzzle_info.json")

    with (data_dir / "test.csv").open(encoding="utf-8", newline="") as f:
        states = {
            int(row["initial_state_id"]):
            tuple(int(x) for x in row["initial_state"].split(","))
            for row in csv.DictReader(f)
        }

    measured: dict[int, tuple[int, int]] = {}
    for epoch in range(100, 1501, 100):
        path = results / f"q1m_f1_ep{epoch:04d}.csv"
        assert path.is_file(), f"missing {path.name}"
        measured[epoch] = verify_csv(path, puzzle, states)
        solved, total = measured[epoch]
        print(f"epoch {epoch:4d}: {solved:2d}/15 paths, {total:3d} moves, replay OK")

    with (results / "checkpoint_sweep_summary.csv").open(
            encoding="utf-8", newline="") as f:
        summary = {int(row["epoch"]): row for row in csv.DictReader(f)}
    assert set(summary) == set(measured), "summary epoch set mismatch"
    for epoch, (solved, total) in measured.items():
        row = summary[epoch]
        assert int(row["solved"]) == solved, f"epoch {epoch}: solved mismatch"
        assert int(row["total"]) == total, f"epoch {epoch}: total mismatch"
        expected_status = "complete" if solved == len(EXPECTED_PIDS) else "incomplete"
        assert row["status"] == expected_status, f"epoch {epoch}: status mismatch"

    for epoch in args.four_frame_epochs:
        path = results / f"q1m_f4_h1_ep{epoch:04d}.csv"
        solved, total = verify_csv(path, puzzle, states)
        assert solved == len(EXPECTED_PIDS), f"{path.name}: incomplete"
        print(f"epoch {epoch:4d} four-frame: 15/15 paths, {total:3d} moves, replay OK")

    print("all checkpoint-sweep totals and paths verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
