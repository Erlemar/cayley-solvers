#!/usr/bin/env python3
"""Merge verified solutions from several ensemble progress databases."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import torch

from pilgrim import load_torch_file, parse_generator_spec
from solve_ensemble_submission import (
    BASE_DIR,
    apply_path,
    load_csv_states,
    load_paths,
    path_length,
    state_hash,
    validate_external_paths,
    write_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("databases", type=Path, nargs="+")
    parser.add_argument("--test-csv", type=Path, default=BASE_DIR / "test.csv")
    parser.add_argument(
        "--baseline-submission",
        type=Path,
        default=BASE_DIR / "baseline_submission.csv",
    )
    parser.add_argument(
        "--generator-file", type=Path, default=BASE_DIR / "generators/p002.json"
    )
    parser.add_argument(
        "--target-file", type=Path, default=BASE_DIR / "targets/p002-t000.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=BASE_DIR / "results/submission_merged.csv"
    )
    args = parser.parse_args()

    ordered_states = load_csv_states(args.test_csv)
    states_by_id = dict(ordered_states)
    with args.generator_file.open("r", encoding="utf-8") as handle:
        moves, move_names = parse_generator_spec(json.load(handle))
    all_moves_cpu = torch.tensor(moves, dtype=torch.int64)
    move_to_idx = {name: index for index, name in enumerate(move_names)}
    target_cpu = load_torch_file(
        args.target_file, weights_only=True, map_location="cpu"
    ).long()

    baseline = validate_external_paths(
        label=str(args.baseline_submission),
        paths=load_paths(args.baseline_submission),
        states_by_id=states_by_id,
        target_cpu=target_cpu,
        move_to_idx=move_to_idx,
        all_moves_cpu=all_moves_cpu,
    )
    merged: dict[int, str] = {}
    accepted_per_database: dict[str, int] = {}
    for database in args.databases:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        accepted = 0
        for initial_state_id, stored_hash, path in connection.execute(
            "SELECT initial_state_id, state_hash, path FROM solutions"
        ):
            initial_state_id = int(initial_state_id)
            state = states_by_id.get(initial_state_id)
            if state is None or state_hash(state) != stored_hash:
                continue
            try:
                verified = torch.equal(
                    apply_path(state, path, move_to_idx, all_moves_cpu), target_cpu
                )
            except ValueError:
                verified = False
            if not verified:
                continue
            if initial_state_id not in merged or path_length(path) < path_length(
                merged[initial_state_id]
            ):
                merged[initial_state_id] = path
            accepted += 1
        connection.close()
        accepted_per_database[str(database)] = accepted
        print(f"{database}: accepted {accepted} verified paths")

    report = write_outputs(
        output=args.output,
        ordered_states=ordered_states,
        database_paths=merged,
        baseline_paths=baseline,
        run_config={
            "operation": "merge",
            "databases": [str(path) for path in args.databases],
            "accepted_per_database": accepted_per_database,
        },
    )
    print(
        f"Saved {args.output}: ensemble_n={len(merged)}, "
        f"total={report['best_total_length']}, "
        f"saved={report['moves_saved_vs_baseline']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
