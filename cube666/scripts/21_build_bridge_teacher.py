"""Build a replay-verified PID-split trajectory store for local bridge training."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.bridge import (  # noqa: E402
    BridgeTrajectoryDataset,
    TERMINAL_ACTION,
    digest_files,
)
from cube666.classical import position_orbits  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument(
        "--submission",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "cube666" / "training" / "bridge_v1" / "trajectories.npz",
    )
    parser.add_argument("--validation-pids", type=int, default=100)
    parser.add_argument("--test-pids", type=int, default=112)
    parser.add_argument("--seed", type=int, default=666)
    return parser.parse_args()


def read_paths(path: Path, known_moves: set[str]) -> list[tuple[int, tuple[str, ...]]]:
    rows: list[tuple[int, tuple[str, ...]]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"initial_state_id", "path"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("submission must contain initial_state_id and path columns")
        for row in reader:
            pid = int(row["initial_state_id"])
            moves = tuple(token for token in row["path"].split(".") if token)
            unknown = set(moves).difference(known_moves)
            if unknown:
                raise ValueError(f"pid {pid} contains unknown moves: {sorted(unknown)}")
            rows.append((pid, moves))
    rows.sort(key=lambda item: item[0])
    if len({pid for pid, _ in rows}) != len(rows):
        raise ValueError("submission contains duplicate puzzle IDs")
    return rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle_info = args.data_dir / "puzzle_info.json"
    test_csv = args.data_dir / "test.csv"
    puzzle = Cube666Puzzle.load(puzzle_info)
    initial_states = {int(pid): state for pid, state in puzzle.iter_test_states(test_csv)}
    paths = read_paths(args.submission, set(puzzle.move_names))
    submission_pids = {pid for pid, _ in paths}
    if submission_pids != set(initial_states):
        missing = sorted(set(initial_states).difference(submission_pids))
        extra = sorted(submission_pids.difference(initial_states))
        raise ValueError(f"submission coverage mismatch: missing={missing}, extra={extra}")
    if args.validation_pids <= 0 or args.test_pids <= 0:
        raise ValueError("validation-pids and test-pids must be positive")
    if args.validation_pids + args.test_pids >= len(paths):
        raise ValueError("validation and test splits leave no training paths")

    move_to_action = {name: index for index, name in enumerate(puzzle.move_names)}
    inverse_actions = np.asarray(
        [move_to_action[puzzle.inverse_name(name)] for name in puzzle.move_names],
        dtype=np.uint8,
    )
    path_pids = np.asarray([pid for pid, _ in paths], dtype=np.int32)
    shuffled = np.random.default_rng(args.seed).permutation(len(paths))
    path_splits = np.zeros(len(paths), dtype=np.uint8)
    path_splits[shuffled[: args.validation_pids]] = 1
    path_splits[
        shuffled[args.validation_pids : args.validation_pids + args.test_pids]
    ] = 2

    lengths = np.asarray([len(path) for _, path in paths], dtype=np.int64)
    path_offsets = np.empty(len(paths) + 1, dtype=np.int64)
    path_offsets[0] = 0
    np.cumsum(lengths + 1, out=path_offsets[1:])
    states = np.empty((int(path_offsets[-1]), puzzle.size), dtype=np.uint16)
    next_actions = np.full(len(states), TERMINAL_ACTION, dtype=np.uint8)
    remaining_lengths = np.empty(len(states), dtype=np.uint16)

    for path_index, (pid, path) in enumerate(paths):
        write = int(path_offsets[path_index])
        current = initial_states[pid]
        states[write] = current
        path_length = len(path)
        remaining_lengths[write] = path_length
        for depth, move in enumerate(path):
            next_actions[write] = move_to_action[move]
            current = puzzle.apply_move(current, move)
            write += 1
            states[write] = current
            remaining_lengths[write] = path_length - depth - 1
        if current != puzzle.solved_state:
            raise ValueError(f"pid {pid} does not replay to the exact solved state")
        expected_stop = int(path_offsets[path_index + 1])
        if write + 1 != expected_stop:
            raise AssertionError("trajectory write cursor disagrees with path offsets")

    dataset = BridgeTrajectoryDataset(
        states=states,
        next_actions=next_actions,
        remaining_lengths=remaining_lengths,
        path_offsets=path_offsets,
        path_pids=path_pids,
        path_splits=path_splits,
        orbit_positions=np.asarray(position_orbits(puzzle.generators), dtype=np.uint16),
        generator_permutations=np.asarray(
            [puzzle.generators[name] for name in puzzle.move_names],
            dtype=np.uint16,
        ),
        inverse_actions=inverse_actions,
        move_names=puzzle.move_names,
        source_digest=digest_files((puzzle_info, test_csv, args.submission)),
    )
    dataset.save(args.out)
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "move_count": int(lengths.sum()),
        "output": str(args.out),
        "path_count": dataset.path_count,
        "source_digest": dataset.source_digest,
        "split_paths": {
            "test": int(np.sum(path_splits == 2)),
            "train": int(np.sum(path_splits == 0)),
            "validation": int(np.sum(path_splits == 1)),
        },
        "split_states": {
            "test": int(sum(lengths[path_splits == 2]) + np.sum(path_splits == 2)),
            "train": int(sum(lengths[path_splits == 0]) + np.sum(path_splits == 0)),
            "validation": int(sum(lengths[path_splits == 1]) + np.sum(path_splits == 1)),
        },
        "state_count": dataset.state_count,
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
