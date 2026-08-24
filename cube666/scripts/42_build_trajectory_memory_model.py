"""Build an exact trajectory-memory value model from verified cube666 paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument(
        "--submission", type=Path, action="append", default=[],
        help="replay-verified CSV source; may be repeated",
    )
    parser.add_argument(
        "--result-dir", type=Path, action="append", default=[],
        help="directory containing NNNN.path.txt candidates; may be repeated",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def load_submission(path: Path) -> dict[int, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"initial_state_id", "path"}:
        raise ValueError(f"{path}: expected initial_state_id,path")
    return {
        int(row["initial_state_id"]): parse_path(row["path"])
        for row in rows
    }


def main() -> None:
    args = parse_args()
    if not args.submission:
        raise ValueError("at least one --submission is required for complete coverage")
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    initial_states = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    move_names = tuple(puzzle.move_names)
    move_to_index = {move: index for index, move in enumerate(move_names)}

    candidates: defaultdict[int, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    source_manifest: list[dict[str, object]] = []
    for path in args.submission:
        rows = load_submission(path)
        for pid, candidate in rows.items():
            candidates[pid].append((path.stem, candidate))
        source_manifest.append(
            {
                "kind": "submission",
                "path": str(path),
                "sha256": sha256_file(path),
                "paths": len(rows),
            }
        )
    for directory in args.result_dir:
        path_files = sorted(directory.glob("*.path.txt"))
        accepted = 0
        for path_file in path_files:
            try:
                pid = int(path_file.name.split(".", 1)[0])
            except ValueError:
                continue
            candidates[pid].append((directory.name, parse_path(path_file.read_text())))
            accepted += 1
        source_manifest.append(
            {
                "kind": "result_dir",
                "path": str(directory),
                "paths": accepted,
            }
        )

    missing = sorted(set(initial_states).difference(candidates))
    if missing:
        raise ValueError(f"missing candidate paths for PIDs: {missing[:10]}")

    all_states: list[np.ndarray] = []
    all_pids: list[int] = []
    all_distances: list[int] = []
    all_action_masks: list[int] = []
    offsets = [0]
    per_pid_rows: list[dict[str, object]] = []
    replay_verified = 0

    for pid in sorted(initial_states):
        initial = initial_states[pid]
        node_states: list[tuple[int, ...]] = []
        node_index: dict[tuple[int, ...], int] = {}
        edges: defaultdict[int, set[tuple[int, int]]] = defaultdict(set)
        unique_paths: set[tuple[str, ...]] = set()
        source_lengths: list[int] = []

        def intern(state: tuple[int, ...]) -> int:
            position = node_index.get(state)
            if position is None:
                position = len(node_states)
                node_index[state] = position
                node_states.append(state)
            return position

        for _, path in candidates[pid]:
            if path in unique_paths:
                continue
            unique_paths.add(path)
            current = initial
            current_index = intern(current)
            for move in path:
                action = move_to_index.get(move)
                if action is None:
                    raise ValueError(f"PID {pid}: unknown move {move!r}")
                next_state = puzzle.apply_move(current, move)
                next_index = intern(next_state)
                edges[current_index].add((action, next_index))
                current = next_state
                current_index = next_index
            if current != puzzle.solved_state:
                raise AssertionError(f"PID {pid}: source candidate failed exact replay")
            replay_verified += 1
            source_lengths.append(len(path))

        # Multiple trajectories can pass one move apart without containing the
        # same state.  Add every exact generator edge whose endpoint is already
        # represented in this PID's memory, allowing safe path recombination.
        if len(unique_paths) > 1:
            for source, state in enumerate(tuple(node_states)):
                for action, move in enumerate(move_names):
                    target = node_index.get(puzzle.apply_move(state, move))
                    if target is not None:
                        edges[source].add((action, target))

        solved_index = node_index.get(puzzle.solved_state)
        initial_index = node_index.get(initial)
        if solved_index is None or initial_index is None:
            raise AssertionError(f"PID {pid}: missing terminal model nodes")
        reverse: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for source, outgoing in edges.items():
            for action, target in outgoing:
                reverse[target].append((source, action))
        unreachable = np.iinfo(np.uint16).max
        distances = np.full(len(node_states), unreachable, dtype=np.uint16)
        distances[solved_index] = 0
        queue: deque[int] = deque((solved_index,))
        while queue:
            target = queue.popleft()
            next_distance = int(distances[target]) + 1
            for source, _ in reverse.get(target, ()):
                if distances[source] != unreachable:
                    continue
                distances[source] = next_distance
                queue.append(source)
        if distances[initial_index] == unreachable:
            raise AssertionError(f"PID {pid}: memory graph cannot reach solved state")

        action_masks = np.zeros(len(node_states), dtype=np.uint64)
        for source, outgoing in edges.items():
            source_distance = int(distances[source])
            if source_distance in (0, unreachable):
                continue
            for action, target in outgoing:
                if int(distances[target]) + 1 == source_distance:
                    action_masks[source] |= np.uint64(1) << np.uint64(action)
        if np.any((distances != 0) & (distances != unreachable) & (action_masks == 0)):
            raise AssertionError(f"PID {pid}: reachable state has no optimal action")

        all_states.extend(np.asarray(node_states, dtype=np.uint8))
        all_pids.extend([pid] * len(node_states))
        all_distances.extend(int(value) for value in distances)
        all_action_masks.extend(int(value) for value in action_masks)
        offsets.append(len(all_states))
        per_pid_rows.append(
            {
                "best_source_moves": min(source_lengths),
                "model_moves": int(distances[initial_index]),
                "nodes": len(node_states),
                "pid": pid,
                "saving_vs_best_source": min(source_lengths)
                - int(distances[initial_index]),
                "unique_paths": len(unique_paths),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.npz"
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            states=np.asarray(all_states, dtype=np.uint8),
            pids=np.asarray(all_pids, dtype=np.int16),
            distances=np.asarray(all_distances, dtype=np.uint16),
            action_masks=np.asarray(all_action_masks, dtype=np.uint64),
            offsets=np.asarray(offsets, dtype=np.int64),
        )
    temporary.replace(model_path)
    model_sha256 = sha256_file(model_path)
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "kind": "cube666_trajectory_memory_value_model_v1",
        "model": str(model_path),
        "model_sha256": model_sha256,
        "move_names": list(move_names),
        "nodes": len(all_states),
        "pids": len(per_pid_rows),
        "replay_verified_source_paths": replay_verified,
        "rows": per_pid_rows,
        "source_manifest": source_manifest,
        "total_best_source_moves": sum(
            int(row["best_source_moves"]) for row in per_pid_rows
        ),
        "total_model_moves": sum(int(row["model_moves"]) for row in per_pid_rows),
        "total_saving_vs_best_sources": sum(
            int(row["saving_vs_best_source"]) for row in per_pid_rows
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
