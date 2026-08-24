"""Extract exact relative short-macro windows from a replay-valid submission."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macros import analyze_corner_fixing_macro  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


@dataclass(frozen=True)
class BoundaryAction:
    before_prefix: int
    after_prefix: int
    action: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--action-dir", type=Path, required=True)
    parser.add_argument("--window-actions", type=int, default=5)
    parser.add_argument("--top-windows", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def apply_effect(state: np.ndarray, effect: np.ndarray) -> np.ndarray:
    return np.take_along_axis(state, effect, axis=-1)


def main() -> None:
    args = parse_args()
    if args.window_actions <= 0 or args.top_windows <= 0:
        raise ValueError("window sizes must be positive")
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    test_states = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
    effects = np.load(args.action_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.action_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    inverse = np.load(
        args.action_dir / "inverse_actions.npy", allow_pickle=False
    ).astype(np.int64, copy=False)
    library = json.loads(
        (args.action_dir / "action_library.json").read_text(encoding="utf-8")
    )
    action_paths = [tuple(path) for path in library["paths"]]
    if len(action_paths) != len(effects):
        raise ValueError("action library and effect array disagree")
    action_by_effect = {effect.tobytes(): index for index, effect in enumerate(effects)}

    with open(args.submission, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates: list[dict[str, object]] = []
    mapped_boundaries = 0
    total_boundaries = 0
    replayed_moves = 0
    for row_position, row in enumerate(rows):
        pid = int(row["initial_state_id"])
        path = tuple(token for token in row["path"].split(".") if token)
        current = test_states[str(pid)]
        boundaries: list[tuple[int, tuple[int, ...]]] = []
        for prefix in range(len(path) + 1):
            if all(
                current[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            ):
                report = residual_report(current, puzzle.solved_state, decomposition)
                if report.all_even:
                    boundaries.append((prefix, current))
            if prefix < len(path):
                current = puzzle.apply_move(current, path[prefix])
        if current != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: source submission failed replay")
        replayed_moves += len(path)
        adjacent: list[BoundaryAction | None] = []
        for boundary_position in range(len(boundaries) - 1):
            before_prefix, before_state = boundaries[boundary_position]
            after_prefix, after_state = boundaries[boundary_position + 1]
            segment = path[before_prefix:after_prefix]
            total_boundaries += 1
            if not segment:
                adjacent.append(None)
                continue
            macro = analyze_corner_fixing_macro(
                segment, puzzle.generators, decomposition
            )
            effect = np.asarray(macro.cluster_permutations, dtype=np.uint8)
            action = action_by_effect.get(effect.tobytes())
            if action is None:
                adjacent.append(None)
                continue
            replacement_state = puzzle.apply_path(before_state, action_paths[action])
            if replacement_state != after_state:
                adjacent.append(None)
                continue
            mapped_boundaries += 1
            adjacent.append(BoundaryAction(before_prefix, after_prefix, action))
        for start in range(0, len(adjacent) - args.window_actions + 1):
            pieces = adjacent[start : start + args.window_actions]
            if any(piece is None for piece in pieces):
                continue
            typed = [piece for piece in pieces if piece is not None]
            before_prefix = typed[0].before_prefix
            after_prefix = typed[-1].after_prefix
            target_actions = tuple(piece.action for piece in typed)
            source_moves = after_prefix - before_prefix
            target_cost = int(costs[np.asarray(target_actions)].sum())
            candidates.append(
                {
                    "after_prefix": after_prefix,
                    "before_prefix": before_prefix,
                    "pid": pid,
                    "row_position": row_position,
                    "source_moves": source_moves,
                    "target_actions": target_actions,
                    "target_cost": target_cost,
                }
            )

    selected = sorted(
        candidates,
        key=lambda item: (
            -int(item["source_moves"]),
            -int(item["target_cost"]),
            int(item["pid"]),
            int(item["before_prefix"]),
        ),
    )[: args.top_windows]
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()
    relative_states: list[np.ndarray] = []
    teacher_actions = np.full(
        (len(selected), args.window_actions), -1, dtype=np.int32
    )
    upper_bounds = np.empty(len(selected), dtype=np.float32)
    for position, item in enumerate(selected):
        target = tuple(int(value) for value in item["target_actions"])
        relative = identity.copy()
        for action in reversed(target):
            relative = apply_effect(relative, effects[inverse[action]])
        replay = relative.copy()
        for action in target:
            replay = apply_effect(replay, effects[action])
        if not np.array_equal(replay, identity):
            raise AssertionError(f"window {position}: relative action replay failed")
        relative_states.append(relative)
        teacher_actions[position, : len(target)] = target
        upper_bounds[position] = float(item["target_cost"])
        item["index"] = position
        item["target_actions"] = list(target)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "windows.npz",
        states=np.stack(relative_states),
        search_value_targets=upper_bounds,
        walk_depths=np.full(len(selected), args.window_actions, dtype=np.int16),
        teacher_solution_actions=teacher_actions,
    )
    np.save(args.out_dir / "source_state_ids.npy", np.asarray(
        [int(item["pid"]) for item in selected], dtype=np.int64
    ))
    metadata = {
        "action_dir": str(args.action_dir),
        "candidate_windows": len(candidates),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "mapped_boundaries": mapped_boundaries,
        "replayed_moves": replayed_moves,
        "selected": selected,
        "submission": str(args.submission),
        "top_windows": len(selected),
        "total_boundaries": total_boundaries,
        "window_actions": args.window_actions,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "selected"}, indent=2))


if __name__ == "__main__":
    main()
