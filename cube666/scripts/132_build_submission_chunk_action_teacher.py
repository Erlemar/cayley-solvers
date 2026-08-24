"""Build a <=5-step hybrid action teacher from all incumbent boundary chunks.

The existing short actions stay at their original indices.  Every net effect between
two corner-fixed/even boundaries is then added (with its inverse) as a reusable
trajectory chunk.  Multiple <=5-chunk factorizations of each incumbent residual
become short-horizon policy traces.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macros import analyze_corner_fixing_macro, invert_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--short-action-dir", type=Path, required=True)
    parser.add_argument("--maximum-depth", type=int, default=5)
    parser.add_argument("--maximum-decompositions-per-pid", type=int, default=32)
    parser.add_argument("--seed", type=int, default=132666)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def inverse_effect(effect: np.ndarray) -> np.ndarray:
    return np.argsort(effect, axis=1).astype(np.uint8, copy=False)


def apply_effect(state: np.ndarray, effect: np.ndarray) -> np.ndarray:
    return np.take_along_axis(state, effect, axis=-1)


def main() -> None:
    args = parse_args()
    if args.maximum_depth <= 0 or args.maximum_decompositions_per_pid <= 0:
        raise ValueError("depth and decomposition limit must be positive")
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    test_states = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()

    short_effects = np.load(
        args.short_action_dir / "action_effects.npy", allow_pickle=False
    ).astype(np.uint8, copy=False)
    short_costs = np.load(
        args.short_action_dir / "action_costs.npy", allow_pickle=False
    ).astype(np.int32, copy=False)
    short_library = json.loads(
        (args.short_action_dir / "action_library.json").read_text(encoding="utf-8")
    )
    short_paths = [tuple(path) for path in short_library["paths"]]
    if not (len(short_effects) == len(short_costs) == len(short_paths)):
        raise ValueError("short action artifacts disagree")

    path_by_effect: dict[bytes, tuple[str, ...]] = {
        effect.tobytes(): short_paths[index] for index, effect in enumerate(short_effects)
    }
    base_index_by_effect = {
        effect.tobytes(): index for index, effect in enumerate(short_effects)
    }
    trajectory_rows: list[dict[str, object]] = []
    skipped_pids: list[int] = []
    with args.submission.open(newline="", encoding="utf-8") as handle:
        submission_rows = list(csv.DictReader(handle))
    replayed_moves = 0
    total_spans = 0
    for row_position, row in enumerate(submission_rows):
        pid = int(row["initial_state_id"])
        path = tuple(token for token in row["path"].split(".") if token)
        current = test_states[str(pid)]
        boundary_prefixes: list[int] = []
        for prefix in range(len(path) + 1):
            if all(
                current[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            ):
                report = residual_report(current, puzzle.solved_state, decomposition)
                if report.all_even:
                    boundary_prefixes.append(prefix)
            if prefix < len(path):
                current = puzzle.apply_move(current, path[prefix])
        if current != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: source submission failed replay")
        replayed_moves += len(path)
        if len(boundary_prefixes) < 2:
            skipped_pids.append(pid)
            continue

        span_keys: dict[tuple[int, int], bytes] = {}
        for start in range(len(boundary_prefixes) - 1):
            for end in range(start + 1, len(boundary_prefixes)):
                before = boundary_prefixes[start]
                after = boundary_prefixes[end]
                segment = path[before:after]
                macro = analyze_corner_fixing_macro(segment, puzzle.generators, decomposition)
                effect = np.asarray(macro.cluster_permutations, dtype=np.uint8)
                key = effect.tobytes()
                inverse = inverse_effect(effect)
                inverse_key = inverse.tobytes()
                inverse_path = invert_path(segment)
                if key not in path_by_effect or len(segment) < len(path_by_effect[key]):
                    path_by_effect[key] = segment
                if inverse_key not in path_by_effect or len(inverse_path) < len(path_by_effect[inverse_key]):
                    path_by_effect[inverse_key] = inverse_path
                span_keys[(start, end)] = key
                total_spans += 1

        intervals = len(boundary_prefixes) - 1
        cut_sets: list[tuple[int, ...]] = [()]
        for chunks in range(2, min(args.maximum_depth, intervals) + 1):
            cut_sets.extend(itertools.combinations(range(1, intervals), chunks - 1))
        if len(cut_sets) > args.maximum_decompositions_per_pid:
            mandatory = {(), tuple(range(1, intervals))} if intervals <= args.maximum_depth else {()}
            evenly_spaced: set[tuple[int, ...]] = set()
            for chunks in range(2, min(args.maximum_depth, intervals) + 1):
                cuts = tuple(
                    sorted(
                        {
                            int(round(position * intervals / chunks))
                            for position in range(1, chunks)
                        }
                    )
                )
                if len(cuts) == chunks - 1 and all(0 < cut < intervals for cut in cuts):
                    evenly_spaced.add(cuts)
            retained = list(mandatory | evenly_spaced)
            remaining = [cuts for cuts in cut_sets if cuts not in set(retained)]
            slots = args.maximum_decompositions_per_pid - len(retained)
            if slots > 0:
                choices = rng.choice(len(remaining), size=min(slots, len(remaining)), replace=False)
                retained.extend(remaining[int(position)] for position in choices)
            cut_sets = retained
        cut_sets = sorted(set(cut_sets), key=lambda cuts: (len(cuts), cuts))
        trajectory_rows.append(
            {
                "after_prefix": boundary_prefixes[-1],
                "before_prefix": boundary_prefixes[0],
                "cut_sets": cut_sets,
                "intervals": intervals,
                "pid": pid,
                "row_position": row_position,
                "span_keys": span_keys,
            }
        )

    base_keys = [effect.tobytes() for effect in short_effects]
    new_keys = sorted(
        (key for key in path_by_effect if key not in base_index_by_effect),
        key=lambda key: (len(path_by_effect[key]), path_by_effect[key], key),
    )
    ordered_keys = base_keys + new_keys
    effect_to_action = {key: index for index, key in enumerate(ordered_keys)}
    effects = np.stack(
        [short_effects[index] for index in range(len(short_effects))]
        + [np.frombuffer(key, dtype=np.uint8).reshape(6, 24) for key in new_keys]
    )
    paths = [path_by_effect[key] for key in ordered_keys]
    costs = np.asarray([len(path) for path in paths], dtype=np.int16)
    inverse_actions = np.empty(len(effects), dtype=np.int32)
    for action, effect in enumerate(effects):
        inverse_key = inverse_effect(effect).tobytes()
        inverse_action = effect_to_action.get(inverse_key)
        if inverse_action is None:
            raise AssertionError(f"action {action}: inverse is absent")
        inverse_actions[action] = inverse_action
        replay = apply_effect(effect, effects[inverse_action])
        if not np.array_equal(replay, identity):
            raise AssertionError(f"action {action}: inverse replay failed")

    states_out: list[np.ndarray] = []
    solutions_out: list[list[int]] = []
    values_out: list[float] = []
    groups_out: list[int] = []
    query_metadata: list[dict[str, object]] = []
    for item in trajectory_rows:
        intervals = int(item["intervals"])
        span_keys = item["span_keys"]
        for cuts in item["cut_sets"]:
            endpoints = (0, *cuts, intervals)
            solution = [
                effect_to_action[span_keys[(start, end)]]
                for start, end in zip(endpoints[:-1], endpoints[1:])
            ]
            relative = identity.copy()
            for action in reversed(solution):
                relative = apply_effect(relative, effects[inverse_actions[action]])
            replay = relative.copy()
            for action in solution:
                replay = apply_effect(replay, effects[action])
            if not np.array_equal(replay, identity):
                raise AssertionError(f"PID {item['pid']}: trajectory replay failed")
            states_out.append(relative)
            solutions_out.append(solution)
            values_out.append(float(costs[np.asarray(solution)].sum()))
            groups_out.append(int(item["pid"]))
            query_metadata.append(
                {
                    "after_prefix": int(item["after_prefix"]),
                    "before_prefix": int(item["before_prefix"]),
                    "cuts": list(cuts),
                    "index": len(states_out) - 1,
                    "pid": int(item["pid"]),
                    "row_position": int(item["row_position"]),
                    "source_moves": int(item["after_prefix"]) - int(item["before_prefix"]),
                }
            )

    maximum_solution_depth = max(len(solution) for solution in solutions_out)
    solution_array = np.full(
        (len(solutions_out), maximum_solution_depth), -1, dtype=np.int32
    )
    for row, solution in enumerate(solutions_out):
        solution_array[row, : len(solution)] = solution
    depth_array = np.asarray([len(solution) for solution in solutions_out], dtype=np.int16)
    action_digest = hashlib.sha256(
        json.dumps(paths, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "teacher.npz",
        states=np.stack(states_out),
        teacher_actions=solution_array[:, :1],
        teacher_action_counts=np.ones(len(states_out), dtype=np.int16),
        search_value_targets=np.asarray(values_out, dtype=np.float32),
        walk_depths=depth_array,
        teacher_solution_actions=solution_array,
    )
    np.save(args.out_dir / "source_state_ids.npy", np.asarray(groups_out, dtype=np.int64))
    np.save(args.out_dir / "action_effects.npy", effects)
    np.save(args.out_dir / "action_costs.npy", costs)
    np.save(args.out_dir / "inverse_actions.npy", inverse_actions)
    (args.out_dir / "action_library.json").write_text(
        json.dumps(
            {
                "action_digest": action_digest,
                "format_version": 1,
                "paths": paths,
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "action_count": len(effects),
        "action_digest": action_digest,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "maximum_action_cost": int(costs.max()),
        "maximum_depth": maximum_solution_depth,
        "mean_depth": float(depth_array.mean()),
        "mean_primitive_target": float(np.mean(values_out)),
        "new_actions": len(new_keys),
        "query_count": len(states_out),
        "query_metadata": query_metadata,
        "replayed_moves": replayed_moves,
        "short_actions": len(short_effects),
        "skipped_pids": skipped_pids,
        "source_pids": len(trajectory_rows),
        "submission": str(args.submission),
        "total_boundary_spans": total_spans,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "query_metadata"}, indent=2))


if __name__ == "__main__":
    main()
