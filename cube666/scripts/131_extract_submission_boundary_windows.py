"""Extract arbitrary corner-fixed/even submission spans as short-macro search queries.

Unlike the mapped-action extractor, this does not require the incumbent segment to
already belong to the learned action vocabulary.  Equal net effects are deduplicated
so one learned replacement can be reused at every compatible path occurrence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macros import analyze_corner_fixing_macro  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--action-dir", type=Path, required=True)
    parser.add_argument("--maximum-boundary-span", type=int, default=5)
    parser.add_argument("--minimum-source-moves", type=int, default=8)
    parser.add_argument("--maximum-source-moves", type=int, default=70)
    parser.add_argument(
        "--top-queries",
        type=int,
        default=0,
        help="0 retains every unique query; otherwise keep this many ranked effects",
    )
    parser.add_argument("--search-depth", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def apply_effect(state: np.ndarray, effect: np.ndarray) -> np.ndarray:
    return np.take_along_axis(state, effect, axis=-1)


def main() -> None:
    args = parse_args()
    if args.maximum_boundary_span <= 0 or args.search_depth <= 0:
        raise ValueError("span and search depth must be positive")
    if args.minimum_source_moves <= 0 or args.maximum_source_moves < args.minimum_source_moves:
        raise ValueError("invalid source-move range")
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    test_states = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()

    with args.submission.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    occurrences_by_effect: dict[bytes, list[dict[str, int]]] = defaultdict(list)
    relative_by_effect: dict[bytes, np.ndarray] = {}
    candidate_spans = 0
    eligible_boundaries = 0
    replayed_moves = 0
    for row_position, row in enumerate(rows):
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
        eligible_boundaries += len(boundary_prefixes)
        for start in range(len(boundary_prefixes) - 1):
            for end in range(start + 1, min(len(boundary_prefixes), start + args.maximum_boundary_span + 1)):
                before_prefix = boundary_prefixes[start]
                after_prefix = boundary_prefixes[end]
                source_moves = after_prefix - before_prefix
                if not args.minimum_source_moves <= source_moves <= args.maximum_source_moves:
                    continue
                segment = path[before_prefix:after_prefix]
                macro = analyze_corner_fixing_macro(segment, puzzle.generators, decomposition)
                effect = np.asarray(macro.cluster_permutations, dtype=np.uint8)
                relative = np.argsort(effect, axis=1).astype(np.uint8, copy=False)
                if not np.array_equal(apply_effect(relative, effect), identity):
                    raise AssertionError(f"PID {pid} [{before_prefix},{after_prefix}): inverse effect failed")
                key = effect.tobytes()
                relative_by_effect[key] = relative
                occurrences_by_effect[key].append(
                    {
                        "after_prefix": after_prefix,
                        "before_prefix": before_prefix,
                        "pid": pid,
                        "row_position": row_position,
                        "source_moves": source_moves,
                    }
                )
                candidate_spans += 1

    grouped: list[dict[str, object]] = []
    for key, occurrences in occurrences_by_effect.items():
        source_lengths = [item["source_moves"] for item in occurrences]
        grouped.append(
            {
                "effect_key": key,
                "effect_sha256": hashlib.sha256(key).hexdigest(),
                "maximum_source_moves": max(source_lengths),
                "occurrence_count": len(occurrences),
                "occurrences": sorted(
                    occurrences,
                    key=lambda item: (item["pid"], item["before_prefix"], item["after_prefix"]),
                ),
                "source_moves": max(source_lengths),
                "total_source_moves": sum(source_lengths),
            }
        )
    grouped.sort(
        key=lambda item: (
            -int(item["maximum_source_moves"]),
            -int(item["occurrence_count"]),
            -int(item["total_source_moves"]),
            str(item["effect_sha256"]),
        )
    )
    if args.top_queries:
        grouped = grouped[: args.top_queries]

    selected: list[dict[str, object]] = []
    states: list[np.ndarray] = []
    group_ids: list[int] = []
    upper_bounds: list[float] = []
    for index, item in enumerate(grouped):
        key = item.pop("effect_key")
        item["index"] = index
        selected.append(item)
        states.append(relative_by_effect[key])
        group_ids.append(int(item["occurrences"][0]["pid"]))
        upper_bounds.append(float(item["maximum_source_moves"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_array = np.stack(states) if states else np.empty((0, 6, 24), dtype=np.uint8)
    np.savez_compressed(
        args.out_dir / "windows.npz",
        states=state_array,
        search_value_targets=np.asarray(upper_bounds, dtype=np.float32),
        walk_depths=np.full(len(states), args.search_depth, dtype=np.int16),
        teacher_solution_actions=np.full(
            (len(states), args.search_depth), -1, dtype=np.int32
        ),
    )
    np.save(args.out_dir / "source_state_ids.npy", np.asarray(group_ids, dtype=np.int64))
    metadata = {
        "action_dir": str(args.action_dir),
        "candidate_spans": candidate_spans,
        "eligible_boundaries": eligible_boundaries,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "maximum_boundary_span": args.maximum_boundary_span,
        "maximum_source_moves": args.maximum_source_moves,
        "minimum_source_moves": args.minimum_source_moves,
        "replayed_moves": replayed_moves,
        "search_depth": args.search_depth,
        "selected": selected,
        "submission": str(args.submission),
        "unique_queries": len(selected),
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "selected"}, indent=2))


if __name__ == "__main__":
    main()
