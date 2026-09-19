"""Describe the reusable permutation geometry behind cube666 teacher labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--action-costs", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def cycle_lengths(permutation: np.ndarray) -> tuple[int, ...]:
    seen = np.zeros(len(permutation), dtype=bool)
    lengths: list[int] = []
    for start in range(len(permutation)):
        if seen[start]:
            continue
        cursor = start
        length = 0
        while not seen[cursor]:
            seen[cursor] = True
            cursor = int(permutation[cursor])
            length += 1
        if length > 1:
            lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def geometry(effect: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(cycle_lengths(cluster) for cluster in effect)


def summarize(counter: Counter[object], limit: int = 20) -> list[dict[str, object]]:
    return [
        {"geometry": repr(key), "count": count}
        for key, count in counter.most_common(limit)
    ]


def main() -> None:
    args = parse_args()
    with np.load(args.teacher, allow_pickle=False) as payload:
        labels = payload["teacher_actions"].astype(np.int32, copy=False)
        counts = payload["teacher_action_counts"].astype(np.int16, copy=False)
    effects = np.load(args.action_effects, allow_pickle=False)
    costs = np.load(args.action_costs, allow_pickle=False)
    library = json.loads(args.action_library.read_text(encoding="utf-8"))
    paths = library["paths"]
    valid_mask = np.arange(labels.shape[1])[None] < counts[:, None]
    occurrences = labels[valid_mask]
    target_actions, frequencies = np.unique(occurrences, return_counts=True)
    all_geometries = [geometry(effect) for effect in effects]
    target_geometries = [all_geometries[int(action)] for action in target_actions]
    occurrence_geometries = [all_geometries[int(action)] for action in occurrences]
    changed_clusters = np.asarray(
        [sum(bool(cycles) for cycles in item) for item in all_geometries]
    )
    moved_points = np.asarray(
        [sum(sum(cycles) for cycles in item) for item in all_geometries]
    )
    target_costs = costs[target_actions]
    long_mask = target_costs > 14
    long_actions = target_actions[long_mask]
    long_geometries = [all_geometries[int(action)] for action in long_actions]
    report = {
        "actions": len(effects),
        "teacher_label_occurrences": len(occurrences),
        "teacher_target_actions": len(target_actions),
        "all_action_changed_cluster_histogram": {
            str(key): int(value)
            for key, value in Counter(changed_clusters.tolist()).items()
        },
        "all_action_moved_point_histogram": {
            str(key): int(value)
            for key, value in Counter(moved_points.tolist()).most_common()
        },
        "target_changed_cluster_histogram": {
            str(key): int(value)
            for key, value in Counter(changed_clusters[target_actions].tolist()).items()
        },
        "target_moved_point_histogram": {
            str(key): int(value)
            for key, value in Counter(moved_points[target_actions].tolist()).most_common()
        },
        "unique_target_geometries": len(set(target_geometries)),
        "top_target_geometries": summarize(Counter(target_geometries)),
        "top_occurrence_geometries": summarize(Counter(occurrence_geometries)),
        "long_target_actions": len(long_actions),
        "long_unique_geometries": len(set(long_geometries)),
        "top_long_geometries": summarize(Counter(long_geometries)),
        "long_cost_minimum": int(costs[long_actions].min()),
        "long_cost_mean": float(costs[long_actions].mean()),
        "long_cost_maximum": int(costs[long_actions].max()),
        "long_path_length_matches_cost": int(
            sum(len(paths[int(action)]) == int(costs[int(action)]) for action in long_actions)
        ),
        "target_frequency_maximum": int(frequencies.max()),
        "target_frequency_median": float(np.median(frequencies)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
