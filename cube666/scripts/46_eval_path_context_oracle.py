"""Evaluate state-only versus path-aware ranking on the KMC rough-word oracle."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_oracle_gate16_v1/"
            "unique_rough_rows_insertion_context.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "cube666/reports/path_context_insertion_gate16_v1.json",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=100.0)
    return parser.parse_args()


def configuration_features(row: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [
            float(row["alpha"]),
            math.log10(float(row["anneal_steps"])),
            float(int(row["anneal_seed"]) == 50002),
            float(bool(row["anneal_keep_best"])),
        ],
        dtype=np.float64,
    )


def state_summary_features(row: dict[str, object]) -> np.ndarray:
    features = [
        float(row["rough_moves"]),
        float(row["rough_reduced_moves"]),
        float(row["residual_three_cycles"]),
        float(row["mismatches"]),
    ]
    for cluster in row["clusters"]:
        cycles = cluster["cycles"]
        features.extend(
            (
                float(cluster["misplaced"]),
                float(cluster["three_cycles"]),
                float(len(cycles)),
                float(max((len(cycle) for cycle in cycles), default=0)),
            )
        )
        histogram = [0.0] * 23
        for cycle in cycles:
            histogram[len(cycle) - 2] += 1.0
        features.extend(histogram)
    return np.asarray(features, dtype=np.float64)


def exact_state_features(row: dict[str, object]) -> np.ndarray:
    summary = state_summary_features(row)
    permutations = np.asarray(row["cluster_permutations"], dtype=np.int64)
    one_hot = np.eye(24, dtype=np.float64)[permutations].reshape(-1)
    return np.concatenate((summary, one_hot))


def path_features(row: dict[str, object]) -> np.ndarray:
    state = exact_state_features(row)
    raw = row["path_features"]
    action_ids = [int(value) for value in raw["action_ids"]]
    bigrams = np.zeros(36 * 36, dtype=np.float64)
    for left, right in zip(action_ids, action_ids[1:]):
        bigrams[left * 36 + right] += 1.0
    endpoints = np.zeros(16 * 36, dtype=np.float64)
    prefix = action_ids[:8]
    suffix = action_ids[-8:]
    for position, action in enumerate(prefix):
        endpoints[position * 36 + action] = 1.0
    for position, action in enumerate(reversed(suffix), start=8):
        endpoints[position * 36 + action] = 1.0
    scalars = np.asarray(
        [
            *raw["move_counts"],
            *raw["axis_counts"],
            *raw["layer_counts"],
            *raw["direction_counts"],
            raw["same_axis_transitions"],
            raw["same_layer_transitions"],
            raw["inverse_transitions"],
            raw["max_axis_run"],
        ],
        dtype=np.float64,
    )
    return np.concatenate((state, scalars, bigrams, endpoints))


def insertion_context_features(row: dict[str, object]) -> np.ndarray:
    """Compact the sorted exact insertion frontier into fixed-size features."""

    raw = row["insertion_context"]
    added = np.asarray(raw["added_lengths"], dtype=np.float64)
    clusters = np.asarray(raw["cluster_indices"], dtype=np.int64)
    positions = np.asarray(raw["insertion_indices"], dtype=np.float64)
    returned = max(int(raw["descriptors_returned"]), 1)
    path_length = max(int(raw["path_length"]), 1)

    rank_values = []
    for rank in (0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 511):
        rank_values.append(float(added[min(rank, len(added) - 1)]) if len(added) else 32.0)
    prefix_moments = []
    for width in (4, 8, 16, 32, 64, 128, 256, 512):
        prefix = added[:width]
        prefix_moments.extend(
            (
                float(prefix.mean()) if len(prefix) else 32.0,
                float(prefix.std()) if len(prefix) else 0.0,
            )
        )
    # Insertion words have even length and local cancellation changes that
    # length by an even amount.  Half-open odd boundaries isolate the common
    # deltas 0, 2, 4, ... without relying on that invariant.
    delta_histogram = np.histogram(
        added, bins=np.asarray([-1000, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 1000])
    )[0].astype(np.float64) / returned

    cluster_best = np.asarray(
        [32.0 if value is None else float(value) for value in raw["cluster_best_added_lengths"]],
        dtype=np.float64,
    )
    cluster_counts = np.asarray(raw["cluster_descriptor_counts"], dtype=np.float64) / returned
    position_counts = np.asarray(raw["position_descriptor_counts"], dtype=np.float64) / returned

    frontier_counts = []
    for width in (32, 128):
        prefix_clusters = clusters[:width]
        frontier_counts.extend(
            np.bincount(prefix_clusters, minlength=6).astype(np.float64)
            / max(len(prefix_clusters), 1)
        )
        prefix_positions = positions[:width]
        bins = np.minimum(
            7,
            (prefix_positions.astype(np.int64) * 8) // (path_length + 1),
        )
        frontier_counts.extend(
            np.bincount(bins, minlength=8).astype(np.float64) / max(len(bins), 1)
        )

    scalars = np.asarray(
        [
            float(raw["path_length"]),
            float(raw["residual_cost"]),
            float(raw["descriptors_returned"]) / float(raw["descriptor_limit"]),
            float(raw["distinct_desired_cycles"]) / returned,
            *rank_values,
            *prefix_moments,
            *cluster_best,
            *cluster_counts,
            *position_counts,
            *frontier_counts,
        ],
        dtype=np.float64,
    )
    return np.concatenate((scalars, delta_histogram))


def pair_rows(
    features: np.ndarray,
    targets: np.ndarray,
    pids: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    differences = []
    target_differences = []
    for pid in np.unique(pids[indices]):
        group = indices[pids[indices] == pid]
        for left, right in itertools.combinations(group, 2):
            if targets[left] == targets[right]:
                continue
            differences.append(features[left] - features[right])
            target_differences.append(targets[left] - targets[right])
    return np.asarray(differences), np.asarray(target_differences)


def fit_pairwise_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    pids: np.ndarray,
    train_indices: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = features[train_indices]
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    differences, target_differences = pair_rows(
        standardized, targets, pids, train_indices
    )
    gram = differences @ differences.T
    dual = np.linalg.solve(
        gram + ridge * np.eye(len(gram), dtype=np.float64), target_differences
    )
    weights = differences.T @ dual
    return standardized @ weights, mean, scale


def score_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    pids: np.ndarray,
) -> dict[str, float | int]:
    selected_total = 0
    oracle_total = 0
    winner_hits = 0
    pair_correct = 0
    pair_total = 0
    for pid in np.unique(pids):
        group = np.flatnonzero(pids == pid)
        selected = group[int(np.argmin(predictions[group]))]
        oracle = float(targets[group].min())
        selected_total += int(targets[selected])
        oracle_total += int(oracle)
        winner_hits += int(targets[selected] == oracle)
        for left, right in itertools.combinations(group, 2):
            actual = targets[left] - targets[right]
            if actual == 0:
                continue
            predicted = predictions[left] - predictions[right]
            pair_correct += int(actual * predicted > 0)
            pair_total += 1
    return {
        "oracle_total": oracle_total,
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "pairwise_comparisons": pair_total,
        "selected_regret": selected_total - oracle_total,
        "selected_total": selected_total,
        "winner_hit_fraction": winner_hits / len(np.unique(pids)),
        "winner_hits": winner_hits,
    }


def cross_validated_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    pids: np.ndarray,
    folds: int,
    ridge: float,
) -> np.ndarray:
    unique_pids = np.asarray(sorted(np.unique(pids)))
    predictions = np.full(len(targets), np.nan, dtype=np.float64)
    for fold in range(folds):
        validation_pids = unique_pids[fold::folds]
        validation = np.flatnonzero(np.isin(pids, validation_pids))
        train = np.flatnonzero(~np.isin(pids, validation_pids))
        fold_predictions, _, _ = fit_pairwise_ridge(
            features, targets, pids, train, ridge
        )
        predictions[validation] = fold_predictions[validation]
    if np.isnan(predictions).any():
        raise AssertionError("cross-validation did not predict every row")
    return predictions


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    targets = np.asarray([row["best_final_moves"] for row in rows], dtype=np.float64)
    pids = np.asarray([row["pid"] for row in rows], dtype=np.int64)
    feature_sets = {
        "configuration_only": np.stack([configuration_features(row) for row in rows]),
        "state_summary": np.stack([state_summary_features(row) for row in rows]),
        "exact_state": np.stack([exact_state_features(row) for row in rows]),
        "exact_state_plus_path": np.stack([path_features(row) for row in rows]),
    }
    if all("insertion_context" in row for row in rows):
        insertion = np.stack([insertion_context_features(row) for row in rows])
        feature_sets.update(
            {
                "insertion_context": insertion,
                "state_summary_plus_insertion_context": np.stack(
                    [
                        np.concatenate((state_summary_features(row), insertion[index]))
                        for index, row in enumerate(rows)
                    ]
                ),
                "exact_state_plus_insertion_context": np.stack(
                    [
                        np.concatenate((exact_state_features(row), insertion[index]))
                        for index, row in enumerate(rows)
                    ]
                ),
                "exact_state_plus_path_plus_insertion_context": np.stack(
                    [
                        np.concatenate((path_features(row), insertion[index]))
                        for index, row in enumerate(rows)
                    ]
                ),
            }
        )
    evaluations = {}
    for name, features in feature_sets.items():
        predictions = cross_validated_predictions(
            features, targets, pids, args.folds, args.ridge
        )
        evaluations[name] = {
            "feature_count": int(features.shape[1]),
            **score_predictions(predictions, targets, pids),
        }

    proxy_predictions = np.asarray(
        [
            float(row["rough_reduced_moves"])
            + 4.34 * float(row["residual_three_cycles"])
            for row in rows
        ]
    )
    evaluations["fixed_rough_plus_residual_proxy"] = {
        "feature_count": 2,
        **score_predictions(proxy_predictions, targets, pids),
    }
    if "insertion_context" in feature_sets:
        # Learn only a correction to the strong fixed proxy.  This is the
        # relevant promotion test: insertion features must add held-out signal,
        # not merely relearn rough length and residual cost from 16 PIDs.
        residual_targets = targets - proxy_predictions
        correction_predictions = cross_validated_predictions(
            feature_sets["insertion_context"],
            residual_targets,
            pids,
            args.folds,
            args.ridge,
        )
        evaluations["fixed_proxy_plus_insertion_correction"] = {
            "feature_count": int(feature_sets["insertion_context"].shape[1]) + 2,
            **score_predictions(
                proxy_predictions + correction_predictions, targets, pids
            ),
        }

    state_groups: defaultdict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        state_groups[(int(row["pid"]), str(row["rough_state_digest"]))].append(row)
    ambiguous = []
    for (pid, state_digest), group in state_groups.items():
        costs = sorted(int(row["best_final_moves"]) for row in group)
        paths = {str(row["rough_path_digest"]) for row in group}
        lengths = {int(row["rough_moves"]) for row in group}
        if len(paths) > 1 and len(set(costs)) > 1:
            ambiguous.append(
                {
                    "completion_costs": costs,
                    "cost_spread": max(costs) - min(costs),
                    "path_count": len(paths),
                    "pid": pid,
                    "rough_lengths": sorted(lengths),
                    "state_digest": state_digest,
                }
            )
    ambiguous.sort(key=lambda row: int(row["pid"]))
    exact_state = evaluations["exact_state"]
    path_aware = evaluations["exact_state_plus_path"]
    fixed_proxy = evaluations["fixed_rough_plus_residual_proxy"]
    insertion_correction = evaluations.get("fixed_proxy_plus_insertion_correction")
    report = {
        "ambiguous_same_state_groups": ambiguous,
        "ambiguous_same_state_group_count": len(ambiguous),
        "dataset": str(args.dataset),
        "evaluations": evaluations,
        "folds": args.folds,
        "gate_passed": bool(
            ambiguous
            and int(path_aware["selected_regret"]) < int(exact_state["selected_regret"])
        ),
        "insertion_gate_passed": bool(
            insertion_correction
            and int(insertion_correction["selected_regret"])
            < int(fixed_proxy["selected_regret"])
        ),
        "pids": int(len(np.unique(pids))),
        "ridge": args.ridge,
        "rows": len(rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
