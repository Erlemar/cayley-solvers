"""Train a strongly regularized correction to the rough/residual proxy.

Hyperparameters are selected only by PID-grouped cross-validation on the
corpus.  The separate 16-PID gate is evaluated once after selection and never
participates in fitting or model selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_corpus32_v1/"
            "unique_rough_rows_insertion_context.json"
        ),
    )
    parser.add_argument(
        "--test",
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
        default=PROJECT / "cube666/reports/path_context_ranker_v1.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT / "models/cube666_path_context_ranker_v1/model.json",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--target-field",
        default="best_final_moves",
        help="Row field containing the comparable final-path label.",
    )
    parser.add_argument(
        "--disallow-config-features",
        action="store_true",
        help=(
            "Exclude feature variants containing annealer configuration identity. "
            "Use this for state-quality models intended to generalize to new generators."
        ),
    )
    parser.add_argument(
        "--skip-heldout-gate",
        action="store_true",
        help="Fit from PID-grouped CV without evaluating --test.",
    )
    return parser.parse_args()


def proxy(row: dict[str, object]) -> float:
    return float(row["rough_reduced_moves"]) + 4.34 * float(
        row["residual_three_cycles"]
    )


def entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return float(-(positive * np.log(positive)).sum())


def tiny_frontier_features(row: dict[str, object]) -> np.ndarray:
    raw = row["insertion_context"]
    added = np.asarray(raw["added_lengths"], dtype=np.float64)
    returned = max(len(added), 1)

    def prefix_mean(width: int) -> float:
        values = added[:width]
        return float(values.mean()) if len(values) else 32.0

    def prefix_std(width: int) -> float:
        values = added[:width]
        return float(values.std()) if len(values) else 0.0

    cluster_best = np.asarray(
        [32.0 if value is None else float(value) for value in raw["cluster_best_added_lengths"]]
    )
    cluster_distribution = (
        np.asarray(raw["cluster_descriptor_counts"], dtype=np.float64) / returned
    )
    position_distribution = (
        np.asarray(raw["position_descriptor_counts"], dtype=np.float64) / returned
    )
    return np.asarray(
        [
            float(added[0]) if len(added) else 32.0,
            prefix_mean(8),
            prefix_mean(32),
            prefix_mean(128),
            prefix_mean(512),
            prefix_std(32),
            prefix_std(512),
            float(np.mean(added <= 2)) if len(added) else 0.0,
            float(np.mean(added <= 4)) if len(added) else 0.0,
            float(cluster_best.min()),
            float(cluster_best.mean()),
            float(cluster_best.std()),
            entropy(cluster_distribution),
            entropy(position_distribution),
            float(raw["distinct_desired_cycles"]) / returned,
        ],
        dtype=np.float64,
    )


def compact_frontier_features(row: dict[str, object]) -> np.ndarray:
    raw = row["insertion_context"]
    added = np.asarray(raw["added_lengths"], dtype=np.float64)
    returned = max(len(added), 1)
    features = list(tiny_frontier_features(row))
    for rank in (1, 3, 7, 15, 31, 63, 127, 255, 511):
        features.append(
            float(added[min(rank, len(added) - 1)]) if len(added) else 32.0
        )
    for threshold in (0, 2, 4, 6, 8, 10):
        features.append(float(np.mean(added <= threshold)) if len(added) else 0.0)
    features.extend(
        float(value) / returned for value in raw["cluster_descriptor_counts"]
    )
    features.extend(
        float(value) / returned for value in raw["position_descriptor_counts"]
    )
    return np.asarray(features, dtype=np.float64)


def path_summary_features(row: dict[str, object]) -> np.ndarray:
    raw = row["path_features"]
    length = max(int(row["rough_moves"]), 1)
    absolute = np.asarray(
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
    return np.concatenate((absolute, absolute / length))


def configuration_features(row: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [
            float(row["alpha"]),
            math.log10(float(row["anneal_steps"])),
            float(int(row["anneal_seed"]) == 50002),
            float(int(row["anneal_seed"]) == 50003),
            float(bool(row["anneal_keep_best"])),
        ],
        dtype=np.float64,
    )


def cluster_structure_features(row: dict[str, object]) -> np.ndarray:
    clusters = list(row["clusters"])
    three_cycles = np.asarray(
        [cluster["three_cycles"] for cluster in clusters], dtype=np.float64
    )
    misplaced = np.asarray(
        [cluster["misplaced"] for cluster in clusters], dtype=np.float64
    )
    cycle_counts = np.asarray(
        [len(cluster["cycles"]) for cluster in clusters], dtype=np.float64
    )

    def summary(values: np.ndarray) -> list[float]:
        return [
            float(values.min()),
            float(values.mean()),
            float(np.median(values)),
            float(values.std()),
            float(values.max()),
            float(values.sum()),
            float(np.mean(values > 0)),
        ]

    scalars = [
        float(row["rough_moves"]),
        float(row["rough_reduced_moves"]),
        float(row["residual_three_cycles"]),
        float(row["mismatches"]),
        float(len(clusters)),
        *summary(three_cycles),
        *summary(misplaced),
        *summary(cycle_counts),
    ]
    return np.asarray(
        [*scalars, *three_cycles, *misplaced, *cycle_counts], dtype=np.float64
    )


def feature_variants(rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    tiny = np.stack([tiny_frontier_features(row) for row in rows])
    compact = np.stack([compact_frontier_features(row) for row in rows])
    paths = np.stack([path_summary_features(row) for row in rows])
    configs = np.stack([configuration_features(row) for row in rows])
    clusters = np.stack([cluster_structure_features(row) for row in rows])
    return {
        "tiny_frontier": tiny,
        "tiny_frontier_plus_cluster_structure": np.concatenate(
            (tiny, clusters), axis=1
        ),
        "tiny_frontier_plus_config": np.concatenate((tiny, configs), axis=1),
        "compact_frontier": compact,
        "compact_frontier_plus_cluster_structure": np.concatenate(
            (compact, clusters), axis=1
        ),
        "compact_frontier_plus_config": np.concatenate((compact, configs), axis=1),
        "compact_frontier_plus_path_summary": np.concatenate(
            (compact, paths), axis=1
        ),
        "compact_frontier_plus_cluster_structure_plus_path_summary": np.concatenate(
            (compact, clusters, paths), axis=1
        ),
        "compact_frontier_plus_path_summary_plus_config": np.concatenate(
            (compact, paths, configs), axis=1
        ),
    }


def pair_rows(
    features: np.ndarray,
    targets: np.ndarray,
    pids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    differences = []
    target_differences = []
    for pid in np.unique(pids):
        group = np.flatnonzero(pids == pid)
        for left, right in itertools.combinations(group, 2):
            if targets[left] == targets[right]:
                continue
            differences.append(features[left] - features[right])
            target_differences.append(targets[left] - targets[right])
    return np.asarray(differences), np.asarray(target_differences)


def fit_pairwise_ridge(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_pids: np.ndarray,
    predict_features: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized_train = (train_features - mean) / scale
    standardized_predict = (predict_features - mean) / scale
    differences, target_differences = pair_rows(
        standardized_train, train_targets, train_pids
    )
    gram = differences @ differences.T
    dual = np.linalg.solve(
        gram + ridge * np.eye(len(gram), dtype=np.float64), target_differences
    )
    weights = differences.T @ dual
    return standardized_predict @ weights, mean, scale, weights


def cross_validated_corrections(
    features: np.ndarray,
    residual_targets: np.ndarray,
    pids: np.ndarray,
    folds: int,
    ridge: float,
) -> np.ndarray:
    unique_pids = np.asarray(sorted(np.unique(pids)))
    predictions = np.full(len(pids), np.nan, dtype=np.float64)
    for fold in range(folds):
        validation_pids = unique_pids[fold::folds]
        validation = np.flatnonzero(np.isin(pids, validation_pids))
        train = np.flatnonzero(~np.isin(pids, validation_pids))
        fold_predictions, _, _, _ = fit_pairwise_ridge(
            features[train],
            residual_targets[train],
            pids[train],
            features[validation],
            ridge,
        )
        predictions[validation] = fold_predictions
    if np.isnan(predictions).any():
        raise AssertionError("cross-validation did not predict every row")
    return predictions


def score_predictions(
    predictions: np.ndarray, targets: np.ndarray, pids: np.ndarray
) -> dict[str, float | int]:
    selected_total = 0
    oracle_total = 0
    winner_hits = 0
    pair_correct = 0
    pair_total = 0
    for pid in np.unique(pids):
        group = np.flatnonzero(pids == pid)
        selected = group[int(np.argmin(predictions[group]))]
        oracle = int(targets[group].min())
        selected_total += int(targets[selected])
        oracle_total += oracle
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
        "selected_regret": selected_total - oracle_total,
        "selected_total": selected_total,
        "winner_hits": winner_hits,
        "winner_hit_fraction": winner_hits / len(np.unique(pids)),
    }


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    train_rows = json.loads(args.train.read_text(encoding="utf-8"))
    test_rows = (
        []
        if args.skip_heldout_gate
        else json.loads(args.test.read_text(encoding="utf-8"))
    )
    train_features = feature_variants(train_rows)
    test_features = feature_variants(test_rows) if test_rows else {}
    if args.disallow_config_features:
        train_features = {
            name: values
            for name, values in train_features.items()
            if "config" not in name
        }
        if test_features:
            test_features = {
                name: values
                for name, values in test_features.items()
                if "config" not in name
            }
    if not train_features:
        raise ValueError("no feature variants remain after filtering")
    missing_train = [
        index for index, row in enumerate(train_rows) if args.target_field not in row
    ]
    missing_test = [
        index for index, row in enumerate(test_rows) if args.target_field not in row
    ]
    if missing_train or missing_test:
        raise KeyError(
            f"target field {args.target_field!r} missing from "
            f"{len(missing_train)} train and {len(missing_test)} test rows"
        )
    train_targets = np.asarray(
        [row[args.target_field] for row in train_rows], dtype=np.float64
    )
    test_targets = (
        np.asarray([row[args.target_field] for row in test_rows], dtype=np.float64)
        if test_rows
        else None
    )
    train_pids = np.asarray([row["pid"] for row in train_rows], dtype=np.int64)
    test_pids = (
        np.asarray([row["pid"] for row in test_rows], dtype=np.int64)
        if test_rows
        else None
    )
    train_proxy = np.asarray([proxy(row) for row in train_rows])
    test_proxy = (
        np.asarray([proxy(row) for row in test_rows]) if test_rows else None
    )
    residual_targets = train_targets - train_proxy

    ridge_grid = (0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0)
    shrink_grid = (0.125, 0.25, 0.5, 0.75, 1.0)
    candidates = []
    for name, features in train_features.items():
        for ridge in ridge_grid:
            corrections = cross_validated_corrections(
                features, residual_targets, train_pids, args.folds, ridge
            )
            for shrink in shrink_grid:
                metrics = score_predictions(
                    train_proxy + shrink * corrections, train_targets, train_pids
                )
                candidates.append(
                    {
                        "feature_count": int(features.shape[1]),
                        "name": name,
                        "ridge": ridge,
                        "shrink": shrink,
                        **metrics,
                    }
                )
    candidates.sort(
        key=lambda row: (
            int(row["selected_regret"]),
            -float(row["pairwise_accuracy"]),
            int(row["feature_count"]),
            float(row["ridge"]),
            float(row["shrink"]),
        )
    )
    selected = candidates[0]
    selected_name = str(selected["name"])
    prediction_features = (
        test_features[selected_name]
        if test_rows
        else train_features[selected_name]
    )
    test_corrections, mean, scale, weights = fit_pairwise_ridge(
        train_features[selected_name],
        residual_targets,
        train_pids,
        prediction_features,
        float(selected["ridge"]),
    )
    train_proxy_metrics = score_predictions(train_proxy, train_targets, train_pids)
    if test_rows:
        assert test_proxy is not None and test_targets is not None and test_pids is not None
        test_predictions = test_proxy + float(selected["shrink"]) * test_corrections
        test_proxy_metrics = score_predictions(test_proxy, test_targets, test_pids)
        test_model_metrics = score_predictions(test_predictions, test_targets, test_pids)
        heldout_gate = {
            "dataset": str(args.test),
            "model": test_model_metrics,
            "model_improvement_in_regret": int(test_proxy_metrics["selected_regret"])
            - int(test_model_metrics["selected_regret"]),
            "proxy": test_proxy_metrics,
        }
        promotion_passed: bool | None = bool(
            int(test_model_metrics["selected_regret"])
            < int(test_proxy_metrics["selected_regret"])
            and float(test_model_metrics["pairwise_accuracy"])
            >= float(test_proxy_metrics["pairwise_accuracy"])
        )
    else:
        heldout_gate = None
        promotion_passed = None

    model = {
        "feature_variant": selected_name,
        "format_version": 1,
        "mean": mean.tolist(),
        "proxy_residual_coefficient": 4.34,
        "ridge": selected["ridge"],
        "scale": scale.tolist(),
        "shrink": selected["shrink"],
        "target_field": args.target_field,
        "training_dataset": str(args.train),
        "uses_configuration_features": "config" in selected_name,
        "weights": weights.tolist(),
    }
    atomic_write(args.model, model)
    report = {
        "candidate_count": len(candidates),
        "configuration_features_allowed": not args.disallow_config_features,
        "folds": args.folds,
        "heldout_gate": heldout_gate,
        "model": str(args.model),
        "promotion_passed": promotion_passed,
        "selected_by_train_cv": selected,
        "target_field": args.target_field,
        "top_train_cv_candidates": candidates[:20],
        "train": {
            "dataset": str(args.train),
            "proxy": train_proxy_metrics,
        },
    }
    atomic_write(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
