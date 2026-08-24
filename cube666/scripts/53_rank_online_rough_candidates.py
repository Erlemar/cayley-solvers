"""Rank fresh rough-only candidates with the frozen path-context model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]


def load_ranker_module():
    path = PROJECT / "cube666/scripts/50_train_path_context_ranker.py"
    spec = importlib.util.spec_from_file_location("path_context_ranker", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_online24_v1/"
            "rough_rows_insertion_context.json"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT / "models/cube666_path_context_ranker_v1/model.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            PROJECT / "cube666/reports/path_context_online24_ranked_v1.json"
        ),
    )
    parser.add_argument(
        "--proxy-out",
        type=Path,
        default=(
            PROJECT / "cube666/reports/path_context_online24_proxy_disagreements_v1.json"
        ),
    )
    parser.add_argument(
        "--proxy-all-out",
        type=Path,
        help="Optional completion-ready report containing the proxy choice for every PID.",
    )
    return parser.parse_args()


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    ranker = load_ranker_module()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    model = json.loads(args.model.read_text(encoding="utf-8"))
    variants = ranker.feature_variants(rows)
    name = str(model["feature_variant"])
    if name not in variants:
        raise ValueError(f"model feature variant {name!r} is unavailable")
    features = variants[name]
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if features.shape[1:] != mean.shape or mean.shape != scale.shape or scale.shape != weights.shape:
        raise ValueError("model/feature dimensions differ")
    standardized = (features - mean) / scale
    corrections = standardized @ weights
    proxy_predictions = np.asarray([ranker.proxy(row) for row in rows])
    predictions = proxy_predictions + float(model["shrink"]) * corrections
    pids = np.asarray([row["pid"] for row in rows], dtype=np.int64)

    selections = []
    proxy_disagreement_selections = []
    proxy_all_selections = []
    model_sources: Counter[str] = Counter()
    proxy_sources: Counter[str] = Counter()
    disagreements = 0
    for pid in sorted(np.unique(pids)):
        group = np.flatnonzero(pids == pid)
        selected_index = group[int(np.argmin(predictions[group]))]
        proxy_index = group[int(np.argmin(proxy_predictions[group]))]
        disagreements += int(selected_index != proxy_index)
        model_sources[str(rows[selected_index]["condition"])] += 1
        proxy_sources[str(rows[proxy_index]["condition"])] += 1
        candidate_rows = []
        for index in group:
            row = rows[int(index)]
            candidate_rows.append(
                {
                    "condition": str(row["condition"]),
                    "max_abs_standardized_feature": float(
                        np.max(np.abs(standardized[index]))
                    ),
                    "model_correction": float(
                        float(model["shrink"]) * corrections[index]
                    ),
                    "model_prediction": float(predictions[index]),
                    "proxy_prediction": float(proxy_predictions[index]),
                    "residual_three_cycles": int(row["residual_three_cycles"]),
                    "rough_moves": int(row["rough_moves"]),
                    "rough_path_digest": str(row["rough_path_digest"]),
                }
            )
        candidate_rows.sort(key=lambda row: (float(row["model_prediction"]), row["condition"]))
        selected = rows[int(selected_index)]
        proxy_selected = rows[int(proxy_index)]
        selections.append(
            {
                "candidates": candidate_rows,
                "condition": str(selected["condition"]),
                "model_prediction": float(predictions[selected_index]),
                "pid": int(pid),
                "proxy_condition": str(rows[int(proxy_index)]["condition"]),
                "proxy_prediction": float(proxy_predictions[selected_index]),
                "residual_three_cycles": int(selected["residual_three_cycles"]),
                "rough_moves": int(selected["rough_moves"]),
                "rough_path": list(selected["rough_path"]),
                "rough_path_digest": str(selected["rough_path_digest"]),
            }
        )
        proxy_all_selections.append(
            {
                "condition": str(proxy_selected["condition"]),
                "model_prediction": float(predictions[proxy_index]),
                "pid": int(pid),
                "proxy_condition": str(proxy_selected["condition"]),
                "proxy_prediction": float(proxy_predictions[proxy_index]),
                "residual_three_cycles": int(
                    proxy_selected["residual_three_cycles"]
                ),
                "rough_moves": int(proxy_selected["rough_moves"]),
                "rough_path": list(proxy_selected["rough_path"]),
                "rough_path_digest": str(proxy_selected["rough_path_digest"]),
            }
        )
        if selected_index != proxy_index:
            proxy_disagreement_selections.append(
                {
                    "condition": str(proxy_selected["condition"]),
                    "model_prediction": float(predictions[proxy_index]),
                    "pid": int(pid),
                    "proxy_condition": str(proxy_selected["condition"]),
                    "proxy_prediction": float(proxy_predictions[proxy_index]),
                    "residual_three_cycles": int(
                        proxy_selected["residual_three_cycles"]
                    ),
                    "rough_moves": int(proxy_selected["rough_moves"]),
                    "rough_path": list(proxy_selected["rough_path"]),
                    "rough_path_digest": str(proxy_selected["rough_path_digest"]),
                }
            )
    report = {
        "dataset": str(args.dataset),
        "frozen_model": str(args.model),
        "model_feature_variant": name,
        "model_proxy_disagreements": disagreements,
        "model_source_counts": dict(sorted(model_sources.items())),
        "pids": [int(pid) for pid in sorted(np.unique(pids))],
        "proxy_source_counts": dict(sorted(proxy_sources.items())),
        "selected_before_completion": True,
        "selections": selections,
    }
    atomic_write(args.out, report)
    atomic_write(
        args.proxy_out,
        {
            "dataset": str(args.dataset),
            "pids": [int(row["pid"]) for row in proxy_disagreement_selections],
            "selected_before_completion": True,
            "selection_policy": "fixed_rough_plus_residual_proxy",
            "selections": proxy_disagreement_selections,
        },
    )
    if args.proxy_all_out is not None:
        atomic_write(
            args.proxy_all_out,
            {
                "dataset": str(args.dataset),
                "pids": [int(row["pid"]) for row in proxy_all_selections],
                "selected_before_completion": True,
                "selection_policy": "fixed_rough_plus_residual_proxy",
                "selections": proxy_all_selections,
            },
        )
    print(json.dumps({key: value for key, value in report.items() if key != "selections"}, indent=2))


if __name__ == "__main__":
    main()
