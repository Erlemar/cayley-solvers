"""Materialize replay-verified out-of-fold paths selected by the ranker."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


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
            / "cube666/training/path_context_corpus32_v1/"
            "unique_rough_rows_insertion_context.json"
        ),
    )
    parser.add_argument(
        "--ranker-report",
        type=Path,
        default=PROJECT / "cube666/reports/path_context_ranker_v1.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "cube666/reports/PATH_CONTEXT_CORPUS32_MANIFEST.json",
    )
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=PROJECT / "submissions/cube666_model_hybrid_strictwin_v1.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "submissions/path_context_ranker_oof_candidates_v1.csv",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=PROJECT / "cube666/reports/path_context_ranker_oof_candidates_v1.json",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--folds", type=int, default=4)
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def load_submission(path: Path) -> dict[int, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): parse_path(row["path"])
            for row in csv.DictReader(handle)
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
    ranker = load_ranker_module()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = json.loads(args.ranker_report.read_text(encoding="utf-8"))
    selected_model = report["selected_by_train_cv"]
    features = ranker.feature_variants(rows)[selected_model["name"]]
    targets = np.asarray([row["best_final_moves"] for row in rows], dtype=np.float64)
    pids = np.asarray([row["pid"] for row in rows], dtype=np.int64)
    proxy_predictions = np.asarray([ranker.proxy(row) for row in rows])
    corrections = ranker.cross_validated_corrections(
        features,
        targets - proxy_predictions,
        pids,
        args.folds,
        float(selected_model["ridge"]),
    )
    predictions = proxy_predictions + float(selected_model["shrink"]) * corrections

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    directories = {
        str(condition["label"]): Path(str(condition["directory"]))
        for condition in manifest["conditions"]
    }
    incumbent = load_submission(args.incumbent)
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    states = {
        int(pid): state for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }

    selections = []
    selected_paths: dict[int, tuple[str, ...]] = {}
    source_counts: Counter[str] = Counter()
    for pid in sorted(np.unique(pids)):
        group = np.flatnonzero(pids == pid)
        selected_index = group[int(np.argmin(predictions[group]))]
        row = rows[int(selected_index)]
        labels = [int(value) for value in row["final_labels"]]
        best_label_index = int(np.argmin(labels))
        condition = str(row["conditions"][best_label_index])
        path_file = directories[condition] / f"{int(pid):04d}.path.txt"
        path = reduce_commuting_quarter_turn_path(
            parse_path(path_file.read_text(encoding="utf-8"))
        )
        if len(path) != int(row["best_final_moves"]):
            raise AssertionError(
                f"PID {pid}: materialized length {len(path)} != label "
                f"{row['best_final_moves']}"
            )
        if puzzle.apply_path(states[int(pid)], path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: selected path failed exact replay")
        incumbent_length = len(incumbent[int(pid)])
        selected_paths[int(pid)] = path
        source_counts[condition] += 1
        selections.append(
            {
                "condition": condition,
                "incumbent_moves": incumbent_length,
                "model_prediction": float(predictions[selected_index]),
                "pid": int(pid),
                "proxy_prediction": float(proxy_predictions[selected_index]),
                "selected_moves": len(path),
                "strict_saving": max(incumbent_length - len(path), 0),
                "would_regress": len(path) > incumbent_length,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        for pid, path in sorted(selected_paths.items()):
            writer.writerow({"initial_state_id": pid, "path": ".".join(path)})
    temporary.replace(args.output)

    candidate_total = sum(int(row["selected_moves"]) for row in selections)
    incumbent_total = sum(int(row["incumbent_moves"]) for row in selections)
    output_report = {
        "candidate_csv": str(args.output),
        "incumbent": str(args.incumbent),
        "incumbent_total_on_selected_pids": incumbent_total,
        "model_selected_improvement_before_strict_merge": incumbent_total
        - candidate_total,
        "model_selected_total": candidate_total,
        "pids": len(selections),
        "replay_verified_selected_paths": len(selections),
        "selections": selections,
        "source_counts": dict(sorted(source_counts.items())),
        "strict_merge_savings": sum(int(row["strict_saving"]) for row in selections),
        "strict_wins": sum(int(row["strict_saving"]) > 0 for row in selections),
        "would_regress_count": sum(bool(row["would_regress"]) for row in selections),
    }
    atomic_write(args.out_report, output_report)
    print(json.dumps({k: v for k, v in output_report.items() if k != "selections"}, indent=2))


if __name__ == "__main__":
    main()
