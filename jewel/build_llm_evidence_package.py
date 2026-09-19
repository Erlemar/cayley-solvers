"""Build the compact evidence archive referenced by LLM_TRAINING_PLAN.md."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .ball import ExactBall
from .official import OfficialPuzzle, parse_official_state
from .pdb import EdgePatternDatabase, combined_heuristic, ring_heuristic
from .puzzle import permutation_parity


ROOT = Path(__file__).resolve().parents[1]
JEWEL = ROOT / "jewel"
RESULTS = JEWEL / "results"
PACKAGES = JEWEL / "packages"
ARCHIVE_NAME = "christophers_jewel_llm_evidence_20260811.zip"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    return {
        "min": int(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "sum": int(values.sum()),
    }


def build_lower_bounds() -> tuple[Path, Path, dict]:
    official = OfficialPuzzle.load(JEWEL / "data" / "puzzle_info.json")
    pdbs = [
        EdgePatternDatabase.load(JEWEL / "artifacts" / "pdb_edges_0_4_official.npy"),
        EdgePatternDatabase.load(JEWEL / "artifacts" / "pdb_edges_5_9_official.npy"),
    ]
    ball = ExactBall.load(JEWEL / "artifacts" / "ball_d8_official")

    with (JEWEL / "data" / "test.csv").open(encoding="utf-8", newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    with (JEWEL / "data" / "public_16490.csv").open(encoding="utf-8", newline="") as handle:
        incumbent = {row["initial_state_id"]: len(row["path"].split(".")) if row["path"] else 0 for row in csv.DictReader(handle)}

    rows: list[dict[str, int | str]] = []
    raw_bounds: list[int] = []
    parity_bounds: list[int] = []
    certified_bounds: list[int] = []
    exact_count = 0
    parity_raised = 0
    for row in test_rows:
        state = official.to_structured(parse_official_state(row["initial_state"]))
        ring = ring_heuristic(state)
        pdb_0_4 = pdbs[0].heuristic(state)
        pdb_5_9 = pdbs[1].heuristic(state)
        raw = combined_heuristic(state, pdbs)
        parity = permutation_parity(state.edge_perm)
        parity_tightened = raw + ((parity - raw) & 1)
        exact = ball.distance(state)
        certified = exact if exact is not None else parity_tightened
        if exact is not None:
            exact_count += 1
        elif parity_tightened > raw:
            parity_raised += 1
        upper = incumbent[row["initial_state_id"]]
        rows.append(
            {
                "initial_state_id": int(row["initial_state_id"]),
                "state_rank": state.rank(),
                "ring_lower_bound": ring,
                "pdb_edges_0_4_lower_bound": pdb_0_4,
                "pdb_edges_5_9_lower_bound": pdb_5_9,
                "raw_component_max_lower_bound": raw,
                "edge_permutation_parity": parity,
                "parity_tightened_lower_bound": parity_tightened,
                "exact_ball_distance": "" if exact is None else exact,
                "certified_admissible_lower_bound": certified,
                "public_solution_upper_bound": upper,
                "certified_gap": upper - certified,
            }
        )
        raw_bounds.append(raw)
        parity_bounds.append(parity_tightened)
        certified_bounds.append(certified)

    csv_path = RESULTS / "admissible_lower_bounds_1000.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    raw_arr = np.asarray(raw_bounds)
    parity_arr = np.asarray(parity_bounds)
    certified_arr = np.asarray(certified_bounds)
    summary = {
        "definition": (
            "max(ring bound, five-edge PDB 0..4, five-edge PDB 5..9), raised to "
            "the reachable solution-length parity; exact depth replaces the bound for states in the depth-8 ball"
        ),
        "admissibility": (
            "Each component is a lower bound, every generator flips edge-permutation parity, "
            "and an exact-ball distance is exact; their stated combination cannot exceed true distance."
        ),
        "count": len(rows),
        "exact_ball_states": exact_count,
        "states_raised_one_by_parity_outside_ball": parity_raised,
        "raw_component_max": _percentiles(raw_arr),
        "parity_tightened": _percentiles(parity_arr),
        "certified_admissible_lower_bound": {
            **_percentiles(certified_arr),
            "histogram": {str(int(value)): int((certified_arr == value).sum()) for value in np.unique(certified_arr)},
        },
    }
    summary_path = RESULTS / "admissible_lower_bounds_1000_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, summary_path, summary


def build_validation_metrics() -> tuple[Path, dict]:
    dataset_path = JEWEL / "artifacts" / "train_mixed_v4_public.npz"
    checkpoint_path = JEWEL / "models" / "transformer_v4_public" / "best.pt"
    data = np.load(dataset_path)
    validation = data["is_validation"].astype(bool)
    exact = data["complete_mask"].astype(bool)
    exact_validation = validation & exact
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metric = float(checkpoint["metrics"]["exact_top1"])
    exact_rows = int(exact_validation.sum())
    hits = int(round(metric * exact_rows))

    # Count unique exact-validation states without relying on Python objects.
    from .puzzle import rank_states

    ranks = rank_states(
        data["edge_perm"][exact_validation],
        data["edge_ori"][exact_validation],
        data["ring_ori"][exact_validation],
    )
    unique_ranks, multiplicities = np.unique(ranks, return_counts=True)
    distances = data["distance"][exact_validation]
    report = {
        "reported_exact_top1": metric,
        "reported_exact_top1_percent": 100.0 * metric,
        "numerator_top1_descending_hits": hits,
        "denominator_exact_validation_rows": exact_rows,
        "validation_rows_all_sources": int(validation.sum()),
        "validation_rows_nonexact": int((validation & ~exact).sum()),
        "unique_exact_validation_states": int(len(unique_ranks)),
        "duplicate_exact_validation_rows": int(exact_rows - len(unique_ranks)),
        "maximum_state_multiplicity": int(multiplicities.max()),
        "depth_histogram_rows": {
            str(int(depth)): int((distances == depth).sum()) for depth in np.unique(distances)
        },
        "metric_definition": (
            "Argmax of the 12 policy logits is counted correct when it belongs to the complete set "
            "of exact distance-descending actions. The result is row-weighted."
        ),
        "split_definition": (
            "Approximately 95/5 validation assignment is a stable function of the 51-bit state rank, "
            "so duplicate states cannot cross between training and validation."
        ),
        "important_limitation": (
            "Exact states were sampled with replacement. This is not unique-state-weighted accuracy, "
            "not accuracy on the 1,000 competition states, and not an end-to-end solve rate."
        ),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_step": int(checkpoint["step"]),
        "other_checkpoint_metrics": {
            key: value
            for key, value in checkpoint["metrics"].items()
            if key.startswith("exact_") and key != "exact_top1"
        },
        "artifact_sha256": {
            "train_mixed_v4_public.npz": _sha256(dataset_path),
            "best.pt": _sha256(checkpoint_path),
        },
    }
    path = RESULTS / "model_validation_metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path, report


def build_solver_metrics() -> tuple[Path, dict]:
    source = RESULTS / "competition_v4_public_full.json"
    result = json.loads(source.read_text(encoding="utf-8"))
    expanded = np.asarray([row["expanded"] for row in result["rows"]], dtype=np.int64)
    seconds = np.asarray([row["seconds"] for row in result["rows"]], dtype=np.float64)
    widths = Counter(int(row["width"]) for row in result["rows"])
    report = {
        "candidate_score_moves": int(result["candidate_score"]),
        "competition_states": int(result["count"]),
        "solved_and_officially_verified": int(result["officially_verified"]),
        "hardware": "local NVIDIA RTX 4090 Laptop GPU (16 GB)",
        "search_budget": {
            "adaptive_widths": [1, 4, 16, 64],
            "max_learned_layers_per_width": 18,
            "branch_actions": {"width_1": 1, "width_4_16_64": 4},
            "try_all_widths": True,
            "raw_any_solution": True,
            "incumbent_length_bound": None,
            "explicit_node_cap": None,
            "structural_max_expanded_parent_states_per_case": 1449,
            "endgame": "exact depth-8 ball",
            "ranking": "policy log-probability + predicted regret + admissible PDB/ring lower bound",
            "deduplication": "exact 51-bit state rank",
            "immediate_inverse_moves": "forbidden",
        },
        "expanded_parent_states": _percentiles(expanded),
        "zero_neural_expansion_states": int((expanded == 0).sum()),
        "total_seconds": float(seconds.sum()),
        "mean_seconds_per_state": float(seconds.mean()),
        "end_to_end_expanded_parent_states_per_second": float(expanded.sum() / seconds.sum()),
        "reported_best_width_counts": {str(width): count for width, count in sorted(widths.items())},
        "node_definition": (
            "One expanded parent state is one frontier state presented to the neural scorer. "
            "It is not a generated child edge. Generated-edge totals were not stored in this run."
        ),
        "score_definition": "sum of the 1,000 independently replay-verified candidate path lengths",
        "source_result_sha256": _sha256(source),
    }
    path = RESULTS / "compact_solver_benchmark.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path, report


def _write_archive(entries: dict[str, Path], archive: Path) -> None:
    payloads = {name: path.read_bytes() for name, path in entries.items()}
    manifest_lines = [f"{hashlib.sha256(payload).hexdigest()}  {name}" for name, payload in sorted(payloads.items())]
    payloads["MANIFEST.sha256"] = ("\n".join(manifest_lines) + "\n").encode("utf-8")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for name, payload in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, payload)


def main() -> None:
    lower_csv, lower_summary, lower_report = build_lower_bounds()
    validation_path, validation_report = build_validation_metrics()
    solver_path, solver_report = build_solver_metrics()
    archive = PACKAGES / ARCHIVE_NAME
    entries = {
        "README.md": JEWEL / "LLM_EVIDENCE_PACKAGE.md",
        "PROJECT_README.md": JEWEL / "README.md",
        "source/jewel/__init__.py": JEWEL / "__init__.py",
        "source/jewel/official.py": JEWEL / "official.py",
        "source/jewel/puzzle.py": JEWEL / "puzzle.py",
        "source/jewel/pdb.py": JEWEL / "pdb.py",
        "source/jewel/ball.py": JEWEL / "ball.py",
        "data/puzzle_info.json": JEWEL / "data" / "puzzle_info.json",
        "data/test.csv": JEWEL / "data" / "test.csv",
        "data/public_16490.csv": JEWEL / "data" / "public_16490.csv",
        "artifacts/ball_d8_official_meta.json": JEWEL / "artifacts" / "ball_d8_official" / "meta.json",
        "artifacts/pdb_edges_0_4_official.npy": JEWEL / "artifacts" / "pdb_edges_0_4_official.npy",
        "artifacts/pdb_edges_0_4_official.json": JEWEL / "artifacts" / "pdb_edges_0_4_official.json",
        "artifacts/pdb_edges_5_9_official.npy": JEWEL / "artifacts" / "pdb_edges_5_9_official.npy",
        "artifacts/pdb_edges_5_9_official.json": JEWEL / "artifacts" / "pdb_edges_5_9_official.json",
        "metrics/admissible_lower_bounds_1000.csv": lower_csv,
        "metrics/admissible_lower_bounds_1000_summary.json": lower_summary,
        "metrics/model_validation_metrics.json": validation_path,
        "metrics/compact_solver_benchmark.json": solver_path,
        "evidence/competition_v4_public_full.json": RESULTS / "competition_v4_public_full.json",
        "evidence/transformer_v4_public_history.json": JEWEL / "models" / "transformer_v4_public" / "history.json",
        "evidence/train_mixed_v4_public_metadata.json": JEWEL / "artifacts" / "train_mixed_v4_public.json",
    }
    _write_archive(entries, archive)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": _sha256(archive),
                "entry_count": len(entries) + 1,
                "lower_bounds": lower_report,
                "validation_top1": {
                    "hits": validation_report["numerator_top1_descending_hits"],
                    "rows": validation_report["denominator_exact_validation_rows"],
                    "unique_states": validation_report["unique_exact_validation_states"],
                },
                "solver": {
                    "score": solver_report["candidate_score_moves"],
                    "expanded": solver_report["expanded_parent_states"]["sum"],
                    "expanded_per_second": solver_report["end_to_end_expanded_parent_states_per_second"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
