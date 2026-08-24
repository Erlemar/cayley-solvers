"""Combine comparable macro-frontier KMC labels into a grouped dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-file",
        type=Path,
        required=True,
        help="JSON list of objects with frontier_dir and query_dir",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = json.loads(args.pairs_file.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs-file must contain a non-empty JSON list")
    states: list[np.ndarray] = []
    pids: list[int] = []
    action_costs: list[int] = []
    targets: list[float] = []
    rows_out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for pair_index, pair in enumerate(pairs):
        frontier_dir = Path(pair["frontier_dir"])
        query_dir = Path(pair["query_dir"])
        frontier = json.loads(
            (frontier_dir / "report.json").read_text(encoding="utf-8")
        )
        query_report = json.loads(
            (query_dir / "report.json").read_text(encoding="utf-8")
        )
        expected = {
            "beam": 1000,
            "anneal_steps": 1_000_000,
            "alpha": 3.0,
            "anneal_seed": 89043280,
        }
        for key, value in expected.items():
            if query_report.get(key) != value:
                raise ValueError(
                    f"{query_dir}: {key}={query_report.get(key)!r}, expected {value!r}"
                )
        with np.load(frontier_dir / "frontiers.npz", allow_pickle=False) as payload:
            cluster_states = payload["cluster_states"].astype(np.uint8, copy=False)
        frontier_rows = frontier["rows"]
        if len(cluster_states) != len(frontier_rows):
            raise ValueError(f"{frontier_dir}: state/report row mismatch")
        for index, row in enumerate(frontier_rows):
            query_id = str(row["query_id"])
            metadata_file = query_dir / "metadata" / f"{query_id}.json"
            if not metadata_file.exists():
                continue
            key = (str(frontier_dir.resolve()), query_id)
            if key in seen:
                raise ValueError(f"duplicate frontier/query row {key}")
            seen.add(key)
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if not metadata.get("replay_verified"):
                raise ValueError(f"{query_id}: KMC label is not replay verified")
            states.append(cluster_states[index])
            pids.append(int(row["pid"]))
            action_costs.append(int(row["macro_primitive_moves"]))
            targets.append(float(metadata["moves"]))
            rows_out.append(
                {
                    "action_cost": int(row["macro_primitive_moves"]),
                    "action_id": int(row["action_id"]),
                    "frontier_dir": str(frontier_dir),
                    "pair_index": pair_index,
                    "pid": int(row["pid"]),
                    "query_dir": str(query_dir),
                    "query_id": query_id,
                    "target_completion_moves": int(metadata["moves"]),
                }
            )
    if not states:
        raise RuntimeError("no comparable completion labels were found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "completion.npz",
        states=np.stack(states).astype(np.uint8, copy=False),
        pids=np.asarray(pids, dtype=np.int32),
        action_costs=np.asarray(action_costs, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.float32),
    )
    (args.out_dir / "rows.json").write_text(
        json.dumps(rows_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    unique_pids, counts = np.unique(np.asarray(pids), return_counts=True)
    report = {
        "holdout_mod5_pids": [int(pid) for pid in unique_pids if pid % 5 == 0],
        "label_maximum": max(targets),
        "label_mean": float(np.mean(targets)),
        "label_minimum": min(targets),
        "pairs": pairs,
        "pid_counts": {
            str(int(pid)): int(count) for pid, count in zip(unique_pids, counts, strict=True)
        },
        "pids": [int(pid) for pid in unique_pids],
        "rows": len(states),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
