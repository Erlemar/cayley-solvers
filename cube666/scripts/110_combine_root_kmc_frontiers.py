"""Combine per-PID root rollout frontiers into one resumable query artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-dirs", help="comma-separated directories")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--pids", help="comma-separated PIDs under --root")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frontier_dirs:
        directories = [
            Path(token.strip())
            for token in args.frontier_dirs.split(",")
            if token.strip()
        ]
    elif args.root is not None and args.pids:
        pid_tokens = args.pids.replace("_", ",").split(",")
        directories = [
            args.root / f"macro_kmc_root_pid{int(token):03d}_v1"
            for token in pid_tokens
            if token.strip()
        ]
    else:
        directories = []
    if not directories:
        raise ValueError("frontier-dirs cannot be empty")
    arrays: dict[str, list[np.ndarray]] = {
        "full_states": [],
        "cluster_states": [],
        "source_pids": [],
        "initial_residuals": [],
        "frontier_residuals": [],
    }
    rows: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    seen_queries: set[str] = set()
    seen_pids: set[int] = set()
    for directory in directories:
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        pid = int(report["pid"])
        if pid in seen_pids:
            raise ValueError(f"duplicate PID {pid}")
        seen_pids.add(pid)
        with np.load(directory / "frontiers.npz", allow_pickle=False) as payload:
            for name in arrays:
                arrays[name].append(payload[name])
        case_rows = list(report["rows"])
        if len(case_rows) != len(arrays["source_pids"][-1]):
            raise ValueError(f"{directory}: report/array row mismatch")
        start = len(rows)
        for row in case_rows:
            query_id = str(row["query_id"])
            if query_id in seen_queries:
                raise ValueError(f"duplicate query ID {query_id}")
            seen_queries.add(query_id)
            rows.append(
                {
                    **row,
                    "setup_moves": int(report["setup_moves"]),
                    "setup_path": list(report["setup_path"]),
                }
            )
        cases.append(
            {
                "initial_residual": int(report["initial_residual"]),
                "pid": pid,
                "row_start": start,
                "row_stop": len(rows),
                "setup_moves": int(report["setup_moves"]),
                "setup_path": list(report["setup_path"]),
                "source_dir": str(directory),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "frontiers.npz",
        **{name: np.concatenate(parts, axis=0) for name, parts in arrays.items()},
    )
    combined = {
        "cases": cases,
        "frontiers": len(rows),
        "pids": sorted(seen_pids),
        "rows": rows,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in combined.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
