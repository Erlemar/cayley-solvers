"""Combine PID-disjoint uniformly labeled path-context corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--target-field", default="uniform_final_moves")
    parser.add_argument(
        "--out",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_uniform72_v3/rough_rows_uniform_b20000.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "cube666/reports/path_context_uniform72_v3.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def condition_label(row: dict[str, object]) -> str:
    if row.get("condition") is not None:
        return str(row["condition"])
    conditions = [str(value) for value in row.get("conditions", ["unknown"])]
    if "control" in conditions:
        return "control"
    return conditions[0]


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    seen_pids: set[int] = set()
    sources = []
    for dataset in args.dataset:
        rows = json.loads(dataset.read_text(encoding="utf-8"))
        pids = {int(row["pid"]) for row in rows}
        overlap = seen_pids & pids
        if overlap:
            raise ValueError(
                f"PID leakage between corpora at {dataset}: {sorted(overlap)}"
            )
        missing = [
            index for index, row in enumerate(rows) if args.target_field not in row
        ]
        if missing:
            raise KeyError(
                f"{dataset}: target {args.target_field!r} missing from "
                f"{len(missing)} rows"
            )
        seen_pids.update(pids)
        all_rows.extend(rows)
        sources.append(
            {
                "path": str(dataset),
                "pids": len(pids),
                "rows": len(rows),
                "sha256": sha256_file(dataset),
            }
        )

    keys = [(int(row["pid"]), str(row["rough_path_digest"])) for row in all_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate PID/rough-path key across combined corpora")
    rows_per_pid = Counter(int(row["pid"]) for row in all_rows)
    if len(set(rows_per_pid.values())) != 1:
        raise ValueError(f"unequal candidate counts per PID: {rows_per_pid}")
    all_rows.sort(
        key=lambda row: (
            int(row["pid"]),
            condition_label(row),
            str(row["rough_path_digest"]),
        )
    )
    atomic_write(args.out, all_rows)
    targets = [int(row[args.target_field]) for row in all_rows]
    report = {
        "candidates_per_pid": next(iter(rows_per_pid.values())),
        "condition_counts": dict(
            sorted(Counter(condition_label(row) for row in all_rows).items())
        ),
        "out": str(args.out),
        "pids": len(rows_per_pid),
        "rows": len(all_rows),
        "sources": sources,
        "target_field": args.target_field,
        "target_max": max(targets),
        "target_mean": statistics.fmean(targets),
        "target_min": min(targets),
        "target_sum": sum(targets),
    }
    atomic_write(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
