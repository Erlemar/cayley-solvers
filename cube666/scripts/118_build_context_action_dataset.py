"""Build independent path-context action labels from corrected KMC campaigns.

Each campaign is a combined root frontier plus a candidate-specific KMC query
directory.  Rows are grouped against the matched no-op from the same PID and
solver configuration.  The resulting artifact is suitable for a joint
state/action/path ranker; it deliberately retains source labels so learned
proposals can be compared with random controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        help="FRONTIER_DIR=QUERY_DIR; repeat for independently keyed campaigns",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--wide-query-dir",
        action="append",
        type=Path,
        default=[],
        help="optional query directory whose matching paths supply wide labels",
    )
    parser.add_argument("--wide-query-root", type=Path)
    parser.add_argument(
        "--wide-query-pattern", default="*context_kmc20k*n40m*ctxkey_v2"
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def parse_campaign(specification: str) -> tuple[Path, Path]:
    if "=" not in specification:
        raise ValueError("campaign must be FRONTIER_DIR=QUERY_DIR")
    frontier, query = specification.split("=", 1)
    return Path(frontier), Path(query)


def read_path(path: Path) -> list[str]:
    return [token for token in path.read_text(encoding="utf-8").strip().split(".") if token]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    wide_lengths: dict[str, int] = {}
    wide_dirs = list(args.wide_query_dir)
    if args.wide_query_root is not None:
        wide_dirs.extend(
            path
            for path in args.wide_query_root.glob(args.wide_query_pattern)
            if path.is_dir()
        )
    for wide_dir in sorted(set(wide_dirs)):
        for path_file in (wide_dir / "paths").glob("*.path.txt"):
            query_id = path_file.name.removesuffix(".path.txt")
            moves = len(read_path(path_file))
            previous = wide_lengths.get(query_id)
            wide_lengths[query_id] = moves if previous is None else min(previous, moves)
    output_rows: list[dict[str, object]] = []
    campaign_summaries: list[dict[str, object]] = []
    for campaign_index, specification in enumerate(args.campaign):
        frontier_dir, query_dir = parse_campaign(specification)
        frontier_report = json.loads(
            (frontier_dir / "report.json").read_text(encoding="utf-8")
        )
        frontier_rows = list(frontier_report["rows"])
        with np.load(frontier_dir / "frontiers.npz", allow_pickle=False) as payload:
            cluster_states = payload["cluster_states"].astype(np.uint8, copy=False)
        if cluster_states.shape[:2] != (len(frontier_rows), 6):
            raise ValueError(f"{frontier_dir}: inconsistent cluster-state artifact")

        present: list[tuple[int, dict[str, object], list[str], dict[str, object]]] = []
        missing: list[str] = []
        for index, row in enumerate(frontier_rows):
            query_id = str(row["query_id"])
            path_file = query_dir / "paths" / f"{query_id}.path.txt"
            metadata_file = query_dir / "metadata" / f"{query_id}.json"
            if not path_file.exists() or not metadata_file.exists():
                missing.append(query_id)
                continue
            path = read_path(path_file)
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if int(metadata["moves"]) != len(path):
                raise ValueError(f"{query_id}: path/metadata length mismatch")
            if not bool(metadata.get("replay_verified")):
                raise ValueError(f"{query_id}: query is not replay verified")
            present.append((index, row, path, metadata))
        if missing and not args.allow_partial:
            raise FileNotFoundError(
                f"{query_dir}: missing {len(missing)} of {len(frontier_rows)} rows"
            )

        noops: dict[int, tuple[int, dict[str, object], list[str], dict[str, object]]] = {}
        for item in present:
            _, row, _, _ = item
            if int(row["action_id"]) == -1:
                pid = int(row["pid"])
                if pid in noops:
                    raise ValueError(f"{query_dir}: duplicate no-op for PID {pid}")
                noops[pid] = item
        completed_pids = sorted(
            pid
            for pid in {int(row["pid"]) for _, row, _, _ in present}
            if pid in noops
        )
        kept = 0
        for index, row, path, metadata in present:
            pid = int(row["pid"])
            if pid not in noops:
                if args.allow_partial:
                    continue
                raise ValueError(f"{query_dir}: PID {pid} has no matched no-op")
            noop_index, _, noop_path, _ = noops[pid]
            setup_path = [str(move) for move in row.get("setup_path", [])]
            macro_path = [str(move) for move in row["macro_path"]]
            completion_moves = len(path)
            noop_moves = len(noop_path)
            output_rows.append(
                {
                    "action_id": int(row["action_id"]),
                    "campaign": campaign_index,
                    "cluster_permutations": cluster_states[index].tolist(),
                    "completion_moves": completion_moves,
                    "context_digest": str(metadata.get("context_digest", "")),
                    "context_path": setup_path + macro_path,
                    "frame": int(metadata["symmetry_index"]),
                    "index": index,
                    "macro_path": macro_path,
                    "macro_primitive_moves": int(row["macro_primitive_moves"]),
                    "noop_completion_moves": noop_moves,
                    "pid": pid,
                    "predicted_child_value": float(row["predicted_child_value"]),
                    "query_id": str(row["query_id"]),
                    "relative_moves": completion_moves - noop_moves,
                    "root_cluster_permutations": cluster_states[noop_index].tolist(),
                    "selection_sources": [str(source) for source in row["selection_sources"]],
                    "setup_moves": len(setup_path),
                    "wide_completion_moves": wide_lengths.get(str(row["query_id"])),
                }
            )
            kept += 1
        campaign_summaries.append(
            {
                "completed_pids": completed_pids,
                "frontier_dir": str(frontier_dir),
                "missing": len(missing),
                "query_dir": str(query_dir),
                "rows": kept,
            }
        )

    if not output_rows:
        raise ValueError("no matched context rows were found")
    pids = sorted({int(row["pid"]) for row in output_rows})
    payload = {
        "campaigns": campaign_summaries,
        "rows": output_rows,
        "summary": {
            "actions": sum(int(row["action_id"]) >= 0 for row in output_rows),
            "pids": len(pids),
            "rows": len(output_rows),
            "wide_labels": sum(
                row["wide_completion_moves"] is not None for row in output_rows
            ),
        },
    }
    atomic_json(args.out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
