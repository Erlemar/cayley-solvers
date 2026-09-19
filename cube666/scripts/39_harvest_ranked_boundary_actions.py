"""Harvest several model-ranked one-step actions from incumbent boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macro_beam import rank_one_step_macro_actions  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.macros import reduce_quarter_turn_path, state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--source-frontier-dir", type=Path, required=True)
    parser.add_argument("--source-indices", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--proposal-width", type=int, default=256)
    parser.add_argument("--candidates-per-boundary", type=int, default=16)
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument("--move-cost-weight", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source_indices = tuple(
        int(token) for token in args.source_indices.split(",") if token.strip()
    )
    if not source_indices:
        raise ValueError("at least one source index is required")
    if args.proposal_width <= 0 or args.candidates_per_boundary <= 0:
        raise ValueError("invalid candidate dimensions")
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library, puzzle.generators, decomposition
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != table.digest:
        raise ValueError("checkpoint and action library digests differ")
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    test_states = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
    source_report = json.loads(
        (args.source_frontier_dir / "report.json").read_text(encoding="utf-8")
    )
    source_rows = source_report["rows"]
    if any(index < 0 or index >= len(source_rows) for index in source_indices):
        raise ValueError("source index is outside the frontier report")

    full_frontiers: list[np.ndarray] = []
    cluster_frontiers: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for source_index in source_indices:
        source = source_rows[source_index]
        pid = int(source["pid"])
        prefix = int(source["boundary_prefix"])
        prefix_path = tuple(str(move) for move in source["prefix_path"])
        if len(prefix_path) != prefix:
            raise AssertionError(f"PID {pid}: prefix metadata length mismatch")
        boundary_state = puzzle.apply_path(test_states[str(pid)], prefix_path)
        boundary_report = residual_report(
            boundary_state, puzzle.solved_state, decomposition
        )
        if not boundary_report.all_even or any(
            boundary_state[position] != puzzle.solved_state[position]
            for position in decomposition.corner_orbit
        ):
            raise AssertionError(f"PID {pid}: source is not a normalized boundary")
        cluster_state = np.asarray(
            state_cluster_permutations(
                boundary_state, puzzle.solved_state, decomposition
            ),
            dtype=np.uint8,
        )
        candidates = rank_one_step_macro_actions(
            cluster_state,
            model,
            table,
            proposal_width=args.proposal_width,
            limit=args.candidates_per_boundary,
            policy_nll_weight=args.policy_nll_weight,
            move_cost_weight=args.move_cost_weight,
            model_batch_size=2048,
        )
        for candidate_rank, candidate in enumerate(candidates):
            scout_path = reduce_quarter_turn_path(table.paths[candidate.action])
            frontier = puzzle.apply_path(boundary_state, scout_path)
            replay_clusters = np.asarray(
                state_cluster_permutations(
                    frontier, puzzle.solved_state, decomposition
                ),
                dtype=np.uint8,
            )
            if not np.array_equal(replay_clusters, candidate.state):
                raise AssertionError(f"PID {pid}: ranked action projection mismatch")
            frontier_report = residual_report(
                frontier, puzzle.solved_state, decomposition
            )
            if not frontier_report.all_even or any(
                frontier[position] != puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            ):
                raise AssertionError(f"PID {pid}: ranked action broke normalization")
            key = np.asarray(frontier, dtype=np.uint8).tobytes()
            if key in seen:
                continue
            seen.add(key)
            full_frontiers.append(np.asarray(frontier, dtype=np.uint8))
            cluster_frontiers.append(replay_clusters)
            row = {
                "baseline_suffix_moves": int(source["baseline_suffix_moves"]),
                "beam_actions": [candidate.action],
                "boundary_prefix": prefix,
                "boundary_residual": int(boundary_report.unrestricted_three_cycles),
                "candidate_rank": candidate_rank,
                "frontier_residual": int(frontier_report.unrestricted_three_cycles),
                "macro_primitive_moves": len(scout_path),
                "pid": pid,
                "policy_nll": round(candidate.policy_nll, 6),
                "predicted_completion_moves": round(candidate.predicted_value, 4),
                "predicted_rank": round(candidate.rank, 4),
                "prefix_path": list(prefix_path),
                "query_id": (
                    f"pid{pid:04d}_b{prefix:03d}_rank{candidate_rank:02d}"
                    f"_a{candidate.action:05d}"
                ),
                "replay_verified": True,
                "scout_depth": 1,
                "source_index": source_index,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    if not rows:
        raise RuntimeError("no ranked frontier candidates were produced")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.out_dir / "frontiers.npz.tmp"
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            full_states=np.stack(full_frontiers),
            cluster_states=np.stack(cluster_frontiers),
            source_pids=np.asarray([int(row["pid"]) for row in rows], dtype=np.int32),
        )
    temporary.replace(args.out_dir / "frontiers.npz")
    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "candidates_per_boundary": args.candidates_per_boundary,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "frontiers": len(rows),
        "move_cost_weight": args.move_cost_weight,
        "policy_nll_weight": args.policy_nll_weight,
        "proposal_width": args.proposal_width,
        "rows": rows,
        "source_frontier_dir": str(args.source_frontier_dir),
        "source_indices": list(source_indices),
    }
    atomic_json(args.out_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
