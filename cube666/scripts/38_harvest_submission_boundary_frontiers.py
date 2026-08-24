"""Harvest model scouts from exact normalized boundaries on an incumbent path."""

from __future__ import annotations

import argparse
import csv
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
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.macros import reduce_quarter_turn_path, state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--pids", required=True)
    parser.add_argument("--step-depths", default="0,1")
    parser.add_argument(
        "--boundary-mode",
        choices=("first", "all"),
        default="first",
        help="Harvest the first eligible incumbent boundary or every eligible boundary.",
    )
    parser.add_argument(
        "--min-baseline-suffix",
        type=int,
        default=1,
        help="Ignore boundaries with fewer incumbent suffix moves than this.",
    )
    parser.add_argument(
        "--max-boundaries-per-pid",
        type=int,
        default=0,
        help="Optional cap after filtering; zero keeps every eligible boundary.",
    )
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--branch-width", type=int, default=64)
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
    if ":" in args.pids:
        range_parts = tuple(int(token) for token in args.pids.split(":"))
        if len(range_parts) not in (2, 3):
            raise ValueError("PID range must use start:stop[:step]")
        pids = tuple(range(*range_parts))
    else:
        pids = tuple(int(token) for token in args.pids.split(",") if token.strip())
    depths = tuple(
        sorted({int(token) for token in args.step_depths.split(",") if token.strip()})
    )
    if not pids or not depths or min(depths) < 0:
        raise ValueError("PIDs and non-negative step depths are required")
    if args.min_baseline_suffix < 1 or args.max_boundaries_per_pid < 0:
        raise ValueError("invalid boundary filtering options")
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
    states = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
    with open(args.submission, newline="", encoding="utf-8") as handle:
        paths = {
            int(row["initial_state_id"]): tuple(
                token for token in row["path"].split(".") if token
            )
            for row in csv.DictReader(handle)
        }

    full_frontiers: list[np.ndarray] = []
    cluster_frontiers: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    seen: set[bytes] = set()
    skipped_pids: list[int] = []
    for pid in pids:
        initial = states[str(pid)]
        incumbent_path = paths[pid]
        boundaries: list[tuple[int, tuple[int, ...], int]] = []
        current = initial
        for prefix in range(len(incumbent_path) + 1):
            if 0 < prefix < len(incumbent_path) and all(
                current[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            ):
                report = residual_report(current, puzzle.solved_state, decomposition)
                if report.all_even and report.unrestricted_three_cycles is not None:
                    boundaries.append(
                        (prefix, current, int(report.unrestricted_three_cycles))
                    )
            if prefix < len(incumbent_path):
                current = puzzle.apply_move(current, incumbent_path[prefix])
        if current != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: incumbent path failed replay")
        boundaries = [
            boundary
            for boundary in boundaries
            if len(incumbent_path) - boundary[0] >= args.min_baseline_suffix
        ]
        if not boundaries:
            skipped_pids.append(pid)
            print(
                json.dumps(
                    {
                        "pid": pid,
                        "skipped": "no eligible incumbent boundary",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        if args.boundary_mode == "first":
            boundaries = boundaries[:1]
        if args.max_boundaries_per_pid:
            boundaries = boundaries[: args.max_boundaries_per_pid]
        for boundary_rank, (prefix, boundary_state, boundary_residual) in enumerate(
            boundaries
        ):
            cluster_state = np.asarray(
                state_cluster_permutations(
                    boundary_state, puzzle.solved_state, decomposition
                ),
                dtype=np.uint8,
            )
            prefix_path = incumbent_path[:prefix]
            for depth in depths:
                case_started = time.perf_counter()
                beam = learned_macro_beam_search(
                    cluster_state,
                    model,
                    table,
                    beam_width=args.beam_width,
                    branch_width=args.branch_width,
                    max_steps=depth,
                    policy_nll_weight=args.policy_nll_weight,
                    exact_cost_weight=None,
                    move_cost_weight=args.move_cost_weight,
                    model_batch_size=2048,
                    fallback_exact_cost=False,
                )
                scout_path = reduce_quarter_turn_path(
                    tuple(move for action in beam.actions for move in table.paths[action])
                )
                frontier = puzzle.apply_path(boundary_state, scout_path)
                frontier_report = residual_report(
                    frontier, puzzle.solved_state, decomposition
                )
                if not frontier_report.all_even:
                    raise AssertionError(f"PID {pid}: scout changed boundary parity")
                if any(
                    frontier[position] != puzzle.solved_state[position]
                    for position in decomposition.corner_orbit
                ):
                    raise AssertionError(f"PID {pid}: scout moved solved corners")
                replay_clusters = np.asarray(
                    state_cluster_permutations(
                        frontier, puzzle.solved_state, decomposition
                    ),
                    dtype=np.uint8,
                )
                if not np.array_equal(replay_clusters, beam.final_state):
                    raise AssertionError(f"PID {pid}: scout projection replay mismatch")
                key = np.asarray(frontier, dtype=np.uint8).tobytes()
                if key in seen:
                    print(
                        json.dumps(
                            {
                                "boundary_prefix": prefix,
                                "duplicate_frontier": True,
                                "pid": pid,
                                "scout_depth": depth,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    continue
                seen.add(key)
                full_frontiers.append(np.asarray(frontier, dtype=np.uint8))
                cluster_frontiers.append(replay_clusters)
                row = {
                    "baseline_suffix_moves": len(incumbent_path) - prefix,
                    "beam_actions": list(beam.actions),
                    "beam_expanded_states": beam.expanded_states,
                    "beam_generated_states": beam.generated_states,
                    "boundary_prefix": prefix,
                    "boundary_rank": boundary_rank,
                    "boundary_residual": boundary_residual,
                    "elapsed_seconds": round(time.perf_counter() - case_started, 4),
                    "frontier_residual": int(
                        frontier_report.unrestricted_three_cycles
                    ),
                    "macro_primitive_moves": len(scout_path),
                    "pid": pid,
                    "predicted_completion_moves": round(
                        float(beam.final_predicted_value), 4
                    ),
                    "prefix_path": list(prefix_path),
                    "query_id": f"pid{pid:04d}_b{prefix:03d}_step{depth:02d}",
                    "replay_verified": True,
                    "scout_depth": depth,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

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
        "beam_width": args.beam_width,
        "boundary_mode": args.boundary_mode,
        "branch_width": args.branch_width,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "frontiers": len(rows),
        "max_boundaries_per_pid": args.max_boundaries_per_pid,
        "min_baseline_suffix": args.min_baseline_suffix,
        "move_cost_weight": args.move_cost_weight,
        "policy_nll_weight": args.policy_nll_weight,
        "rows": rows,
        "skipped_pids": skipped_pids,
        "step_depths": list(depths),
        "submission": str(args.submission),
    }
    atomic_json(args.out_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
