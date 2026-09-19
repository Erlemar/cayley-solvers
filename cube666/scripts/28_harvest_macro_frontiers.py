"""Harvest failed on-policy macro-beam frontiers for KMC DAgger queries.

Each saved frontier is a complete 216-sticker state, not merely the six-cluster
projection.  The script replays every selected macro on the full cube and checks
that corners remain solved, cluster parity remains even, and the projected state
matches the beam result exactly.
"""

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

from cube666.classical import (  # noqa: E402
    apply_path,
    build_decomposition,
    parity_repair_path,
    residual_report,
)
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.macros import reduce_quarter_turn_path, state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT / "cayley-py-666-cube",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--pids", default="201,212,223,234")
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--branch-width", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--step-depths",
        help="optional comma-separated scout depths; produces one frontier per PID/depth",
    )
    parser.add_argument("--exact-cost-weight", type=float, default=8.0)
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument("--move-cost-weight", type=float, default=0.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "training" / "dagger_frontiers_pilot_v1",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    pids = tuple(int(token) for token in args.pids.split(",") if token.strip())
    if not pids:
        raise ValueError("pids cannot be empty")
    if len(set(pids)) != len(pids):
        raise ValueError("pids must be unique")
    step_depths = (
        tuple(
            sorted(
                {
                    int(token)
                    for token in args.step_depths.split(",")
                    if token.strip()
                }
            )
        )
        if args.step_depths
        else (args.max_steps,)
    )
    if not step_depths or min(step_depths) < 0:
        raise ValueError("scout depths must be non-negative")
    if args.exact_cost_weight < 0 or args.move_cost_weight < 0:
        raise ValueError("search weights must be non-negative")
    if args.exact_cost_weight > 0 and args.move_cost_weight > 0:
        raise ValueError("move-cost-weight is only valid with learned-value ranking")

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library,
        puzzle.generators,
        decomposition,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != table.digest:
        raise ValueError("checkpoint and action library digests differ")
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    states_by_id = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }

    full_frontiers: list[np.ndarray] = []
    cluster_frontiers: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    seen: set[bytes] = set()
    for pid in pids:
        case_started = time.perf_counter()
        if pid not in states_by_id:
            raise KeyError(f"PID {pid} is not present in test.csv")
        state = states_by_id[pid]
        corner_path = corner_solver.solve(state, puzzle.solved_state)
        corner_state = apply_path(state, puzzle.generators, corner_path)
        parity_path = parity_repair_path(
            residual_report(corner_state, puzzle.solved_state, decomposition).parity_vector,
            decomposition,
        )
        normalized = apply_path(corner_state, puzzle.generators, parity_path)
        normalized_report = residual_report(normalized, puzzle.solved_state, decomposition)
        if not normalized_report.all_even or normalized_report.unrestricted_three_cycles is None:
            raise AssertionError(f"PID {pid}: normalization did not produce an even state")
        initial_clusters = np.asarray(
            state_cluster_permutations(normalized, puzzle.solved_state, decomposition),
            dtype=np.uint8,
        )
        for scout_depth in step_depths:
            depth_started = time.perf_counter()
            beam = learned_macro_beam_search(
                initial_clusters,
                model,
                table,
                beam_width=args.beam_width,
                branch_width=args.branch_width,
                max_steps=scout_depth,
                policy_nll_weight=args.policy_nll_weight,
                exact_cost_weight=(
                    args.exact_cost_weight if args.exact_cost_weight > 0 else None
                ),
                move_cost_weight=args.move_cost_weight,
                model_batch_size=2048,
                fallback_exact_cost=args.exact_cost_weight > 0,
            )
            macro_path = reduce_quarter_turn_path(
                tuple(move for action in beam.actions for move in table.paths[action])
            )
            frontier = apply_path(normalized, puzzle.generators, macro_path)
            report = residual_report(frontier, puzzle.solved_state, decomposition)
            if not report.all_even or report.unrestricted_three_cycles is None:
                raise AssertionError(f"PID {pid}: harvested frontier does not have even parity")
            corner_mismatches = sum(
                frontier[position] != puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            )
            if corner_mismatches:
                raise AssertionError(f"PID {pid}: harvested frontier moved {corner_mismatches} corners")
            replay_clusters = np.asarray(
                state_cluster_permutations(frontier, puzzle.solved_state, decomposition),
                dtype=np.uint8,
            )
            if not np.array_equal(replay_clusters, beam.final_state):
                raise AssertionError(f"PID {pid}: macro/full-state replay mismatch")
            key = np.asarray(frontier, dtype=np.uint8).tobytes()
            if key in seen:
                raise ValueError(f"PID {pid}: duplicate harvested frontier at depth {scout_depth}")
            seen.add(key)
            full_frontiers.append(np.asarray(frontier, dtype=np.uint8))
            cluster_frontiers.append(replay_clusters)
            row = {
                "beam_actions": list(beam.actions),
                "beam_expanded_states": beam.expanded_states,
                "beam_generated_states": beam.generated_states,
                "beam_solved": beam.solved,
                "elapsed_seconds": round(time.perf_counter() - depth_started, 4),
                "frontier_residual": int(report.unrestricted_three_cycles),
                "predicted_completion_moves": round(
                    float(beam.final_predicted_value), 4
                ),
                "initial_residual": int(normalized_report.unrestricted_three_cycles),
                "macro_primitive_moves": len(macro_path),
                "pid": pid,
                "query_id": f"pid{pid:04d}_step{scout_depth:02d}",
                "replay_verified": True,
                "scout_depth": scout_depth,
                "setup_moves": len(reduce_quarter_turn_path(corner_path + parity_path)),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        args.out_dir / "frontiers.npz",
        full_states=np.stack(full_frontiers),
        cluster_states=np.stack(cluster_frontiers),
        source_pids=np.asarray([int(row["pid"]) for row in rows], dtype=np.int32),
        initial_residuals=np.asarray(
            [int(row["initial_residual"]) for row in rows], dtype=np.int16
        ),
        frontier_residuals=np.asarray(
            [int(row["frontier_residual"]) for row in rows], dtype=np.int16
        ),
    )
    summary = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "exact_cost_weight": args.exact_cost_weight,
        "frontiers": len(rows),
        "max_steps": args.max_steps,
        "step_depths": list(step_depths),
        "policy_nll_weight": args.policy_nll_weight,
        "move_cost_weight": args.move_cost_weight,
        "rows": rows,
    }
    atomic_json(args.out_dir / "report.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
