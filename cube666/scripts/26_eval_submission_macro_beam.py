"""Evaluate a submission-trained macro beam against exact classical paths."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
from cube666.macros import (  # noqa: E402
    finish_with_inserted_three_cycles,
    load_three_cycle_library,
    reduce_quarter_turn_path,
    state_cluster_permutations,
)
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
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--stage1-action-library", type=Path)
    parser.add_argument(
        "--classical-submission",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument(
        "--finisher-library",
        type=Path,
        default=PROJECT / "cube666" / "artifacts" / "three_cycle_library.json",
    )
    parser.add_argument("--pids", default="200,210,220,230,240")
    parser.add_argument("--beam-width", type=int, default=512)
    parser.add_argument("--branch-width", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument(
        "--exact-cost-weight",
        type=float,
        default=3.0,
        help="positive exact-residual weight, or zero to rank by the learned value",
    )
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument(
        "--continuation-steps",
        type=int,
        default=0,
        help=(
            "after the exact/value-guided stage, continue from its frontier "
            "using learned value and policy ranking only"
        ),
    )
    parser.add_argument(
        "--continuation-policy-nll-weight",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--continuation-move-cost-weight",
        type=float,
        default=0.0,
        help="primitive moves already spent per unit of learned continuation rank",
    )
    parser.add_argument(
        "--continuation-continue-after-solution",
        action="store_true",
        help="keep the first exact solution as a cost bound and seek a shorter one",
    )
    parser.add_argument("--continuation-beam-width", type=int)
    parser.add_argument("--continuation-branch-width", type=int)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "submission_macro_beam.json",
    )
    return parser.parse_args()


def load_submission(path: Path) -> dict[int, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): tuple(
                token for token in row["path"].split(".") if token
            )
            for row in csv.DictReader(handle)
        }


def main() -> None:
    args = parse_args()
    if (args.stage1_checkpoint is None) != (args.stage1_action_library is None):
        raise ValueError("stage1 checkpoint and action library must be supplied together")
    pids = tuple(int(token) for token in args.pids.split(",") if token.strip())
    if not pids:
        raise ValueError("pids cannot be empty")
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
    stage1_model = model
    stage1_table = table
    if args.stage1_checkpoint is not None:
        _, stage1_table = load_macro_action_library(
            args.stage1_action_library,
            puzzle.generators,
            decomposition,
        )
        stage1_checkpoint = torch.load(
            args.stage1_checkpoint, map_location="cpu", weights_only=True
        )
        if stage1_checkpoint["action_digest"] != stage1_table.digest:
            raise ValueError("stage1 checkpoint and action library digests differ")
        stage1_model = build_macro_policy_model(stage1_checkpoint["model_config"])
        stage1_model.load_state_dict(stage1_checkpoint["model_state_dict"])
        stage1_model.to(device).eval()
    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    finisher = load_three_cycle_library(
        args.finisher_library,
        puzzle.generators,
        decomposition,
    )
    states = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    classical = load_submission(args.classical_submission)

    rows: list[dict[str, object]] = []
    for pid in pids:
        case_started = time.perf_counter()
        state = states[pid]
        classical_path = classical[pid]
        if puzzle.apply_path(state, classical_path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: classical path failed replay")
        corner_path = corner_solver.solve(state, puzzle.solved_state)
        corner_state = apply_path(state, puzzle.generators, corner_path)
        parity_path = parity_repair_path(
            residual_report(
                corner_state,
                puzzle.solved_state,
                decomposition,
            ).parity_vector,
            decomposition,
        )
        normalized = apply_path(corner_state, puzzle.generators, parity_path)
        setup_path = reduce_quarter_turn_path(corner_path + parity_path)
        initial_clusters = np.asarray(
            state_cluster_permutations(
                normalized,
                puzzle.solved_state,
                decomposition,
            ),
            dtype=np.uint8,
        )
        initial_cost = int(
            residual_report(
                normalized,
                puzzle.solved_state,
                decomposition,
            ).unrestricted_three_cycles
        )
        stage1 = learned_macro_beam_search(
            initial_clusters,
            stage1_model,
            stage1_table,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            max_steps=args.max_steps,
            policy_nll_weight=args.policy_nll_weight,
            exact_cost_weight=(
                args.exact_cost_weight if args.exact_cost_weight > 0 else None
            ),
            model_batch_size=2048,
            fallback_exact_cost=True,
        )
        continuation = None
        if args.continuation_steps > 0 and not stage1.solved:
            continuation = learned_macro_beam_search(
                stage1.final_state,
                model,
                table,
                beam_width=args.continuation_beam_width or args.beam_width,
                branch_width=args.continuation_branch_width or args.branch_width,
                max_steps=args.continuation_steps,
                policy_nll_weight=args.continuation_policy_nll_weight,
                exact_cost_weight=None,
                move_cost_weight=args.continuation_move_cost_weight,
                model_batch_size=2048,
                fallback_exact_cost=True,
                stop_on_first_solution=(
                    not args.continuation_continue_after_solution
                ),
            )
        beam_action_count = len(stage1.actions) + (
            len(continuation.actions) if continuation is not None else 0
        )
        final_state = (
            continuation.final_state if continuation is not None else stage1.final_state
        )
        final_exact_cost = (
            continuation.final_exact_cost
            if continuation is not None
            else stage1.final_exact_cost
        )
        solved = continuation.solved if continuation is not None else stage1.solved
        expanded_states = stage1.expanded_states + (
            continuation.expanded_states if continuation is not None else 0
        )
        generated_states = stage1.generated_states + (
            continuation.generated_states if continuation is not None else 0
        )
        macro_path = reduce_quarter_turn_path(
            tuple(move for action in stage1.actions for move in stage1_table.paths[action])
            + tuple(
                move
                for action in (continuation.actions if continuation is not None else ())
                for move in table.paths[action]
            )
        )
        stage1_raw_moves = sum(
            len(stage1_table.paths[action]) for action in stage1.actions
        )
        continuation_raw_moves = sum(
            len(table.paths[action])
            for action in (continuation.actions if continuation is not None else ())
        )
        macro_state = apply_path(normalized, puzzle.generators, macro_path)
        replay_clusters = np.asarray(
            state_cluster_permutations(
                macro_state,
                puzzle.solved_state,
                decomposition,
            ),
            dtype=np.uint8,
        )
        if not np.array_equal(replay_clusters, final_state):
            raise AssertionError(f"PID {pid}: macro/full-state replay mismatch")
        finished = finish_with_inserted_three_cycles(
            tuple(tuple(int(value) for value in row) for row in replay_clusters),
            reduce_quarter_turn_path(setup_path + macro_path),
            finisher,
            puzzle.generators,
            decomposition,
        )
        if puzzle.apply_path(state, finished.path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: learned candidate failed exact replay")
        row = {
            "beam_actions": beam_action_count,
            "beam_expanded_states": expanded_states,
            "beam_generated_states": generated_states,
            "beam_solved_bulk": solved,
            "classical_moves": len(classical_path),
            "delta_vs_classical": len(finished.path) - len(classical_path),
            "elapsed_seconds": round(time.perf_counter() - case_started, 4),
            "final_moves": len(finished.path),
            "final_residual": final_exact_cost,
            "initial_residual": initial_cost,
            "macro_primitive_moves": len(macro_path),
            "pid": pid,
            "replay_verified": True,
            "setup_moves": len(setup_path),
            "stage1_actions": len(stage1.actions),
            "stage1_raw_primitive_moves": stage1_raw_moves,
            "stage1_residual": stage1.final_exact_cost,
            "continuation_actions": (
                len(continuation.actions) if continuation is not None else 0
            ),
            "continuation_raw_primitive_moves": continuation_raw_moves,
            "continuation_solved": (
                continuation.solved if continuation is not None else False
            ),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "continuation_policy_nll_weight": args.continuation_policy_nll_weight,
        "continuation_move_cost_weight": args.continuation_move_cost_weight,
        "continuation_continue_after_solution": (
            args.continuation_continue_after_solution
        ),
        "continuation_steps": args.continuation_steps,
        "continuation_beam_width": args.continuation_beam_width or args.beam_width,
        "continuation_branch_width": (
            args.continuation_branch_width or args.branch_width
        ),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "exact_cost_weight": args.exact_cost_weight,
        "max_steps": args.max_steps,
        "mean_classical_moves": statistics.fmean(
            int(row["classical_moves"]) for row in rows
        ),
        "mean_delta_vs_classical": statistics.fmean(
            int(row["delta_vs_classical"]) for row in rows
        ),
        "mean_final_moves": statistics.fmean(int(row["final_moves"]) for row in rows),
        "replay_verified": all(bool(row["replay_verified"]) for row in rows),
        "rows": rows,
        "states": len(rows),
        "stage1_action_count": stage1_table.action_count,
        "stage1_action_digest": stage1_table.digest,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
