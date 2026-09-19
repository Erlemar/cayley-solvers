"""Run autonomous learned beam search on synthetic or full normalized 666 states."""

from __future__ import annotations

import argparse
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
from cube666.macro_data import (  # noqa: E402
    MacroTeacherDataset,
    cluster_costs,
    load_macro_action_library,
    validate_factorized_action_table,
)
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.macros import reduce_quarter_turn_path, state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_indices(raw: str | None) -> list[int]:
    if not raw:
        return []
    return [int(value) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--teacher-indices", default="0,16,33,50,67")
    parser.add_argument("--puzzle-indices")
    parser.add_argument(
        "--all-puzzles",
        action="store_true",
        help="evaluate every row in test.csv",
    )
    parser.add_argument("--beam-width", type=int, default=512)
    parser.add_argument("--branch-width", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=72)
    parser.add_argument(
        "--rescue-beam-width",
        type=int,
        help="retry unsolved cases with this wider beam",
    )
    parser.add_argument(
        "--rescue-branch-width",
        type=int,
        help="branch width for rescue attempts; defaults to --branch-width",
    )
    parser.add_argument(
        "--rescue-max-steps",
        type=int,
        help="depth for rescue attempts; defaults to --max-steps",
    )
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument("--model-batch-size", type=int, default=1024)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="print one result every N cases; use 0 to suppress per-case output",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit the full row list from the final stdout summary",
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    if checkpoint["model_config"].get("architecture") == "factorized":
        validate_factorized_action_table(table)
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    cases: list[tuple[str, np.ndarray, tuple[int, ...] | None, tuple[int, ...] | None]] = []
    if args.teacher:
        dataset = MacroTeacherDataset.load(args.teacher)
        if dataset.action_digest != table.digest:
            raise ValueError("teacher and action library digests differ")
        for index in parse_indices(args.teacher_indices):
            cases.append((f"teacher:{index}", dataset.states[index], None, None))

    if args.all_puzzles and args.puzzle_indices:
        raise ValueError("use either --all-puzzles or --puzzle-indices, not both")
    puzzle_indices = parse_indices(args.puzzle_indices)
    if args.all_puzzles:
        puzzle_indices = list(
            range(sum(1 for _ in puzzle.iter_test_states(args.data_dir / "test.csv")))
        )
    if puzzle_indices:
        corner_solver = ExactCornerSolver.build(
            CornerCoordinateSystem.discover(puzzle.generators, decomposition)
        )
        all_states = list(puzzle.iter_test_states(args.data_dir / "test.csv"))
        for index in puzzle_indices:
            state_id, state = all_states[index]
            corner_path = corner_solver.solve(state, puzzle.solved_state)
            corner_state = apply_path(state, puzzle.generators, corner_path)
            parity_path = parity_repair_path(
                residual_report(corner_state, puzzle.solved_state, decomposition).parity_vector,
                decomposition,
            )
            normalized = apply_path(corner_state, puzzle.generators, parity_path)
            clusters = np.asarray(
                state_cluster_permutations(normalized, puzzle.solved_state, decomposition),
                dtype=np.uint8,
            )
            cases.append((f"puzzle:{state_id}", clusters, state, corner_path + parity_path))
    if not cases:
        raise ValueError("provide --teacher, --puzzle-indices, or --all-puzzles")

    rows: list[dict[str, object]] = []
    for label, state, full_state, setup_path in cases:
        case_started = time.perf_counter()
        initial_cost = int(cluster_costs(state).sum())
        result = learned_macro_beam_search(
            state,
            model,
            table,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            max_steps=args.max_steps,
            policy_nll_weight=args.policy_nll_weight,
            model_batch_size=args.model_batch_size,
        )
        attempts = [
            {
                "beam_width": args.beam_width,
                "branch_width": args.branch_width,
                "expanded_states": result.expanded_states,
                "final_exact_cost": int(cluster_costs(result.final_state).sum()),
                "generated_states": result.generated_states,
                "macro_steps": result.macro_steps,
                "max_steps": args.max_steps,
                "solved": result.solved,
            }
        ]
        effective_beam_width = args.beam_width
        effective_branch_width = args.branch_width
        effective_max_steps = args.max_steps
        if not result.solved and args.rescue_beam_width is not None:
            effective_beam_width = args.rescue_beam_width
            effective_branch_width = args.rescue_branch_width or args.branch_width
            effective_max_steps = args.rescue_max_steps or args.max_steps
            result = learned_macro_beam_search(
                state,
                model,
                table,
                beam_width=effective_beam_width,
                branch_width=effective_branch_width,
                max_steps=effective_max_steps,
                policy_nll_weight=args.policy_nll_weight,
                model_batch_size=args.model_batch_size,
            )
            attempts.append(
                {
                    "beam_width": effective_beam_width,
                    "branch_width": effective_branch_width,
                    "expanded_states": result.expanded_states,
                    "final_exact_cost": int(cluster_costs(result.final_state).sum()),
                    "generated_states": result.generated_states,
                    "macro_steps": result.macro_steps,
                    "max_steps": effective_max_steps,
                    "solved": result.solved,
                }
            )
        final_cost = int(cluster_costs(result.final_state).sum())
        replay_verified = False
        primitive_moves: int | None = None
        if result.solved:
            macro_path = tuple(
                move
                for action in result.actions
                for move in table.paths[action]
            )
            if full_state is None:
                replay_state = state.copy()
                for action in result.actions:
                    replay_state = table.apply(replay_state, action)
                replay_verified = bool(
                    np.array_equal(
                        replay_state,
                        np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)),
                    )
                )
                primitive_moves = len(reduce_quarter_turn_path(macro_path))
            else:
                if setup_path is None:
                    raise AssertionError("full puzzle case has no setup path")
                full_path = reduce_quarter_turn_path(setup_path + macro_path)
                replay_verified = puzzle.apply_path(full_state, full_path) == puzzle.solved_state
                primitive_moves = len(full_path)
        row = {
            "attempts": attempts,
            "beam_width": effective_beam_width,
            "branch_width": effective_branch_width,
            "case": label,
            "elapsed_seconds": round(time.perf_counter() - case_started, 4),
            "expanded_states": sum(attempt["expanded_states"] for attempt in attempts),
            "final_exact_cost": final_cost,
            "generated_states": sum(attempt["generated_states"] for attempt in attempts),
            "initial_exact_cost": initial_cost,
            "macro_steps": result.macro_steps,
            "max_steps": effective_max_steps,
            "primitive_moves": primitive_moves,
            "replay_verified": replay_verified,
            "rescue_used": len(attempts) > 1,
            "solved": result.solved,
        }
        rows.append(row)
        if args.progress_every > 0 and (
            len(rows) % args.progress_every == 0 or not result.solved
        ):
            print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "action_digest": table.digest,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "rescue_beam_width": args.rescue_beam_width,
        "rescue_branch_width": args.rescue_branch_width,
        "rescue_max_steps": args.rescue_max_steps,
        "rescue_used": sum(bool(row["rescue_used"]) for row in rows),
        "replay_verified_solutions": sum(
            bool(row["solved"] and row["replay_verified"])
            for row in rows
        ),
        "rows": rows,
        "solved": sum(bool(row["solved"]) for row in rows),
        "solve_rate": statistics.fmean(bool(row["solved"]) for row in rows),
        "total": len(rows),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.out)
    stdout_report = (
        {key: value for key, value in report.items() if key != "rows"}
        if args.summary_only
        else report
    )
    print(json.dumps(stdout_report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
