"""Matched KMC-prefix beam comparison with and without appended Walton actions."""

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
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macro_policy import MacroPolicyConfig, MacroPolicyValueNet  # noqa: E402
from cube666.macros import (  # noqa: E402
    enumerate_basic_corner_fixing_commutators,
    enumerate_conjugated_macros,
    finish_with_inserted_three_cycles,
    greedy_macro_reduce,
    load_three_cycle_library,
    reduce_quarter_turn_path,
    state_cluster_permutations,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT / "models" / "cube666_kmc_macro_policy_grouped_v1" / "checkpoint.pt",
    )
    parser.add_argument(
        "--model-action-library",
        type=Path,
        default=PROJECT
        / "cube666"
        / "training"
        / "kmc_macro_teacher_v1"
        / "action_library.json",
    )
    parser.add_argument(
        "--augmented-action-library",
        type=Path,
        default=PROJECT
        / "cube666"
        / "training"
        / "kmc_walton_macro4"
        / "action_library.json",
    )
    parser.add_argument(
        "--finisher-library",
        type=Path,
        default=PROJECT / "cube666" / "artifacts" / "three_cycle_library.json",
    )
    parser.add_argument("--pids", default="597,808,854,906")
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--branch-width", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--greedy-steps", type=int, default=20)
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "cube666" / "results" / "kmc_walton_macro4" / "beam_comparison.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pids = tuple(int(token) for token in args.pids.split(",") if token.strip())
    if not pids:
        raise ValueError("pids cannot be empty")
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, model_table = load_macro_action_library(
        args.model_action_library,
        puzzle.generators,
        decomposition,
    )
    _, augmented_table = load_macro_action_library(
        args.augmented_action_library,
        puzzle.generators,
        decomposition,
    )
    if augmented_table.action_count <= model_table.action_count:
        raise ValueError("augmented library has no appended actions")
    if not np.array_equal(
        augmented_table.effects[: model_table.action_count],
        model_table.effects,
    ) or augmented_table.paths[: model_table.action_count] != model_table.paths:
        raise ValueError("augmented library does not preserve model action indices")
    extra_actions = tuple(range(model_table.action_count, augmented_table.action_count))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != model_table.digest:
        raise ValueError("checkpoint and model action library digests differ")
    model = MacroPolicyValueNet(MacroPolicyConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    finisher = load_three_cycle_library(
        args.finisher_library,
        puzzle.generators,
        decomposition,
    )
    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    classical_bulk_macros = enumerate_conjugated_macros(
        enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition),
        puzzle.generators,
        decomposition,
        max_conjugator_depth=1,
    )
    all_states = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }

    rows: list[dict[str, object]] = []
    for pid in pids:
        state = all_states[pid]
        corner_path = corner_solver.solve(state, puzzle.solved_state)
        corner_state = apply_path(state, puzzle.generators, corner_path)
        parity_path = parity_repair_path(
            residual_report(corner_state, puzzle.solved_state, decomposition).parity_vector,
            decomposition,
        )
        normalized = apply_path(corner_state, puzzle.generators, parity_path)
        setup_path = reduce_quarter_turn_path(corner_path + parity_path)
        initial_clusters = np.asarray(
            state_cluster_permutations(normalized, puzzle.solved_state, decomposition),
            dtype=np.uint8,
        )

        condition_rows: dict[str, dict[str, object]] = {}
        for condition, table, extras in (
            ("control", model_table, ()),
            ("augmented", augmented_table, extra_actions),
        ):
            condition_started = time.perf_counter()
            beam = learned_macro_beam_search(
                initial_clusters,
                model,
                table,
                beam_width=args.beam_width,
                branch_width=args.branch_width,
                max_steps=args.max_steps,
                policy_nll_weight=args.policy_nll_weight,
                extra_action_indices=extras,
                fallback_exact_cost=True,
            )
            beam_path = reduce_quarter_turn_path(
                tuple(
                    move
                    for action in beam.actions
                    for move in table.paths[action]
                )
            )
            beam_state = apply_path(normalized, puzzle.generators, beam_path)
            replay_clusters = np.asarray(
                state_cluster_permutations(beam_state, puzzle.solved_state, decomposition),
                dtype=np.uint8,
            )
            if not np.array_equal(replay_clusters, beam.final_state):
                raise AssertionError(f"PID {pid} {condition}: cluster/full replay mismatch")

            greedy = greedy_macro_reduce(
                beam.final_state,
                classical_bulk_macros,
                max_macros=args.greedy_steps,
            )
            bulk_path = reduce_quarter_turn_path(beam_path + greedy.path)
            bulk_state = apply_path(normalized, puzzle.generators, bulk_path)
            bulk_clusters = state_cluster_permutations(
                bulk_state,
                puzzle.solved_state,
                decomposition,
            )
            finished = finish_with_inserted_three_cycles(
                bulk_clusters,
                reduce_quarter_turn_path(setup_path + bulk_path),
                finisher,
                puzzle.generators,
                decomposition,
            )
            if puzzle.apply_path(state, finished.path) != puzzle.solved_state:
                raise AssertionError(f"PID {pid} {condition}: exact replay failed")
            novel_used = [action for action in beam.actions if action >= model_table.action_count]
            condition_rows[condition] = {
                "beam_actions": list(beam.actions),
                "beam_exact_cost": beam.final_exact_cost,
                "beam_expanded_states": beam.expanded_states,
                "beam_generated_states": beam.generated_states,
                "beam_macro_steps": beam.macro_steps,
                "beam_primitive_moves": len(beam_path),
                "elapsed_seconds": round(time.perf_counter() - condition_started, 4),
                "final_moves": len(finished.path),
                "final_path": list(finished.path),
                "greedy_actions": greedy.macro_count,
                "greedy_final_cost": greedy.final_cost,
                "novel_action_indices_used": novel_used,
                "novel_actions_used": len(novel_used),
                "replay_verified": True,
            }
        control = condition_rows["control"]
        augmented = condition_rows["augmented"]
        row = {
            "augmented": augmented,
            "beam_cost_delta": int(augmented["beam_exact_cost"]) - int(control["beam_exact_cost"]),
            "control": control,
            "final_move_delta": int(augmented["final_moves"]) - int(control["final_moves"]),
            "pid": pid,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "beam_cost_delta": row["beam_cost_delta"],
                    "final_move_delta": row["final_move_delta"],
                    "pid": pid,
                    "walton_actions_used": augmented["novel_actions_used"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "augmented_action_count": augmented_table.action_count,
        "augmented_action_digest": augmented_table.digest,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "extra_action_count": len(extra_actions),
        "max_steps": args.max_steps,
        "mean_beam_cost_delta": statistics.fmean(int(row["beam_cost_delta"]) for row in rows),
        "mean_final_move_delta": statistics.fmean(int(row["final_move_delta"]) for row in rows),
        "model_action_count": model_table.action_count,
        "model_action_digest": model_table.digest,
        "pids": list(pids),
        "policy_nll_weight": args.policy_nll_weight,
        "replay_verified": True,
        "rows": rows,
        "total_novel_actions_used": sum(
            int(row["augmented"]["novel_actions_used"])
            for row in rows
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
