"""Replay-verified end-to-end pilot for the learned KMC bulk-macro policy."""

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
from cube666.macro_data import cluster_costs, load_macro_action_library  # noqa: E402
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument(
        "--augmented-action-library",
        type=Path,
        help=(
            "checkpoint-compatible action library with the model actions first and "
            "additional actions appended; appended actions are always exact-scored"
        ),
    )
    parser.add_argument(
        "--finisher-library",
        type=Path,
        default=PROJECT / "cube666" / "artifacts" / "three_cycle_library.json",
    )
    parser.add_argument("--indices", default="200,210,220,230,240")
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--max-bulk-steps", type=int, default=20)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "results" / "macro_policy_pilot_v1",
    )
    return parser.parse_args()


def replay_verify(
    puzzle: Cube666Puzzle,
    state_id: str,
    state: tuple[int, ...],
    path: tuple[str, ...],
) -> None:
    final = puzzle.apply_path(state, path)
    if final != puzzle.solved_state:
        mismatches = sum(
            left != right
            for left, right in zip(final, puzzle.solved_state, strict=True)
        )
        raise ValueError(f"state {state_id}: replay failed with {mismatches} mismatches")


def main() -> None:
    args = parse_args()
    selected_indices = [int(value) for value in args.indices.split(",") if value.strip()]
    if not selected_indices:
        raise ValueError("indices cannot be empty")
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, model_table = load_macro_action_library(
        args.action_library,
        puzzle.generators,
        decomposition,
    )
    search_table = model_table
    extra_action_indices = np.empty(0, dtype=np.int64)
    if args.augmented_action_library is not None:
        _, search_table = load_macro_action_library(
            args.augmented_action_library,
            puzzle.generators,
            decomposition,
        )
        if search_table.action_count <= model_table.action_count:
            raise ValueError("augmented action library has no appended actions")
        if not np.array_equal(
            search_table.effects[: model_table.action_count],
            model_table.effects,
        ):
            raise ValueError("augmented action library does not preserve model action indices")
        if search_table.paths[: model_table.action_count] != model_table.paths:
            raise ValueError("augmented action library changed a model action path")
        extra_action_indices = np.arange(
            model_table.action_count,
            search_table.action_count,
            dtype=np.int64,
        )
    finisher = load_three_cycle_library(
        args.finisher_library,
        puzzle.generators,
        decomposition,
    )
    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != model_table.digest:
        raise ValueError("checkpoint and action library digests differ")
    model = MacroPolicyValueNet(MacroPolicyConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    top_k = min(args.top_k, model_table.action_count)
    classical_bulk_macros = enumerate_conjugated_macros(
        enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition),
        puzzle.generators,
        decomposition,
        max_conjugator_depth=1,
    )

    all_states = list(puzzle.iter_test_states(args.data_dir / "test.csv"))
    rows: list[dict[str, object]] = []
    solution_rows: list[dict[str, str]] = []
    for index in selected_indices:
        state_started = time.perf_counter()
        state_id, state = all_states[index]
        corner_path = corner_solver.solve(state, puzzle.solved_state)
        corner_state = apply_path(state, puzzle.generators, corner_path)
        parity_path = parity_repair_path(
            residual_report(corner_state, puzzle.solved_state, decomposition).parity_vector,
            decomposition,
        )
        normalized = apply_path(corner_state, puzzle.generators, parity_path)
        setup_path = reduce_quarter_turn_path(corner_path + parity_path)
        normalized_report = residual_report(normalized, puzzle.solved_state, decomposition)
        if not normalized_report.all_even or normalized_report.unrestricted_three_cycles is None:
            raise AssertionError("normalization did not produce six even clusters")

        initial_clusters = np.asarray(
            state_cluster_permutations(normalized, puzzle.solved_state, decomposition),
            dtype=np.uint8,
        )
        baseline_finished = finish_with_inserted_three_cycles(
            initial_clusters,
            setup_path,
            finisher,
            puzzle.generators,
            decomposition,
        )
        replay_verify(puzzle, state_id, state, baseline_finished.path)

        greedy_bulk = greedy_macro_reduce(
            initial_clusters,
            classical_bulk_macros,
            max_macros=args.max_bulk_steps,
        )
        greedy_state = apply_path(normalized, puzzle.generators, greedy_bulk.path)
        greedy_clusters = state_cluster_permutations(
            greedy_state,
            puzzle.solved_state,
            decomposition,
        )
        greedy_finished = finish_with_inserted_three_cycles(
            greedy_clusters,
            reduce_quarter_turn_path(setup_path + greedy_bulk.path),
            finisher,
            puzzle.generators,
            decomposition,
        )
        replay_verify(puzzle, state_id, state, greedy_finished.path)

        clusters = initial_clusters.copy()
        current_cost = int(cluster_costs(clusters).sum())
        bulk_actions: list[int] = []
        bulk_path: list[str] = []
        for _ in range(args.max_bulk_steps):
            state_tensor = torch.from_numpy(clusters[None]).to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _, _ = model(state_tensor)
            model_proposed = logits.topk(top_k, dim=1).indices[0].cpu().numpy()
            proposed = np.concatenate((model_proposed, extra_action_indices))
            action, next_cost = search_table.exact_rerank(clusters, proposed)
            if next_cost >= current_cost:
                break
            clusters = search_table.apply(clusters, action)
            bulk_actions.append(action)
            bulk_path.extend(search_table.paths[action])
            current_cost = next_cost
            if current_cost == 0:
                break

        reduced_bulk_path = reduce_quarter_turn_path(bulk_path)
        bulk_state = apply_path(normalized, puzzle.generators, reduced_bulk_path)
        replay_clusters = np.asarray(
            state_cluster_permutations(bulk_state, puzzle.solved_state, decomposition),
            dtype=np.uint8,
        )
        if not np.array_equal(replay_clusters, clusters):
            raise AssertionError("cluster-only policy rollout disagrees with full replay")
        policy_finished = finish_with_inserted_three_cycles(
            clusters,
            reduce_quarter_turn_path(setup_path + reduced_bulk_path),
            finisher,
            puzzle.generators,
            decomposition,
        )
        replay_verify(puzzle, state_id, state, policy_finished.path)

        hybrid_greedy = greedy_macro_reduce(
            clusters,
            classical_bulk_macros,
            max_macros=args.max_bulk_steps,
        )
        hybrid_bulk_path = reduce_quarter_turn_path(
            reduced_bulk_path + hybrid_greedy.path
        )
        hybrid_state = apply_path(normalized, puzzle.generators, hybrid_bulk_path)
        hybrid_clusters = state_cluster_permutations(
            hybrid_state,
            puzzle.solved_state,
            decomposition,
        )
        hybrid_finished = finish_with_inserted_three_cycles(
            hybrid_clusters,
            reduce_quarter_turn_path(setup_path + hybrid_bulk_path),
            finisher,
            puzzle.generators,
            decomposition,
        )
        replay_verify(puzzle, state_id, state, hybrid_finished.path)
        rows.append(
            {
                "baseline_moves": len(baseline_finished.path),
                "bulk_action_indices": bulk_actions,
                "bulk_actions": len(bulk_actions),
                "bulk_moves": len(reduced_bulk_path),
                "elapsed_seconds": round(time.perf_counter() - state_started, 4),
                "final_residual": current_cost,
                "greedy_bulk_actions": greedy_bulk.macro_count,
                "greedy_final_residual": greedy_bulk.final_cost,
                "greedy_moves": len(greedy_finished.path),
                "hybrid_greedy_actions": hybrid_greedy.macro_count,
                "hybrid_moves": len(hybrid_finished.path),
                "initial_residual": normalized_report.unrestricted_three_cycles,
                "move_delta": len(policy_finished.path) - len(baseline_finished.path),
                "move_delta_vs_greedy": len(policy_finished.path) - len(greedy_finished.path),
                "hybrid_delta_vs_greedy": len(hybrid_finished.path) - len(greedy_finished.path),
                "novel_action_indices_used": [
                    action
                    for action in bulk_actions
                    if action >= model_table.action_count
                ],
                "novel_actions_used": sum(
                    action >= model_table.action_count
                    for action in bulk_actions
                ),
                "policy_moves": len(policy_finished.path),
                "state_id": state_id,
            }
        )
        solution_rows.append(
            {
                "initial_state_id": state_id,
                "path": ".".join(
                    min(
                        (policy_finished.path, hybrid_finished.path, greedy_finished.path),
                        key=lambda path: (len(path), path),
                    )
                ),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "solutions.csv.tmp", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        writer.writerows(solution_rows)
    (args.out_dir / "solutions.csv.tmp").replace(args.out_dir / "solutions.csv")
    report = {
        "action_digest": search_table.digest,
        "augmented_action_library": (
            str(args.augmented_action_library)
            if args.augmented_action_library is not None
            else None
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "extra_action_count": int(extra_action_indices.size),
        "mean_baseline_moves": statistics.fmean(int(row["baseline_moves"]) for row in rows),
        "mean_move_delta": statistics.fmean(int(row["move_delta"]) for row in rows),
        "mean_move_delta_vs_greedy": statistics.fmean(
            int(row["move_delta_vs_greedy"])
            for row in rows
        ),
        "mean_greedy_moves": statistics.fmean(int(row["greedy_moves"]) for row in rows),
        "mean_hybrid_moves": statistics.fmean(int(row["hybrid_moves"]) for row in rows),
        "mean_hybrid_delta_vs_greedy": statistics.fmean(
            int(row["hybrid_delta_vs_greedy"])
            for row in rows
        ),
        "mean_policy_moves": statistics.fmean(int(row["policy_moves"]) for row in rows),
        "model_action_count": model_table.action_count,
        "model_action_digest": model_table.digest,
        "replay_verified": True,
        "rows": rows,
        "states": len(rows),
        "top_k": top_k,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
