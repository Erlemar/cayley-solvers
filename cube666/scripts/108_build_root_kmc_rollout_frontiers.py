"""Build diverse one-step macro children at a puzzle's real normalized root.

The resulting full states are suitable for exact KMC completion queries.  A
no-op frontier is included as a positive control, and every report contains the
real corner/parity setup path needed to materialize an end-to-end solution.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from cube666.macro_dual_policy import load_macro_dual_policy_checkpoint  # noqa: E402
from cube666.macro_factorized_value import (  # noqa: E402
    load_macro_factorized_value_checkpoint,
)
from cube666.macros import reduce_quarter_turn_path, state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument(
        "--base-actions",
        default="",
        help=(
            "underscore- or comma-separated macro action IDs already present in "
            "the prefix; proposed actions are appended after them"
        ),
    )
    parser.add_argument("--shortlist", type=int, default=4096)
    parser.add_argument("--value-candidates", type=int, default=8)
    parser.add_argument("--policy-candidates", type=int, default=4)
    parser.add_argument(
        "--random-candidates",
        type=int,
        default=0,
        help="add deterministic random short-macro controls to the learned shortlist",
    )
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument(
        "--random-max-cost",
        type=int,
        default=14,
        help="maximum primitive cost for random control macros",
    )
    parser.add_argument(
        "--skip-parity-repair",
        action="store_true",
        help="stop at the exact-corner boundary used by production KMC",
    )
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def predict_values(
    model: torch.nn.Module, states: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, clusters = model(batch)
        moves = 0.5 * (
            total.float() * model.config.total_scale
            + clusters.float().sum(dim=1) * model.config.cluster_scale
        )
        output.append(moves.clamp_min(0).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(
        args.shortlist,
        args.value_candidates,
        args.policy_candidates,
        args.inference_batch_size,
    ) <= 0:
        raise ValueError("candidate and batch sizes must be positive")
    if args.random_candidates < 0 or args.random_max_cost <= 0:
        raise ValueError("random candidate count/cost must be non-negative/positive")

    device = torch.device("cuda")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    policy, _ = load_macro_dual_policy_checkpoint(args.checkpoint, device)
    value_model, _ = load_macro_factorized_value_checkpoint(
        args.value_checkpoint, device
    )
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.int32, copy=False
    )
    library = json.loads(
        (args.teacher_dir / "action_library.json").read_text(encoding="utf-8")
    )
    paths = library["paths"]
    if len(paths) != len(effects) or len(costs) != len(effects):
        raise ValueError("macro action artifacts have inconsistent lengths")
    base_action_ids = [
        int(token)
        for token in args.base_actions.replace("_", ",").split(",")
        if token.strip()
    ]
    if any(action < 0 or action >= len(paths) for action in base_action_ids):
        raise IndexError("base action is outside the macro library")
    base_macro_path = tuple(
        move for action in base_action_ids for move in paths[action]
    )
    base_macro_cost = int(sum(int(costs[action]) for action in base_action_ids))
    query_prefix = (
        f"pid{args.pid:04d}_root"
        if not base_action_ids
        else f"pid{args.pid:04d}_d{len(base_action_ids)}b"
        + "x".join(f"{action:05d}" for action in base_action_ids)
    )

    states_by_id = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    if args.pid not in states_by_id:
        raise KeyError(f"PID {args.pid} is absent from test.csv")
    initial_state = states_by_id[args.pid]
    corner_solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    corner_path = corner_solver.solve(initial_state, puzzle.solved_state)
    corner_state = apply_path(initial_state, puzzle.generators, corner_path)
    parity_path = (
        ()
        if args.skip_parity_repair
        else parity_repair_path(
            residual_report(
                corner_state, puzzle.solved_state, decomposition
            ).parity_vector,
            decomposition,
        )
    )
    setup_path = reduce_quarter_turn_path(corner_path + parity_path)
    normalized = apply_path(initial_state, puzzle.generators, setup_path)
    normalized = apply_path(normalized, puzzle.generators, base_macro_path)
    normalized_report = residual_report(normalized, puzzle.solved_state, decomposition)
    if not normalized_report.all_even and not args.skip_parity_repair:
        raise AssertionError("corner/parity setup did not produce an even state")
    initial_residual = (
        -1
        if normalized_report.unrestricted_three_cycles is None
        else int(normalized_report.unrestricted_three_cycles)
    )
    state = np.asarray(
        state_cluster_permutations(normalized, puzzle.solved_state, decomposition),
        dtype=np.uint8,
    )

    key_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(effects), args.inference_batch_size):
            stop = min(start + args.inference_batch_size, len(effects))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                key_parts.append(
                    policy.encode_actions(
                        torch.from_numpy(effects[start:stop]).to(device),
                        torch.from_numpy(costs[start:stop]).to(device),
                    )
                )
        action_keys = torch.cat(key_parts)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            query = policy.encode_states(torch.from_numpy(state[None]).to(device))
        retrieved = (
            (query @ action_keys.transpose(0, 1))
            .topk(min(args.shortlist, len(effects)), dim=1)
            .indices[0]
            .cpu()
            .numpy()
        )

    children = np.take_along_axis(
        np.broadcast_to(state, (len(retrieved),) + state.shape),
        effects[retrieved],
        axis=-1,
    )
    predicted = predict_values(
        value_model, children, args.inference_batch_size, device
    )
    q_scores = costs[retrieved] + predicted
    value_actions = retrieved[np.argsort(q_scores)[: args.value_candidates]]
    policy_actions = retrieved[: args.policy_candidates]
    learned_actions = np.unique(np.concatenate((value_actions, policy_actions))).astype(
        np.int32
    )
    random_actions = np.empty(0, dtype=np.int32)
    if args.random_candidates:
        eligible = np.flatnonzero(costs <= args.random_max_cost).astype(np.int32)
        eligible = eligible[~np.isin(eligible, learned_actions)]
        if len(eligible) < args.random_candidates:
            raise ValueError(
                f"only {len(eligible)} random controls satisfy cost <= "
                f"{args.random_max_cost}"
            )
        seed = (
            int(args.random_seed)
            + 1_000_003 * int(args.pid)
            + 97 * sum((index + 1) * action for index, action in enumerate(base_action_ids))
        )
        rng = np.random.default_rng(seed)
        random_actions = np.sort(
            rng.choice(eligible, size=args.random_candidates, replace=False)
        ).astype(np.int32)
    selected = np.unique(
        np.concatenate((learned_actions, random_actions))
    ).astype(np.int32)
    selection_sources: dict[int, list[str]] = {int(action): [] for action in selected}
    for action in value_actions:
        selection_sources[int(action)].append("value")
    for action in policy_actions:
        selection_sources[int(action)].append("policy")
    for action in random_actions:
        selection_sources[int(action)].append("random_control")

    full_frontiers: list[np.ndarray] = [np.asarray(normalized, dtype=np.uint8)]
    cluster_frontiers: list[np.ndarray] = [state]
    rows: list[dict[str, object]] = [
        {
            "action_id": -1,
            "frontier_residual": initial_residual,
            "macro_path": list(base_macro_path),
            "macro_primitive_moves": base_macro_cost,
            "pid": args.pid,
            "predicted_child_value": float(
                predict_values(
                    value_model, state[None], args.inference_batch_size, device
                )[0]
            ),
            "query_id": f"{query_prefix}_noop",
            "replay_verified": True,
            "selection_sources": ["noop"],
        }
    ]
    selected_children = np.take_along_axis(
        np.broadcast_to(state, (len(selected),) + state.shape),
        effects[selected],
        axis=-1,
    )
    selected_predicted = predict_values(
        value_model, selected_children, args.inference_batch_size, device
    )
    predicted_by_action = {
        int(action): float(value)
        for action, value in zip(selected, selected_predicted, strict=True)
    }
    for action in selected:
        child_macro_path = tuple(paths[int(action)])
        macro_path = base_macro_path + child_macro_path
        frontier = puzzle.apply_path(normalized, child_macro_path)
        clusters = np.asarray(
            state_cluster_permutations(frontier, puzzle.solved_state, decomposition),
            dtype=np.uint8,
        )
        expected = np.take_along_axis(state, effects[action], axis=-1)
        if not np.array_equal(clusters, expected):
            raise AssertionError(f"action {action}: full/cluster replay mismatch")
        exact = residual_report(frontier, puzzle.solved_state, decomposition)
        if not exact.all_even and not args.skip_parity_repair:
            raise AssertionError(f"action {action}: frontier parity is not even")
        frontier_residual = (
            -1
            if exact.unrestricted_three_cycles is None
            else int(exact.unrestricted_three_cycles)
        )
        query_id = f"{query_prefix}_a{int(action):05d}"
        full_frontiers.append(np.asarray(frontier, dtype=np.uint8))
        cluster_frontiers.append(clusters)
        rows.append(
            {
                "action_id": int(action),
                "frontier_residual": frontier_residual,
                "macro_path": list(macro_path),
                "macro_primitive_moves": base_macro_cost + int(costs[action]),
                "pid": args.pid,
                "predicted_child_value": predicted_by_action[int(action)],
                "query_id": query_id,
                "replay_verified": True,
                "selection_sources": selection_sources[int(action)],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "frontiers.npz",
        full_states=np.stack(full_frontiers),
        cluster_states=np.stack(cluster_frontiers),
        source_pids=np.full(len(rows), args.pid, dtype=np.int32),
        initial_residuals=np.full(len(rows), initial_residual, dtype=np.int16),
        frontier_residuals=np.asarray(
            [int(row["frontier_residual"]) for row in rows], dtype=np.int16
        ),
    )
    report = {
        "candidate_config": {
            "checkpoint": str(args.checkpoint),
            "policy_candidates": args.policy_candidates,
            "random_candidates": args.random_candidates,
            "random_max_cost": args.random_max_cost,
            "random_seed": args.random_seed,
            "shortlist": args.shortlist,
            "value_candidates": args.value_candidates,
            "value_checkpoint": str(args.value_checkpoint),
        },
        "initial_residual": initial_residual,
        "base_action_ids": base_action_ids,
        "base_macro_path": list(base_macro_path),
        "parity_repaired": not args.skip_parity_repair,
        "pid": args.pid,
        "rows": rows,
        "setup_moves": len(setup_path),
        "setup_path": list(setup_path),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "frontiers": len(rows),
                "initial_residual": report["initial_residual"],
                "pid": args.pid,
                "setup_moves": report["setup_moves"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
