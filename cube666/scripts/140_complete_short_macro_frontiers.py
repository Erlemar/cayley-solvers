"""Replay-verify bounded neural completions for harvested cube666 beam states.

Only solved candidates receive return labels.  Unsolved searches are recorded for
resume and diagnostics but are deliberately not converted into negative labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument(
        "--direct-action-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--beam", type=int, default=512)
    parser.add_argument("--direct-branch", type=int, default=512)
    parser.add_argument("--direct-root-branch", type=int, default=0)
    parser.add_argument("--maximum-completion-depth", type=int, default=4)
    parser.add_argument("--path-cost-weight", type=float, default=0.02)
    parser.add_argument("--policy-nll-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--value-prebeam-multiplier", type=int, default=32)
    parser.add_argument(
        "--per-root-limit",
        type=int,
        default=0,
        help="complete only the first N retained candidates per root; zero uses all",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=140667)
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cube666_frontier_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def persist(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def selected_frontier_indices(
    root_source_indices: np.ndarray, per_root_limit: int, limit: int
) -> np.ndarray:
    selected: list[int] = []
    counts: dict[int, int] = {}
    for index, root_value in enumerate(root_source_indices):
        root = int(root_value)
        count = counts.get(root, 0)
        if per_root_limit and count >= per_root_limit:
            continue
        counts[root] = count + 1
        selected.append(index)
        if limit and len(selected) >= limit:
            break
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = args.out_dir / "attempts.jsonl"
    report_path = args.out_dir / "report.json"
    solved_path = args.out_dir / "solved_completions.npz"

    evaluator = load_module(args.eval_script)
    factorized_model, _ = evaluator.load_macro_factorized_value_checkpoint(
        str(args.critic), device="cuda"
    )
    model = evaluator.FactorizedValueOnlyAdapter(factorized_model).eval()
    direct_models = [
        evaluator.load_macro_action_policy_checkpoint(str(path), device="cuda")[0]
        for path in args.direct_action_checkpoint
    ]

    effects_np = np.load(
        args.dataset_dir / "action_effects.npy", allow_pickle=False
    ).astype(np.uint8, copy=False)
    costs_np = np.load(
        args.dataset_dir / "action_costs.npy", allow_pickle=False
    ).astype(np.int32, copy=False)
    inverse_np = np.load(
        args.dataset_dir / "inverse_actions.npy", allow_pickle=False
    ).astype(np.int64, copy=False)
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).float().cuda()
    inverse_actions = torch.from_numpy(inverse_np).cuda()
    with np.load(args.frontier, allow_pickle=False) as payload:
        frontier = {name: payload[name] for name in payload.files}
    required = {
        "states",
        "forward_costs",
        "cumulative_nll",
        "paths",
        "root_source_indices",
        "teacher_upper_bounds",
    }
    missing = required.difference(frontier)
    if missing:
        raise ValueError(f"frontier is missing arrays: {sorted(missing)}")
    candidate_indices = selected_frontier_indices(
        frontier["root_source_indices"], args.per_root_limit, args.limit
    )
    if not len(candidate_indices):
        raise ValueError("frontier selection is empty")

    attempts: dict[int, dict[str, object]] = {}
    if attempt_path.exists():
        for line in attempt_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                attempts[int(row["frontier_index"])] = row

    rng = np.random.default_rng(args.seed)
    zobrist = torch.from_numpy(
        rng.integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=(144, 24),
            dtype=np.int64,
        )
    ).cuda()
    identity = np.broadcast_to(
        np.arange(24, dtype=np.uint8), frontier["states"].shape[1:]
    )
    started = time.perf_counter()
    completed_this_run = 0
    with attempt_path.open("a", encoding="utf-8") as attempt_handle:
        for ordinal, frontier_index_value in enumerate(candidate_indices, start=1):
            frontier_index = int(frontier_index_value)
            if frontier_index in attempts:
                continue
            state = frontier["states"][frontier_index]
            item_started = time.perf_counter()
            continuation, stats = evaluator.solve(
                None,
                model,
                [],
                direct_models,
                None,
                state,
                effects,
                costs,
                inverse_actions,
                beam_width=args.beam,
                branch=0,
                direct_branch=args.direct_branch,
                direct_early_branch=0,
                direct_early_branch_depths=0,
                direct_root_branch=args.direct_root_branch,
                direct_cost_stratified_branch=128,
                direct_cost_maximum=6,
                direct_cost_stratified_depths=3,
                direct_score_mode="log_probability",
                ensemble_direct_actions=False,
                maximum_depth=args.maximum_completion_depth,
                path_cost_weight=args.path_cost_weight,
                value_weight=args.value_weight,
                value_prebeam_multiplier=args.value_prebeam_multiplier,
                policy_nll_weight=args.policy_nll_weight,
                policy_step_nll_cap=0.0,
                policy_tail_slope=0.01,
                prebeam_multiplier=1,
                cycle_cost_weight=0.0,
                cycle_proposal_branch=0,
                cycle_proposal_cost_weight=0.0,
                cycle_proposal_score_scale=0.2,
                cycle_proposal_depths=0,
                backup_prebeam_multiplier=1,
                backup_branch=0,
                backup_state_batch_size=128,
                backup_weight=0.0,
                action_chunk_size=1024,
                inference_batch_size=16384,
                zobrist=zobrist,
                oracle_actions=None,
                one_macro_endgame=True,
                exact_root_two_macro=True,
                continue_after_solution=False,
            )
            replay = state.copy()
            if continuation is not None:
                for action in continuation:
                    replay = np.take_along_axis(replay, effects_np[action], axis=-1)
            replay_verified = continuation is not None and np.array_equal(
                replay, identity
            )
            if continuation is not None and not replay_verified:
                raise AssertionError(
                    f"frontier {frontier_index}: completion failed exact replay"
                )
            completion_cost = (
                int(costs_np[np.asarray(continuation, dtype=np.int64)].sum())
                if continuation is not None
                else None
            )
            if completion_cost is not None and completion_cost != int(
                stats["primitive_cost"]
            ):
                raise AssertionError("reported and replayed completion costs disagree")
            prefix_cost = int(frontier["forward_costs"][frontier_index])
            teacher_upper_bound = float(
                frontier["teacher_upper_bounds"][frontier_index]
            )
            row: dict[str, object] = {
                "frontier_index": frontier_index,
                "ordinal": ordinal,
                "root_source_index": int(
                    frontier["root_source_indices"][frontier_index]
                ),
                "prefix_cost": prefix_cost,
                "continuation": continuation,
                "completion_cost": completion_cost,
                "total_cost": (
                    prefix_cost + completion_cost
                    if completion_cost is not None
                    else None
                ),
                "teacher_upper_bound": teacher_upper_bound,
                "improvement": (
                    teacher_upper_bound - prefix_cost - completion_cost
                    if completion_cost is not None
                    else None
                ),
                "replay_verified": replay_verified,
                "solved": continuation is not None,
                "elapsed_seconds": time.perf_counter() - item_started,
                "search_stats": stats,
            }
            attempt_handle.write(json.dumps(row, sort_keys=True) + "\n")
            attempt_handle.flush()
            attempts[frontier_index] = row
            completed_this_run += 1
            if completed_this_run % args.checkpoint_every == 0:
                solved_count = sum(bool(item["solved"]) for item in attempts.values())
                print(
                    json.dumps(
                        {
                            "attempted": len(attempts),
                            "selected": len(candidate_indices),
                            "solved": solved_count,
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    selected_attempts = [
        attempts[int(index)]
        for index in candidate_indices
        if int(index) in attempts
    ]
    solved_rows = [row for row in selected_attempts if bool(row["solved"])]
    root_count = int(
        np.unique(frontier["root_source_indices"][candidate_indices]).size
    )
    solved_roots = len({int(row["root_source_index"]) for row in solved_rows})
    improvements = np.asarray(
        [float(row["improvement"]) for row in solved_rows], dtype=np.float32
    )
    report = {
        "attempted": len(selected_attempts),
        "selected": len(candidate_indices),
        "roots": root_count,
        "solved": len(solved_rows),
        "solved_fraction": len(solved_rows) / max(1, len(selected_attempts)),
        "solved_roots": solved_roots,
        "solved_root_fraction": solved_roots / max(1, root_count),
        "strict_improvements": int((improvements > 0).sum()),
        "equal_teacher": int((improvements == 0).sum()),
        "improvement_mean": (
            float(improvements.mean()) if len(improvements) else None
        ),
        "improvement_median": (
            float(np.median(improvements)) if len(improvements) else None
        ),
        "beam": args.beam,
        "maximum_completion_depth": args.maximum_completion_depth,
        "per_root_limit": args.per_root_limit,
        "value_weight": args.value_weight,
        "value_prebeam_multiplier": args.value_prebeam_multiplier,
        "frontier": str(args.frontier),
        "solved_completions": str(solved_path),
    }
    if solved_rows:
        max_depth = args.maximum_completion_depth
        completion_actions = np.full(
            (len(solved_rows), max_depth), -1, dtype=np.int32
        )
        for output_index, row in enumerate(solved_rows):
            actions = np.asarray(row["continuation"], dtype=np.int32)
            completion_actions[output_index, : len(actions)] = actions
        solved_indices = np.asarray(
            [int(row["frontier_index"]) for row in solved_rows], dtype=np.int64
        )
        np.savez_compressed(
            solved_path,
            frontier_indices=solved_indices,
            root_source_indices=frontier["root_source_indices"][solved_indices],
            states=frontier["states"][solved_indices],
            prefix_paths=frontier["paths"][solved_indices],
            prefix_costs=frontier["forward_costs"][solved_indices],
            cumulative_nll=frontier["cumulative_nll"][solved_indices],
            completion_actions=completion_actions,
            completion_lengths=np.asarray(
                [len(row["continuation"]) for row in solved_rows], dtype=np.int16
            ),
            completion_costs=np.asarray(
                [int(row["completion_cost"]) for row in solved_rows], dtype=np.int32
            ),
            total_costs=np.asarray(
                [int(row["total_cost"]) for row in solved_rows], dtype=np.int32
            ),
            teacher_upper_bounds=frontier["teacher_upper_bounds"][solved_indices],
            improvements=improvements,
        )
    persist(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
