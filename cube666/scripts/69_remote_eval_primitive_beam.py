"""Matched autonomous beam gate for clean primitive-cost cube666 models."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


CLUSTER_COUNT = 6
CLUSTER_SIZE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--group-ids", type=Path, required=True)
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--action-costs", type=Path, required=True)
    parser.add_argument("--append-effects", type=Path)
    parser.add_argument("--append-costs", type=Path)
    parser.add_argument("--extra-actions", type=Path)
    parser.add_argument("--train-module", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--action-digest", required=True)
    parser.add_argument("--eval-puzzles", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--branch-width", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument("--proposal-chunk-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=1024)
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cube666_effect_eval_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load training module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_model(
    module: ModuleType,
    checkpoint_path: Path,
    action_count: int,
    device: torch.device,
) -> torch.nn.Module:
    model = module.MacroEffectPolicyValueNet(
        module.Config(action_count=action_count)
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def inverse_action_indices(effects: np.ndarray) -> np.ndarray:
    by_effect = {effect.tobytes(): index for index, effect in enumerate(effects)}
    identity = np.broadcast_to(
        np.arange(CLUSTER_SIZE, dtype=np.uint8),
        effects.shape,
    )
    inverse = np.empty_like(effects)
    np.put_along_axis(inverse, effects, identity, axis=-1)
    result = np.empty(len(effects), dtype=np.int32)
    for action, effect in enumerate(inverse):
        try:
            result[action] = by_effect[effect.tobytes()]
        except KeyError as exc:
            raise ValueError(f"action {action} lacks an inverse") from exc
    return result


def reconstruct_teacher_paths(
    states: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    effects: np.ndarray,
    action_costs: np.ndarray,
) -> tuple[dict[int, tuple[int, ...]], dict[str, int]]:
    identity = np.broadcast_to(
        np.arange(CLUSTER_SIZE, dtype=np.uint8),
        (CLUSTER_COUNT, CLUSTER_SIZE),
    )
    by_pid_state = {
        (int(groups[index]), states[index].tobytes()): index
        for index in range(len(states))
    }
    memo: dict[int, tuple[int, ...] | None] = {}

    def solve(index: int) -> tuple[int, ...] | None:
        cached = memo.get(index, ...)
        if cached is not ...:
            return cached
        memo[index] = None
        state = states[index]
        pid = int(groups[index])
        limit = float(values[index]) + 1.0e-5
        best: tuple[int, tuple[int, ...]] | None = None
        for action in labels[index, : int(counts[index])]:
            action = int(action)
            cost = int(action_costs[action])
            child = np.take_along_axis(state, effects[action], axis=-1)
            if np.array_equal(child, identity):
                suffix: tuple[int, ...] = ()
            else:
                child_index = by_pid_state.get((pid, child.tobytes()))
                if child_index is None or values[child_index] >= values[index] - 1.0e-5:
                    continue
                child_path = solve(child_index)
                if child_path is None:
                    continue
                suffix = child_path
            path = (action,) + suffix
            total = sum(int(action_costs[item]) for item in path)
            if total > limit:
                continue
            candidate = (total, path)
            if best is None or candidate < best:
                best = candidate
        memo[index] = None if best is None else best[1]
        return memo[index]

    paths: dict[int, tuple[int, ...]] = {}
    replay_failures = 0
    for index in range(len(states)):
        path = solve(index)
        if path is None:
            continue
        replay = states[index].copy()
        for action in path:
            replay = np.take_along_axis(replay, effects[action], axis=-1)
        if not np.array_equal(replay, identity):
            replay_failures += 1
            continue
        paths[index] = path
    return paths, {
        "covered_rows": len(paths),
        "replay_failures": replay_failures,
        "rows": len(states),
    }


@torch.no_grad()
def policy_topk(
    module: ModuleType,
    model: torch.nn.Module,
    states: np.ndarray,
    last_actions: np.ndarray,
    inverse_indices: np.ndarray,
    action_effects: torch.Tensor,
    extra_actions: torch.Tensor,
    *,
    branch_width: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    state_tensor = torch.from_numpy(states).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _, _ = model(state_tensor)
    best_scores = torch.empty((len(states), 0), device=device)
    best_actions = torch.empty((len(states), 0), dtype=torch.long, device=device)
    forbidden = np.full(len(states), -1, dtype=np.int32)
    has_last = last_actions >= 0
    forbidden[has_last] = inverse_indices[last_actions[has_last]]
    for start in range(0, len(action_effects), chunk_size):
        stop = min(start + chunk_size, len(action_effects))
        candidates = action_effects[start:stop][None].expand(len(states), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        rows_np = np.flatnonzero((forbidden >= start) & (forbidden < stop))
        if len(rows_np):
            rows = torch.from_numpy(rows_np).to(device)
            columns = torch.from_numpy(forbidden[rows_np] - start).to(device)
            scores[rows, columns] = -torch.inf
        action_ids = torch.arange(start, stop, device=device)[None].expand(len(states), -1)
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, action_ids), dim=1)
        keep = min(branch_width, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    if len(extra_actions):
        extra_candidates = action_effects[extra_actions][None].expand(
            len(states), -1, -1, -1
        )
        extra_scores = module.score_candidates(logits, extra_candidates)
        if np.any(has_last):
            extra_np = extra_actions.cpu().numpy()
            matches = forbidden[:, None] == extra_np[None, :]
            rows_np, columns_np = np.nonzero(matches)
            if len(rows_np):
                rows = torch.from_numpy(rows_np).to(device)
                columns = torch.from_numpy(columns_np).to(device)
                extra_scores[rows, columns] = -torch.inf
        best_actions = torch.cat(
            (best_actions, extra_actions[None].expand(len(states), -1)),
            dim=1,
        )
        best_scores = torch.cat((best_scores, extra_scores), dim=1)
    return best_actions.cpu().numpy().astype(np.int32), best_scores.cpu().numpy()


@torch.no_grad()
def predict_values(
    model: torch.nn.Module,
    states: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        state_tensor = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, total, clusters = model(state_tensor)
        total_moves = total.float() * 72.0
        cluster_moves = clusters.float().sum(dim=1) * 12.0
        values = ((total_moves + cluster_moves) * 0.5).clamp_min(0.0)
        batches.append(values.cpu().numpy())
    return np.concatenate(batches)


@torch.no_grad()
def beam_search(
    module: ModuleType,
    model: torch.nn.Module,
    initial_state: np.ndarray,
    effects_np: np.ndarray,
    effects: torch.Tensor,
    action_costs: np.ndarray,
    inverse_indices: np.ndarray,
    extra_actions: torch.Tensor,
    *,
    beam_width: int,
    branch_width: int,
    max_steps: int,
    policy_nll_weight: float,
    proposal_chunk_size: int,
    inference_batch_size: int,
    device: torch.device,
) -> dict[str, int | float | bool | list[int]]:
    identity = np.broadcast_to(
        np.arange(CLUSTER_SIZE, dtype=np.uint8),
        (CLUSTER_COUNT, CLUSTER_SIZE),
    )
    states = initial_state[None].copy()
    paths: list[tuple[int, ...]] = [()]
    cumulative_cost = np.zeros(1, dtype=np.int32)
    cumulative_nll = np.zeros(1, dtype=np.float32)
    last_actions = np.full(1, -1, dtype=np.int32)
    best_g = {initial_state.tobytes(): 0}
    best_solution: tuple[int, float, tuple[int, ...]] | None = None
    generated = 0
    started = time.perf_counter()
    for step in range(1, max_steps + 1):
        proposed, policy_scores = policy_topk(
            module,
            model,
            states,
            last_actions,
            inverse_indices,
            effects,
            extra_actions,
            branch_width=branch_width,
            chunk_size=proposal_chunk_size,
            device=device,
        )
        actions_per_parent = proposed.shape[1]
        parent_indices = np.repeat(np.arange(len(states)), actions_per_parent)
        flat_actions = proposed.reshape(-1)
        parent_states = states[parent_indices]
        children = np.take_along_axis(
            parent_states,
            effects_np[flat_actions],
            axis=-1,
        )
        child_cost = cumulative_cost[parent_indices] + action_costs[flat_actions]
        child_nll = cumulative_nll[parent_indices] - policy_scores.reshape(-1)
        generated += len(children)
        solved = np.all(children == identity, axis=(1, 2))
        for position in np.flatnonzero(solved):
            parent = int(parent_indices[position])
            candidate = (
                int(child_cost[position]),
                float(child_nll[position]),
                paths[parent] + (int(flat_actions[position]),),
            )
            if best_solution is None or candidate < best_solution:
                best_solution = candidate
        predicted = predict_values(
            model,
            children,
            device=device,
            batch_size=inference_batch_size,
        )
        ranks = child_cost + predicted + policy_nll_weight * child_nll
        order = np.lexsort((child_nll, child_cost, ranks))
        next_states: list[np.ndarray] = []
        next_paths: list[tuple[int, ...]] = []
        next_costs: list[int] = []
        next_nll: list[float] = []
        next_last: list[int] = []
        for position in order:
            if solved[position]:
                continue
            cost = int(child_cost[position])
            if best_solution is not None and cost >= best_solution[0]:
                continue
            key = children[position].tobytes()
            old_cost = best_g.get(key)
            if old_cost is not None and old_cost <= cost:
                continue
            best_g[key] = cost
            parent = int(parent_indices[position])
            next_states.append(children[position])
            next_paths.append(paths[parent] + (int(flat_actions[position]),))
            next_costs.append(cost)
            next_nll.append(float(child_nll[position]))
            next_last.append(int(flat_actions[position]))
            if len(next_states) == beam_width:
                break
        if not next_states:
            break
        states = np.stack(next_states)
        paths = next_paths
        cumulative_cost = np.asarray(next_costs, dtype=np.int32)
        cumulative_nll = np.asarray(next_nll, dtype=np.float32)
        last_actions = np.asarray(next_last, dtype=np.int32)
    if best_solution is None:
        return {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "generated": generated,
            "path": [],
            "primitive_cost": -1,
            "solved": False,
        }
    replay = initial_state.copy()
    for action in best_solution[2]:
        replay = np.take_along_axis(replay, effects_np[action], axis=-1)
    if not np.array_equal(replay, identity):
        raise AssertionError("beam path failed exact residual replay")
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "generated": generated,
        "macro_steps": len(best_solution[2]),
        "path": list(best_solution[2]),
        "primitive_cost": best_solution[0],
        "solved": True,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    module = load_module(args.train_module)
    with np.load(args.teacher, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        labels = payload["teacher_actions"].astype(np.int32, copy=False)
        counts = payload["teacher_action_counts"].astype(np.int16, copy=False)
        values = payload["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.group_ids, allow_pickle=False)
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(np.uint8, copy=False)
    action_costs = np.load(args.action_costs, allow_pickle=False).astype(np.int32, copy=False)
    if (args.append_effects is None) != (args.append_costs is None):
        raise ValueError("append-effects and append-costs must be supplied together")
    if args.append_effects is not None:
        effects_np = np.concatenate(
            (
                effects_np,
                np.load(args.append_effects, allow_pickle=False).astype(np.uint8, copy=False),
            )
        )
        action_costs = np.concatenate(
            (
                action_costs,
                np.load(args.append_costs, allow_pickle=False).astype(np.int32, copy=False),
            )
        )
    if effects_np.shape != (len(action_costs), CLUSTER_COUNT, CLUSTER_SIZE):
        raise ValueError("action effect/cost shapes disagree")
    inverse_indices = inverse_action_indices(effects_np)
    extra_actions_np = (
        np.load(args.extra_actions, allow_pickle=False).astype(np.int64, copy=False)
        if args.extra_actions is not None
        else np.empty(0, dtype=np.int64)
    )
    if extra_actions_np.ndim != 1 or np.any(extra_actions_np < 0) or np.any(
        extra_actions_np >= len(effects_np)
    ):
        raise ValueError("extra actions are outside the search table")
    teacher_paths, teacher_audit = reconstruct_teacher_paths(
        states,
        labels,
        counts,
        values,
        groups,
        effects_np,
        action_costs,
    )
    heldout = [index for index in teacher_paths if int(groups[index]) % 10 == 0]
    best_by_pid: dict[int, int] = {}
    for index in heldout:
        pid = int(groups[index])
        current = best_by_pid.get(pid)
        if current is None or (values[index], -index) > (values[current], -current):
            best_by_pid[pid] = index
    selected = sorted(
        best_by_pid.values(),
        key=lambda index: (-float(values[index]), int(groups[index]), index),
    )[: args.eval_puzzles]
    if len(selected) < args.eval_puzzles:
        raise RuntimeError("not enough held-out states have replayable teacher paths")

    effects = torch.from_numpy(effects_np).to(device)
    extra_actions = torch.from_numpy(extra_actions_np).to(device)
    models = {
        "control_mixed_units": load_model(
            module,
            args.control_checkpoint,
            len(effects_np),
            device,
        ),
        "candidate_primitive": load_model(
            module,
            args.candidate_checkpoint,
            len(effects_np),
            device,
        ),
    }
    started = time.perf_counter()
    results: dict[str, list[dict[str, object]]] = {name: [] for name in models}
    for name, model in models.items():
        for index in selected:
            teacher_path = teacher_paths[index]
            teacher_cost = sum(int(action_costs[action]) for action in teacher_path)
            result = beam_search(
                module,
                model,
                states[index],
                effects_np,
                effects,
                action_costs,
                inverse_indices,
                extra_actions,
                beam_width=args.beam_width,
                branch_width=args.branch_width,
                max_steps=args.max_steps,
                policy_nll_weight=args.policy_nll_weight,
                proposal_chunk_size=args.proposal_chunk_size,
                inference_batch_size=args.inference_batch_size,
                device=device,
            )
            result.update(
                {
                    "pid": int(groups[index]),
                    "row": index,
                    "teacher_macro_steps": len(teacher_path),
                    "teacher_primitive_cost": teacher_cost,
                    "teacher_value_target": float(values[index]),
                }
            )
            if bool(result["solved"]):
                result["improvement_vs_teacher"] = (
                    int(result["primitive_cost"]) - teacher_cost
                )
            results[name].append(result)
            print(json.dumps({"model": name, **result}, sort_keys=True), flush=True)

    summaries = {}
    for name, rows in results.items():
        solved_rows = [row for row in rows if bool(row["solved"])]
        summaries[name] = {
            "improvement_sum_vs_teacher_on_solved": sum(
                int(row["improvement_vs_teacher"]) for row in solved_rows
            ),
            "mean_elapsed_seconds": float(
                np.mean([float(row["elapsed_seconds"]) for row in rows])
            ),
            "solved": len(solved_rows),
            "trials": len(rows),
        }
    report = {
        "action_count": len(effects_np),
        "action_digest": args.action_digest,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "extra_actions": len(extra_actions_np),
        "max_steps": args.max_steps,
        "policy_nll_weight": args.policy_nll_weight,
        "results": results,
        "selected_pids": [int(groups[index]) for index in selected],
        "summaries": summaries,
        "teacher_audit": teacher_audit,
        "value_unit": "primitive_moves",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
