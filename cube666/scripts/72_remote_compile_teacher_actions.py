"""Compile singleton cube666 teacher macros into two reusable short macros on GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

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
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--action-digest", required=True)
    parser.add_argument("--maximum-component-cost", type=int, default=14)
    parser.add_argument("--target-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=72666)
    return parser.parse_args()


def hash_effects(
    effects: torch.Tensor,
    zobrist: torch.Tensor,
) -> torch.Tensor:
    flat = effects.reshape(*effects.shape[:-2], CLUSTER_COUNT * CLUSTER_SIZE).long()
    positions = torch.arange(flat.shape[-1], device=flat.device)
    return zobrist[positions, flat].sum(dim=-1)


def compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.take_along_axis(left, right, axis=-1)


def compile_targets(
    effects_np: np.ndarray,
    costs_np: np.ndarray,
    target_actions: np.ndarray,
    *,
    maximum_component_cost: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    raw_candidate_ids_np = np.flatnonzero(
        costs_np <= maximum_component_cost
    ).astype(np.int64)
    raw_candidate_effects = effects_np[raw_candidate_ids_np]
    _, inverse_groups = np.unique(
        raw_candidate_effects.reshape(len(raw_candidate_effects), -1),
        axis=0,
        return_inverse=True,
    )
    best_positions = np.full(int(inverse_groups.max()) + 1, -1, dtype=np.int64)
    for position, group in enumerate(inverse_groups):
        incumbent = int(best_positions[group])
        if incumbent < 0 or (
            int(costs_np[raw_candidate_ids_np[position]]),
            int(raw_candidate_ids_np[position]),
        ) < (
            int(costs_np[raw_candidate_ids_np[incumbent]]),
            int(raw_candidate_ids_np[incumbent]),
        ):
            best_positions[group] = position
    candidate_ids_np = raw_candidate_ids_np[best_positions]
    candidate_effects_np = effects_np[candidate_ids_np]
    candidate_costs_np = costs_np[candidate_ids_np].astype(np.int64, copy=False)
    identity = np.broadcast_to(
        np.arange(CLUSTER_SIZE, dtype=np.uint8),
        candidate_effects_np.shape,
    )
    candidate_inverse_np = np.empty_like(candidate_effects_np)
    np.put_along_axis(
        candidate_inverse_np,
        candidate_effects_np,
        identity,
        axis=-1,
    )

    rng = np.random.default_rng(seed)
    zobrist_np = rng.integers(
        np.iinfo(np.int64).min,
        np.iinfo(np.int64).max,
        size=(CLUSTER_COUNT * CLUSTER_SIZE, CLUSTER_SIZE),
        dtype=np.int64,
    )
    candidates = torch.from_numpy(candidate_effects_np).to(device)
    inverses = torch.from_numpy(candidate_inverse_np).to(device)
    candidate_ids = torch.from_numpy(candidate_ids_np).to(device)
    candidate_costs = torch.from_numpy(candidate_costs_np).to(device)
    zobrist = torch.from_numpy(zobrist_np).to(device)
    candidate_hashes = hash_effects(candidates, zobrist)
    sorted_hashes, order = candidate_hashes.sort()
    sorted_ids = candidate_ids[order]
    sorted_costs = candidate_costs[order]

    words = np.full((len(target_actions), 2), -1, dtype=np.int32)
    compiled_costs = np.empty(len(target_actions), dtype=np.int32)
    collisions = 0
    matches = 0
    for start in range(0, len(target_actions), batch_size):
        stop = min(start + batch_size, len(target_actions))
        target_ids_np = target_actions[start:stop]
        target_effects = torch.from_numpy(effects_np[target_ids_np]).to(device)
        target_expanded = target_effects[:, None].expand(
            -1, len(candidates), -1, -1
        )
        inverse_expanded = inverses[None].expand(
            len(target_effects), -1, -1, -1
        )
        needed = inverse_expanded.gather(-1, target_expanded.long())
        needed_hashes = hash_effects(needed, zobrist)
        positions = torch.searchsorted(sorted_hashes, needed_hashes)
        in_range = positions < len(sorted_hashes)
        safe_positions = positions.clamp_max(len(sorted_hashes) - 1)
        hash_match = in_range & (
            sorted_hashes[safe_positions] == needed_hashes
        )
        right_ids = sorted_ids[safe_positions]
        pair_costs = candidate_costs[None] + sorted_costs[safe_positions]
        pair_costs = pair_costs.masked_fill(~hash_match, torch.iinfo(torch.int64).max)
        best_costs, best_left_positions = pair_costs.min(dim=1)
        best_right_ids = right_ids.gather(1, best_left_positions[:, None]).squeeze(1)
        best_left_ids = candidate_ids[best_left_positions]
        best_costs_np = best_costs.cpu().numpy()
        best_left_np = best_left_ids.cpu().numpy()
        best_right_np = best_right_ids.cpu().numpy()
        for offset, target_action in enumerate(target_ids_np):
            output = start + offset
            baseline_cost = int(costs_np[target_action])
            if best_costs_np[offset] == np.iinfo(np.int64).max:
                words[output, 0] = int(target_action)
                compiled_costs[output] = baseline_cost
                continue
            left = int(best_left_np[offset])
            right = int(best_right_np[offset])
            if not np.array_equal(
                compose(effects_np[left], effects_np[right]),
                effects_np[target_action],
            ):
                collisions += 1
                words[output, 0] = int(target_action)
                compiled_costs[output] = baseline_cost
                continue
            pair_cost = int(costs_np[left]) + int(costs_np[right])
            if pair_cost < baseline_cost:
                words[output] = (left, right)
                compiled_costs[output] = pair_cost
                matches += 1
            else:
                words[output, 0] = int(target_action)
                compiled_costs[output] = baseline_cost
        print(
            json.dumps(
                {
                    "compiled": stop,
                    "improved": matches,
                    "targets": len(target_actions),
                }
            ),
            flush=True,
        )
    return words, compiled_costs, {
        "candidate_actions": len(candidate_ids_np),
        "candidate_actions_before_effect_deduplication": len(raw_candidate_ids_np),
        "hash_collisions": collisions,
        "improved_targets": matches,
    }


def build_compiled_teacher(
    states: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    effects: np.ndarray,
    costs: np.ndarray,
    target_actions: np.ndarray,
    words: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int | float]]:
    word_by_action = {
        int(action): tuple(int(value) for value in word if value >= 0)
        for action, word in zip(target_actions, words, strict=True)
    }
    merged: dict[tuple[int, bytes], dict[str, object]] = {}

    def add_row(pid: int, state: np.ndarray, target: float, action: int) -> None:
        key = (pid, state.tobytes())
        current = merged.get(key)
        if current is None or target < float(current["target"]) - 1.0e-5:
            merged[key] = {
                "actions": {action},
                "state": state.copy(),
                "target": float(target),
            }
        elif abs(target - float(current["target"])) <= 1.0e-5:
            current["actions"].add(action)  # type: ignore[union-attr]

    baseline_cost_sum = 0
    compiled_cost_sum = 0
    intermediate_rows = 0
    for index in range(len(states)):
        valid_actions = labels[index, : int(counts[index])]
        action = min(
            (int(value) for value in valid_actions),
            key=lambda value: (
                sum(costs[item] for item in word_by_action[value]),
                word_by_action[value],
            ),
        )
        word = word_by_action[action]
        original_cost = int(costs[action])
        word_cost = sum(int(costs[item]) for item in word)
        suffix = max(float(values[index]) - original_cost, 0.0)
        compiled_target = suffix + word_cost
        add_row(int(groups[index]), states[index], compiled_target, word[0])
        baseline_cost_sum += original_cost
        compiled_cost_sum += word_cost
        if len(word) == 2:
            child = np.take_along_axis(states[index], effects[word[0]], axis=-1)
            add_row(
                int(groups[index]),
                child,
                suffix + int(costs[word[1]]),
                word[1],
            )
            intermediate_rows += 1

    ordered = sorted(
        merged.items(),
        key=lambda item: (item[0][0], float(item[1]["target"]), item[0][1]),
    )
    maximum_labels = max(len(row["actions"]) for _, row in ordered)  # type: ignore[arg-type]
    output_states = np.stack([row["state"] for _, row in ordered])
    output_labels = np.full((len(ordered), maximum_labels), -1, dtype=np.int32)
    output_counts = np.empty(len(ordered), dtype=np.int16)
    output_values = np.asarray([row["target"] for _, row in ordered], dtype=np.float32)
    output_groups = np.asarray([key[0] for key, _ in ordered], dtype=np.int32)
    for index, (_, row) in enumerate(ordered):
        actions = sorted(row["actions"])  # type: ignore[arg-type]
        output_labels[index, : len(actions)] = actions
        output_counts[index] = len(actions)
    payload = {
        "search_cluster_targets": np.broadcast_to(
            (output_values / 6.0)[:, None],
            (len(output_values), CLUSTER_COUNT),
        ).copy(),
        "search_value_targets": output_values,
        "states": output_states.astype(np.uint8, copy=False),
        "teacher_action_counts": output_counts,
        "teacher_actions": output_labels,
    }
    unique_labels, frequencies = np.unique(output_labels[output_labels >= 0], return_counts=True)
    report = {
        "baseline_labeled_action_cost_sum": baseline_cost_sum,
        "compiled_labeled_action_cost_sum": compiled_cost_sum,
        "compiled_rows": len(output_states),
        "intermediate_rows_before_deduplication": intermediate_rows,
        "label_singletons": int(np.count_nonzero(frequencies == 1)),
        "unique_labels": len(unique_labels),
    }
    return payload, output_groups, report


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.perf_counter()
    device = torch.device("cuda")
    with np.load(args.teacher, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        labels = payload["teacher_actions"].astype(np.int32, copy=False)
        counts = payload["teacher_action_counts"].astype(np.int16, copy=False)
        values = payload["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.group_ids, allow_pickle=False)
    effects = np.load(args.action_effects, allow_pickle=False).astype(np.uint8, copy=False)
    costs = np.load(args.action_costs, allow_pickle=False).astype(np.int32, copy=False)
    if effects.shape != (len(costs), CLUSTER_COUNT, CLUSTER_SIZE):
        raise ValueError("action artifacts disagree")
    target_actions = np.unique(labels[labels >= 0]).astype(np.int32)
    long_targets = target_actions[costs[target_actions] > args.maximum_component_cost]
    long_words, long_compiled_costs, compiler_report = compile_targets(
        effects,
        costs,
        long_targets,
        maximum_component_cost=args.maximum_component_cost,
        batch_size=args.target_batch_size,
        seed=args.seed,
        device=device,
    )
    words = np.full((len(target_actions), 2), -1, dtype=np.int32)
    words[:, 0] = target_actions
    target_positions = {int(action): index for index, action in enumerate(target_actions)}
    for action, word in zip(long_targets, long_words, strict=True):
        words[target_positions[int(action)]] = word
    for action, word in zip(target_actions, words, strict=True):
        replay = effects[int(word[0])]
        if word[1] >= 0:
            replay = compose(replay, effects[int(word[1])])
        if not np.array_equal(replay, effects[int(action)]):
            raise AssertionError(f"compiled action {action} failed replay")

    teacher_payload, compiled_groups, teacher_report = build_compiled_teacher(
        states,
        labels,
        counts,
        values,
        groups,
        effects,
        costs,
        target_actions,
        words,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "teacher.npz", **teacher_payload)
    np.save(args.out_dir / "source_state_ids.npy", compiled_groups)
    np.savez_compressed(
        args.out_dir / "compiled_actions.npz",
        compiled_costs=np.asarray(
            [sum(int(costs[item]) for item in word if item >= 0) for word in words],
            dtype=np.int16,
        ),
        target_actions=target_actions,
        words=words,
    )
    report = {
        "action_count": len(effects),
        "action_digest": args.action_digest,
        "compiler": compiler_report,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "format_version": 1,
        "long_targets": len(long_targets),
        "maximum_component_cost": args.maximum_component_cost,
        "target_actions": len(target_actions),
        "teacher": teacher_report,
        "teacher_digest": hashlib.sha256(
            teacher_payload["states"].tobytes()
            + teacher_payload["teacher_actions"].tobytes()
            + teacher_payload["search_value_targets"].tobytes()
        ).hexdigest(),
        "value_unit": "primitive_moves",
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
