"""Evaluate a macro policy with exact shortlist re-ranking on held-out states."""

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

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    MacroActionTable,
    MacroTeacherDataset,
    cluster_costs,
    load_macro_action_library,
)
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.macros import (  # noqa: E402
    enumerate_basic_corner_fixing_commutators,
    enumerate_conjugated_macros,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--action-library", type=Path)
    parser.add_argument("--group-ids", type=Path)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--validation-folds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--top-k", default="1,16,64,256")
    parser.add_argument("--gate-mode", choices=("exact", "demonstration"), default="exact")
    parser.add_argument("--rollout-k", type=int, default=256)
    parser.add_argument("--max-rollout-steps", type=int, default=20)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def describe(values: list[float]) -> dict[str, float]:
    return {
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
    }


def main() -> None:
    args = parse_args()
    ks = tuple(sorted({int(value) for value in args.top_k.split(",")}))
    if not ks or min(ks) <= 0:
        raise ValueError("top-k must contain positive integers")
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    if args.action_library:
        _, table = load_macro_action_library(
            args.action_library,
            puzzle.generators,
            decomposition,
        )
    else:
        macros = enumerate_conjugated_macros(
            enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition),
            puzzle.generators,
            decomposition,
            max_conjugator_depth=1,
        )
        table = MacroActionTable.from_macros(macros)
    dataset = MacroTeacherDataset.load(args.teacher)
    dataset.validate(table.action_count)
    if args.group_ids:
        groups = np.load(args.group_ids, allow_pickle=False)
        if groups.shape != (dataset.sample_count,):
            raise ValueError("group-ids must contain one value per teacher sample")
        fold = args.validation_fold % args.validation_folds
        selected = np.flatnonzero(np.mod(groups, args.validation_folds) == fold)
        if len(selected) == 0:
            raise ValueError("selected validation fold is empty")
        dataset = MacroTeacherDataset(
            states=dataset.states[selected],
            teacher_actions=dataset.teacher_actions[selected],
            teacher_action_counts=dataset.teacher_action_counts[selected],
            cluster_cost_targets=dataset.cluster_cost_targets[selected],
            teacher_next_costs=dataset.teacher_next_costs[selected],
            walk_depths=dataset.walk_depths[selected],
            action_digest=dataset.action_digest,
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != table.digest or dataset.action_digest != table.digest:
        raise ValueError("checkpoint, teacher data, and macro action table digests differ")
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    max_k = min(max(ks), table.action_count)
    proposed_batches: list[np.ndarray] = []
    value_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, dataset.sample_count, args.batch_size):
            states = torch.from_numpy(dataset.states[start : start + args.batch_size]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, values, _ = model(states)
            proposed_batches.append(logits.topk(max_k, dim=1).indices.cpu().numpy())
            value_predictions.append((values.float() * 72.0).cpu().numpy())
    proposals = np.concatenate(proposed_batches)
    predicted_values = np.concatenate(value_predictions)
    current_costs = dataset.cluster_cost_targets.sum(axis=1)
    optimal_costs = dataset.teacher_next_costs

    shortlist_report: dict[str, dict[str, float]] = {}
    teacher_reductions = current_costs - optimal_costs
    mean_teacher_reduction = float(teacher_reductions.mean())
    for k in ks:
        effective_k = min(k, max_k)
        regrets: list[float] = []
        reductions: list[float] = []
        exact_optimal = 0
        for index, state in enumerate(dataset.states):
            _, cost = table.exact_rerank(state, proposals[index, :effective_k])
            regret = float(cost - optimal_costs[index])
            regrets.append(regret)
            reductions.append(float(current_costs[index] - cost))
            exact_optimal += regret == 0
        shortlist_report[str(k)] = {
            "exact_optimal_fraction": exact_optimal / dataset.sample_count,
            "mean_exact_reduction": statistics.fmean(reductions),
            "mean_regret": statistics.fmean(regrets),
            "median_regret": statistics.median(regrets),
            "not_worse_than_teacher_fraction": (
                sum(value <= 0 for value in regrets) / dataset.sample_count
            ),
            "teacher_reduction_ratio": (
                statistics.fmean(reductions) / mean_teacher_reduction
            ),
            "reducing_fraction": sum(value > 0 for value in reductions) / dataset.sample_count,
        }

    value_errors = np.abs(predicted_values - current_costs)
    gate_metrics = shortlist_report[str(64)] if 64 in ks else shortlist_report[str(max(ks))]
    if args.gate_mode == "exact":
        gate_criteria = {
            "exact_optimal_fraction_min": 0.90,
            "mean_regret_max": 0.25,
            "reducing_fraction_min": 0.99,
        }
        gate_passed = (
            gate_metrics["exact_optimal_fraction"] >= 0.90
            and gate_metrics["mean_regret"] <= 0.25
            and gate_metrics["reducing_fraction"] >= 0.99
        )
    else:
        gate_criteria = {
            "teacher_reduction_ratio_min": 0.90,
            "mean_regret_max": 1.0,
            "reducing_fraction_min": 0.99,
        }
        gate_passed = (
            gate_metrics["teacher_reduction_ratio"] >= 0.90
            and gate_metrics["mean_regret"] <= 1.0
            and gate_metrics["reducing_fraction"] >= 0.99
        )

    rollout_initial: list[int] = []
    rollout_final: list[int] = []
    rollout_steps: list[int] = []
    rollout_k = min(args.rollout_k, table.action_count)
    model.eval()
    for raw_state in dataset.states:
        state = raw_state.copy()
        initial_cost = int(cluster_costs(state).sum())
        current_cost = initial_cost
        steps = 0
        for _ in range(args.max_rollout_steps):
            state_tensor = torch.from_numpy(state[None]).to(device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                state_logits, _, _ = model(state_tensor)
            proposed = state_logits.topk(rollout_k, dim=1).indices[0].cpu().numpy()
            action, next_cost = table.exact_rerank(state, proposed)
            if next_cost >= current_cost:
                break
            state = table.apply(state, action)
            current_cost = next_cost
            steps += 1
            if current_cost == 0:
                break
        rollout_initial.append(initial_cost)
        rollout_final.append(current_cost)
        rollout_steps.append(steps)
    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "gate": {
            "criteria": gate_criteria,
            "mode": args.gate_mode,
            "passed": gate_passed,
        },
        "mean_teacher_reduction": mean_teacher_reduction,
        "rollout": {
            "k": rollout_k,
            "maximum_steps": args.max_rollout_steps,
            "mean_final_cost": statistics.fmean(rollout_final),
            "mean_initial_cost": statistics.fmean(rollout_initial),
            "mean_reduction": statistics.fmean(
                before - after
                for before, after in zip(rollout_initial, rollout_final, strict=True)
            ),
            "mean_steps": statistics.fmean(rollout_steps),
            "solved_fraction": sum(value == 0 for value in rollout_final) / len(rollout_final),
        },
        "samples": dataset.sample_count,
        "shortlists": shortlist_report,
        "teacher": str(args.teacher),
        "value_absolute_error": describe(value_errors.tolist()),
    }
    output = args.out or args.checkpoint.with_name("eval_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
