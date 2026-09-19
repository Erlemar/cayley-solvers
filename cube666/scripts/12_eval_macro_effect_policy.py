"""Exact held-out evaluation for the compositional macro-effect policy."""

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
    load_macro_action_library,
)
from cube666.macro_policy import (  # noqa: E402
    MacroEffectPolicyConfig,
    MacroEffectPolicyValueNet,
    score_macro_effect_candidates,
)
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--action-chunk-size", type=int, default=512)
    parser.add_argument("--top-k", default="1,16,64,256")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


@torch.no_grad()
def top_action_proposals(
    effect_logits: torch.Tensor,
    action_effects: torch.Tensor,
    *,
    top_k: int,
    chunk_size: int,
) -> np.ndarray:
    batch_size = effect_logits.shape[0]
    best_scores = torch.empty((batch_size, 0), device=effect_logits.device)
    best_actions = torch.empty((batch_size, 0), dtype=torch.long, device=effect_logits.device)
    for start in range(0, len(action_effects), chunk_size):
        stop = min(start + chunk_size, len(action_effects))
        chunk = action_effects[start:stop]
        candidates = chunk[None].expand(batch_size, -1, -1, -1)
        scores = score_macro_effect_candidates(effect_logits, candidates)
        action_ids = torch.arange(start, stop, device=effect_logits.device)
        action_ids = action_ids[None].expand(batch_size, -1)
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, action_ids), dim=1)
        keep = min(top_k, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    return best_actions.cpu().numpy()


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
        if args.validation_folds <= 1:
            raise ValueError("validation-folds must exceed one")
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
    config_payload = dict(checkpoint["model_config"])
    if config_payload.pop("architecture", None) != "effect":
        raise ValueError("checkpoint is not a macro-effect policy")
    model = MacroEffectPolicyValueNet(MacroEffectPolicyConfig(**config_payload))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    action_effects = torch.from_numpy(table.effects).to(device)

    proposal_batches: list[np.ndarray] = []
    value_batches: list[np.ndarray] = []
    for start in range(0, dataset.sample_count, args.batch_size):
        states = torch.from_numpy(dataset.states[start : start + args.batch_size]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            effect_logits, values, _ = model(states)
        proposal_batches.append(
            top_action_proposals(
                effect_logits,
                action_effects,
                top_k=max(ks),
                chunk_size=args.action_chunk_size,
            )
        )
        value_batches.append((values.float() * 72.0).detach().cpu().numpy())
    proposals = np.concatenate(proposal_batches)
    predicted_values = np.concatenate(value_batches)
    current_costs = dataset.cluster_cost_targets.sum(axis=1)
    optimal_costs = dataset.teacher_next_costs

    shortlist_report: dict[str, dict[str, float]] = {}
    for k in ks:
        regrets: list[float] = []
        reductions: list[float] = []
        for index, state in enumerate(dataset.states):
            _, cost = table.exact_rerank(state, proposals[index, :k])
            regrets.append(float(cost - optimal_costs[index]))
            reductions.append(float(current_costs[index] - cost))
        shortlist_report[str(k)] = {
            "exact_optimal_fraction": sum(value == 0 for value in regrets) / len(regrets),
            "mean_exact_reduction": statistics.fmean(reductions),
            "mean_regret": statistics.fmean(regrets),
            "median_regret": statistics.median(regrets),
            "reducing_fraction": sum(value > 0 for value in reductions) / len(reductions),
        }

    gate_metrics = shortlist_report[str(64)] if 64 in ks else shortlist_report[str(max(ks))]
    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "architecture": "effect",
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "gate": {
            "criteria": {
                "exact_optimal_fraction_min": 0.90,
                "mean_regret_max": 0.25,
                "reducing_fraction_min": 0.99,
            },
            "passed": (
                gate_metrics["exact_optimal_fraction"] >= 0.90
                and gate_metrics["mean_regret"] <= 0.25
                and gate_metrics["reducing_fraction"] >= 0.99
            ),
        },
        "samples": dataset.sample_count,
        "shortlists": shortlist_report,
        "teacher": str(args.teacher),
        "value_mae": float(np.abs(predicted_values - current_costs).mean()),
    }
    output = args.out or args.checkpoint.with_name("effect_eval_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
