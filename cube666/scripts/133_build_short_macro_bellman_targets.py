"""Build fitted-value-iteration targets from short-macro Bellman backups."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_action_policy import load_macro_action_policy_checkpoint  # noqa: E402
from cube666.macro_factorized_value import (  # noqa: E402
    MacroFactorizedPrimitiveValueNet,
    load_macro_factorized_value_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument("--proposal-branch", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=133666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def combined_value(
    model: MacroFactorizedPrimitiveValueNet,
    outputs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    total, clusters = outputs
    return 0.5 * (
        total.float() * model.config.total_scale
        + clusters.float().sum(dim=1) * model.config.cluster_scale
    )


@torch.inference_mode()
def predict_values(
    model: MacroFactorizedPrimitiveValueNet,
    states: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            parts.append(combined_value(model, model(states[start : start + batch_size])))
    return torch.cat(parts)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.proposal_branch <= 0:
        raise ValueError("samples and proposal branch must be positive")
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    value_model, _ = load_macro_factorized_value_checkpoint(
        str(args.value_checkpoint), device="cuda"
    )
    value_model.eval()
    policy_model, _ = load_macro_action_policy_checkpoint(
        str(args.policy_checkpoint), device="cuda"
    )
    policy_model.eval()
    effects_np = np.load(args.dataset_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs_np = np.load(args.dataset_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).cuda()
    with np.load(args.dataset_dir / "teacher.npz", allow_pickle=False) as payload:
        all_states = payload["states"].astype(np.uint8, copy=False)
        all_values = payload["search_value_targets"].astype(np.float32, copy=False)
        all_depths = payload["walk_depths"].astype(np.int16, copy=False)
        all_solutions = payload["teacher_solution_actions"].astype(np.int32, copy=False)
    groups = np.load(args.dataset_dir / "source_state_ids.npy", allow_pickle=False)
    eligible = np.flatnonzero(all_depths == args.depth)
    if not len(eligible):
        raise ValueError(f"no depth-{args.depth} states")
    selected = rng.choice(eligible, size=min(args.samples, len(eligible)), replace=False)
    states_out = all_states[selected].copy()
    values_out = np.empty(len(selected), dtype=np.float32)
    actions_out = np.empty(len(selected), dtype=np.int32)
    teacher_selected = 0
    predicted_improvements = 0
    for start in range(0, len(selected), args.batch_size):
        stop = min(start + args.batch_size, len(selected))
        rows = selected[start:stop]
        states = torch.from_numpy(all_states[rows]).cuda()
        remaining_depth = torch.full(
            (len(rows),), args.depth, dtype=torch.long, device="cuda"
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = policy_model(states, remaining_depth=remaining_depth)
        proposed = logits.topk(min(args.proposal_branch, logits.shape[1]), dim=1).indices
        teacher = torch.from_numpy(all_solutions[rows, 0].astype(np.int64)).cuda()
        proposed = torch.cat((proposed, teacher[:, None]), dim=1)
        selected_effects = effects[proposed]
        children = states[:, None].expand(-1, proposed.shape[1], -1, -1).gather(
            -1, selected_effects.long()
        )
        child_values = predict_values(
            value_model, children.flatten(0, 1), args.inference_batch_size
        ).reshape(len(rows), proposed.shape[1]).clamp_min(0)
        q = costs[proposed] + child_values
        exact_parent_values = torch.from_numpy(all_values[rows]).cuda()
        q[:, -1] = exact_parent_values
        best = q.argmin(dim=1)
        chosen_actions = proposed.gather(1, best[:, None]).squeeze(1)
        chosen_values = q.gather(1, best[:, None]).squeeze(1)
        actions_out[start:stop] = chosen_actions.cpu().numpy().astype(np.int32)
        values_out[start:stop] = chosen_values.cpu().numpy().astype(np.float32)
        teacher_selected += int(chosen_actions.eq(teacher).sum())
        predicted_improvements += int(chosen_values.lt(exact_parent_values - 1e-4).sum())
        if stop == len(selected) or stop % (args.batch_size * 20) == 0:
            print(json.dumps({"processed": stop, "samples": len(selected)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        states=states_out,
        bellman_actions=actions_out,
        bellman_value_targets=values_out,
        constructive_value_targets=all_values[selected],
        remaining_depths=all_depths[selected],
        source_indices=selected.astype(np.int64),
        source_state_ids=groups[selected],
    )
    report = {
        "constructive_mean": float(all_values[selected].mean()),
        "depth": args.depth,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "mean_bellman_target": float(values_out.mean()),
        "mean_predicted_improvement": float((all_values[selected] - values_out).mean()),
        "policy_checkpoint": str(args.policy_checkpoint),
        "predicted_improvements": predicted_improvements,
        "proposal_branch": args.proposal_branch,
        "samples": len(selected),
        "teacher_selected": teacher_selected,
        "value_checkpoint": str(args.value_checkpoint),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
