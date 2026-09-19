"""Audit every exact solution step of shallow short-macro curriculum states."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_factorized_value import (  # noqa: E402
    MacroFactorizedPrimitiveValueNet,
    load_macro_factorized_value_checkpoint,
)
from cube666.macro_action_policy import (  # noqa: E402
    load_macro_action_policy_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--proposal-checkpoint", type=Path, action="append", default=[]
    )
    parser.add_argument("--factorized-value-checkpoint", type=Path)
    parser.add_argument(
        "--direct-action-checkpoint", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--train-module",
        type=Path,
        default=Path("cube666/scripts/33_remote_train_macro_effect.py"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--action-effects", type=Path)
    parser.add_argument("--action-costs", type=Path)
    parser.add_argument("--inverse-actions", type=Path)
    parser.add_argument("--indices", default="")
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset_dir is not None:
        defaults = {
            "teacher": "teacher.npz",
            "action_effects": "action_effects.npy",
            "action_costs": "action_costs.npy",
            "inverse_actions": "inverse_actions.npy",
        }
        for name, filename in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, args.dataset_dir / filename)
    missing = [
        name
        for name in ("teacher", "action_effects", "action_costs", "inverse_actions")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "provide --dataset-dir or each dataset path; missing " + ", ".join(missing)
        )
    if not args.indices and args.indices_file is None:
        parser.error("provide --indices or --indices-file")
    return args


class FactorizedValueAdapter(torch.nn.Module):
    def __init__(self, policy_model, value_model: MacroFactorizedPrimitiveValueNet):
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model

    def forward(self, states):
        logits, _, _ = self.policy_model(states)
        total, clusters = self.value_model(states)
        return (
            logits,
            total * self.value_model.config.total_scale / 72.0,
            clusters * self.value_model.config.cluster_scale / 12.0,
        )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("short_macro_trace_audit_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def values(model: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, total, clusters = model(states)
    return 0.5 * (total.float() * 72.0 + clusters.float().sum(dim=1) * 12.0)


def main() -> None:
    args = parse_args()
    module = load_module(args.train_module)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = module.Config(action_count=checkpoint["model_config"]["action_count"])
    policy_model = module.MacroEffectPolicyValueNet(config).cuda()
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    policy_model.eval()
    proposal_models = []
    for proposal_path in args.proposal_checkpoint:
        proposal_checkpoint = torch.load(
            proposal_path, map_location="cpu", weights_only=True
        )
        proposal_model = module.MacroEffectPolicyValueNet(config).cuda()
        proposal_model.load_state_dict(proposal_checkpoint["model_state_dict"])
        proposal_model.eval()
        proposal_models.append(proposal_model)
    if args.factorized_value_checkpoint is not None:
        factorized_model, _ = load_macro_factorized_value_checkpoint(
            str(args.factorized_value_checkpoint), device="cuda"
        )
        model = FactorizedValueAdapter(policy_model, factorized_model).eval()
    else:
        model = policy_model
    direct_action_models = [
        load_macro_action_policy_checkpoint(str(path), device="cuda")[0]
        for path in args.direct_action_checkpoint
    ]
    index_text = args.indices
    if args.indices_file is not None:
        index_text = index_text + "," + args.indices_file.read_text(encoding="utf-8")
    indices = [
        int(token)
        for token in index_text.replace("_", ",").replace("\n", ",").split(",")
        if token.strip()
    ]
    with np.load(args.teacher, allow_pickle=False) as payload:
        initial_states = payload["states"].astype(np.uint8, copy=False)
        depths = payload["walk_depths"].astype(np.int64, copy=False)
        traces = payload["teacher_solution_actions"].astype(np.int64, copy=False)
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs_np = np.load(args.action_costs, allow_pickle=False).astype(
        np.float32, copy=False
    )
    inverse_np = np.load(args.inverse_actions, allow_pickle=False).astype(
        np.int64, copy=False
    )
    states_out: list[np.ndarray] = []
    metadata: list[dict[str, int | float]] = []
    for index in indices:
        state = initial_states[index].copy()
        trace = traces[index, : depths[index]]
        remaining = float(costs_np[trace].sum())
        previous = -1
        for step, action_value in enumerate(trace):
            action = int(action_value)
            states_out.append(state.copy())
            metadata.append(
                {
                    "action": action,
                    "action_cost": float(costs_np[action]),
                    "expected_child_value": remaining - float(costs_np[action]),
                    "expected_parent_value": remaining,
                    "index": index,
                    "previous_action": previous,
                    "remaining_macro_depth": int(len(trace) - step),
                    "step": step,
                }
            )
            state = np.take_along_axis(state, effects_np[action], axis=-1)
            remaining -= float(costs_np[action])
            previous = action
        if not np.array_equal(
            state, np.broadcast_to(np.arange(24, dtype=np.uint8), state.shape)
        ):
            raise AssertionError(f"index {index} trace failed exact replay")

    states = torch.from_numpy(np.stack(states_out)).cuda()
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).cuda()
    labels = torch.as_tensor([int(row["action"]) for row in metadata], device="cuda")
    previous = torch.as_tensor(
        [int(row["previous_action"]) for row in metadata], device="cuda"
    )
    inverse = torch.from_numpy(inverse_np).cuda()
    forbidden = torch.where(
        previous >= 0,
        inverse[previous.clamp_min(0)],
        torch.full_like(previous, -1),
    )
    policy_ranks: list[np.ndarray] = []
    policy_cost_ranks: list[np.ndarray] = []
    for rank_model in [policy_model, *proposal_models]:
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits, _, _ = rank_model(states)
        label_scores = module.score_candidates(logits, effects[labels][:, None])[:, 0]
        higher = torch.zeros(len(states), dtype=torch.int64, device="cuda")
        cost_higher = torch.zeros(len(states), dtype=torch.int64, device="cuda")
        for start in range(0, len(effects), args.chunk_size):
            stop = min(start + args.chunk_size, len(effects))
            action_ids = torch.arange(start, stop, device="cuda")[None]
            candidates = effects[start:stop][None].expand(len(states), -1, -1, -1)
            scores = module.score_candidates(logits, candidates)
            scores = scores.masked_fill(action_ids.eq(forbidden[:, None]), -torch.inf)
            higher += (scores > label_scores[:, None]).sum(dim=1)
            same_cost = costs[start:stop][None].eq(costs[labels][:, None])
            cost_higher += ((scores > label_scores[:, None]) & same_cost).sum(dim=1)
        policy_ranks.append((higher + 1).cpu().numpy())
        policy_cost_ranks.append((cost_higher + 1).cpu().numpy())
    for rank_model in direct_action_models:
        remaining_depths = torch.as_tensor(
            [int(row["remaining_macro_depth"]) for row in metadata],
            device="cuda",
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = rank_model(states, remaining_depth=remaining_depths)
        label_scores = logits.gather(1, labels[:, None]).squeeze(1)
        scores = logits.masked_fill(
            torch.arange(logits.shape[1], device="cuda")[None].eq(
                forbidden[:, None]
            ),
            -torch.inf,
        )
        higher = (scores > label_scores[:, None]).sum(dim=1)
        same_cost = costs[None].eq(costs[labels][:, None])
        cost_higher = ((scores > label_scores[:, None]) & same_cost).sum(dim=1)
        policy_ranks.append((higher + 1).cpu().numpy())
        policy_cost_ranks.append((cost_higher + 1).cpu().numpy())
    children = states.gather(-1, effects[labels].long())
    parent_values = values(model, states).cpu().numpy()
    child_values = values(model, children).cpu().numpy()
    ranks = np.stack(policy_ranks)
    cost_ranks = np.stack(policy_cost_ranks)
    rows = []
    for position, base in enumerate(metadata):
        rows.append(
            {
                **base,
                "child_predicted_value": float(child_values[position]),
                "parent_predicted_value": float(parent_values[position]),
                "policy_rank": int(ranks[:, position].min()),
                "policy_ranks": [int(value) for value in ranks[:, position]],
                "policy_same_cost_rank": int(cost_ranks[:, position].min()),
                "policy_same_cost_ranks": [
                    int(value) for value in cost_ranks[:, position]
                ],
            }
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "factorized_value_checkpoint": (
            str(args.factorized_value_checkpoint)
            if args.factorized_value_checkpoint is not None
            else None
        ),
        "direct_action_checkpoints": [
            str(path) for path in args.direct_action_checkpoint
        ],
        "maximum_policy_rank": max(int(row["policy_rank"]) for row in rows),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
