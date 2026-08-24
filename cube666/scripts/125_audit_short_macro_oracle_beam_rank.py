"""Report the exact curriculum trace's global rank at every beam layer."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--factorized-value-checkpoint", type=Path)
    parser.add_argument(
        "--proposal-checkpoint", type=Path, action="append", default=[]
    )
    parser.add_argument("--train-module", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--action-costs", type=Path, required=True)
    parser.add_argument("--inverse-actions", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--beam", type=int, default=1024)
    parser.add_argument("--branch", type=int, default=512)
    parser.add_argument("--maximum-layer", type=int, default=0)
    parser.add_argument("--action-chunk-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=125666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def model_outputs(model, states, batch_size, retain_logits):
    logits_out = []
    values_out = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, total, clusters = model(states[start : start + batch_size])
        values_out.append(
            0.5 * (total.float() * 72.0 + clusters.float().sum(dim=1) * 12.0)
        )
        if retain_logits:
            logits_out.append(logits)
    return (torch.cat(logits_out) if retain_logits else None, torch.cat(values_out))


@torch.inference_mode()
def policy_topk(module, logits, effects, topk, chunk_size, forbidden):
    best_scores = torch.empty((len(logits), 0), device="cuda")
    best_actions = torch.empty((len(logits), 0), dtype=torch.long, device="cuda")
    for start in range(0, len(effects), chunk_size):
        stop = min(start + chunk_size, len(effects))
        candidates = effects[start:stop][None].expand(len(logits), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        action_ids = torch.arange(start, stop, device="cuda")[None].expand(
            len(logits), -1
        )
        scores = scores.masked_fill(action_ids.eq(forbidden[:, None]), -torch.inf)
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, action_ids), dim=1)
        keep = min(topk, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    return best_actions


def main() -> None:
    args = parse_args()
    module = load_module(args.train_module, "oracle_beam_train_module")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = module.Config(action_count=checkpoint["model_config"]["action_count"])
    policy_model = module.MacroEffectPolicyValueNet(config).cuda()
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    policy_model.eval()
    if args.factorized_value_checkpoint is not None:
        factorized_model, _ = load_macro_factorized_value_checkpoint(
            str(args.factorized_value_checkpoint), device="cuda"
        )
        model = FactorizedValueAdapter(policy_model, factorized_model).eval()
    else:
        model = policy_model
    proposal_models = []
    for proposal_path in args.proposal_checkpoint:
        proposal_checkpoint = torch.load(
            proposal_path, map_location="cpu", weights_only=True
        )
        proposal_model = module.MacroEffectPolicyValueNet(config).cuda()
        proposal_model.load_state_dict(proposal_checkpoint["model_state_dict"])
        proposal_model.eval()
        proposal_models.append(proposal_model)
    with np.load(args.teacher, allow_pickle=False) as payload:
        initial_np = payload["states"][args.index].astype(np.uint8, copy=False)
        depth = int(payload["walk_depths"][args.index])
        trace_np = payload["teacher_solution_actions"][args.index, :depth].astype(
            np.int64, copy=False
        )
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs_np = np.load(args.action_costs, allow_pickle=False).astype(
        np.float32, copy=False
    )
    inverse_np = np.load(args.inverse_actions, allow_pickle=False).astype(
        np.int64, copy=False
    )
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).cuda()
    inverse = torch.from_numpy(inverse_np).cuda()
    states = torch.from_numpy(initial_np[None]).cuda()
    logits, _ = model_outputs(model, states, args.inference_batch_size, True)
    cumulative_cost = torch.zeros(1, device="cuda")
    last_actions = torch.full((1,), -1, dtype=torch.long, device="cuda")
    oracle_state = initial_np.copy()
    rng = np.random.default_rng(args.seed)
    zobrist = torch.from_numpy(
        rng.integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=(144, 24),
            dtype=np.int64,
        )
    ).cuda()
    positions = torch.arange(144, device="cuda")
    rows = []
    for layer, oracle_action_value in enumerate(trace_np, start=1):
        if args.maximum_layer and layer > args.maximum_layer:
            break
        oracle_action = int(oracle_action_value)
        oracle_state = np.take_along_axis(
            oracle_state, effects_np[oracle_action], axis=-1
        )
        forbidden = torch.where(
            last_actions >= 0,
            inverse[last_actions.clamp_min(0)],
            torch.full_like(last_actions, -1),
        )
        proposed_parts = [policy_topk(
            module,
            logits,
            effects,
            args.branch,
            args.action_chunk_size,
            forbidden,
        )]
        for proposal_model in proposal_models:
            proposal_logits, _ = model_outputs(
                proposal_model, states, args.inference_batch_size, True
            )
            proposed_parts.append(
                policy_topk(
                    module,
                    proposal_logits,
                    effects,
                    args.branch,
                    args.action_chunk_size,
                    forbidden,
                )
            )
        proposed = torch.cat(proposed_parts, dim=1)
        candidate_width = proposed.shape[1]
        children = states[:, None].expand(-1, candidate_width, -1, -1).gather(
            -1, effects[proposed].long()
        )
        flat_children = children.flatten(0, 1)
        flat_actions = proposed.flatten()
        parent_indices = torch.arange(len(states), device="cuda").repeat_interleave(
            candidate_width
        )
        child_cost = (cumulative_cost[:, None] + costs[proposed]).flatten()
        _, child_values = model_outputs(
            model, flat_children, args.inference_batch_size, False
        )
        ranks = child_cost + child_values.clamp_min(0)
        oracle_tensor = torch.from_numpy(oracle_state).cuda()
        matches = flat_children.eq(oracle_tensor).all(dim=(1, 2))
        if matches.any():
            oracle_score = ranks[matches].min()
            oracle_rank = int((ranks < oracle_score).sum()) + 1
            oracle_proposed = True
        else:
            oracle_score = torch.tensor(float("inf"), device="cuda")
            oracle_rank = None
            oracle_proposed = False
        flat = flat_children.reshape(len(flat_children), -1).long()
        hashes = zobrist[positions, flat].sum(dim=1)
        order = torch.argsort(ranks)
        ordered_hashes = hashes[order].cpu().numpy()
        keep_positions = []
        seen = set()
        for ordered_position, state_hash in enumerate(ordered_hashes):
            key = int(state_hash)
            if key in seen:
                continue
            seen.add(key)
            keep_positions.append(ordered_position)
            if len(keep_positions) == args.beam:
                break
        keep = order[torch.as_tensor(keep_positions, device="cuda")]
        states = flat_children[keep]
        cumulative_cost = child_cost[keep]
        last_actions = flat_actions[keep]
        retained = bool(states.eq(oracle_tensor).all(dim=(1, 2)).any())
        row = {
            "beam_candidates": len(flat_children),
            "layer": layer,
            "oracle_action": oracle_action,
            "oracle_global_rank": oracle_rank,
            "oracle_proposed": oracle_proposed,
            "oracle_retained": retained,
            "oracle_score": float(oracle_score) if oracle_proposed else None,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        if not retained:
            break
        logits, _ = model_outputs(model, states, args.inference_batch_size, True)
    report = {
        "beam": args.beam,
        "branch": args.branch,
        "checkpoint": str(args.checkpoint),
        "factorized_value_checkpoint": (
            str(args.factorized_value_checkpoint)
            if args.factorized_value_checkpoint is not None
            else None
        ),
        "index": args.index,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
