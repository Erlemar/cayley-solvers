"""Harvest actual short-macro beam states with certified multi-step returns."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
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
    MacroActionPolicyNet,
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
    parser.add_argument("--direct-branch", type=int, default=0)
    parser.add_argument("--direct-cost-stratified-branch", type=int, default=0)
    parser.add_argument("--direct-cost-maximum", type=int, default=0)
    parser.add_argument("--direct-cost-stratified-depths", type=int, default=0)
    parser.add_argument(
        "--train-module",
        type=Path,
        default=Path("cube666/scripts/33_remote_train_macro_effect.py"),
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--group-ids", type=Path)
    parser.add_argument("--action-effects", type=Path)
    parser.add_argument("--action-costs", type=Path)
    parser.add_argument("--inverse-actions", type=Path)
    parser.add_argument("--indices", default="")
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument(
        "--random-root-count",
        type=int,
        default=0,
        help="also sample this many training-split roots from the teacher dataset",
    )
    parser.add_argument(
        "--root-depth",
        type=int,
        default=0,
        help="when randomly sampling roots, require this macro depth; zero allows all",
    )
    parser.add_argument("--beam", type=int, default=1024)
    parser.add_argument("--branch", type=int, default=512)
    parser.add_argument("--maximum-depth", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, default=256)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123666)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset_dir is not None:
        defaults = {
            "teacher": "teacher.npz",
            "group_ids": "source_state_ids.npy",
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
    if not args.indices and args.indices_file is None and not args.random_root_count:
        parser.error("provide indices or --random-root-count")
    if args.random_root_count and args.group_ids is None:
        parser.error("--random-root-count requires --group-ids or --dataset-dir")
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


@torch.inference_mode()
def direct_action_logits(model, states: torch.Tensor, batch_size: int) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output.append(model(states[start : start + batch_size]).float())
    return torch.cat(output)


@torch.inference_mode()
def direct_policy_topk(
    logits: torch.Tensor,
    costs: torch.Tensor,
    forbidden: torch.Tensor,
    *,
    topk: int,
    cost_stratified_topk: int,
    cost_maximum: int,
) -> torch.Tensor:
    action_ids = torch.arange(logits.shape[1], device=logits.device)
    masked = logits.masked_fill(
        action_ids[None].eq(forbidden[:, None]), -torch.inf
    )
    parts: list[torch.Tensor] = []
    if topk:
        parts.append(masked.topk(min(topk, masked.shape[1]), dim=1).indices)
    if cost_stratified_topk and cost_maximum:
        for cost_value in torch.unique(costs[costs <= cost_maximum]).tolist():
            allowed = costs.eq(cost_value)
            keep = min(cost_stratified_topk, int(allowed.sum()))
            parts.append(
                masked.masked_fill(~allowed[None], -torch.inf)
                .topk(keep, dim=1)
                .indices
            )
    return (
        torch.cat(parts, dim=1)
        if parts
        else torch.empty((len(logits), 0), dtype=torch.long, device=logits.device)
    )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("short_macro_beam_harvest_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def model_outputs(
    model: torch.nn.Module,
    states: torch.Tensor,
    batch_size: int,
    *,
    retain_logits: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    logits_out: list[torch.Tensor] = []
    values_out: list[torch.Tensor] = []
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
def policy_topk(
    module,
    logits: torch.Tensor,
    effects: torch.Tensor,
    *,
    topk: int,
    chunk_size: int,
    forbidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_scores = torch.empty((len(logits), 0), device=logits.device)
    best_actions = torch.empty(
        (len(logits), 0), dtype=torch.long, device=logits.device
    )
    for start in range(0, len(effects), chunk_size):
        stop = min(start + chunk_size, len(effects))
        candidates = effects[start:stop][None].expand(len(logits), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        action_ids = torch.arange(start, stop, device=logits.device)[None].expand(
            len(logits), -1
        )
        scores = scores.masked_fill(action_ids.eq(forbidden[:, None]), -torch.inf)
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, action_ids), dim=1)
        keep = min(topk, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    return best_actions, best_scores


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    module = load_module(args.train_module)
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
    proposal_models: list[torch.nn.Module] = []
    for proposal_path in args.proposal_checkpoint:
        proposal_checkpoint = torch.load(
            proposal_path, map_location="cpu", weights_only=True
        )
        proposal_model = module.MacroEffectPolicyValueNet(config).cuda()
        proposal_model.load_state_dict(proposal_checkpoint["model_state_dict"])
        proposal_model.eval()
        proposal_models.append(proposal_model)
    direct_action_models: list[MacroActionPolicyNet] = []
    for direct_path in args.direct_action_checkpoint:
        direct_model, _ = load_macro_action_policy_checkpoint(
            str(direct_path), device="cuda"
        )
        if direct_model.config.action_count != config.action_count:
            raise ValueError("direct-action checkpoint action count disagrees")
        direct_action_models.append(direct_model)
    index_text = args.indices
    if args.indices_file is not None:
        index_text = index_text + "," + args.indices_file.read_text(encoding="utf-8")
    indices = [
        int(token)
        for token in index_text.replace("_", ",").replace("\n", ",").split(",")
        if token.strip()
    ]
    with np.load(args.teacher, allow_pickle=False) as payload:
        source_states = payload["states"].astype(np.uint8, copy=False)
        source_values = payload["search_value_targets"].astype(np.float32, copy=False)
        source_depths = (
            payload["walk_depths"].astype(np.int16, copy=False)
            if args.random_root_count
            else None
        )
    rng = np.random.default_rng(args.seed)
    if args.random_root_count:
        if source_depths is None or args.group_ids is None:
            raise AssertionError("random-root metadata missing")
        groups = np.load(args.group_ids, allow_pickle=False)
        eligible = np.mod(groups, 10) != 0
        if args.root_depth:
            eligible &= source_depths == args.root_depth
        if indices:
            eligible[np.asarray(indices, dtype=np.int64)] = False
        candidates = np.flatnonzero(eligible)
        if len(candidates) < args.random_root_count:
            raise ValueError(
                f"requested {args.random_root_count} roots from {len(candidates)} eligible"
            )
        sampled = rng.choice(
            candidates, size=args.random_root_count, replace=False
        ).astype(np.int64, copy=False)
        indices.extend(int(value) for value in sampled)
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
    zobrist = torch.from_numpy(
        rng.integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=(144, 24),
            dtype=np.int64,
        )
    ).cuda()
    positions = torch.arange(144, device="cuda")
    harvested_states: list[np.ndarray] = []
    harvested_targets: list[np.ndarray] = []
    harvested_forward_costs: list[np.ndarray] = []
    harvested_depths: list[np.ndarray] = []
    harvested_roots: list[np.ndarray] = []
    started = time.perf_counter()
    for root_position, index in enumerate(indices):
        states = torch.from_numpy(source_states[index][None]).cuda()
        logits, _ = model_outputs(
            model, states, args.inference_batch_size, retain_logits=True
        )
        if logits is None:
            raise AssertionError("root logits missing")
        forward_cost = torch.zeros(1, device="cuda")
        undo_cost = torch.zeros(1, device="cuda")
        last_actions = torch.full((1,), -1, dtype=torch.long, device="cuda")
        for depth in range(1, args.maximum_depth + 1):
            forbidden = torch.where(
                last_actions >= 0,
                inverse[last_actions.clamp_min(0)],
                torch.full_like(last_actions, -1),
            )
            proposed_parts = [policy_topk(
                module,
                logits,
                effects,
                topk=args.branch,
                chunk_size=args.action_chunk_size,
                forbidden=forbidden,
            )[0]]
            for proposal_model in proposal_models:
                proposal_logits, _ = model_outputs(
                    proposal_model,
                    states,
                    args.inference_batch_size,
                    retain_logits=True,
                )
                if proposal_logits is None:
                    raise AssertionError("proposal logits missing")
                proposed_parts.append(
                    policy_topk(
                        module,
                        proposal_logits,
                        effects,
                        topk=args.branch,
                        chunk_size=args.action_chunk_size,
                        forbidden=forbidden,
                    )[0]
                )
            for direct_model in direct_action_models:
                direct_logits = direct_action_logits(
                    direct_model, states, args.inference_batch_size
                )
                proposed_parts.append(
                    direct_policy_topk(
                        direct_logits,
                        costs,
                        forbidden,
                        topk=args.direct_branch,
                        cost_stratified_topk=(
                            args.direct_cost_stratified_branch
                            if not args.direct_cost_stratified_depths
                            or depth <= args.direct_cost_stratified_depths
                            else 0
                        ),
                        cost_maximum=args.direct_cost_maximum,
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
            child_forward = (
                forward_cost[:, None] + costs[proposed]
            ).flatten()
            child_undo = (
                undo_cost[:, None] + costs[inverse[proposed]]
            ).flatten()
            _, child_values = model_outputs(
                model, flat_children, args.inference_batch_size, retain_logits=False
            )
            ranks = child_forward + child_values.clamp_min(0)
            flat = flat_children.reshape(len(flat_children), -1).long()
            hashes = zobrist[positions, flat].sum(dim=1)
            order = torch.argsort(ranks)
            ordered_hashes = hashes[order].cpu().numpy()
            keep_positions: list[int] = []
            seen: set[int] = set()
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
            forward_cost = child_forward[keep]
            undo_cost = child_undo[keep]
            last_actions = flat_actions[keep]
            targets = float(source_values[index]) + undo_cost.cpu().numpy()
            harvested_states.append(states.cpu().numpy().astype(np.uint8, copy=False))
            harvested_targets.append(targets.astype(np.float32, copy=False))
            harvested_forward_costs.append(
                forward_cost.cpu().numpy().astype(np.float32, copy=False)
            )
            harvested_depths.append(np.full(len(states), depth, dtype=np.int16))
            harvested_roots.append(np.full(len(states), index, dtype=np.int64))
            logits, _ = model_outputs(
                model, states, args.inference_batch_size, retain_logits=True
            )
            if logits is None:
                raise AssertionError("beam logits missing")
        print(
            json.dumps(
                {
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "index": index,
                    "root": root_position + 1,
                    "roots": len(indices),
                }
            ),
            flush=True,
        )
    states_out = np.concatenate(harvested_states)
    targets_out = np.concatenate(harvested_targets)
    forward_costs_out = np.concatenate(harvested_forward_costs)
    depths_out = np.concatenate(harvested_depths)
    roots_out = np.concatenate(harvested_roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        states=states_out,
        return_cost_targets=targets_out,
        depths=depths_out,
        forward_costs=forward_costs_out,
        root_indices=roots_out,
    )
    report = {
        "beam": args.beam,
        "branch": args.branch,
        "checkpoint": str(args.checkpoint),
        "factorized_value_checkpoint": (
            str(args.factorized_value_checkpoint)
            if args.factorized_value_checkpoint is not None
            else None
        ),
        "direct_action_checkpoints": [
            str(path) for path in args.direct_action_checkpoint
        ],
        "direct_branch": args.direct_branch,
        "direct_cost_maximum": args.direct_cost_maximum,
        "direct_cost_stratified_branch": args.direct_cost_stratified_branch,
        "direct_cost_stratified_depths": args.direct_cost_stratified_depths,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "maximum_depth": args.maximum_depth,
        "roots": len(indices),
        "proposal_checkpoints": [str(path) for path in args.proposal_checkpoint],
        "samples": len(states_out),
        "target_maximum": float(targets_out.max()),
        "target_mean": float(targets_out.mean()),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
