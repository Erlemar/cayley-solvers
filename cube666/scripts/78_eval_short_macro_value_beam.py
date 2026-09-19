"""Autonomous learned beam over the inverse-closed short cube666 macro graph."""

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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="main effect-policy checkpoint; optional for direct-action-only search",
    )
    parser.add_argument(
        "--factorized-value-checkpoint",
        type=Path,
        help="optional factorized value checkpoint; --checkpoint still supplies policy",
    )
    parser.add_argument(
        "--root-transition-ranker-checkpoint",
        type=Path,
        help="optional factorized child-state ranker used only for the first layer",
    )
    parser.add_argument("--root-transition-score-scale", type=float, default=1.0)
    parser.add_argument(
        "--root-transition-policy-weight",
        type=float,
        default=0.0,
        help="retain this multiple of the original root policy NLL alongside transition Q",
    )
    parser.add_argument(
        "--transition-rerank-branch",
        type=int,
        default=0,
        help="retain this many child-Q-ranked direct candidates per model and parent",
    )
    parser.add_argument(
        "--transition-rerank-depths",
        type=int,
        default=0,
        help="apply transition reranking through this layer; zero is unlimited",
    )
    parser.add_argument("--transition-rerank-state-batch-size", type=int, default=32)
    parser.add_argument(
        "--proposal-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="additional same-architecture policy checkpoints; each contributes branch actions",
    )
    parser.add_argument(
        "--direct-action-checkpoint", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--ensemble-direct-actions",
        action="store_true",
        help="average direct-policy logits, then propose and score once",
    )
    parser.add_argument("--direct-branch", type=int, default=0)
    parser.add_argument("--direct-early-branch", type=int, default=0)
    parser.add_argument(
        "--direct-early-branch-depths",
        type=int,
        default=0,
        help="use --direct-early-branch through this layer before --direct-branch",
    )
    parser.add_argument(
        "--direct-root-branch",
        type=int,
        default=0,
        help="per-model direct proposal width at layer 1; zero uses --direct-branch",
    )
    parser.add_argument(
        "--direct-score-mode",
        choices=("log_probability", "global_logit_gap", "stratified_logit_gap"),
        default="log_probability",
    )
    parser.add_argument("--direct-cost-stratified-branch", type=int, default=0)
    parser.add_argument("--direct-cost-maximum", type=int, default=0)
    parser.add_argument(
        "--direct-cost-stratified-depths",
        type=int,
        default=0,
        help="limit cost-stratified proposals to the first N layers; zero is unlimited",
    )
    parser.add_argument(
        "--train-module",
        type=Path,
        default=Path("cube666/scripts/33_remote_train_macro_effect.py"),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="directory containing teacher.npz and the short-action arrays",
    )
    parser.add_argument("--states", type=Path)
    parser.add_argument("--group-ids", type=Path)
    parser.add_argument("--action-effects", type=Path)
    parser.add_argument("--action-costs", type=Path)
    parser.add_argument("--inverse-actions", type=Path)
    parser.add_argument("--indices", default="")
    parser.add_argument("--indices-file", type=Path)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument(
        "--index-limit", type=int, default=0,
        help="also evaluate this contiguous slice; 0 disables the slice",
    )
    parser.add_argument("--pids", default="")
    parser.add_argument(
        "--random-count",
        type=int,
        default=0,
        help="evaluate this many deterministic random source rows",
    )
    parser.add_argument(
        "--random-depth",
        type=int,
        default=0,
        help="with --random-count, restrict source rows to this walk depth",
    )
    parser.add_argument(
        "--all-pids",
        action="store_true",
        help="evaluate the maximum-target row for every distinct group id",
    )
    parser.add_argument("--beam", type=int, default=128)
    parser.add_argument(
        "--root-beam",
        type=int,
        default=0,
        help="retained width after layer 1; zero uses --beam",
    )
    parser.add_argument("--branch", type=int, default=256)
    parser.add_argument(
        "--root-action-diversity",
        action="store_true",
        help=(
            "reserve an equal beam quota for every retained first-action branch "
            "before filling spare slots globally"
        ),
    )
    parser.add_argument(
        "--root-action-diversity-depths",
        type=int,
        default=0,
        help="limit root-branch quotas to the first N layers; zero is unlimited",
    )
    parser.add_argument(
        "--root-action-diversity-groups",
        type=int,
        default=0,
        help=(
            "protect only the N first-action groups whose best child ranks highest; "
            "zero protects every group"
        ),
    )
    parser.add_argument(
        "--root-action-diversity-quota",
        type=int,
        default=0,
        help=(
            "fixed protected states per selected first-action group; zero divides "
            "the whole beam equally across the selected groups"
        ),
    )
    parser.add_argument("--maximum-depth", type=int, default=40)
    parser.add_argument(
        "--bidirectional-meet-depth",
        type=int,
        default=0,
        help=(
            "run two model-guided beams (root->goal and goal->root) and join "
            "their retained frontiers at this depth; zero uses ordinary search"
        ),
    )
    parser.add_argument("--path-cost-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument(
        "--value-prebeam-multiplier",
        type=int,
        default=1,
        help=(
            "when >1, evaluate value only on this many beam-widths selected "
            "by path cost plus policy score"
        ),
    )
    parser.add_argument("--policy-nll-weight", type=float, default=0.02)
    parser.add_argument("--policy-step-nll-cap", type=float, default=0.0)
    parser.add_argument("--policy-tail-slope", type=float, default=0.01)
    parser.add_argument("--prebeam-multiplier", type=int, default=1)
    parser.add_argument("--cycle-cost-weight", type=float, default=0.0)
    parser.add_argument("--cycle-proposal-branch", type=int, default=0)
    parser.add_argument("--cycle-proposal-cost-weight", type=float, default=0.0)
    parser.add_argument("--cycle-proposal-score-scale", type=float, default=0.2)
    parser.add_argument(
        "--cycle-proposal-depths",
        type=int,
        default=0,
        help="use exhaustive structural action proposals for the first N layers",
    )
    parser.add_argument("--backup-prebeam-multiplier", type=int, default=1)
    parser.add_argument("--backup-branch", type=int, default=0)
    parser.add_argument("--backup-state-batch-size", type=int, default=128)
    parser.add_argument("--backup-weight", type=float, default=0.0)
    parser.add_argument("--action-chunk-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=78666)
    parser.add_argument("--audit-oracle-retention", action="store_true")
    parser.add_argument(
        "--one-macro-endgame",
        action="store_true",
        help="before pruning, exactly join any generated state to the goal by one macro",
    )
    parser.add_argument(
        "--exact-root-two-macro",
        action="store_true",
        help="exhaustively test every one- and two-macro route before neural search",
    )
    parser.add_argument(
        "--continue-after-solution",
        action="store_true",
        help="retain the cheapest solution and continue through maximum depth",
    )
    parser.add_argument(
        "--harvest-frontier-depth",
        type=int,
        default=0,
        help="save the retained beam at this depth; zero disables harvesting",
    )
    parser.add_argument(
        "--harvest-frontier-per-root",
        type=int,
        default=0,
        help="maximum retained states saved per root; zero saves the full beam",
    )
    parser.add_argument(
        "--harvest-frontier-out",
        type=Path,
        help="compressed NPZ receiving harvested states, costs, NLLs, and paths",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.dataset_dir is not None:
        defaults = {
            "states": "teacher.npz",
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
        for name in (
            "states",
            "group_ids",
            "action_effects",
            "action_costs",
            "inverse_actions",
        )
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "provide --dataset-dir or each dataset path; missing " + ", ".join(missing)
        )
    if args.harvest_frontier_depth < 0 or args.harvest_frontier_per_root < 0:
        parser.error("frontier harvest values must be non-negative")
    if bool(args.harvest_frontier_depth) != bool(args.harvest_frontier_out):
        parser.error(
            "--harvest-frontier-depth and --harvest-frontier-out must be used together"
        )
    if args.harvest_frontier_depth > args.maximum_depth:
        parser.error("frontier harvest depth exceeds --maximum-depth")
    if args.bidirectional_meet_depth < 0:
        parser.error("bidirectional meet depth must be non-negative")
    if args.bidirectional_meet_depth and (
        args.harvest_frontier_depth or args.harvest_frontier_out
    ):
        parser.error("bidirectional meet mode owns the frontier harvest")
    if args.bidirectional_meet_depth > args.maximum_depth:
        parser.error("bidirectional meet depth exceeds --maximum-depth")
    if args.root_beam < 0:
        parser.error("root beam must be non-negative")
    if (
        args.transition_rerank_branch < 0
        or args.transition_rerank_depths < 0
        or args.transition_rerank_state_batch_size <= 0
    ):
        parser.error("transition rerank values must be non-negative and batch positive")
    if args.transition_rerank_branch and args.root_transition_ranker_checkpoint is None:
        parser.error("transition reranking requires a root transition ranker")
    if bool(args.direct_early_branch) != bool(args.direct_early_branch_depths):
        parser.error("direct early branch and depths must be used together")
    return args


class FactorizedValueAdapter(torch.nn.Module):
    """Use an effect model's policy heads with a factorized primitive-cost value."""

    def __init__(
        self,
        policy_model: torch.nn.Module,
        value_model: MacroFactorizedPrimitiveValueNet,
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model

    def forward(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, _, _ = self.policy_model(states)
        total_scaled, cluster_scaled = self.value_model(states)
        total_effect_scale = (
            total_scaled * self.value_model.config.total_scale / 72.0
        )
        cluster_effect_scale = (
            cluster_scaled * self.value_model.config.cluster_scale / 12.0
        )
        return logits, total_effect_scale, cluster_effect_scale


class FactorizedValueOnlyAdapter(torch.nn.Module):
    """Expose a standalone factorized critic to direct-action-only search."""

    def __init__(self, value_model: MacroFactorizedPrimitiveValueNet) -> None:
        super().__init__()
        self.value_model = value_model

    def forward(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_scaled, cluster_scaled = self.value_model(states)
        total_effect_scale = (
            total_scaled * self.value_model.config.total_scale / 72.0
        )
        cluster_effect_scale = (
            cluster_scaled * self.value_model.config.cluster_scale / 12.0
        )
        return (
            torch.empty((len(states), 0), device=states.device),
            total_effect_scale,
            cluster_effect_scale,
        )


class NullPolicyValue(torch.nn.Module):
    """Zero-cost adapter for searches driven exclusively by direct action heads."""

    def forward(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((len(states), 0), device=states.device),
            torch.zeros(len(states), device=states.device),
            torch.zeros((len(states), 6), device=states.device),
        )


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cube666_short_macro_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
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
            ((total.float() * 72.0 + clusters.float().sum(dim=1) * 12.0) * 0.5)
        )
        if retain_logits:
            logits_out.append(logits)
    return (
        torch.cat(logits_out) if retain_logits else None,
        torch.cat(values_out),
    )


@torch.inference_mode()
def policy_topk(
    module: ModuleType,
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


@torch.inference_mode()
def direct_action_logits(
    model: MacroActionPolicyNet,
    states: torch.Tensor,
    batch_size: int,
    remaining_depth: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output.append(
                model(
                    states[start : start + batch_size],
                    remaining_depth=remaining_depth,
                ).float()
            )
    return torch.cat(output)


@torch.inference_mode()
def factorized_combined_values(
    model: MacroFactorizedPrimitiveValueNet,
    states: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, clusters = model(states[start : start + batch_size])
        output.append(
            0.5
            * (
                total.float() * model.config.total_scale
                + clusters.float().sum(dim=1) * model.config.cluster_scale
            )
        )
    return torch.cat(output)


@torch.inference_mode()
def rerank_direct_transitions(
    ranker: MacroFactorizedPrimitiveValueNet,
    states: torch.Tensor,
    actions: torch.Tensor,
    effects: torch.Tensor,
    costs: torch.Tensor,
    *,
    keep: int,
    state_batch_size: int,
    inference_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream a parent-local child-state Q rerank without a global child tensor."""

    action_parts: list[torch.Tensor] = []
    q_parts: list[torch.Tensor] = []
    retained = min(keep, actions.shape[1])
    for start in range(0, len(states), state_batch_size):
        stop = min(start + state_batch_size, len(states))
        batch_actions = actions[start:stop]
        selected_effects = effects[batch_actions]
        children = states[start:stop, None].expand(
            -1, batch_actions.shape[1], -1, -1
        ).gather(-1, selected_effects.long())
        values = factorized_combined_values(
            ranker,
            children.flatten(0, 1),
            inference_batch_size,
        ).view(stop - start, -1)
        q = costs[batch_actions] + values.clamp_min(0)
        best_q, positions = q.topk(retained, dim=1, largest=False)
        action_parts.append(batch_actions.gather(1, positions))
        q_parts.append(best_q)
    return torch.cat(action_parts), torch.cat(q_parts)


@torch.inference_mode()
def direct_policy_topk(
    logits: torch.Tensor,
    costs: torch.Tensor,
    forbidden: torch.Tensor,
    *,
    topk: int,
    cost_stratified_topk: int,
    cost_maximum: int,
    score_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_ids = torch.arange(logits.shape[1], device=logits.device)
    masked_logits = logits.masked_fill(
        action_ids[None].eq(forbidden[:, None]), -torch.inf
    )
    if score_mode == "log_probability":
        masked = masked_logits.log_softmax(dim=1)
        global_reference = None
    else:
        masked = masked_logits
        global_reference = masked_logits.max(dim=1, keepdim=True).values
    action_parts: list[torch.Tensor] = []
    score_parts: list[torch.Tensor] = []
    if topk:
        scores, actions = masked.topk(min(topk, masked.shape[1]), dim=1)
        if global_reference is not None:
            scores = scores - global_reference
        action_parts.append(actions)
        score_parts.append(scores)
    if cost_stratified_topk and cost_maximum:
        for cost_value in torch.unique(costs[costs <= cost_maximum]).tolist():
            allowed = costs.eq(cost_value)
            count = int(allowed.sum())
            keep = min(cost_stratified_topk, count)
            bucket = masked.masked_fill(~allowed[None], -torch.inf)
            scores, actions = bucket.topk(keep, dim=1)
            if score_mode == "global_logit_gap":
                if global_reference is None:
                    raise AssertionError("global logit reference missing")
                scores = scores - global_reference
            elif score_mode == "stratified_logit_gap":
                scores = scores - bucket.max(dim=1, keepdim=True).values
            action_parts.append(actions)
            score_parts.append(scores)
    if not action_parts:
        return (
            torch.empty((len(logits), 0), dtype=torch.long, device=logits.device),
            torch.empty((len(logits), 0), device=logits.device),
        )
    return torch.cat(action_parts, dim=1), torch.cat(score_parts, dim=1)


@torch.inference_mode()
def three_cycle_cost(states: torch.Tensor) -> torch.Tensor:
    """Exact sum of permutation 3-cycle lower bounds over the six clusters."""
    starts = torch.arange(24, device=states.device).view(1, 1, 24)
    starts = starts.expand(len(states), 6, -1)
    cursor = starts
    minimum = starts
    for _ in range(24):
        cursor = states.gather(-1, cursor).long()
        minimum = torch.minimum(minimum, cursor)
    cycles = minimum.eq(starts).sum(dim=2)
    return (24 - cycles).float().sum(dim=1) * 0.5


@torch.inference_mode()
def cycle_q_topk(
    states: torch.Tensor,
    effects: torch.Tensor,
    costs: torch.Tensor,
    forbidden: torch.Tensor,
    *,
    topk: int,
    action_chunk_size: int,
    action_cost_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exhaustively shortlist actions by exact child cycle cost plus action cost."""
    if not topk:
        empty_actions = torch.empty((len(states), 0), dtype=torch.long, device=states.device)
        return empty_actions, torch.empty((len(states), 0), device=states.device)
    keep = min(topk, len(effects))
    best_q = torch.full((len(states), keep), torch.inf, device=states.device)
    best_actions = torch.full(
        (len(states), keep), -1, dtype=torch.long, device=states.device
    )
    for start in range(0, len(effects), action_chunk_size):
        stop = min(start + action_chunk_size, len(effects))
        chunk_effects = effects[start:stop]
        children = states[:, None].expand(-1, stop - start, -1, -1).gather(
            -1, chunk_effects[None].expand(len(states), -1, -1, -1).long()
        )
        q = three_cycle_cost(children.flatten(0, 1)).reshape(len(states), stop - start)
        if action_cost_weight:
            q = q + action_cost_weight * costs[start:stop][None]
        chunk_actions = torch.arange(start, stop, device=states.device)[None].expand(
            len(states), -1
        )
        q = q.masked_fill(chunk_actions.eq(forbidden[:, None]), torch.inf)
        merged_q = torch.cat((best_q, q), dim=1)
        merged_actions = torch.cat((best_actions, chunk_actions), dim=1)
        best_q, positions = torch.topk(merged_q, keep, dim=1, largest=False)
        best_actions = merged_actions.gather(1, positions)
    return best_actions, best_q


@torch.inference_mode()
def state_hashes(
    states: torch.Tensor, zobrist: torch.Tensor, position_chunk_size: int = 12
) -> torch.Tensor:
    """Hash states without materializing an (N, 144) int64 lookup result."""
    flat_states = states.reshape(len(states), -1)
    hashes = torch.zeros(len(states), dtype=torch.int64, device=states.device)
    for start in range(0, flat_states.shape[1], position_chunk_size):
        stop = min(start + position_chunk_size, flat_states.shape[1])
        positions = torch.arange(start, stop, device=states.device)
        hashes += zobrist[positions, flat_states[:, start:stop].long()].sum(dim=1)
    return hashes


@torch.inference_mode()
def bellman_backup_values(
    module: ModuleType,
    model: torch.nn.Module,
    states: torch.Tensor,
    last_actions: torch.Tensor,
    effects: torch.Tensor,
    costs: torch.Tensor,
    inverse_actions: torch.Tensor,
    *,
    branch: int,
    state_batch_size: int,
    action_chunk_size: int,
    inference_batch_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(states), state_batch_size):
        stop = min(start + state_batch_size, len(states))
        batch = states[start:stop]
        logits, _ = model_outputs(
            model, batch, inference_batch_size, retain_logits=True
        )
        if logits is None:
            raise AssertionError("backup logits missing")
        previous = last_actions[start:stop]
        forbidden = torch.where(
            previous >= 0,
            inverse_actions[previous.clamp_min(0)],
            torch.full_like(previous, -1),
        )
        actions, _ = policy_topk(
            module,
            logits,
            effects,
            topk=branch,
            chunk_size=action_chunk_size,
            forbidden=forbidden,
        )
        grandchildren = batch[:, None].expand(-1, branch, -1, -1).gather(
            -1, effects[actions].long()
        )
        _, grandchild_values = model_outputs(
            model,
            grandchildren.flatten(0, 1),
            inference_batch_size,
            retain_logits=False,
        )
        q_values = costs[actions] + grandchild_values.reshape(len(batch), branch)
        output.append(q_values.min(dim=1).values)
    return torch.cat(output)


@torch.inference_mode()
def solve(
    module: ModuleType | None,
    model: torch.nn.Module,
    proposal_models: list[torch.nn.Module],
    direct_action_models: list[MacroActionPolicyNet],
    root_transition_ranker: MacroFactorizedPrimitiveValueNet | None,
    initial: np.ndarray,
    effects: torch.Tensor,
    costs: torch.Tensor,
    inverse_actions: torch.Tensor,
    *,
    beam_width: int,
    branch: int,
    direct_branch: int,
    direct_early_branch: int,
    direct_early_branch_depths: int,
    direct_root_branch: int,
    direct_cost_stratified_branch: int,
    direct_cost_maximum: int,
    direct_cost_stratified_depths: int,
    direct_score_mode: str,
    ensemble_direct_actions: bool,
    maximum_depth: int,
    path_cost_weight: float,
    value_weight: float,
    value_prebeam_multiplier: int,
    policy_nll_weight: float,
    policy_step_nll_cap: float,
    policy_tail_slope: float,
    prebeam_multiplier: int,
    cycle_cost_weight: float,
    cycle_proposal_branch: int,
    cycle_proposal_cost_weight: float,
    cycle_proposal_score_scale: float,
    cycle_proposal_depths: int,
    backup_prebeam_multiplier: int,
    backup_branch: int,
    backup_state_batch_size: int,
    backup_weight: float,
    action_chunk_size: int,
    inference_batch_size: int,
    zobrist: torch.Tensor,
    oracle_actions: np.ndarray | None,
    one_macro_endgame: bool,
    exact_root_two_macro: bool,
    continue_after_solution: bool,
    root_action_diversity: bool = False,
    root_action_diversity_depths: int = 0,
    root_action_diversity_groups: int = 0,
    root_action_diversity_quota: int = 0,
    root_beam_width: int = 0,
    root_transition_score_scale: float = 1.0,
    root_transition_policy_weight: float = 0.0,
    transition_rerank_branch: int = 0,
    transition_rerank_depths: int = 0,
    transition_rerank_state_batch_size: int = 32,
    policy_horizon: int = 0,
    harvest_frontier_depth: int = 0,
    harvest_frontier_limit: int = 0,
) -> tuple[list[int] | None, dict[str, object]]:
    device = effects.device
    identity = torch.arange(24, dtype=torch.uint8, device=device)[None].expand(6, -1)
    states = torch.from_numpy(initial[None]).to(device)
    logits, values = model_outputs(
        model, states, inference_batch_size, retain_logits=True
    )
    if logits is None:
        raise AssertionError("initial logits missing")
    cumulative_cost = torch.zeros(1, device=device)
    cumulative_nll = torch.zeros(1, device=device)
    last_actions = torch.full((1,), -1, dtype=torch.long, device=device)
    root_actions = torch.full((1,), -1, dtype=torch.long, device=device)
    paths = torch.full((1, maximum_depth), -1, dtype=torch.int32, device=device)
    expanded = 0
    oracle_audit: list[dict[str, object]] = []
    oracle_state = states.clone() if oracle_actions is not None else None
    best_result: list[int] | None = None
    best_cost = float("inf")
    best_depth: int | None = None
    harvest_payload: dict[str, np.ndarray] | None = None
    if one_macro_endgame or exact_root_two_macro:
        # Applying action a to effects[inverse_actions[a]] gives the identity.
        # Sort the tiny predecessor table once so each generated layer can be
        # joined to it with a vectorized hash lookup.  Every hash hit is still
        # verified against the full state below.
        goal_predecessors = effects[inverse_actions]
        predecessor_hashes = state_hashes(goal_predecessors, zobrist)
        sorted_predecessor_hashes, predecessor_order = torch.sort(predecessor_hashes)
    else:
        goal_predecessors = None
        sorted_predecessor_hashes = None
        predecessor_order = None
    started = time.perf_counter()
    if exact_root_two_macro:
        if (
            goal_predecessors is None
            or sorted_predecessor_hashes is None
            or predecessor_order is None
        ):
            raise AssertionError("exact root endgame table is missing")
        exact_result: list[int] | None = None
        exact_cost = float("inf")

        initial_hash = state_hashes(states, zobrist)
        initial_position = torch.searchsorted(sorted_predecessor_hashes, initial_hash)
        if int(initial_position) < len(sorted_predecessor_hashes):
            finish = predecessor_order[initial_position]
            if states[0].eq(goal_predecessors[finish]).all():
                exact_result = [int(finish)]
                exact_cost = int(costs[finish])

        root_children = states.expand(len(effects), -1, -1).gather(-1, effects.long())
        expanded += len(root_children)
        root_hashes = state_hashes(root_children, zobrist)
        positions = torch.searchsorted(sorted_predecessor_hashes, root_hashes)
        in_range = positions.lt(len(sorted_predecessor_hashes))
        safe_positions = positions.clamp_max(len(sorted_predecessor_hashes) - 1)
        hash_matches = in_range & sorted_predecessor_hashes[safe_positions].eq(root_hashes)
        first_actions = torch.nonzero(hash_matches, as_tuple=False).flatten()
        if len(first_actions):
            finishing_actions = predecessor_order[safe_positions[first_actions]]
            exact_matches = root_children[first_actions].eq(
                goal_predecessors[finishing_actions]
            ).all(dim=(1, 2))
            first_actions = first_actions[exact_matches]
            finishing_actions = finishing_actions[exact_matches]
        if len(first_actions):
            joined_costs = costs[first_actions] + costs[finishing_actions]
            position = joined_costs.argmin()
            joined_cost = int(joined_costs[position])
            if joined_cost < exact_cost:
                exact_result = [
                    int(first_actions[position]),
                    int(finishing_actions[position]),
                ]
                exact_cost = joined_cost
        if exact_result is not None:
            best_result = exact_result
            best_cost = exact_cost
            best_depth = len(exact_result)
            if not continue_after_solution:
                return exact_result, {
                    "depth": len(exact_result),
                    "elapsed_seconds": time.perf_counter() - started,
                    "exact_root_two_macro": True,
                    "expanded_children": expanded,
                    "primitive_cost": int(exact_cost),
                    "oracle_audit": oracle_audit,
                }
    for depth in range(1, maximum_depth + 1):
        layer_beam_width = root_beam_width if depth == 1 and root_beam_width else beam_width
        forbidden = torch.where(
            last_actions >= 0,
            inverse_actions[last_actions.clamp_min(0)],
            torch.full_like(last_actions, -1),
        )
        proposed_parts: list[torch.Tensor] = []
        score_parts: list[torch.Tensor] = []
        if branch:
            if module is None:
                raise AssertionError("main policy module is missing")
            proposed, scores = policy_topk(
                module,
                logits,
                effects,
                topk=branch,
                chunk_size=action_chunk_size,
                forbidden=forbidden,
            )
            proposed_parts.append(proposed)
            score_parts.append(scores)
        for proposal_model in proposal_models if branch else []:
            if module is None:
                raise AssertionError("proposal policy module is missing")
            proposal_logits, _ = model_outputs(
                proposal_model,
                states,
                inference_batch_size,
                retain_logits=True,
            )
            if proposal_logits is None:
                raise AssertionError("proposal logits missing")
            extra_actions, extra_scores = policy_topk(
                module,
                proposal_logits,
                effects,
                topk=branch,
                chunk_size=action_chunk_size,
                forbidden=forbidden,
            )
            proposed_parts.append(extra_actions)
            score_parts.append(extra_scores)
        direct_logits_parts = [
            direct_action_logits(
                direct_model,
                states,
                inference_batch_size,
                remaining_depth=(policy_horizon or maximum_depth) - depth + 1,
            )
            for direct_model in direct_action_models
        ]
        if ensemble_direct_actions and direct_logits_parts:
            direct_logits_parts = [torch.stack(direct_logits_parts).mean(dim=0)]
        for direct_logits in direct_logits_parts:
            extra_actions, extra_scores = direct_policy_topk(
                direct_logits,
                costs,
                forbidden,
                topk=(
                    direct_root_branch
                    if depth == 1 and direct_root_branch
                    else (
                        direct_early_branch
                        if direct_early_branch
                        and depth <= direct_early_branch_depths
                        else direct_branch
                    )
                ),
                cost_stratified_topk=(
                    direct_cost_stratified_branch
                    if not direct_cost_stratified_depths
                    or depth <= direct_cost_stratified_depths
                    else 0
                ),
                cost_maximum=direct_cost_maximum,
                score_mode=direct_score_mode,
            )
            if (
                root_transition_ranker is not None
                and transition_rerank_branch
                and depth >= 2
                and (
                    not transition_rerank_depths
                    or depth <= transition_rerank_depths
                )
            ):
                extra_actions, transition_q = rerank_direct_transitions(
                    root_transition_ranker,
                    states,
                    extra_actions,
                    effects,
                    costs,
                    keep=transition_rerank_branch,
                    state_batch_size=transition_rerank_state_batch_size,
                    inference_batch_size=inference_batch_size,
                )
                extra_scores = -root_transition_score_scale * (
                    transition_q - transition_q.min(dim=1, keepdim=True).values
                )
            proposed_parts.append(extra_actions)
            score_parts.append(extra_scores)
        if cycle_proposal_branch and (
            not cycle_proposal_depths or depth <= cycle_proposal_depths
        ):
            cycle_actions, cycle_q = cycle_q_topk(
                states,
                effects,
                costs,
                forbidden,
                topk=cycle_proposal_branch,
                action_chunk_size=action_chunk_size,
                action_cost_weight=cycle_proposal_cost_weight,
            )
            proposed_parts.append(cycle_actions)
            score_parts.append(-cycle_proposal_score_scale * cycle_q)
        proposed = torch.cat(proposed_parts, dim=1)
        scores = torch.cat(score_parts, dim=1)
        candidate_width = proposed.shape[1]
        selected_effects = effects[proposed]
        children = states[:, None].expand(-1, candidate_width, -1, -1).gather(
            -1, selected_effects.long()
        )
        flat_children = children.flatten(0, 1)
        flat_actions = proposed.flatten()
        parent_indices = torch.arange(len(states), device=device).repeat_interleave(
            candidate_width
        )
        child_root_actions = (
            flat_actions if depth == 1 else root_actions[parent_indices]
        )
        child_cost = (cumulative_cost[:, None] + costs[proposed]).flatten()
        step_nll = -scores
        if depth == 1 and root_transition_ranker is not None:
            root_transition_values = factorized_combined_values(
                root_transition_ranker,
                flat_children,
                inference_batch_size,
            ).clamp_min(0)
            root_transition_q = child_cost + root_transition_values
            step_nll = (
                root_transition_policy_weight * step_nll
                + root_transition_score_scale
                * (root_transition_q - root_transition_q.min())
            )
        if policy_step_nll_cap > 0:
            capped = torch.full_like(step_nll, policy_step_nll_cap)
            step_nll = torch.minimum(step_nll, capped) + policy_tail_slope * torch.relu(
                step_nll - policy_step_nll_cap
            )
        child_nll = (cumulative_nll[:, None] + step_nll).flatten()
        expanded += len(flat_children)
        solved = flat_children.eq(identity).all(dim=(1, 2))
        if solved.any():
            solved_indices = torch.nonzero(solved, as_tuple=False).flatten()
            best = solved_indices[child_cost[solved_indices].argmin()]
            parent = int(parent_indices[best])
            result = paths[parent, : depth - 1].cpu().tolist()
            result.append(int(flat_actions[best]))
            solution_cost = int(child_cost[best])
            if solution_cost < best_cost:
                best_result = result
                best_cost = solution_cost
                best_depth = depth
            if not continue_after_solution:
                return result, {
                    "depth": depth,
                    "elapsed_seconds": time.perf_counter() - started,
                    "expanded_children": expanded,
                    "primitive_cost": solution_cost,
                    "oracle_audit": oracle_audit,
                }
        base_ranks = path_cost_weight * child_cost + policy_nll_weight * child_nll
        if value_weight == 0.0 and backup_weight == 0.0:
            child_values = torch.zeros(len(flat_children), device=device)
            ranks = base_ranks
        elif value_weight and value_prebeam_multiplier > 1:
            if backup_weight:
                raise ValueError("value prebeam is incompatible with Bellman backup")
            value_prebeam = min(
                len(base_ranks), layer_beam_width * value_prebeam_multiplier
            )
            value_keep = torch.topk(
                base_ranks, value_prebeam, largest=False
            ).indices
            _, kept_values = model_outputs(
                model,
                flat_children[value_keep],
                inference_batch_size,
                retain_logits=False,
            )
            child_values = torch.full(
                (len(flat_children),), torch.inf, device=device
            )
            child_values[value_keep] = kept_values
            ranks = torch.full_like(base_ranks, torch.inf)
            ranks[value_keep] = (
                base_ranks[value_keep]
                + value_weight * kept_values.clamp_min(0)
            )
        else:
            _, child_values = model_outputs(
                model, flat_children, inference_batch_size, retain_logits=False
            )
            ranks = base_ranks + value_weight * child_values.clamp_min(0)
        hashes = state_hashes(flat_children, zobrist)
        if one_macro_endgame and depth < maximum_depth:
            if (
                goal_predecessors is None
                or sorted_predecessor_hashes is None
                or predecessor_order is None
            ):
                raise AssertionError("one-macro endgame table is missing")
            positions = torch.searchsorted(sorted_predecessor_hashes, hashes)
            in_range = positions.lt(len(sorted_predecessor_hashes))
            safe_positions = positions.clamp_max(len(sorted_predecessor_hashes) - 1)
            hash_matches = in_range & sorted_predecessor_hashes[safe_positions].eq(hashes)
            match_indices = torch.nonzero(hash_matches, as_tuple=False).flatten()
            if len(match_indices):
                finishing_actions = predecessor_order[safe_positions[match_indices]]
                exact_matches = flat_children[match_indices].eq(
                    goal_predecessors[finishing_actions]
                ).all(dim=(1, 2))
                match_indices = match_indices[exact_matches]
                finishing_actions = finishing_actions[exact_matches]
            if len(match_indices):
                joined_costs = child_cost[match_indices] + costs[finishing_actions]
                best_position = joined_costs.argmin()
                best = match_indices[best_position]
                finish = finishing_actions[best_position]
                parent = int(parent_indices[best])
                result = paths[parent, : depth - 1].cpu().tolist()
                result.extend((int(flat_actions[best]), int(finish)))
                solution_cost = int(joined_costs[best_position])
                if solution_cost < best_cost:
                    best_result = result
                    best_cost = solution_cost
                    best_depth = depth + 1
                if not continue_after_solution:
                    return result, {
                        "depth": depth + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        "expanded_children": expanded,
                        "primitive_cost": solution_cost,
                        "one_macro_endgame": True,
                        "oracle_audit": oracle_audit,
                    }
        if continue_after_solution and solved.any():
            ranks = ranks.masked_fill(solved, torch.inf)
        if backup_weight and backup_branch and backup_prebeam_multiplier > 1:
            prebeam = min(len(ranks), layer_beam_width * backup_prebeam_multiplier)
            prekeep = torch.topk(ranks, prebeam, largest=False).indices
            backed_up = bellman_backup_values(
                module,
                model,
                flat_children[prekeep],
                flat_actions[prekeep],
                effects,
                costs,
                inverse_actions,
                branch=backup_branch,
                state_batch_size=backup_state_batch_size,
                action_chunk_size=action_chunk_size,
                inference_batch_size=inference_batch_size,
            )
            direct = child_values[prekeep].clamp_min(0)
            adjusted = ranks[prekeep] + backup_weight * (backed_up - direct)
            order = prekeep[torch.argsort(adjusted)]
        elif cycle_cost_weight and prebeam_multiplier > 1:
            prebeam = min(len(ranks), layer_beam_width * prebeam_multiplier)
            prekeep = torch.topk(ranks, prebeam, largest=False).indices
            structural = three_cycle_cost(flat_children[prekeep])
            adjusted = ranks[prekeep] + cycle_cost_weight * structural
            order = prekeep[torch.argsort(adjusted)]
        else:
            order = torch.argsort(ranks)
        ordered_hashes = hashes[order].cpu().numpy()
        keep_positions: list[int] = []
        seen: set[int] = set()
        if (
            root_action_diversity
            and depth >= 2
            and (
                not root_action_diversity_depths
                or depth <= root_action_diversity_depths
            )
        ):
            ordered_root_actions = child_root_actions[order].cpu().numpy()
            group_values, group_first_positions, group_candidate_counts = np.unique(
                ordered_root_actions, return_index=True, return_counts=True
            )
            group_rank_order = np.argsort(group_first_positions)
            if root_action_diversity_groups > 0:
                group_rank_order = group_rank_order[:root_action_diversity_groups]
            protected_groups = group_values[group_rank_order]
            protected_candidate_counts = group_candidate_counts[group_rank_order]
            quota = root_action_diversity_quota or max(
                1, layer_beam_width // max(len(protected_groups), 1)
            )
            reserved_target = min(
                layer_beam_width,
                int(np.minimum(protected_candidate_counts, quota).sum()),
            )
            protected_group_set = {int(group) for group in protected_groups}
            group_counts: dict[int, int] = {}
            for ordered_position, (state_hash, group_value) in enumerate(
                zip(ordered_hashes, ordered_root_actions, strict=True)
            ):
                key = int(state_hash)
                group = int(group_value)
                if (
                    key in seen
                    or group not in protected_group_set
                    or group_counts.get(group, 0) >= quota
                ):
                    continue
                seen.add(key)
                group_counts[group] = group_counts.get(group, 0) + 1
                keep_positions.append(ordered_position)
                if len(keep_positions) == reserved_target:
                    break
            if len(keep_positions) < layer_beam_width:
                for ordered_position, state_hash in enumerate(ordered_hashes):
                    key = int(state_hash)
                    if key in seen:
                        continue
                    seen.add(key)
                    keep_positions.append(ordered_position)
                    if len(keep_positions) == layer_beam_width:
                        break
        else:
            for ordered_position, state_hash in enumerate(ordered_hashes):
                key = int(state_hash)
                if key in seen:
                    continue
                seen.add(key)
                keep_positions.append(ordered_position)
                if len(keep_positions) == layer_beam_width:
                    break
        keep = order[torch.as_tensor(keep_positions, device=device)]
        if (
            oracle_state is not None
            and oracle_actions is not None
            and depth <= len(oracle_actions)
        ):
            oracle_action = int(oracle_actions[depth - 1])
            oracle_state = oracle_state.gather(
                -1, effects[oracle_action][None].long()
            )
            oracle_matches = flat_children.eq(oracle_state).all(dim=(1, 2))
            if oracle_matches.any():
                oracle_score = ranks[oracle_matches].min()
                oracle_rank = int((ranks < oracle_score).sum()) + 1
                oracle_proposed = True
                oracle_root_action = int(oracle_actions[0])
                oracle_root_group_mask = child_root_actions.eq(oracle_root_action)
                oracle_root_group_local_rank = (
                    int((ranks[oracle_root_group_mask] < oracle_score).sum()) + 1
                )
                root_group_best_scores = torch.full(
                    (len(costs),), torch.inf, device=device, dtype=ranks.dtype
                )
                root_group_best_scores.scatter_reduce_(
                    0,
                    child_root_actions.long(),
                    ranks,
                    reduce="amin",
                    include_self=True,
                )
                oracle_root_group_best_score = root_group_best_scores[
                    oracle_root_action
                ]
                oracle_root_group_global_rank = (
                    int(
                        (
                            root_group_best_scores
                            < oracle_root_group_best_score
                        ).sum()
                    )
                    + 1
                )
                counterfactual_ranks = {
                    f"minus_{delta}": int((ranks < oracle_score - delta).sum()) + 1
                    for delta in (1.0, 2.0, 4.0)
                }
            else:
                oracle_score = torch.tensor(float("inf"), device=device)
                oracle_rank = None
                oracle_proposed = False
                oracle_root_group_local_rank = None
                oracle_root_group_global_rank = None
                counterfactual_ranks = None
            retained = bool(flat_children[keep].eq(oracle_state).all(dim=(1, 2)).any())
            audit_row = {
                "layer": depth,
                "oracle_action": oracle_action,
                "oracle_global_rank": oracle_rank,
                "oracle_root_group_local_rank": oracle_root_group_local_rank,
                "oracle_root_group_global_rank": oracle_root_group_global_rank,
                "oracle_proposed": oracle_proposed,
                "oracle_retained": retained,
                "oracle_score": (
                    float(oracle_score) if oracle_proposed else None
                ),
                "oracle_counterfactual_ranks": counterfactual_ranks,
            }
            oracle_audit.append(audit_row)
            print(json.dumps({"oracle_audit": audit_row}), flush=True)
        selected_parents = parent_indices[keep]
        states = flat_children[keep]
        cumulative_cost = child_cost[keep]
        cumulative_nll = child_nll[keep]
        last_actions = flat_actions[keep]
        root_actions = child_root_actions[keep]
        paths = paths[selected_parents].clone()
        paths[:, depth - 1] = last_actions.to(torch.int32)
        if depth == harvest_frontier_depth:
            harvest_count = min(
                len(states), harvest_frontier_limit or len(states)
            )
            harvest_payload = {
                "states": states[:harvest_count].cpu().numpy(),
                "forward_costs": cumulative_cost[:harvest_count]
                .to(torch.int32)
                .cpu()
                .numpy(),
                "cumulative_nll": cumulative_nll[:harvest_count]
                .float()
                .cpu()
                .numpy(),
                "paths": paths[:harvest_count, :depth].cpu().numpy(),
            }
        del (
            child_cost,
            child_nll,
            child_values,
            child_root_actions,
            children,
            flat_actions,
            flat_children,
            hashes,
            order,
            parent_indices,
            proposed,
            proposed_parts,
            ranks,
            scores,
            score_parts,
            selected_effects,
            solved,
        )
        if "adjusted" in locals():
            del adjusted
        if "direct_logits" in locals():
            del direct_logits
        if "oracle_matches" in locals():
            del oracle_matches
        if "ordered_root_actions" in locals():
            del ordered_root_actions
        if "root_transition_q" in locals():
            del root_transition_q
        if "root_transition_values" in locals():
            del root_transition_values
        torch.cuda.empty_cache()
        logits, values = model_outputs(
            model, states, inference_batch_size, retain_logits=True
        )
        if logits is None:
            raise AssertionError("beam logits missing")
        if depth == 1 or depth % 5 == 0:
            print(
                json.dumps(
                    {
                        "beam": len(states),
                        "depth": depth,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "minimum_cost": int(cumulative_cost.min()),
                        "minimum_value": round(float(values.min()), 3),
                    }
                ),
                flush=True,
            )
    final_stats: dict[str, object] = {
        "depth": best_depth if best_depth is not None else maximum_depth,
        "elapsed_seconds": time.perf_counter() - started,
        "expanded_children": expanded,
        **(
            {"primitive_cost": int(best_cost), "continued_after_solution": True}
            if best_result is not None
            else {}
        ),
        "oracle_audit": oracle_audit,
    }
    if harvest_payload is not None:
        final_stats["_harvest_payload"] = harvest_payload
    return best_result, final_stats


def join_bidirectional_frontiers(
    root: np.ndarray,
    forward: dict[str, np.ndarray],
    backward_relative: dict[str, np.ndarray],
    inverse_actions: np.ndarray,
    action_costs: np.ndarray,
) -> tuple[list[int] | None, dict[str, int]]:
    """Join root->goal and inverse(root)->goal beams in the original state frame."""

    forward_states = np.asarray(forward["states"], dtype=np.uint8)
    backward_relative_states = np.asarray(
        backward_relative["states"], dtype=np.uint8
    )
    root_batch = np.broadcast_to(root, backward_relative_states.shape)
    backward_states = np.take_along_axis(
        root_batch, backward_relative_states, axis=2
    )
    forward_paths = np.asarray(forward["paths"], dtype=np.int64)
    backward_paths = np.asarray(backward_relative["paths"], dtype=np.int64)
    forward_costs = np.asarray(forward["forward_costs"], dtype=np.int64)
    backward_costs = np.asarray(
        backward_relative["forward_costs"], dtype=np.int64
    )

    best_forward: dict[bytes, int] = {}
    for position, state in enumerate(forward_states):
        key = state.tobytes()
        previous = best_forward.get(key)
        if previous is None or forward_costs[position] < forward_costs[previous]:
            best_forward[key] = position

    best_path: list[int] | None = None
    best_cost = np.iinfo(np.int64).max
    exact_matches = 0
    for backward_position, state in enumerate(backward_states):
        forward_position = best_forward.get(state.tobytes())
        if forward_position is None:
            continue
        exact_matches += 1
        tail = inverse_actions[backward_paths[backward_position, ::-1]]
        candidate = forward_paths[forward_position].tolist() + tail.tolist()
        candidate_cost = int(action_costs[np.asarray(candidate, dtype=np.int64)].sum())
        expected_cost = int(
            forward_costs[forward_position] + backward_costs[backward_position]
        )
        if candidate_cost != expected_cost:
            raise AssertionError("inverse-closed action costs disagree at meet")
        if candidate_cost < best_cost:
            best_path = candidate
            best_cost = candidate_cost
    return best_path, {
        "bidirectional_exact_matches": exact_matches,
        "bidirectional_forward_states": len(forward_states),
        "bidirectional_backward_states": len(backward_states),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    needs_main_policy = bool(
        args.branch
        or args.proposal_checkpoint
        or args.backup_branch
    )
    if args.checkpoint is None:
        if needs_main_policy:
            raise ValueError(
                "--checkpoint is required for main proposals, factorized value, or backups"
            )
        module: ModuleType | None = None
        checkpoint = None
        config = None
        if args.factorized_value_checkpoint is not None:
            factorized_value_model, _ = load_macro_factorized_value_checkpoint(
                str(args.factorized_value_checkpoint), device="cuda"
            )
            model: torch.nn.Module = FactorizedValueOnlyAdapter(
                factorized_value_model
            ).eval()
        else:
            model = NullPolicyValue().cuda().eval()
    else:
        module = load_module(args.train_module)
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        config = module.Config(
            action_count=checkpoint["model_config"]["action_count"]
        )
        policy_model = module.MacroEffectPolicyValueNet(config).cuda()
        policy_model.load_state_dict(checkpoint["model_state_dict"])
        policy_model.eval()
        if args.factorized_value_checkpoint is not None:
            factorized_value_model, _ = load_macro_factorized_value_checkpoint(
                str(args.factorized_value_checkpoint), device="cuda"
            )
            model = FactorizedValueAdapter(policy_model, factorized_value_model).eval()
        else:
            model = policy_model
    proposal_models: list[torch.nn.Module] = []
    for proposal_path in args.proposal_checkpoint:
        if module is None or checkpoint is None or config is None:
            raise AssertionError("proposal checkpoint requires a main policy")
        proposal_checkpoint = torch.load(
            proposal_path, map_location="cpu", weights_only=True
        )
        if proposal_checkpoint["model_config"] != checkpoint["model_config"]:
            raise ValueError("proposal checkpoint architecture disagrees")
        proposal_model = module.MacroEffectPolicyValueNet(config).cuda()
        proposal_model.load_state_dict(proposal_checkpoint["model_state_dict"])
        proposal_model.eval()
        proposal_models.append(proposal_model)
    direct_action_models: list[MacroActionPolicyNet] = []
    for direct_path in args.direct_action_checkpoint:
        direct_model, _ = load_macro_action_policy_checkpoint(
            str(direct_path), device="cuda"
        )
        direct_action_models.append(direct_model)
    root_transition_ranker: MacroFactorizedPrimitiveValueNet | None = None
    if args.root_transition_ranker_checkpoint is not None:
        root_transition_ranker, _ = load_macro_factorized_value_checkpoint(
            str(args.root_transition_ranker_checkpoint), device="cuda"
        )
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(np.uint8, copy=False)
    costs_np = np.load(args.action_costs, allow_pickle=False).astype(np.int32, copy=False)
    inverse_np = np.load(args.inverse_actions, allow_pickle=False).astype(np.int64, copy=False)
    if any(
        direct_model.config.action_count != len(effects_np)
        for direct_model in direct_action_models
    ):
        raise ValueError("direct-action checkpoint and action table disagree")
    if config is not None and len(effects_np) != config.action_count and (
        args.branch or args.backup_branch or args.proposal_checkpoint
    ):
        raise ValueError(
            "main policy/proposal action count disagrees with the action table"
        )
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).float().cuda()
    inverse_actions = torch.from_numpy(inverse_np).cuda()
    with np.load(args.states, allow_pickle=False) as payload:
        source_states = payload["states"].astype(np.uint8, copy=False)
        source_values = payload["search_value_targets"].astype(np.float32, copy=False)
        source_depths = (
            payload["walk_depths"].astype(np.int16, copy=False)
            if args.audit_oracle_retention or args.random_count
            else None
        )
        source_oracle_actions = (
            payload["teacher_solution_actions"].astype(np.int64, copy=False)
            if args.audit_oracle_retention
            else None
        )
    groups = np.load(args.group_ids, allow_pickle=False)
    rng = np.random.default_rng(args.seed)
    selected: list[tuple[str, int]] = []
    index_text = args.indices
    if args.indices_file is not None:
        index_text = index_text + "," + args.indices_file.read_text(encoding="utf-8")
    if index_text:
        selected.extend(
            (f"index:{value}", int(value))
            for value in index_text.replace("_", ",").replace("\n", ",").split(",")
            if value.strip()
        )
    if args.index_limit:
        if args.index_start < 0 or args.index_limit < 0:
            raise ValueError("index slice values must be non-negative")
        stop = min(len(source_states), args.index_start + args.index_limit)
        selected.extend(
            (f"index:{value}", value) for value in range(args.index_start, stop)
        )
    if args.random_count:
        if args.random_count < 0 or args.random_depth < 0:
            raise ValueError("random source selection values must be non-negative")
        if source_depths is None:
            raise AssertionError("walk depths were not loaded for random selection")
        candidates = (
            np.flatnonzero(source_depths == args.random_depth)
            if args.random_depth
            else np.arange(len(source_states), dtype=np.int64)
        )
        if args.random_count > len(candidates):
            raise ValueError("random-count exceeds the eligible source rows")
        random_indices = rng.choice(
            candidates, size=args.random_count, replace=False
        )
        selected.extend(
            (f"index:{int(value)}", int(value)) for value in random_indices
        )
    if args.pids:
        for token in args.pids.replace("_", ",").split(","):
            pid = int(token)
            candidates = np.flatnonzero(groups == pid)
            if not len(candidates):
                raise ValueError(f"PID {pid} is absent from the state source")
            index = int(candidates[np.argmax(source_values[candidates])])
            selected.append((f"pid:{pid}", index))
    if args.all_pids:
        for pid_value in np.unique(groups):
            pid = int(pid_value)
            candidates = np.flatnonzero(groups == pid_value)
            index = int(candidates[np.argmax(source_values[candidates])])
            selected.append((f"pid:{pid}", index))
    if not selected:
        raise ValueError("provide --indices or --pids")
    zobrist = torch.from_numpy(
        rng.integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=(144, 24),
            dtype=np.int64,
        )
    ).cuda()
    rows: list[dict[str, object]] = []
    harvested: list[dict[str, np.ndarray]] = []
    for label, index in selected:
        oracle_actions = (
            source_oracle_actions[index, : int(source_depths[index])]
            if source_oracle_actions is not None and source_depths is not None
            else None
        )
        solve_kwargs: dict[str, object] = {
            "beam_width": args.beam,
            "root_beam_width": args.root_beam,
            "root_transition_score_scale": args.root_transition_score_scale,
            "root_transition_policy_weight": args.root_transition_policy_weight,
            "transition_rerank_branch": args.transition_rerank_branch,
            "transition_rerank_depths": args.transition_rerank_depths,
            "transition_rerank_state_batch_size": (
                args.transition_rerank_state_batch_size
            ),
            "branch": args.branch,
            "direct_branch": args.direct_branch,
            "direct_early_branch": args.direct_early_branch,
            "direct_early_branch_depths": args.direct_early_branch_depths,
            "direct_root_branch": args.direct_root_branch,
            "direct_cost_stratified_branch": args.direct_cost_stratified_branch,
            "direct_cost_maximum": args.direct_cost_maximum,
            "direct_cost_stratified_depths": args.direct_cost_stratified_depths,
            "direct_score_mode": args.direct_score_mode,
            "ensemble_direct_actions": args.ensemble_direct_actions,
            "maximum_depth": args.maximum_depth,
            "path_cost_weight": args.path_cost_weight,
            "value_weight": args.value_weight,
            "value_prebeam_multiplier": args.value_prebeam_multiplier,
            "policy_nll_weight": args.policy_nll_weight,
            "policy_step_nll_cap": args.policy_step_nll_cap,
            "policy_tail_slope": args.policy_tail_slope,
            "prebeam_multiplier": args.prebeam_multiplier,
            "cycle_cost_weight": args.cycle_cost_weight,
            "cycle_proposal_branch": args.cycle_proposal_branch,
            "cycle_proposal_cost_weight": args.cycle_proposal_cost_weight,
            "cycle_proposal_score_scale": args.cycle_proposal_score_scale,
            "cycle_proposal_depths": args.cycle_proposal_depths,
            "backup_prebeam_multiplier": args.backup_prebeam_multiplier,
            "backup_branch": args.backup_branch,
            "backup_state_batch_size": args.backup_state_batch_size,
            "backup_weight": args.backup_weight,
            "action_chunk_size": args.action_chunk_size,
            "inference_batch_size": args.inference_batch_size,
            "zobrist": zobrist,
            "oracle_actions": oracle_actions,
            "one_macro_endgame": args.one_macro_endgame,
            "exact_root_two_macro": args.exact_root_two_macro,
            "continue_after_solution": args.continue_after_solution,
            "root_action_diversity": args.root_action_diversity,
            "root_action_diversity_depths": args.root_action_diversity_depths,
            "root_action_diversity_groups": args.root_action_diversity_groups,
            "root_action_diversity_quota": args.root_action_diversity_quota,
            "harvest_frontier_depth": args.harvest_frontier_depth,
            "harvest_frontier_limit": args.harvest_frontier_per_root,
        }
        if args.bidirectional_meet_depth:
            meet_depth = args.bidirectional_meet_depth
            meet_kwargs = {
                **solve_kwargs,
                "maximum_depth": meet_depth,
                "policy_horizon": args.maximum_depth,
                "continue_after_solution": True,
                "harvest_frontier_depth": meet_depth,
                "harvest_frontier_limit": 0,
            }
            forward_path, forward_stats = solve(
                module,
                model,
                proposal_models,
                direct_action_models,
                root_transition_ranker,
                source_states[index],
                effects,
                costs,
                inverse_actions,
                **meet_kwargs,
            )
            forward_frontier = forward_stats.pop("_harvest_payload", None)
            inverse_root = np.argsort(source_states[index], axis=1).astype(np.uint8)
            backward_oracle = (
                inverse_np[oracle_actions[::-1]]
                if oracle_actions is not None
                else None
            )
            backward_path, backward_stats = solve(
                module,
                model,
                proposal_models,
                direct_action_models,
                root_transition_ranker,
                inverse_root,
                effects,
                costs,
                inverse_actions,
                **{**meet_kwargs, "oracle_actions": backward_oracle},
            )
            backward_frontier = backward_stats.pop("_harvest_payload", None)
            if not isinstance(forward_frontier, dict) or not isinstance(
                backward_frontier, dict
            ):
                raise AssertionError("bidirectional frontier harvest is missing")
            meet_path, meet_stats = join_bidirectional_frontiers(
                source_states[index],
                forward_frontier,
                backward_frontier,
                inverse_np,
                costs_np,
            )
            candidates: list[list[int]] = []
            if forward_path is not None:
                candidates.append(forward_path)
            if backward_path is not None:
                candidates.append(inverse_np[np.asarray(backward_path)[::-1]].tolist())
            if meet_path is not None:
                candidates.append(meet_path)
            path = min(
                candidates,
                key=lambda candidate: int(
                    costs_np[np.asarray(candidate, dtype=np.int64)].sum()
                ),
                default=None,
            )
            stats = {
                "depth": len(path) if path is not None else 2 * meet_depth,
                "elapsed_seconds": float(forward_stats["elapsed_seconds"])
                + float(backward_stats["elapsed_seconds"]),
                "expanded_children": int(forward_stats["expanded_children"])
                + int(backward_stats["expanded_children"]),
                "oracle_audit": forward_stats["oracle_audit"],
                "bidirectional_backward_oracle_audit": backward_stats[
                    "oracle_audit"
                ],
                "bidirectional_meet_depth": meet_depth,
                **meet_stats,
            }
            if path is not None:
                stats["primitive_cost"] = int(
                    costs_np[np.asarray(path, dtype=np.int64)].sum()
                )
        else:
            path, stats = solve(
                module,
                model,
                proposal_models,
                direct_action_models,
                root_transition_ranker,
                source_states[index],
                effects,
                costs,
                inverse_actions,
                **solve_kwargs,
            )
        harvest_payload = stats.pop("_harvest_payload", None)
        if harvest_payload is not None:
            if not isinstance(harvest_payload, dict):
                raise AssertionError("invalid frontier harvest payload")
            harvest_count = len(harvest_payload["states"])
            harvest_payload["root_source_indices"] = np.full(
                harvest_count, index, dtype=np.int64
            )
            harvest_payload["teacher_upper_bounds"] = np.full(
                harvest_count, source_values[index], dtype=np.float32
            )
            harvested.append(harvest_payload)
        replay = source_states[index].copy()
        if path is not None:
            for action in path:
                replay = np.take_along_axis(replay, effects_np[action], axis=-1)
        replay_ok = path is not None and np.array_equal(
            replay, np.broadcast_to(np.arange(24, dtype=np.uint8), replay.shape)
        )
        if path is not None and not replay_ok:
            raise AssertionError(f"{label}: beam path failed exact effect replay")
        row = {
            **stats,
            "index": index,
            "label": label,
            "path": path,
            "replay_verified": replay_ok,
            "solved": path is not None,
            "teacher_upper_bound": float(source_values[index]),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "beam": args.beam,
        "root_beam": args.root_beam,
        "branch": args.branch,
        "bidirectional_meet_depth": args.bidirectional_meet_depth,
        "root_action_diversity": args.root_action_diversity,
        "root_action_diversity_depths": args.root_action_diversity_depths,
        "root_action_diversity_groups": args.root_action_diversity_groups,
        "root_action_diversity_quota": args.root_action_diversity_quota,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "direct_action_checkpoints": [
            str(path) for path in args.direct_action_checkpoint
        ],
        "ensemble_direct_actions": args.ensemble_direct_actions,
        "direct_branch": args.direct_branch,
        "direct_early_branch": args.direct_early_branch,
        "direct_early_branch_depths": args.direct_early_branch_depths,
        "direct_root_branch": args.direct_root_branch,
        "direct_cost_maximum": args.direct_cost_maximum,
        "direct_cost_stratified_branch": args.direct_cost_stratified_branch,
        "direct_cost_stratified_depths": args.direct_cost_stratified_depths,
        "direct_score_mode": args.direct_score_mode,
        "factorized_value_checkpoint": (
            str(args.factorized_value_checkpoint)
            if args.factorized_value_checkpoint is not None
            else None
        ),
        "root_transition_ranker_checkpoint": (
            str(args.root_transition_ranker_checkpoint)
            if args.root_transition_ranker_checkpoint is not None
            else None
        ),
        "root_transition_score_scale": args.root_transition_score_scale,
        "root_transition_policy_weight": args.root_transition_policy_weight,
        "transition_rerank_branch": args.transition_rerank_branch,
        "transition_rerank_depths": args.transition_rerank_depths,
        "transition_rerank_state_batch_size": (
            args.transition_rerank_state_batch_size
        ),
        "cycle_proposal_branch": args.cycle_proposal_branch,
        "cycle_proposal_cost_weight": args.cycle_proposal_cost_weight,
        "cycle_proposal_score_scale": args.cycle_proposal_score_scale,
        "cycle_proposal_depths": args.cycle_proposal_depths,
        "one_macro_endgame": args.one_macro_endgame,
        "exact_root_two_macro": args.exact_root_two_macro,
        "continue_after_solution": args.continue_after_solution,
        "path_cost_weight": args.path_cost_weight,
        "policy_nll_weight": args.policy_nll_weight,
        "policy_step_nll_cap": args.policy_step_nll_cap,
        "policy_tail_slope": args.policy_tail_slope,
        "value_weight": args.value_weight,
        "value_prebeam_multiplier": args.value_prebeam_multiplier,
        "harvest_frontier_depth": args.harvest_frontier_depth,
        "harvest_frontier_per_root": args.harvest_frontier_per_root,
        "harvest_frontier_out": (
            str(args.harvest_frontier_out)
            if args.harvest_frontier_out is not None
            else None
        ),
        "harvested_states": sum(len(item["states"]) for item in harvested),
        "proposal_checkpoints": [str(path) for path in args.proposal_checkpoint],
        "rows": rows,
        "solved": sum(bool(row["solved"]) for row in rows),
    }
    if args.harvest_frontier_out is not None:
        if not harvested:
            raise RuntimeError("frontier harvest produced no retained states")
        args.harvest_frontier_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.harvest_frontier_out,
            **{
                name: np.concatenate([item[name] for item in harvested], axis=0)
                for name in harvested[0]
            },
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
