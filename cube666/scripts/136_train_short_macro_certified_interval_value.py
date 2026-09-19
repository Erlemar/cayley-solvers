"""Train a short-macro value critic from certified cost intervals and ranks.

The trace supplies constructive upper bounds.  A portfolio of feasible dual
cycle metrics supplies lower bounds.  Candidate children are pairwise ranked
only when the constructive Q upper bound is strictly below the candidate Q
lower bound, so failed searches and unknown children never become negatives.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_action_policy import (  # noqa: E402
    MacroActionPolicyNet,
    load_macro_action_policy_checkpoint,
)
from cube666.macro_factorized_value import (  # noqa: E402
    MacroFactorizedPrimitiveValueNet,
    MacroFactorizedValueConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--direct-action-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proposal-branch", type=int, default=128)
    parser.add_argument("--hard-pairs", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--interval-weight", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=4.0)
    parser.add_argument("--trace-return-weight", type=float, default=0.0)
    parser.add_argument("--minimum-depth", type=int, default=2)
    parser.add_argument("--maximum-depth", type=int, default=6)
    parser.add_argument("--local-dim", type=int, default=256)
    parser.add_argument("--local-blocks", type=int, default=4)
    parser.add_argument("--global-dim", type=int, default=768)
    parser.add_argument("--global-blocks", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=136666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def cycle_distance_vectors(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.uint8)
    flat = states.reshape(-1, 6, 24)
    output = np.empty((len(flat), 6), dtype=np.float32)
    for row in range(len(flat)):
        for cluster in range(6):
            permutation = flat[row, cluster]
            seen = np.zeros(24, dtype=np.bool_)
            cycles = 0
            for start in range(24):
                if seen[start]:
                    continue
                cycles += 1
                cursor = start
                while not seen[cursor]:
                    seen[cursor] = True
                    cursor = int(permutation[cursor])
            output[row, cluster] = (24 - cycles) / 2.0
    return output.reshape(states.shape[:-2] + (6,))


def build_subset_dual_weights(
    action_effects: np.ndarray, action_costs: np.ndarray
) -> np.ndarray:
    action_vectors = cycle_distance_vectors(action_effects)
    weights: list[np.ndarray] = []
    for mask in range(1, 1 << 6):
        selected = np.asarray([(mask >> bit) & 1 for bit in range(6)], dtype=bool)
        distance = action_vectors[:, selected].sum(axis=1)
        active = distance > 0
        ratio = float(np.min(action_costs[active] / distance[active]))
        weight = np.zeros(6, dtype=np.float32)
        weight[selected] = ratio
        weights.append(weight)
    output = np.stack(weights)
    if np.any(action_vectors @ output.T > action_costs[:, None] + 1e-5):
        raise AssertionError("constructed an infeasible cycle dual")
    return output


@torch.inference_mode()
def cycle_cost_lower_bounds(
    states: torch.Tensor, dual_weights: torch.Tensor
) -> torch.Tensor:
    flat = states.reshape(-1, 6, 24)
    starts = torch.arange(24, device=states.device).view(1, 1, 24)
    starts = starts.expand(len(flat), 6, -1)
    cursor = starts
    minimum = starts
    for _ in range(24):
        cursor = flat.gather(-1, cursor.long()).long()
        minimum = torch.minimum(minimum, cursor)
    cycles = minimum.eq(starts).sum(dim=2)
    vectors = (24 - cycles).float() * 0.5
    return (vectors @ dual_weights.T).max(dim=1).values.reshape(states.shape[:-2])


def combined_value(
    model: MacroFactorizedPrimitiveValueNet,
    outputs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    total, clusters = outputs
    return 0.5 * (
        total.float() * model.config.total_scale
        + clusters.float().sum(dim=1) * model.config.cluster_scale
    )


def reconstruct_parents(
    roots: np.ndarray,
    solution_actions: np.ndarray,
    effects: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> np.ndarray:
    parents = roots[rows].copy()
    for step in range(int(prefixes.max(initial=0))):
        active = prefixes > step
        actions = solution_actions[rows[active], step].astype(np.int64, copy=False)
        parents[active] = np.take_along_axis(
            parents[active], effects[actions], axis=-1
        )
    return parents


@torch.inference_mode()
def propose_actions(
    models: list[MacroActionPolicyNet],
    states: torch.Tensor,
    remaining_depths: torch.Tensor,
    branch: int,
) -> torch.Tensor:
    proposals: list[torch.Tensor] = []
    for model in models:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(states, remaining_depth=remaining_depths).float()
        proposals.append(logits.topk(min(branch, logits.shape[1]), dim=1).indices)
    return torch.cat(proposals, dim=1)


@dataclass
class FrontierBatch:
    parents: torch.Tensor
    positive_children: torch.Tensor
    negative_children: torch.Tensor
    parent_lower: torch.Tensor
    parent_upper: torch.Tensor
    positive_lower: torch.Tensor
    positive_upper: torch.Tensor
    negative_lower: torch.Tensor
    negative_upper: torch.Tensor
    positive_cost: torch.Tensor
    negative_cost: torch.Tensor
    pair_margin: torch.Tensor
    pair_mask: torch.Tensor
    certified_pair_count: int
    parents_with_pair: int
    proposal_recall: int


def build_frontier_batch(
    *,
    roots: np.ndarray,
    depths: np.ndarray,
    solution_actions: np.ndarray,
    effects_np: np.ndarray,
    costs_np: np.ndarray,
    inverse_np: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
    effects: torch.Tensor,
    costs: torch.Tensor,
    inverse: torch.Tensor,
    dual_weights: torch.Tensor,
    proposal_models: list[MacroActionPolicyNet],
    proposal_branch: int,
    hard_pairs: int,
    ranking_margin: float,
) -> FrontierBatch:
    parents_np = reconstruct_parents(
        roots, solution_actions, effects_np, rows, prefixes
    )
    teacher_np = solution_actions[rows, prefixes].astype(np.int64, copy=False)
    previous_np = np.full(len(rows), -1, dtype=np.int64)
    has_previous = prefixes > 0
    previous_np[has_previous] = solution_actions[
        rows[has_previous], prefixes[has_previous] - 1
    ]
    forbidden_np = np.full(len(rows), -1, dtype=np.int64)
    forbidden_np[has_previous] = inverse_np[previous_np[has_previous]]
    parent_upper_np = np.asarray(
        [
            costs_np[
                solution_actions[row, prefix : int(depths[row])].astype(
                    np.int64, copy=False
                )
            ].sum(dtype=np.float64)
            for row, prefix in zip(rows, prefixes, strict=True)
        ],
        dtype=np.float32,
    )

    parents = torch.from_numpy(parents_np).cuda()
    teacher = torch.from_numpy(teacher_np).cuda()
    forbidden = torch.from_numpy(forbidden_np).cuda()
    parent_upper = torch.from_numpy(parent_upper_np).cuda()
    remaining = torch.from_numpy(
        (depths[rows] - prefixes).astype(np.int64, copy=False)
    ).cuda()
    proposals = propose_actions(
        proposal_models, parents, remaining, proposal_branch
    ).sort(dim=1).values
    unique = torch.ones_like(proposals, dtype=torch.bool)
    unique[:, 1:] = proposals[:, 1:].ne(proposals[:, :-1])
    valid = unique & proposals.ne(forbidden[:, None])
    proposal_recall = int((valid & proposals.eq(teacher[:, None])).any(dim=1).sum())
    selected_effects = effects[proposals]
    children = parents[:, None].expand_as(selected_effects).gather(
        -1, selected_effects.long()
    )
    child_lower = cycle_cost_lower_bounds(children, dual_weights)
    candidate_q_lower = costs[proposals] + child_lower
    certified = (
        valid
        & proposals.ne(teacher[:, None])
        & candidate_q_lower.gt(parent_upper[:, None])
    )
    gap = (candidate_q_lower - parent_upper[:, None]).masked_fill(
        ~certified, torch.inf
    )
    keep = min(hard_pairs, proposals.shape[1])
    tight_gap, tight_positions = gap.topk(keep, dim=1, largest=False)
    pair_mask = torch.isfinite(tight_gap)
    negative_actions = proposals.gather(1, tight_positions)
    gather_index = tight_positions[:, :, None, None].expand(-1, -1, 6, 24)
    negative_children = children.gather(1, gather_index)
    negative_lower = child_lower.gather(1, tight_positions)
    negative_cost = costs[negative_actions]
    negative_upper = parent_upper[:, None] + costs[inverse[negative_actions]]

    positive_children = parents.gather(-1, effects[teacher].long())
    positive_cost = costs[teacher]
    positive_upper = parent_upper - positive_cost
    parent_lower = cycle_cost_lower_bounds(parents, dual_weights)
    positive_lower = cycle_cost_lower_bounds(positive_children, dual_weights)
    pair_margin = torch.minimum(
        torch.full_like(tight_gap, ranking_margin), tight_gap * 0.5
    ).masked_fill(~pair_mask, 0.0)
    # Proposal and bound helpers intentionally run under inference_mode.  Clone
    # their products after leaving those helper contexts so autograd may safely
    # consume them as constant targets and masks during critic training.
    return FrontierBatch(
        parents=parents.clone(),
        positive_children=positive_children.clone(),
        negative_children=negative_children.clone(),
        parent_lower=parent_lower.clone(),
        parent_upper=parent_upper.clone(),
        positive_lower=positive_lower.clone(),
        positive_upper=positive_upper.clone(),
        negative_lower=negative_lower.clone(),
        negative_upper=negative_upper.clone(),
        positive_cost=positive_cost.clone(),
        negative_cost=negative_cost.clone(),
        pair_margin=pair_margin.clone(),
        pair_mask=pair_mask.clone(),
        certified_pair_count=int(pair_mask.sum()),
        parents_with_pair=int(pair_mask.any(dim=1).sum()),
        proposal_recall=proposal_recall,
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def interval_violation(
    prediction: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    violation = F.smooth_l1_loss(
        prediction,
        torch.maximum(lower, torch.minimum(prediction.detach(), upper)),
        beta=1.0,
        reduction="none",
    )
    if mask is None:
        return violation.mean()
    return masked_mean(violation, mask)


@torch.inference_mode()
def evaluate(
    model: MacroFactorizedPrimitiveValueNet,
    batches: list[FrontierBatch],
) -> dict[str, float | int]:
    model.eval()
    pair_correct = 0
    pair_count = 0
    parent_inside = 0
    positive_inside = 0
    state_count = 0
    lower_errors: list[torch.Tensor] = []
    upper_errors: list[torch.Tensor] = []
    parent_predictions: list[torch.Tensor] = []
    parent_returns: list[torch.Tensor] = []
    for batch in batches:
        count = len(batch.parents)
        hard = batch.negative_children.shape[1]
        inputs = torch.cat(
            (
                batch.parents,
                batch.positive_children,
                batch.negative_children.flatten(0, 1),
            ),
            dim=0,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = combined_value(model, model(inputs))
        parent_value = values[:count]
        positive_value = values[count : 2 * count]
        negative_value = values[2 * count :].reshape(count, hard)
        positive_q = batch.positive_cost + positive_value
        negative_q = batch.negative_cost + negative_value
        correct = negative_q.gt(positive_q[:, None]) & batch.pair_mask
        pair_correct += int(correct.sum())
        pair_count += int(batch.pair_mask.sum())
        parent_inside += int(
            (
                parent_value.ge(batch.parent_lower - 1e-5)
                & parent_value.le(batch.parent_upper + 1e-5)
            ).sum()
        )
        positive_inside += int(
            (
                positive_value.ge(batch.positive_lower - 1e-5)
                & positive_value.le(batch.positive_upper + 1e-5)
            ).sum()
        )
        state_count += count
        lower_errors.append(
            torch.relu(batch.parent_lower - parent_value).detach().cpu()
        )
        upper_errors.append(
            torch.relu(parent_value - batch.parent_upper).detach().cpu()
        )
        parent_predictions.append(parent_value.detach().cpu())
        parent_returns.append(batch.parent_upper.detach().cpu())
    lower_error = torch.cat(lower_errors)
    upper_error = torch.cat(upper_errors)
    predicted = torch.cat(parent_predictions).numpy()
    returns = torch.cat(parent_returns).numpy()
    return_error = predicted - returns
    return {
        "certified_pair_accuracy": pair_correct / max(pair_count, 1),
        "certified_pairs": pair_count,
        "parent_interval_coverage": parent_inside / max(state_count, 1),
        "positive_interval_coverage": positive_inside / max(state_count, 1),
        "parent_lower_violation_mean": float(lower_error.mean()),
        "parent_upper_violation_mean": float(upper_error.mean()),
        "return_bias": float(return_error.mean()),
        "return_correlation": float(np.corrcoef(predicted, returns)[0, 1]),
        "return_mae": float(np.abs(return_error).mean()),
        "samples": state_count,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.hard_pairs <= 0 or args.proposal_branch <= 0:
        raise ValueError("hard-pairs and proposal-branch must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    root = args.dataset_dir
    with np.load(root / "teacher.npz", allow_pickle=False) as payload:
        roots = payload["states"].astype(np.uint8, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        solution_actions = payload["teacher_solution_actions"].astype(
            np.int32, copy=False
        )
    groups = np.load(root / "source_state_ids.npy", allow_pickle=False)
    effects_np = np.load(root / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs_np = np.load(root / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    inverse_np = np.load(root / "inverse_actions.npy", allow_pickle=False).astype(
        np.int64, copy=False
    )
    depth_mask = (depths >= args.minimum_depth) & (depths <= args.maximum_depth)
    train_rows = np.flatnonzero((groups % 10 != 0) & depth_mask)
    heldout_rows = np.flatnonzero((groups % 10 == 0) & depth_mask)
    if not len(train_rows) or not len(heldout_rows):
        raise ValueError("depth filters left an empty train or held-out split")

    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).cuda()
    inverse = torch.from_numpy(inverse_np).cuda()
    dual_weights_np = build_subset_dual_weights(effects_np, costs_np)
    dual_weights = torch.from_numpy(dual_weights_np).cuda()
    proposal_models: list[MacroActionPolicyNet] = []
    proposal_sources: list[dict[str, object]] = []
    for checkpoint_path in args.direct_action_checkpoint:
        proposal_model, payload = load_macro_action_policy_checkpoint(
            str(checkpoint_path), device="cuda"
        )
        if proposal_model.config.action_count != len(effects_np):
            raise ValueError("proposal checkpoint and action table disagree")
        proposal_models.append(proposal_model)
        proposal_sources.append(
            {
                "path": str(checkpoint_path),
                "kind": payload.get("kind"),
                "depth_conditioned": proposal_model.config.depth_conditioned,
            }
        )

    source_payload: dict[str, object] | None = None
    if args.init_checkpoint is not None:
        source_payload = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        config = MacroFactorizedValueConfig(
            **source_payload["factorized_value_config"]
        )
        model = MacroFactorizedPrimitiveValueNet(config).cuda()
        model.load_state_dict(source_payload["model_state_dict"])
    else:
        config = MacroFactorizedValueConfig(
            local_dim=args.local_dim,
            local_blocks=args.local_blocks,
            global_dim=args.global_dim,
            global_blocks=args.global_blocks,
            dropout=args.dropout,
        )
        model = MacroFactorizedPrimitiveValueNet(config).cuda()
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-3)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    eval_rows = rng.choice(
        heldout_rows, size=min(args.eval_samples, len(heldout_rows)), replace=False
    )
    eval_prefixes = np.asarray(
        [rng.integers(0, int(depths[row])) for row in eval_rows], dtype=np.int64
    )
    eval_batches: list[FrontierBatch] = []
    for start in range(0, len(eval_rows), args.batch_size):
        stop = min(start + args.batch_size, len(eval_rows))
        eval_batches.append(
            build_frontier_batch(
                roots=roots,
                depths=depths,
                solution_actions=solution_actions,
                effects_np=effects_np,
                costs_np=costs_np,
                inverse_np=inverse_np,
                rows=eval_rows[start:stop],
                prefixes=eval_prefixes[start:stop],
                effects=effects,
                costs=costs,
                inverse=inverse,
                dual_weights=dual_weights,
                proposal_models=proposal_models,
                proposal_branch=args.proposal_branch,
                hard_pairs=args.hard_pairs,
                ranking_margin=args.ranking_margin,
            )
        )
    before = evaluate(model, eval_batches)
    identity = torch.arange(24, dtype=torch.uint8, device="cuda")[None].expand(
        6, -1
    )
    started = time.perf_counter()
    last: dict[str, float | int] = {}
    total_certified_pairs = 0
    total_parents_with_pair = 0
    total_proposal_recall = 0
    model.train()
    for step in range(1, args.steps + 1):
        rows = rng.choice(train_rows, size=args.batch_size, replace=True)
        prefixes = np.asarray(
            [rng.integers(0, int(depths[row])) for row in rows], dtype=np.int64
        )
        batch = build_frontier_batch(
            roots=roots,
            depths=depths,
            solution_actions=solution_actions,
            effects_np=effects_np,
            costs_np=costs_np,
            inverse_np=inverse_np,
            rows=rows,
            prefixes=prefixes,
            effects=effects,
            costs=costs,
            inverse=inverse,
            dual_weights=dual_weights,
            proposal_models=proposal_models,
            proposal_branch=args.proposal_branch,
            hard_pairs=args.hard_pairs,
            ranking_margin=args.ranking_margin,
        )
        total_certified_pairs += batch.certified_pair_count
        total_parents_with_pair += batch.parents_with_pair
        total_proposal_recall += batch.proposal_recall
        hard = batch.negative_children.shape[1]
        model_inputs = torch.cat(
            (
                batch.parents,
                batch.positive_children,
                batch.negative_children.flatten(0, 1),
                identity[None].expand(args.batch_size, -1, -1),
            ),
            dim=0,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = combined_value(model, train_model(model_inputs))
            parent_value = values[: args.batch_size]
            positive_value = values[args.batch_size : 2 * args.batch_size]
            negative_value = values[
                2 * args.batch_size : 2 * args.batch_size + args.batch_size * hard
            ].reshape(args.batch_size, hard)
            identity_value = values[-args.batch_size :]
            parent_interval = interval_violation(
                parent_value, batch.parent_lower, batch.parent_upper
            )
            positive_interval = interval_violation(
                positive_value, batch.positive_lower, batch.positive_upper
            )
            negative_interval = interval_violation(
                negative_value,
                batch.negative_lower,
                batch.negative_upper,
                batch.pair_mask,
            )
            positive_q = batch.positive_cost + positive_value
            negative_q = batch.negative_cost + negative_value
            ranking_loss = masked_mean(
                F.relu(
                    batch.pair_margin
                    + positive_q[:, None]
                    - negative_q
                ),
                batch.pair_mask,
            )
            trace_return_loss = F.smooth_l1_loss(
                parent_value, batch.parent_upper, beta=2.0
            ) + F.smooth_l1_loss(
                positive_value, batch.positive_upper, beta=2.0
            )
            identity_loss = F.smooth_l1_loss(
                identity_value, torch.zeros_like(identity_value), beta=1.0
            )
            nonnegative = F.relu(-values).mean()
            loss = (
                args.interval_weight
                * (parent_interval + positive_interval + negative_interval)
                + args.ranking_weight * ranking_loss
                + args.trace_return_weight * trace_return_loss
                + identity_loss
                + 0.05 * nonnegative
            ) / 72.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "certified_pairs": batch.certified_pair_count,
            "identity_loss": float(identity_loss.detach()),
            "interval_loss": float(
                (parent_interval + positive_interval + negative_interval).detach()
            ),
            "loss": float(loss.detach()),
            "ranking_loss": float(ranking_loss.detach()),
            "trace_return_loss": float(trace_return_loss.detach()),
            "step": step,
        }
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        **last,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "parents_with_pair_rate": total_parents_with_pair
                        / (step * args.batch_size),
                        "proposal_recall": total_proposal_recall
                        / (step * args.batch_size),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    after = evaluate(model, eval_batches)
    report = {
        "action_count": int(len(effects_np)),
        "after": after,
        "before": before,
        "certified_pairs_seen": total_certified_pairs,
        "dual_count": int(len(dual_weights_np)),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "last_train": last,
        "model_config": config.to_dict(),
        "parents_with_pair_rate": total_parents_with_pair
        / (args.steps * args.batch_size),
        "proposal_recall": total_proposal_recall / (args.steps * args.batch_size),
        "proposal_sources": proposal_sources,
        "source_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "steps": args.steps,
        "trace_return_weight": args.trace_return_weight,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_short_macro_certified_interval_value_v1",
            "factorized_value_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
