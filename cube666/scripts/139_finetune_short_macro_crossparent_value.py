"""Fine-tune the short-macro critic on certified two-layer cross-parent ranks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from torch.nn import functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--direct-action-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--utilities-script", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--root-batch-size", type=int, default=4)
    parser.add_argument("--root-branch", type=int, default=128)
    parser.add_argument("--second-branch", type=int, default=32)
    parser.add_argument("--hard-pairs", type=int, default=32)
    parser.add_argument("--anchor-batch-size", type=int, default=256)
    parser.add_argument("--eval-roots", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=4.0)
    parser.add_argument("--interval-weight", type=float, default=1.0)
    parser.add_argument("--return-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=139666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("crossparent_value_utilities", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def unique_action_mask(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = actions.sort(dim=-1).values
    unique = torch.ones_like(ordered, dtype=torch.bool)
    unique[..., 1:] = ordered[..., 1:].ne(ordered[..., :-1])
    return ordered, unique


@dataclass
class CrossParentBatch:
    positive_states: torch.Tensor
    negative_states: torch.Tensor
    positive_cost: torch.Tensor
    negative_cost: torch.Tensor
    positive_lower: torch.Tensor
    positive_upper: torch.Tensor
    negative_lower: torch.Tensor
    negative_upper: torch.Tensor
    pair_margin: torch.Tensor
    pair_mask: torch.Tensor


def build_crossparent_batch(
    utilities: ModuleType,
    *,
    roots_np: np.ndarray,
    teacher_upper_np: np.ndarray,
    solution_actions: np.ndarray,
    effects: torch.Tensor,
    costs: torch.Tensor,
    inverse: torch.Tensor,
    dual_weights: torch.Tensor,
    proposal_models: list[torch.nn.Module],
    rows: np.ndarray,
    root_branch: int,
    second_branch: int,
    hard_pairs: int,
    ranking_margin: float,
) -> CrossParentBatch:
    roots = torch.from_numpy(roots_np[rows]).cuda()
    batch = len(rows)
    remaining6 = torch.full((batch,), 6, dtype=torch.long, device="cuda")
    root_actions, root_unique = unique_action_mask(
        utilities.propose_actions(
            proposal_models, roots, remaining6, root_branch
        )
    )
    root_effects = effects[root_actions]
    parents = roots[:, None].expand_as(root_effects).gather(
        -1, root_effects.long()
    )
    root_width = root_actions.shape[1]
    flat_parents = parents.flatten(0, 1)
    remaining5 = torch.full(
        (len(flat_parents),), 5, dtype=torch.long, device="cuda"
    )
    second_actions, second_unique = unique_action_mask(
        utilities.propose_actions(
            proposal_models, flat_parents, remaining5, second_branch
        )
    )
    second_width = second_actions.shape[1]
    second_actions = second_actions.reshape(batch, root_width, second_width)
    second_unique = second_unique.reshape(batch, root_width, second_width)
    valid = (
        root_unique[:, :, None]
        & second_unique
        & second_actions.ne(inverse[root_actions][:, :, None])
    )
    selected_effects = effects[second_actions]
    children = parents[:, :, None].expand_as(selected_effects).gather(
        -1, selected_effects.long()
    )
    child_lower = utilities.cycle_cost_lower_bounds(children, dual_weights)
    path_cost = costs[root_actions][:, :, None] + costs[second_actions]
    root_upper = torch.from_numpy(teacher_upper_np[rows]).cuda()
    gap = (path_cost + child_lower - root_upper[:, None, None]).masked_fill(
        ~valid, -torch.inf
    )
    certified = gap > 0
    tight_gap = gap.masked_fill(~certified, torch.inf).flatten(1)
    keep = min(hard_pairs, tight_gap.shape[1])
    selected_gap, selected_flat = tight_gap.topk(keep, dim=1, largest=False)
    pair_mask = torch.isfinite(selected_gap)
    flat_children = children.flatten(1, 2)
    negative_states = flat_children.gather(
        1, selected_flat[:, :, None, None].expand(-1, -1, 6, 24)
    )
    flat_lower = child_lower.flatten(1)
    negative_lower = flat_lower.gather(1, selected_flat)
    flat_cost = path_cost.flatten(1)
    negative_cost = flat_cost.gather(1, selected_flat)
    root_positions = torch.div(
        selected_flat, second_width, rounding_mode="floor"
    )
    second_positions = selected_flat.remainder(second_width)
    selected_root_actions = root_actions.gather(1, root_positions)
    selected_second_actions = second_actions.gather(
        1, root_positions[:, :, None].expand(-1, -1, second_width)
    ).gather(2, second_positions[:, :, None]).squeeze(2)
    negative_upper = (
        root_upper[:, None]
        + costs[inverse[selected_second_actions]]
        + costs[inverse[selected_root_actions]]
    )

    teacher0 = torch.from_numpy(solution_actions[rows, 0].astype(np.int64)).cuda()
    teacher1 = torch.from_numpy(solution_actions[rows, 1].astype(np.int64)).cuda()
    positive_states = roots.gather(-1, effects[teacher0].long())
    positive_states = positive_states.gather(-1, effects[teacher1].long())
    positive_cost = costs[teacher0] + costs[teacher1]
    positive_upper = root_upper - positive_cost
    positive_lower = utilities.cycle_cost_lower_bounds(
        positive_states, dual_weights
    )
    pair_margin = torch.minimum(
        torch.full_like(selected_gap, ranking_margin), selected_gap * 0.5
    ).masked_fill(~pair_mask, 0.0)
    return CrossParentBatch(
        positive_states=positive_states.clone(),
        negative_states=negative_states.clone(),
        positive_cost=positive_cost.clone(),
        negative_cost=negative_cost.clone(),
        positive_lower=positive_lower.clone(),
        positive_upper=positive_upper.clone(),
        negative_lower=negative_lower.clone(),
        negative_upper=negative_upper.clone(),
        pair_margin=pair_margin.clone(),
        pair_mask=pair_mask.clone(),
    )


@torch.inference_mode()
def evaluate_pairs(
    utilities: ModuleType,
    model: torch.nn.Module,
    batches: list[CrossParentBatch],
) -> dict[str, float | int]:
    correct = 0
    pairs = 0
    roots_with_pairs = 0
    for batch in batches:
        root_count = len(batch.positive_states)
        hard = batch.negative_states.shape[1]
        inputs = torch.cat(
            (batch.positive_states, batch.negative_states.flatten(0, 1)), dim=0
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = utilities.combined_value(model, model(inputs))
        positive = values[:root_count]
        negative = values[root_count:].reshape(root_count, hard)
        correct += int(
            (
                batch.negative_cost
                + negative
                > batch.positive_cost[:, None] + positive[:, None]
            ).logical_and(batch.pair_mask).sum()
        )
        pairs += int(batch.pair_mask.sum())
        roots_with_pairs += int(batch.pair_mask.any(dim=1).sum())
    return {
        "accuracy": correct / max(pairs, 1),
        "certified_pairs": pairs,
        "roots_with_pairs": roots_with_pairs,
    }


@torch.inference_mode()
def evaluate_returns(
    utilities: ModuleType,
    model: torch.nn.Module,
    states: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    predictions: list[np.ndarray] = []
    for start in range(0, len(states), 4096):
        batch = torch.from_numpy(states[start : start + 4096]).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = utilities.combined_value(model, model(batch))
        predictions.append(prediction.cpu().numpy())
    predicted = np.concatenate(predictions)
    error = predicted - targets
    return {
        "bias": float(error.mean()),
        "correlation": float(np.corrcoef(predicted, targets)[0, 1]),
        "mae": float(np.abs(error).mean()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    utilities = load_module(args.utilities_script)
    root = args.dataset_dir
    with np.load(root / "teacher.npz", allow_pickle=False) as payload:
        roots_np = payload["states"].astype(np.uint8, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        solution_actions = payload["teacher_solution_actions"].astype(
            np.int32, copy=False
        )
        teacher_upper_np = payload["search_value_targets"].astype(
            np.float32, copy=False
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
    train_depth6 = np.flatnonzero((groups % 10 != 0) & (depths == 6))
    heldout_depth6 = np.flatnonzero((groups % 10 == 0) & (depths == 6))
    anchor_train = np.flatnonzero((groups % 10 != 0) & (depths >= 2))
    eval_rows = rng.choice(
        heldout_depth6, size=min(args.eval_roots, len(heldout_depth6)), replace=False
    )
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).cuda()
    inverse = torch.from_numpy(inverse_np).cuda()
    dual_weights = torch.from_numpy(
        utilities.build_subset_dual_weights(effects_np, costs_np)
    ).cuda()
    proposal_models = [
        utilities.load_macro_action_policy_checkpoint(str(path), device="cuda")[0]
        for path in args.direct_action_checkpoint
    ]
    checkpoint = torch.load(
        args.init_checkpoint, map_location="cpu", weights_only=False
    )
    config = utilities.MacroFactorizedValueConfig(
        **checkpoint["factorized_value_config"]
    )
    model = utilities.MacroFactorizedPrimitiveValueNet(config).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])

    eval_batches = [
        build_crossparent_batch(
            utilities,
            roots_np=roots_np,
            teacher_upper_np=teacher_upper_np,
            solution_actions=solution_actions,
            effects=effects,
            costs=costs,
            inverse=inverse,
            dual_weights=dual_weights,
            proposal_models=proposal_models,
            rows=eval_rows[start : start + args.root_batch_size],
            root_branch=args.root_branch,
            second_branch=args.second_branch,
            hard_pairs=args.hard_pairs,
            ranking_margin=args.ranking_margin,
        )
        for start in range(0, len(eval_rows), args.root_batch_size)
    ]
    before_pairs = evaluate_pairs(utilities, model, eval_batches)
    before_returns = evaluate_returns(
        utilities, model, roots_np[eval_rows], teacher_upper_np[eval_rows]
    )
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
    identity = torch.arange(24, dtype=torch.uint8, device="cuda")[None].expand(
        6, -1
    )
    started = time.perf_counter()
    total_pairs = 0
    roots_with_pairs = 0
    last: dict[str, float | int] = {}
    model.train()
    for step in range(1, args.steps + 1):
        rows = rng.choice(train_depth6, size=args.root_batch_size, replace=True)
        batch = build_crossparent_batch(
            utilities,
            roots_np=roots_np,
            teacher_upper_np=teacher_upper_np,
            solution_actions=solution_actions,
            effects=effects,
            costs=costs,
            inverse=inverse,
            dual_weights=dual_weights,
            proposal_models=proposal_models,
            rows=rows,
            root_branch=args.root_branch,
            second_branch=args.second_branch,
            hard_pairs=args.hard_pairs,
            ranking_margin=args.ranking_margin,
        )
        total_pairs += int(batch.pair_mask.sum())
        roots_with_pairs += int(batch.pair_mask.any(dim=1).sum())
        anchor_rows = rng.choice(
            anchor_train, size=args.anchor_batch_size, replace=True
        )
        anchor_prefixes = np.asarray(
            [rng.integers(0, int(depths[row])) for row in anchor_rows],
            dtype=np.int64,
        )
        anchor_states_np = utilities.reconstruct_parents(
            roots_np,
            solution_actions,
            effects_np,
            anchor_rows,
            anchor_prefixes,
        )
        anchor_targets_np = np.asarray(
            [
                costs_np[
                    solution_actions[row, prefix : int(depths[row])].astype(
                        np.int64, copy=False
                    )
                ].sum(dtype=np.float64)
                for row, prefix in zip(
                    anchor_rows, anchor_prefixes, strict=True
                )
            ],
            dtype=np.float32,
        )
        anchor_states = torch.from_numpy(anchor_states_np).cuda()
        anchor_targets = torch.from_numpy(anchor_targets_np).cuda()
        hard = batch.negative_states.shape[1]
        inputs = torch.cat(
            (
                batch.positive_states,
                batch.negative_states.flatten(0, 1),
                anchor_states,
                identity[None].expand(args.root_batch_size, -1, -1),
            ),
            dim=0,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = utilities.combined_value(model, train_model(inputs))
            positive_value = values[: args.root_batch_size]
            negative_value = values[
                args.root_batch_size : args.root_batch_size * (hard + 1)
            ].reshape(args.root_batch_size, hard)
            anchor_value = values[
                args.root_batch_size * (hard + 1) :
                args.root_batch_size * (hard + 1) + args.anchor_batch_size
            ]
            identity_value = values[-args.root_batch_size :]
            positive_q = batch.positive_cost + positive_value
            negative_q = batch.negative_cost + negative_value
            ranking_loss = utilities.masked_mean(
                F.relu(
                    batch.pair_margin
                    + positive_q[:, None]
                    - negative_q
                ),
                batch.pair_mask,
            )
            positive_interval = utilities.interval_violation(
                positive_value, batch.positive_lower, batch.positive_upper
            )
            negative_interval = utilities.interval_violation(
                negative_value,
                batch.negative_lower,
                batch.negative_upper,
                batch.pair_mask,
            )
            return_loss = F.smooth_l1_loss(
                anchor_value, anchor_targets, beta=2.0
            )
            identity_loss = F.smooth_l1_loss(
                identity_value, torch.zeros_like(identity_value), beta=1.0
            )
            loss = (
                args.ranking_weight * ranking_loss
                + args.interval_weight * (positive_interval + negative_interval)
                + args.return_weight * return_loss
                + identity_loss
            ) / 72.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "identity_loss": float(identity_loss.detach()),
            "interval_loss": float(
                (positive_interval + negative_interval).detach()
            ),
            "loss": float(loss.detach()),
            "ranking_loss": float(ranking_loss.detach()),
            "return_loss": float(return_loss.detach()),
            "step": step,
        }
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        **last,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "pairs_seen": total_pairs,
                        "roots_with_pairs_rate": roots_with_pairs
                        / (step * args.root_batch_size),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    after_pairs = evaluate_pairs(utilities, model, eval_batches)
    after_returns = evaluate_returns(
        utilities, model, roots_np[eval_rows], teacher_upper_np[eval_rows]
    )
    report = {
        "after_pairs": after_pairs,
        "after_returns": after_returns,
        "before_pairs": before_pairs,
        "before_returns": before_returns,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "last_train": last,
        "model_config": config.to_dict(),
        "pairs_seen": total_pairs,
        "roots_with_pairs_rate": roots_with_pairs / (args.steps * args.root_batch_size),
        "source_checkpoint": str(args.init_checkpoint),
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_short_macro_crossparent_value_v1",
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
