"""Fine-tune a cube666 critic on replay-verified off-policy beam returns."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_factorized_value import (  # noqa: E402
    MacroFactorizedPrimitiveValueNet,
    MacroFactorizedValueConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--solved-completions", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--return-batch-size", type=int, default=256)
    parser.add_argument("--pair-batch-size", type=int, default=128)
    parser.add_argument("--anchor-batch-size", type=int, default=256)
    parser.add_argument("--anchor-pool-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--return-weight", type=float, default=4.0)
    parser.add_argument("--ranking-weight", type=float, default=2.0)
    parser.add_argument("--anchor-weight", type=float, default=0.5)
    parser.add_argument("--heldout-root-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=141666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
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


def reconstruct_states(
    roots: np.ndarray,
    solution_actions: np.ndarray,
    effects: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> np.ndarray:
    states = roots[rows].copy()
    for step in range(int(prefixes.max(initial=0))):
        active = prefixes > step
        actions = solution_actions[rows[active], step].astype(np.int64, copy=False)
        states[active] = np.take_along_axis(
            states[active], effects[actions], axis=-1
        )
    return states


def teacher_returns(
    depths: np.ndarray,
    solution_actions: np.ndarray,
    costs: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            costs[
                solution_actions[row, prefix : int(depths[row])].astype(
                    np.int64, copy=False
                )
            ].sum(dtype=np.float64)
            for row, prefix in zip(rows, prefixes, strict=True)
        ],
        dtype=np.float32,
    )


def build_anchor_pool(
    *,
    roots: np.ndarray,
    depths: np.ndarray,
    solution_actions: np.ndarray,
    groups: np.ndarray,
    effects: np.ndarray,
    costs: np.ndarray,
    size: int,
    heldout: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = np.flatnonzero(
        ((groups % 10 == 0) if heldout else (groups % 10 != 0)) & (depths > 0)
    )
    rows = rng.choice(eligible, size=min(size, len(eligible)), replace=False)
    prefixes = np.asarray(
        [rng.integers(0, int(depths[row])) for row in rows], dtype=np.int64
    )
    return (
        reconstruct_states(roots, solution_actions, effects, rows, prefixes),
        teacher_returns(depths, solution_actions, costs, rows, prefixes),
    )


def expand_verified_returns(
    payload: dict[str, np.ndarray],
    effects: np.ndarray,
    costs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    targets: list[float] = []
    roots: list[int] = []
    base_rows: list[int] = []
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24))
    for row in range(len(payload["states"])):
        state = payload["states"][row].copy()
        length = int(payload["completion_lengths"][row])
        actions = payload["completion_actions"][row, :length].astype(
            np.int64, copy=False
        )
        suffix_cost = int(costs[actions].sum())
        if suffix_cost != int(payload["completion_costs"][row]):
            raise AssertionError("completion cost disagrees with action replay")
        root = int(payload["root_source_indices"][row])
        base_rows.append(len(states))
        for step, action in enumerate(actions):
            states.append(state.copy())
            targets.append(float(costs[actions[step:]].sum()))
            roots.append(root)
            state = np.take_along_axis(state, effects[int(action)], axis=-1)
        if not np.array_equal(state, identity):
            raise AssertionError(f"completion row {row} does not reach identity")
    return (
        np.stack(states),
        np.asarray(targets, dtype=np.float32),
        np.asarray(roots, dtype=np.int64),
        np.asarray(base_rows, dtype=np.int64),
    )


def build_pair_indices(
    root_ids: np.ndarray, targets: np.ndarray, base_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    margins: list[float] = []
    base_roots = root_ids[base_rows]
    base_targets = targets[base_rows]
    for root in np.unique(base_roots):
        positions = np.flatnonzero(base_roots == root)
        for first_offset, first in enumerate(positions):
            for second in positions[first_offset + 1 :]:
                difference = float(base_targets[first] - base_targets[second])
                if difference == 0:
                    continue
                better, worse = (first, second) if difference < 0 else (second, first)
                left.append(int(base_rows[better]))
                right.append(int(base_rows[worse]))
                margins.append(min(2.0, abs(difference) * 0.5))
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(margins, dtype=np.float32),
    )


@torch.inference_mode()
def predict(
    model: MacroFactorizedPrimitiveValueNet,
    states: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = combined_value(model, model(batch))
        output.append(values.float().cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def regression_metrics(predicted: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = predicted - targets
    correlation = (
        float(np.corrcoef(predicted, targets)[0, 1])
        if len(predicted) > 1 and predicted.std() > 0 and targets.std() > 0
        else 0.0
    )
    return {
        "bias": float(errors.mean()),
        "correlation": correlation,
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "samples": int(len(predicted)),
    }


def frontier_metrics(
    model: MacroFactorizedPrimitiveValueNet,
    states: np.ndarray,
    targets: np.ndarray,
    roots: np.ndarray,
    base_rows: np.ndarray,
    root_mask: np.ndarray,
) -> dict[str, float | int]:
    chosen_base = base_rows[root_mask[roots[base_rows]]]
    predictions = predict(model, states[chosen_base])
    target_values = targets[chosen_base]
    root_values = roots[chosen_base]
    correct = 0
    pairs = 0
    top1_correct = 0
    top1_regret: list[float] = []
    for root in np.unique(root_values):
        positions = np.flatnonzero(root_values == root)
        root_targets = target_values[positions]
        root_predictions = predictions[positions]
        for first_offset, first in enumerate(positions):
            differences = target_values[positions[first_offset + 1 :]] - target_values[first]
            predicted_differences = (
                predictions[positions[first_offset + 1 :]] - predictions[first]
            )
            non_ties = differences != 0
            correct += int(
                ((differences[non_ties] * predicted_differences[non_ties]) > 0).sum()
            )
            pairs += int(non_ties.sum())
        selected = int(np.argmin(root_predictions))
        best = float(root_targets.min())
        regret = float(root_targets[selected] - best)
        top1_regret.append(regret)
        top1_correct += int(regret == 0)
    return {
        **regression_metrics(predictions, target_values),
        "pair_accuracy": correct / max(pairs, 1),
        "pairs": pairs,
        "roots": int(np.unique(root_values).size),
        "top1_best_recall": top1_correct / max(len(top1_regret), 1),
        "top1_return_regret_mean": float(np.mean(top1_regret)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.heldout_root_fraction < 1:
        raise ValueError("heldout-root-fraction must lie strictly between zero and one")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)

    with np.load(args.solved_completions, allow_pickle=False) as solved_payload:
        solved = {name: solved_payload[name] for name in solved_payload.files}
    with np.load(args.dataset_dir / "teacher.npz", allow_pickle=False) as payload:
        teacher_roots = payload["states"].astype(np.uint8, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        solution_actions = payload["teacher_solution_actions"].astype(
            np.int32, copy=False
        )
    groups = np.load(args.dataset_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.dataset_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.dataset_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    states, targets, root_ids, base_rows = expand_verified_returns(
        solved, effects, costs
    )
    unique_roots = np.unique(root_ids)
    rng.shuffle(unique_roots)
    heldout_count = max(1, round(len(unique_roots) * args.heldout_root_fraction))
    heldout_roots = set(int(value) for value in unique_roots[:heldout_count])
    root_mask = np.zeros(int(root_ids.max()) + 1, dtype=np.bool_)
    root_mask[list(heldout_roots)] = True
    is_heldout_state = root_mask[root_ids]
    train_rows = np.flatnonzero(~is_heldout_state)
    # Upweight layer-2 beam states relative to their easier solved suffix states.
    train_probabilities = np.ones(len(train_rows), dtype=np.float64)
    train_base_set = set(int(value) for value in base_rows if not is_heldout_state[value])
    train_probabilities[
        np.asarray([int(row) in train_base_set for row in train_rows], dtype=np.bool_)
    ] = 4.0
    train_probabilities /= train_probabilities.sum()

    pair_left, pair_right, pair_margins = build_pair_indices(
        root_ids, targets, base_rows
    )
    train_pair_mask = ~root_mask[root_ids[pair_left]]
    pair_left = pair_left[train_pair_mask]
    pair_right = pair_right[train_pair_mask]
    pair_margins = pair_margins[train_pair_mask]

    anchor_states, anchor_targets = build_anchor_pool(
        roots=teacher_roots,
        depths=depths,
        solution_actions=solution_actions,
        groups=groups,
        effects=effects,
        costs=costs,
        size=args.anchor_pool_size,
        heldout=False,
        rng=rng,
    )
    anchor_eval_states, anchor_eval_targets = build_anchor_pool(
        roots=teacher_roots,
        depths=depths,
        solution_actions=solution_actions,
        groups=groups,
        effects=effects,
        costs=costs,
        size=min(2048, args.anchor_pool_size),
        heldout=True,
        rng=rng,
    )

    source = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = MacroFactorizedValueConfig(**source["factorized_value_config"])
    model = MacroFactorizedPrimitiveValueNet(config).cuda()
    model.load_state_dict(source["model_state_dict"])
    before = {
        "frontier_train": frontier_metrics(
            model, states, targets, root_ids, base_rows, ~root_mask
        ),
        "frontier_heldout": frontier_metrics(
            model, states, targets, root_ids, base_rows, root_mask
        ),
        "anchor_heldout": regression_metrics(
            predict(model, anchor_eval_states), anchor_eval_targets
        ),
    }
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-3)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    identity = torch.arange(24, dtype=torch.uint8, device="cuda")[None].expand(
        6, -1
    )
    started = time.perf_counter()
    last: dict[str, float | int] = {}
    model.train()
    for step in range(1, args.steps + 1):
        regression_rows = rng.choice(
            train_rows,
            size=args.return_batch_size,
            replace=True,
            p=train_probabilities,
        )
        if len(pair_left):
            pair_rows = rng.integers(0, len(pair_left), size=args.pair_batch_size)
            left_rows = pair_left[pair_rows]
            right_rows = pair_right[pair_rows]
            margins = torch.from_numpy(pair_margins[pair_rows]).cuda()
        else:
            left_rows = rng.choice(train_rows, size=args.pair_batch_size, replace=True)
            right_rows = left_rows
            margins = torch.zeros(args.pair_batch_size, device="cuda")
        anchor_rows = rng.integers(
            0, len(anchor_states), size=args.anchor_batch_size
        )
        model_inputs = torch.from_numpy(
            np.concatenate(
                (
                    states[regression_rows],
                    states[left_rows],
                    states[right_rows],
                    anchor_states[anchor_rows],
                ),
                axis=0,
            )
        ).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictions = combined_value(model, train_model(model_inputs))
            cursor = 0
            return_predictions = predictions[cursor : cursor + args.return_batch_size]
            cursor += args.return_batch_size
            left_predictions = predictions[cursor : cursor + args.pair_batch_size]
            cursor += args.pair_batch_size
            right_predictions = predictions[cursor : cursor + args.pair_batch_size]
            cursor += args.pair_batch_size
            anchor_predictions = predictions[cursor:]
            return_targets = torch.from_numpy(targets[regression_rows]).cuda()
            anchor_batch_targets = torch.from_numpy(anchor_targets[anchor_rows]).cuda()
            return_loss = F.smooth_l1_loss(
                return_predictions, return_targets, beta=2.0
            )
            ranking_loss = F.relu(
                margins + left_predictions - right_predictions
            ).mean()
            anchor_loss = F.smooth_l1_loss(
                anchor_predictions, anchor_batch_targets, beta=2.0
            )
            identity_value = combined_value(model, train_model(identity[None]))
            identity_loss = F.smooth_l1_loss(
                identity_value, torch.zeros_like(identity_value), beta=1.0
            )
            nonnegative = F.relu(-predictions).mean()
            loss = (
                args.return_weight * return_loss
                + args.ranking_weight * ranking_loss
                + args.anchor_weight * anchor_loss
                + identity_loss
                + 0.05 * nonnegative
            ) / 72.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "anchor_loss": float(anchor_loss.detach()),
            "identity_loss": float(identity_loss.detach()),
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
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    after = {
        "frontier_train": frontier_metrics(
            model, states, targets, root_ids, base_rows, ~root_mask
        ),
        "frontier_heldout": frontier_metrics(
            model, states, targets, root_ids, base_rows, root_mask
        ),
        "anchor_heldout": regression_metrics(
            predict(model, anchor_eval_states), anchor_eval_targets
        ),
    }
    report = {
        "after": after,
        "before": before,
        "base_states": int(len(base_rows)),
        "expanded_return_states": int(len(states)),
        "heldout_roots": int(len(heldout_roots)),
        "last_train": last,
        "model_config": config.to_dict(),
        "pair_count": int(len(pair_left)),
        "source_checkpoint": str(args.init_checkpoint),
        "solved_completions": str(args.solved_completions),
        "steps": args.steps,
        "train_roots": int(len(unique_roots) - len(heldout_roots)),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_short_macro_frontier_return_value_v1",
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
