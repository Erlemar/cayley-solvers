"""Train a direct joint-action policy on exact short-macro trace prefixes."""

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

from cube666.macro_action_policy import (  # noqa: E402
    MacroActionPolicyConfig,
    MacroActionPolicyNet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars="@")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--minimum-train-depth", type=int, default=1)
    parser.add_argument("--maximum-train-depth", type=int, default=5)
    parser.add_argument(
        "--fixed-prefix",
        type=int,
        default=-1,
        help="use this trace prefix for every row; -1 samples prefixes uniformly",
    )
    parser.add_argument("--evaluation-samples", type=int, default=4096)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=0)
    parser.add_argument("--residual-blocks", type=int, default=-1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--initialize-from-scratch", action="store_true")
    parser.add_argument("--depth-conditioned", action="store_true")
    parser.add_argument(
        "--commuting-multilabel",
        action="store_true",
        help=(
            "treat every remaining trace action that commutes leftward to the "
            "current position as a verified positive"
        ),
    )
    parser.add_argument(
        "--inverse-augmentation-probability",
        type=float,
        default=0.0,
        help=(
            "replace this fraction of training examples by the exact trajectory "
            "obtained by group inversion and reversed inverse actions"
        ),
    )
    parser.add_argument("--maximum-conditioned-depth", type=int, default=8)
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="train on every depth-filtered row (evaluation remains the diagnostic split)",
    )
    parser.add_argument("--seed", type=int, default=127666)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if not 0.0 <= args.inverse_augmentation_probability <= 1.0:
        parser.error("inverse augmentation probability must be in [0, 1]")
    if args.inverse_augmentation_probability and args.commuting_multilabel:
        parser.error("inverse augmentation is not yet combined with commuting labels")
    return args


def reconstruct_prefix_batch(
    states: np.ndarray,
    actions: np.ndarray,
    effects: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = states[rows].copy()
    maximum_prefix = int(prefixes.max(initial=0))
    for step in range(maximum_prefix):
        active = prefixes > step
        step_actions = actions[rows[active], step].astype(np.int64, copy=False)
        current[active] = np.take_along_axis(
            current[active], effects[step_actions], axis=-1
        )
    labels = actions[rows, prefixes].astype(np.int64, copy=False)
    return current, labels


def reconstruct_inverse_prefix_batch(
    states: np.ndarray,
    actions: np.ndarray,
    effects: np.ndarray,
    inverse_actions: np.ndarray,
    depths: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct prefixes of the exact inverse-and-reverse solution trajectory."""

    current = np.argsort(states[rows], axis=2).astype(np.uint8)
    row_depths = depths[rows].astype(np.int64, copy=False)
    maximum_prefix = int(prefixes.max(initial=0))
    for step in range(maximum_prefix):
        active = prefixes > step
        original_positions = row_depths[active] - 1 - step
        original_actions = actions[rows[active], original_positions].astype(
            np.int64, copy=False
        )
        step_actions = inverse_actions[original_actions]
        current[active] = np.take_along_axis(
            current[active], effects[step_actions], axis=-1
        )
    label_positions = row_depths - 1 - prefixes
    original_labels = actions[rows, label_positions].astype(np.int64, copy=False)
    labels = inverse_actions[original_labels]
    return current, labels.astype(np.int64, copy=False)


def commuting_label_sets(
    solution_actions: np.ndarray,
    depths: np.ndarray,
    effects: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return actions that can be moved first without changing exact replay."""

    remaining = depths[rows].astype(np.int64) - prefixes
    width = int(remaining.max())
    actions = np.full((len(rows), width), -1, dtype=np.int64)
    valid = np.zeros((len(rows), width), dtype=np.bool_)
    for position in range(width):
        active = remaining > position
        actions[active, position] = solution_actions[
            rows[active], prefixes[active] + position
        ]
        valid[active, position] = True
        for earlier in range(position):
            pair_active = active & valid[:, position]
            if not np.any(pair_active):
                break
            candidate = actions[pair_active, position]
            predecessor = actions[pair_active, earlier]
            candidate_effect = effects[candidate]
            predecessor_effect = effects[predecessor]
            forward = np.take_along_axis(
                candidate_effect, predecessor_effect, axis=-1
            )
            reverse = np.take_along_axis(
                predecessor_effect, candidate_effect, axis=-1
            )
            commutes = np.equal(forward, reverse).all(axis=(1, 2))
            active_rows = np.flatnonzero(pair_active)
            valid[active_rows[~commutes], position] = False
    if np.any(valid[:, 0] == 0):
        raise AssertionError("the original next trace action must remain valid")
    return actions, valid


@torch.inference_mode()
def evaluate(
    model: MacroActionPolicyNet,
    states: np.ndarray,
    labels: np.ndarray,
    remaining_depths: np.ndarray,
    label_mask: np.ndarray | None = None,
    batch_size: int = 1024,
) -> dict[str, float | int]:
    model.eval()
    ranks: list[np.ndarray] = []
    losses: list[float] = []
    for start in range(0, len(states), batch_size):
        batch_states = torch.from_numpy(states[start : start + batch_size]).cuda()
        batch_labels = torch.from_numpy(labels[start : start + batch_size]).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch_depths = torch.from_numpy(
                remaining_depths[start : start + batch_size]
            ).cuda()
            logits = model(batch_states, remaining_depth=batch_depths)
        if batch_labels.ndim == 1:
            label_scores = logits.gather(1, batch_labels[:, None]).squeeze(1)
            batch_loss = F.cross_entropy(logits.float(), batch_labels)
        else:
            if label_mask is None:
                raise AssertionError("multi-label evaluation requires a mask")
            batch_mask = torch.from_numpy(
                label_mask[start : start + batch_size]
            ).cuda()
            gathered = logits.gather(1, batch_labels.clamp_min(0))
            label_scores = gathered.masked_fill(~batch_mask, -torch.inf).max(dim=1).values
            log_probability = logits.float().log_softmax(dim=1).gather(
                1, batch_labels.clamp_min(0)
            )
            set_log_probability = torch.logsumexp(
                log_probability.masked_fill(~batch_mask, -torch.inf), dim=1
            )
            batch_loss = -set_log_probability.mean()
        ranks.append(((logits > label_scores[:, None]).sum(dim=1) + 1).cpu().numpy())
        losses.append(float(batch_loss))
    rank = np.concatenate(ranks)
    return {
        "cross_entropy": float(np.mean(losses)),
        "median_rank": float(np.median(rank)),
        "p90_rank": float(np.quantile(rank, 0.9)),
        "samples": int(len(rank)),
        "top1_recall": float(np.mean(rank <= 1)),
        "top128_recall": float(np.mean(rank <= 128)),
        "top512_recall": float(np.mean(rank <= 512)),
        "top1024_recall": float(np.mean(rank <= 1024)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    with np.load(args.dataset_dir / "teacher.npz", allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        solution_actions = payload["teacher_solution_actions"].astype(
            np.int32, copy=False
        )
    groups = np.load(args.dataset_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.dataset_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    inverse_actions = np.load(
        args.dataset_dir / "inverse_actions.npy", allow_pickle=False
    ).astype(np.int64, copy=False)
    depth_mask = (depths >= args.minimum_train_depth) & (
        (args.maximum_train_depth == 0) | (depths <= args.maximum_train_depth)
    )
    train_indices = np.flatnonzero(depth_mask if args.train_all else ((groups % 10 != 0) & depth_mask))
    heldout_indices = np.flatnonzero((groups % 10 == 0) & depth_mask)
    if not len(train_indices) or not len(heldout_indices):
        raise ValueError("depth filter left an empty split")
    evaluation_rows = rng.choice(
        heldout_indices,
        size=min(args.evaluation_samples, len(heldout_indices)),
        replace=False,
    )
    if args.fixed_prefix >= 0:
        if np.any(depths[evaluation_rows] <= args.fixed_prefix):
            raise ValueError("fixed-prefix must be smaller than every selected depth")
        evaluation_prefixes = np.full(
            len(evaluation_rows), args.fixed_prefix, dtype=np.int64
        )
    else:
        evaluation_prefixes = (
            rng.random(len(evaluation_rows)) * depths[evaluation_rows]
        ).astype(np.int64)
    evaluation_states, evaluation_labels = reconstruct_prefix_batch(
        states,
        solution_actions,
        effects,
        evaluation_rows,
        evaluation_prefixes,
    )
    evaluation_remaining_depths = (
        depths[evaluation_rows] - evaluation_prefixes
    ).astype(np.int64, copy=False)
    inverse_evaluation_states, inverse_evaluation_labels = (
        reconstruct_inverse_prefix_batch(
            states,
            solution_actions,
            effects,
            inverse_actions,
            depths,
            evaluation_rows,
            evaluation_prefixes,
        )
    )
    evaluation_label_mask: np.ndarray | None = None
    if args.commuting_multilabel:
        evaluation_labels, evaluation_label_mask = commuting_label_sets(
            solution_actions,
            depths,
            effects,
            evaluation_rows,
            evaluation_prefixes,
        )
    if args.init_checkpoint is None and not args.initialize_from_scratch:
        raise ValueError("--init-checkpoint is required unless training from scratch")
    initial = (
        torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if args.init_checkpoint is not None
        else None
    )
    initial_config = initial["model_config"] if initial is not None else {}
    config = MacroActionPolicyConfig(
        action_count=len(effects),
        hidden_dim=(
            args.hidden_dim
            if args.hidden_dim
            else int(initial_config.get("hidden_dim", 1024))
        ),
        residual_blocks=(
            args.residual_blocks
            if args.residual_blocks >= 0
            else int(initial_config.get("residual_blocks", 6))
        ),
        dropout=float(initial_config.get("dropout", args.dropout)),
        depth_conditioned=(
            args.depth_conditioned
            if args.initialize_from_scratch
            else bool(initial_config.get("depth_conditioned", False))
        ),
        maximum_depth=(
            args.maximum_conditioned_depth
            if args.initialize_from_scratch
            else int(initial_config.get("maximum_depth", 8))
        ),
    )
    model = MacroActionPolicyNet(config).cuda()
    if args.initialize_from_scratch:
        direct_initialization = False
        initialization = "scratch"
    else:
        if initial is None:
            raise AssertionError("warm-start checkpoint is missing")
        initial_action_count = int(initial_config.get("action_count", 0))
        expanded_direct_head = (
            initial_config.get("architecture") == "direct_action"
            and 0 < initial_action_count < len(effects)
        )
        compatible = {
            key: value
            for key, value in initial["model_state_dict"].items()
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        direct_initialization = initial_config.get("architecture") == "direct_action"
        expected_missing = (
            {"action_head.weight", "action_head.bias"}
            if expanded_direct_head or not direct_initialization
            else set()
        )
        if unexpected or set(missing) != expected_missing:
            raise ValueError(f"unexpected warm-start mismatch: {missing=}, {unexpected=}")
        if expanded_direct_head:
            old_weight = initial["model_state_dict"]["action_head.weight"]
            old_bias = initial["model_state_dict"]["action_head.bias"]
            with torch.no_grad():
                model.action_head.weight[:initial_action_count].copy_(old_weight)
                model.action_head.bias[:initial_action_count].copy_(old_bias)
            initialization = "expanded_direct_checkpoint"
        else:
            initialization = "direct_checkpoint" if direct_initialization else "shared_trunk"
    before = evaluate(
        model,
        evaluation_states,
        evaluation_labels,
        evaluation_remaining_depths,
        label_mask=evaluation_label_mask,
    )
    before_inverse = evaluate(
        model,
        inverse_evaluation_states,
        inverse_evaluation_labels,
        evaluation_remaining_depths,
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
    started = time.perf_counter()
    last_loss = 0.0
    model.train()
    for step in range(1, args.steps + 1):
        rows = rng.choice(train_indices, size=args.batch_size, replace=True)
        if args.fixed_prefix >= 0:
            if np.any(depths[rows] <= args.fixed_prefix):
                raise AssertionError("training row is shorter than fixed-prefix")
            prefixes = np.full(args.batch_size, args.fixed_prefix, dtype=np.int64)
        else:
            prefixes = (rng.random(args.batch_size) * depths[rows]).astype(np.int64)
        batch_states_np, labels_np = reconstruct_prefix_batch(
            states, solution_actions, effects, rows, prefixes
        )
        if args.inverse_augmentation_probability:
            inverse_mask = (
                rng.random(args.batch_size)
                < args.inverse_augmentation_probability
            )
            if np.any(inverse_mask):
                inverse_states_np, inverse_labels_np = (
                    reconstruct_inverse_prefix_batch(
                        states,
                        solution_actions,
                        effects,
                        inverse_actions,
                        depths,
                        rows[inverse_mask],
                        prefixes[inverse_mask],
                    )
                )
                batch_states_np[inverse_mask] = inverse_states_np
                labels_np[inverse_mask] = inverse_labels_np
        batch_states = torch.from_numpy(batch_states_np).cuda()
        label_mask = None
        if args.commuting_multilabel:
            labels_np, label_mask_np = commuting_label_sets(
                solution_actions, depths, effects, rows, prefixes
            )
            label_mask = torch.from_numpy(label_mask_np).cuda()
        labels = torch.from_numpy(labels_np).cuda()
        remaining_depths = torch.from_numpy(
            (depths[rows] - prefixes).astype(np.int64, copy=False)
        ).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = train_model(batch_states, remaining_depth=remaining_depths)
            if labels.ndim == 1:
                label_log_probability = logits.float().log_softmax(dim=1).gather(
                    1, labels[:, None]
                ).squeeze(1)
                per_example_loss = F.cross_entropy(
                    logits.float(),
                    labels,
                    label_smoothing=args.label_smoothing,
                    reduction="none",
                )
            else:
                if label_mask is None:
                    raise AssertionError("multi-label training requires a mask")
                log_probability = logits.float().log_softmax(dim=1)
                positive_log_probability = log_probability.gather(
                    1, labels.clamp_min(0)
                ).masked_fill(~label_mask, -torch.inf)
                label_log_probability = torch.logsumexp(
                    positive_log_probability, dim=1
                )
                uniform_loss = -log_probability.mean(dim=1)
                per_example_loss = (
                    (1.0 - args.label_smoothing) * -label_log_probability
                    + args.label_smoothing * uniform_loss
                )
            if args.focal_gamma:
                focal_weight = (1.0 - label_log_probability.exp()).pow(
                    args.focal_gamma
                )
                loss = (focal_weight * per_example_loss).mean()
            else:
                loss = per_example_loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "loss": last_loss,
                        "step": step,
                    }
                ),
                flush=True,
            )
    after = evaluate(
        model,
        evaluation_states,
        evaluation_labels,
        evaluation_remaining_depths,
        label_mask=evaluation_label_mask,
    )
    after_inverse = evaluate(
        model,
        inverse_evaluation_states,
        inverse_evaluation_labels,
        evaluation_remaining_depths,
    )
    report = {
        "after": after,
        "after_inverse": after_inverse,
        "before": before,
        "before_inverse": before_inverse,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "last_loss": last_loss,
        "maximum_train_depth": args.maximum_train_depth,
        "minimum_train_depth": args.minimum_train_depth,
        "model_config": config.to_dict(),
        "source_checkpoint": (
            str(args.init_checkpoint) if args.init_checkpoint is not None else None
        ),
        "direct_initialization": direct_initialization,
        "commuting_multilabel": args.commuting_multilabel,
        "evaluation_mean_positive_actions": (
            float(evaluation_label_mask.sum(axis=1).mean())
            if evaluation_label_mask is not None
            else 1.0
        ),
        "initialization": initialization,
        "focal_gamma": args.focal_gamma,
        "fixed_prefix": args.fixed_prefix,
        "label_smoothing": args.label_smoothing,
        "inverse_augmentation_probability": args.inverse_augmentation_probability,
        "steps": args.steps,
        "train_all": args.train_all,
        "train_samples": int(len(train_indices)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_macro_direct_action_policy_v1",
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
