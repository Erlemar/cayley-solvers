"""Train the macro-effect network on audited primitive-move cost-to-go labels."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--group-ids", type=Path, required=True)
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--train-module", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--action-digest", required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--reset-value-heads", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negative-actions", type=int, default=512)
    parser.add_argument("--hard-negative-actions", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--cluster-value-weight", type=float, default=1.0)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--policy-eval-samples", type=int, default=64)
    parser.add_argument("--policy-topk", type=int, default=16)
    parser.add_argument("--policy-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=68666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cube666_effect_train_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load training module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def predict_values(
    model: torch.nn.Module,
    states: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, total, clusters = model(batch)
        total_moves = total.float() * 72.0
        cluster_moves = clusters.float().sum(dim=1) * 12.0
        predictions.append(((total_moves + cluster_moves) * 0.5).cpu().numpy())
    return np.concatenate(predictions)


def value_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    if len(predictions) > 1 and np.std(predictions) > 0 and np.std(targets) > 0:
        correlation = float(np.corrcoef(predictions, targets)[0, 1])
    else:
        correlation = 0.0
    return {
        "bias": float(errors.mean()),
        "correlation": correlation,
        "mae": float(np.abs(errors).mean()),
        "prediction_maximum": float(predictions.max()),
        "prediction_mean": float(predictions.mean()),
        "prediction_minimum": float(predictions.min()),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "target_maximum": float(targets.max()),
        "target_mean": float(targets.mean()),
        "target_minimum": float(targets.min()),
    }


@torch.no_grad()
def policy_metrics(
    module: ModuleType,
    model: torch.nn.Module,
    states: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    action_effects: torch.Tensor,
    *,
    device: torch.device,
    topk: int,
    chunk_size: int,
) -> dict[str, float | int]:
    model.eval()
    state_tensor = torch.from_numpy(states).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _, _ = model(state_tensor)
    best_scores = torch.empty((len(states), 0), device=device)
    best_actions = torch.empty((len(states), 0), dtype=torch.long, device=device)
    for start in range(0, len(action_effects), chunk_size):
        stop = min(start + chunk_size, len(action_effects))
        candidates = action_effects[start:stop][None].expand(len(states), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        action_ids = torch.arange(start, stop, device=device)[None].expand(len(states), -1)
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, action_ids), dim=1)
        keep = min(topk, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    proposed = best_actions.cpu().numpy()
    slots = np.arange(labels.shape[1])[None, :]
    valid = slots < counts[:, None]
    hits = proposed[:, :, None] == labels[:, None, :]
    hits &= valid[:, None, :]
    return {
        "samples": len(states),
        "top1_recall": float(hits[:, :1].any(axis=(1, 2)).mean()),
        f"top{topk}_recall": float(hits.any(axis=(1, 2)).mean()),
    }


@torch.no_grad()
def topk_action_ids_from_logits(
    module: ModuleType,
    logits: torch.Tensor,
    action_effects: torch.Tensor,
    *,
    topk: int,
    chunk_size: int,
) -> torch.Tensor:
    best_scores = torch.empty((len(logits), 0), device=logits.device)
    best_actions = torch.empty((len(logits), 0), dtype=torch.long, device=logits.device)
    for start in range(0, len(action_effects), chunk_size):
        stop = min(start + chunk_size, len(action_effects))
        candidates = action_effects[start:stop][None].expand(len(logits), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        action_ids = torch.arange(start, stop, device=logits.device)[None].expand(
            len(logits), -1
        )
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, action_ids), dim=1)
        keep = min(topk, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    return best_actions


@torch.no_grad()
def policy_rank_metrics(
    module: ModuleType,
    model: torch.nn.Module,
    states: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    action_effects: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, float | int]:
    model.eval()
    state_tensor = torch.from_numpy(states).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _, _ = model(state_tensor)
    slots = torch.arange(labels.shape[1], device=device)[None]
    count_tensor = torch.from_numpy(counts).to(device)
    valid = slots < count_tensor[:, None]
    label_tensor = torch.from_numpy(labels).to(device).clamp_min(0)
    positive_scores = module.score_candidates(logits, action_effects[label_tensor])
    positive_scores = positive_scores.masked_fill(~valid, -torch.inf)
    teacher_score = positive_scores.max(dim=1).values
    better = torch.zeros(len(states), dtype=torch.long, device=device)
    for start in range(0, len(action_effects), chunk_size):
        stop = min(start + chunk_size, len(action_effects))
        candidates = action_effects[start:stop][None].expand(len(states), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        better += (scores > teacher_score[:, None]).sum(dim=1)
    ranks = (better + 1).cpu().numpy()
    return {
        "median_teacher_rank": float(np.median(ranks)),
        "p90_teacher_rank": float(np.quantile(ranks, 0.9)),
        "samples": len(states),
        "top1_recall": float(np.mean(ranks <= 1)),
        "top16_recall": float(np.mean(ranks <= 16)),
        "top128_recall": float(np.mean(ranks <= 128)),
        "top512_recall": float(np.mean(ranks <= 512)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("remote primitive training requires CUDA")
    if not 0.0 <= args.high_value_fraction <= 1.0:
        raise ValueError("high-value-fraction must be in [0, 1]")
    if args.reset_value_heads and args.init_checkpoint is None:
        raise ValueError("reset-value-heads requires init-checkpoint")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    module = load_module(args.train_module)

    with np.load(args.teacher, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        labels = payload["teacher_actions"].astype(np.int32, copy=False)
        counts = payload["teacher_action_counts"].astype(np.int16, copy=False)
        values = payload["search_value_targets"].astype(np.float32, copy=False)
        clusters = payload["search_cluster_targets"].astype(np.float32, copy=False)
    groups = np.load(args.group_ids, allow_pickle=False)
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(np.uint8, copy=False)
    if states.shape[1:] != (6, 24) or effects_np.shape[1:] != (6, 24):
        raise ValueError("state/action artifacts have the wrong shape")
    if groups.shape != (len(states),):
        raise ValueError("group ids do not match teacher rows")
    if not np.allclose(clusters.sum(axis=1), values, atol=1.0e-4):
        raise ValueError("cluster targets do not sum to primitive total")
    training_indices = np.flatnonzero(np.mod(groups, 10) != 0)
    heldout_indices = np.flatnonzero(np.mod(groups, 10) == 0)
    if not len(training_indices) or not len(heldout_indices):
        raise ValueError("PID split produced an empty fold")

    config = module.Config(action_count=len(effects_np))
    model = module.MacroEffectPolicyValueNet(config).to(device)
    initialization = "scratch"
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model_state_dict"]
        if args.reset_value_heads:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("total_cost_head.")
                and not key.startswith("cluster_cost_head.")
            }
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            expected_missing = {
                "total_cost_head.weight",
                "total_cost_head.bias",
                "cluster_cost_head.weight",
                "cluster_cost_head.bias",
            }
            if set(missing) != expected_missing or unexpected:
                raise ValueError(
                    f"unexpected warm-start mismatch: missing={missing}, unexpected={unexpected}"
                )
            initialization = "warm_policy_reset_values"
        else:
            model.load_state_dict(state_dict)
            initialization = "warm_all_heads"

    eval_count = min(args.eval_samples, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    policy_count = min(args.policy_eval_samples, len(heldout_indices))
    policy_indices = rng.choice(heldout_indices, size=policy_count, replace=False)
    action_effects = torch.from_numpy(effects_np).to(device)
    before_values = value_metrics(
        predict_values(model, states[eval_indices], device=device),
        values[eval_indices],
    )
    before_policy = policy_metrics(
        module,
        model,
        states[policy_indices],
        labels[policy_indices],
        counts[policy_indices],
        action_effects,
        device=device,
        topk=args.policy_topk,
        chunk_size=args.policy_chunk_size,
    )
    before_policy_ranks = policy_rank_metrics(
        module,
        model,
        states[policy_indices],
        labels[policy_indices],
        counts[policy_indices],
        action_effects,
        device=device,
        chunk_size=args.policy_chunk_size,
    )

    train_model = (
        torch.compile(model, mode="reduce-overhead")
        if args.compile
        else model
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
        fused=True,
    )

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / args.warmup_steps, 1.0e-3)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    training_values = values[training_indices]
    high_weights = np.sqrt(training_values / max(float(training_values.mean()), 1.0e-6))
    high_cdf = np.cumsum(high_weights, dtype=np.float64)
    high_cdf /= high_cdf[-1]
    high_count = int(round(args.batch_size * args.high_value_fraction))
    uniform_count = args.batch_size - high_count
    started = time.perf_counter()
    last_losses: dict[str, float] = {}
    train_model.train()
    for step in range(1, args.steps + 1):
        selected_parts: list[np.ndarray] = []
        if uniform_count:
            selected_parts.append(
                training_indices[rng.integers(len(training_indices), size=uniform_count)]
            )
        if high_count:
            positions = np.searchsorted(high_cdf, rng.random(high_count), side="right")
            selected_parts.append(training_indices[positions])
        selected = np.concatenate(selected_parts)
        rng.shuffle(selected)
        batch_states = torch.from_numpy(states[selected]).to(device)
        batch_labels = torch.from_numpy(labels[selected]).to(device)
        batch_counts = torch.from_numpy(counts[selected]).to(device)
        batch_values = torch.from_numpy(values[selected]).to(device)
        batch_clusters = torch.from_numpy(clusters[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = train_model(batch_states)
            negative_parts: list[torch.Tensor] = []
            if args.hard_negative_actions:
                negative_parts.append(
                    topk_action_ids_from_logits(
                        module,
                        outputs[0].detach(),
                        action_effects,
                        topk=args.hard_negative_actions,
                        chunk_size=args.policy_chunk_size,
                    )
                )
            if args.negative_actions:
                negative_parts.append(
                    torch.randint(
                        len(effects_np),
                        (args.batch_size, args.negative_actions),
                        device=device,
                    )
                )
            if not negative_parts:
                raise ValueError("at least one random or hard negative is required")
            negatives = torch.cat(negative_parts, dim=1)
            loss, last_losses = module.loss_function(
                outputs,
                batch_labels,
                batch_counts,
                batch_values,
                batch_clusters,
                action_effects,
                negatives,
                args.value_weight,
                args.cluster_value_weight,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "step": step,
                        **last_losses,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    after_values = value_metrics(
        predict_values(model, states[eval_indices], device=device),
        values[eval_indices],
    )
    after_policy = policy_metrics(
        module,
        model,
        states[policy_indices],
        labels[policy_indices],
        counts[policy_indices],
        action_effects,
        device=device,
        topk=args.policy_topk,
        chunk_size=args.policy_chunk_size,
    )
    after_policy_ranks = policy_rank_metrics(
        module,
        model,
        states[policy_indices],
        labels[policy_indices],
        counts[policy_indices],
        action_effects,
        device=device,
        chunk_size=args.policy_chunk_size,
    )
    report = {
        "action_count": len(effects_np),
        "action_digest": args.action_digest,
        "after_policy": after_policy,
        "after_policy_ranks": after_policy_ranks,
        "after_values": after_values,
        "batch_size": args.batch_size,
        "before_policy": before_policy,
        "before_policy_ranks": before_policy_ranks,
        "before_values": before_values,
        "compiled": args.compile,
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(elapsed, 4),
        "evaluation_samples": eval_count,
        "heldout_samples": len(heldout_indices),
        "high_value_fraction": args.high_value_fraction,
        "hard_negative_actions": args.hard_negative_actions,
        "initialization": initialization,
        "last_train_losses": last_losses,
        "model_config": config.as_dict(),
        "negative_actions": args.negative_actions,
        "policy_evaluation_samples": policy_count,
        "seed": args.seed,
        "steps": args.steps,
        "teacher_samples": len(states),
        "training_samples": len(training_indices),
        "value_unit": "primitive_moves",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "action_digest": args.action_digest,
            "model_config": config.as_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
