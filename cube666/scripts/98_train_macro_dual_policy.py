"""Train full-library macro retrieval from state/effect geometry."""

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

from cube666.macro_dual_policy import (  # noqa: E402
    MacroDualPolicyConfig,
    MacroDualPolicyValueNet,
)
from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--init-value-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--random-actions", type=int, default=1536)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--cluster-weight", type=float, default=0.5)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=98666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


@torch.inference_mode()
def policy_ranks(
    model: MacroDualPolicyValueNet,
    states: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    effects: np.ndarray,
    costs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    key_parts: list[torch.Tensor] = []
    for start in range(0, len(effects), batch_size):
        effect_batch = torch.from_numpy(effects[start : start + batch_size]).to(device)
        cost_batch = torch.from_numpy(costs[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            keys = model.encode_actions(effect_batch, cost_batch)
        key_parts.append(keys)
    action_keys = torch.cat(key_parts)
    output: list[np.ndarray] = []
    scale = model.logit_scale.exp().clamp_max(100.0)
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        batch = torch.from_numpy(states[start:stop]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            queries = model.encode_states(batch)
        scores = queries @ action_keys.transpose(0, 1) * scale
        batch_labels = torch.from_numpy(labels[start:stop]).to(device).clamp_min(0)
        slots = torch.arange(labels.shape[1], device=device)[None]
        valid = slots < torch.from_numpy(counts[start:stop]).to(device)[:, None]
        positive_scores = scores.gather(1, batch_labels).masked_fill(~valid, -torch.inf)
        best_positive = positive_scores.max(dim=1).values
        ranks = 1 + (scores > best_positive[:, None]).sum(dim=1)
        output.append(ranks.cpu().numpy())
    return np.concatenate(output)


def summarize_ranks(ranks: np.ndarray) -> dict[str, float]:
    return {
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "p90_rank": float(np.quantile(ranks, 0.9)),
        "top1": float(np.mean(ranks <= 1)),
        "top16": float(np.mean(ranks <= 16)),
        "top64": float(np.mean(ranks <= 64)),
        "top256": float(np.mean(ranks <= 256)),
        "top1024": float(np.mean(ranks <= 1024)),
        "top4096": float(np.mean(ranks <= 4096)),
    }


@torch.inference_mode()
def value_metrics(
    model: MacroDualPolicyValueNet,
    states: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    cluster_predictions: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, local = model.value_only(batch)
        predictions.append((total.float() * model.config.total_scale).cpu().numpy())
        cluster_predictions.append(
            (local.float() * model.config.cluster_scale).cpu().numpy()
        )
    prediction = np.concatenate(predictions)
    cluster_prediction = np.concatenate(cluster_predictions)
    errors = prediction - values
    return {
        "bias": float(errors.mean()),
        "cluster_mae": float(np.abs(cluster_prediction - clusters).mean()),
        "correlation": float(np.corrcoef(prediction, values)[0, 1]),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
        clusters = teacher["search_cluster_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    train_indices = np.flatnonzero(np.mod(groups, args.folds) != args.fold)
    heldout_indices = np.flatnonzero(np.mod(groups, args.folds) == args.fold)
    value_model, _ = load_macro_factorized_value_checkpoint(
        args.init_value_checkpoint, device
    )
    config = MacroDualPolicyConfig(
        state_local_dim=value_model.config.local_dim,
        state_local_blocks=value_model.config.local_blocks,
        state_global_dim=value_model.config.global_dim,
        state_global_blocks=value_model.config.global_blocks,
    )
    model = MacroDualPolicyValueNet(config).to(device)
    for name in (
        "local_stem",
        "local_blocks",
        "local_norm",
        "global_stem",
        "global_blocks",
        "global_norm",
        "total_head",
        "cluster_head",
    ):
        getattr(model, name).load_state_dict(getattr(value_model, name).state_dict())
    model.cluster_embedding.data.copy_(value_model.cluster_embedding.data)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    training_values = values[train_indices]
    high_weights = np.sqrt(training_values / max(float(training_values.mean()), 1e-6))
    high_cdf = np.cumsum(high_weights, dtype=np.float64)
    high_cdf /= high_cdf[-1]
    high_count = int(round(args.batch_size * args.high_value_fraction))
    uniform_count = args.batch_size - high_count
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        parts: list[np.ndarray] = []
        if uniform_count:
            parts.append(train_indices[rng.integers(len(train_indices), size=uniform_count)])
        if high_count:
            positions = np.searchsorted(high_cdf, rng.random(high_count), side="right")
            parts.append(train_indices[positions])
        selected = np.concatenate(parts)
        rng.shuffle(selected)
        positive_actions = labels[selected, 0]
        random_actions = rng.integers(len(effects), size=args.random_actions, dtype=np.int32)
        candidate_ids = np.concatenate((positive_actions, random_actions))
        batch_states = torch.from_numpy(states[selected]).to(device)
        candidate_effects = torch.from_numpy(effects[candidate_ids]).to(device)
        candidate_costs = torch.from_numpy(costs[candidate_ids]).to(device)
        state_targets = torch.from_numpy(values[selected]).to(device)
        cluster_targets = torch.from_numpy(clusters[selected]).to(device)
        candidate_id_tensor = torch.from_numpy(candidate_ids).to(device)
        positive_tensor = torch.from_numpy(positive_actions).to(device)
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            queries = train_model.encode_states(batch_states)
            keys = train_model.encode_actions(candidate_effects, candidate_costs)
            logits = queries @ keys.transpose(0, 1)
            logits = logits * train_model.logit_scale.exp().clamp_max(100.0)
            positive_mask = candidate_id_tensor[None].eq(positive_tensor[:, None])
            policy_loss = (
                torch.logsumexp(logits.float(), dim=1)
                - torch.logsumexp(logits.float().masked_fill(~positive_mask, -torch.inf), dim=1)
            ).mean()
            total_scaled, cluster_scaled = train_model.value_only(batch_states)
            value_loss = F.smooth_l1_loss(
                total_scaled.float(), state_targets / config.total_scale, beta=0.05
            )
            cluster_loss = F.smooth_l1_loss(
                cluster_scaled.float(), cluster_targets / config.cluster_scale, beta=0.05
            )
            loss = (
                policy_loss
                + args.value_weight * value_loss
                + args.cluster_weight * cluster_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "cluster_loss": float(cluster_loss.detach()),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "logit_scale": float(model.logit_scale.exp().detach()),
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "step": step,
                "value_loss": float(value_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    eval_count = min(args.eval_samples, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    ranks = policy_ranks(
        model,
        states[eval_indices],
        labels[eval_indices],
        counts[eval_indices],
        effects,
        costs,
        args.eval_batch_size,
        device,
    )
    heldout_policy = summarize_ranks(ranks)
    heldout_values = value_metrics(
        model,
        states[eval_indices],
        values[eval_indices],
        clusters[eval_indices],
        args.eval_batch_size,
        device,
    )
    report = {
        "action_count": len(effects),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "heldout_policy": heldout_policy,
        "heldout_samples": eval_count,
        "heldout_values": heldout_values,
        "history": history,
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_macro_dual_policy_v1",
            "dual_policy_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
