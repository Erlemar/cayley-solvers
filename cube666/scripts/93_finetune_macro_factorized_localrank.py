"""Add local short-macro ranking to the factorized primitive-cost value."""

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

from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negative-actions", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--state-value-weight", type=float, default=5.0)
    parser.add_argument("--cluster-value-weight", type=float, default=1.0)
    parser.add_argument("--child-value-weight", type=float, default=5.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-temperature", type=float, default=4.0)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=93666)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def choose_short_labels(
    labels: np.ndarray, counts: np.ndarray, costs: np.ndarray
) -> np.ndarray:
    chosen = np.full(len(labels), -1, dtype=np.int32)
    for row in range(len(labels)):
        valid = [
            int(action)
            for action in labels[row, : int(counts[row])]
            if action >= 0 and costs[int(action)] <= 14
        ]
        if valid:
            chosen[row] = valid[0]
    return chosen


def sample_negative_actions(
    positive_actions: torch.Tensor,
    short_actions: torch.Tensor,
    count: int,
) -> torch.Tensor:
    batch_size = len(positive_actions)
    random_positions = torch.randint(
        len(short_actions), (batch_size, count // 2), device=positive_actions.device
    )
    random_negatives = short_actions[random_positions]
    in_batch = positive_actions[
        torch.randint(
            batch_size,
            (batch_size, count - count // 2),
            device=positive_actions.device,
        )
    ]
    negatives = torch.cat((random_negatives, in_batch), dim=1)
    replacement = short_actions[
        torch.remainder(random_positions[:, :1] + 1, len(short_actions))
    ]
    return torch.where(negatives.eq(positive_actions[:, None]), replacement, negatives)


@torch.inference_mode()
def evaluate_local(
    model: torch.nn.Module,
    states_np: np.ndarray,
    values_np: np.ndarray,
    chosen_np: np.ndarray,
    effects: torch.Tensor,
    costs: np.ndarray,
    short_actions: torch.Tensor,
    negative_count: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    states = torch.from_numpy(states_np).to(device)
    positives = torch.from_numpy(chosen_np).to(device)
    negatives = sample_negative_actions(positives, short_actions, negative_count)
    positive_children = states.gather(-1, effects[positives].long())
    negative_children = states[:, None].expand(-1, negative_count, -1, -1).gather(
        -1, effects[negatives].long()
    )
    state_scaled, _ = model(states)
    positive_scaled, _ = model(positive_children)
    negative_scaled, _ = model(negative_children.flatten(0, 1))
    negative_scaled = negative_scaled.reshape(len(states), negative_count)
    candidate = torch.cat((positive_scaled[:, None], negative_scaled), dim=1)
    order = torch.argsort(candidate, dim=1)
    ranks = (order == 0).float().argmax(dim=1) + 1
    prediction = state_scaled.float() * model.config.total_scale
    targets = torch.from_numpy(values_np).to(device)
    return {
        "mae": float((prediction - targets).abs().mean()),
        "samples": len(states_np),
        "top1": float((ranks <= 1).float().mean()),
        "top8": float((ranks <= 8).float().mean()),
        "top16": float((ranks <= 16).float().mean()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_macro_factorized_value_checkpoint(
        args.init_checkpoint, device
    )
    config = model.config
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        all_states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        all_values = teacher["search_value_targets"].astype(np.float32, copy=False)
        all_clusters = teacher["search_cluster_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(np.int32)
    effects = torch.from_numpy(
        np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(np.uint8)
    ).to(device)
    chosen = choose_short_labels(labels, counts, costs)
    short_actions_np = np.flatnonzero(costs <= 14).astype(np.int32)
    short_actions = torch.from_numpy(short_actions_np).to(device)
    eligible = np.flatnonzero(chosen >= 0)
    train_indices = eligible[np.mod(groups[eligible], args.folds) != args.fold]
    heldout_indices = eligible[np.mod(groups[eligible], args.folds) == args.fold]
    eval_count = min(args.eval_samples, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    before = evaluate_local(
        model,
        all_states[eval_indices],
        all_values[eval_indices],
        chosen[eval_indices],
        effects,
        costs,
        short_actions,
        args.negative_actions,
        device,
    )
    training_values = all_values[train_indices]
    high_weights = np.sqrt(training_values / max(float(training_values.mean()), 1e-6))
    high_cdf = np.cumsum(high_weights, dtype=np.float64)
    high_cdf /= high_cdf[-1]
    high_count = int(round(args.batch_size * args.high_value_fraction))
    uniform_count = args.batch_size - high_count
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
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
        states = torch.from_numpy(all_states[selected]).to(device)
        positive_actions = torch.from_numpy(chosen[selected]).to(device)
        negatives = sample_negative_actions(
            positive_actions, short_actions, args.negative_actions
        )
        positive_children = states.gather(-1, effects[positive_actions].long())
        negative_children = states[:, None].expand(-1, args.negative_actions, -1, -1).gather(
            -1, effects[negatives].long()
        )
        state_targets = torch.from_numpy(all_values[selected]).to(device)
        cluster_targets = torch.from_numpy(all_clusters[selected]).to(device)
        selected_costs = torch.from_numpy(costs[chosen[selected]]).to(device)
        child_targets = (state_targets - selected_costs).clamp_min(0)
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state_scaled, cluster_scaled = train_model(states)
            positive_scaled, _ = train_model(positive_children)
            negative_scaled, _ = train_model(negative_children.flatten(0, 1))
            negative_scaled = negative_scaled.reshape(args.batch_size, args.negative_actions)
            state_loss = F.smooth_l1_loss(
                state_scaled.float(), state_targets / config.total_scale, beta=0.05
            )
            cluster_loss = F.smooth_l1_loss(
                cluster_scaled.float(), cluster_targets / config.cluster_scale, beta=0.05
            )
            child_loss = F.smooth_l1_loss(
                positive_scaled.float(), child_targets / config.total_scale, beta=0.05
            )
            candidate_moves = torch.cat(
                (positive_scaled.float()[:, None], negative_scaled.float()), dim=1
            ) * config.total_scale
            ranking_loss = F.cross_entropy(
                -candidate_moves / args.ranking_temperature,
                torch.zeros(args.batch_size, dtype=torch.long, device=device),
            )
            nonnegative = F.relu(-candidate_moves).mean()
            loss = (
                args.state_value_weight * state_loss
                + args.cluster_value_weight * cluster_loss
                + args.child_value_weight * child_loss
                + args.ranking_weight * ranking_loss
                + 0.05 * nonnegative
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                ranks = (torch.argsort(candidate_moves, dim=1) == 0).float().argmax(dim=1) + 1
            row = {
                "child_loss": float(child_loss.detach()),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "loss": float(loss.detach()),
                "ranking_loss": float(ranking_loss.detach()),
                "state_loss": float(state_loss.detach()),
                "step": step,
                "train_top1": float((ranks <= 1).float().mean()),
                "train_top8": float((ranks <= 8).float().mean()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    after = evaluate_local(
        model,
        all_states[eval_indices],
        all_values[eval_indices],
        chosen[eval_indices],
        effects,
        costs,
        short_actions,
        args.negative_actions,
        device,
    )
    report = {
        "after": after,
        "before": before,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "history": history,
        "negative_actions": args.negative_actions,
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_macro_factorized_primitive_value_localrank_v1",
            "model_state_dict": model.state_dict(),
            "localrank_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
