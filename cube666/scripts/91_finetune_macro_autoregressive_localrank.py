"""Fine-tune macro decoding/value with local short-macro child ranking."""

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

from cube666.macro_autoregressive import load_macro_autoregressive_checkpoint  # noqa: E402


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
    parser.add_argument("--token-weight", type=float, default=1.0)
    parser.add_argument("--state-value-weight", type=float, default=5.0)
    parser.add_argument("--child-value-weight", type=float, default=5.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-temperature", type=float, default=4.0)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=91666)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_macro_autoregressive_checkpoint(
        args.init_checkpoint, device
    )
    config = model.config
    action_tokens = np.asarray(checkpoint["action_tokens"], dtype=np.int16)
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(np.int32)
    effects_np = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(np.uint8)
    short_actions = np.flatnonzero((costs <= 14) & (action_tokens[:, 0] >= 0)).astype(np.int32)
    effects = torch.from_numpy(effects_np).to(device)
    short_action_tensor = torch.from_numpy(short_actions).to(device)
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        all_states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        all_values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    chosen = np.full(len(labels), -1, dtype=np.int32)
    for row in range(len(labels)):
        valid = [
            int(action)
            for action in labels[row, : int(counts[row])]
            if action >= 0 and costs[int(action)] <= 14
        ]
        if valid:
            chosen[row] = valid[0]
    eligible = np.flatnonzero(chosen >= 0)
    train_indices = eligible[np.mod(groups[eligible], args.folds) != args.fold]
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
        positive_effects = effects[positive_actions]
        positive_children = states.gather(-1, positive_effects.long())
        random_positions = torch.randint(
            len(short_action_tensor),
            (args.batch_size, args.negative_actions // 2),
            device=device,
        )
        random_negatives = short_action_tensor[random_positions]
        in_batch = positive_actions[
            torch.randint(
                args.batch_size,
                (args.batch_size, args.negative_actions - args.negative_actions // 2),
                device=device,
            )
        ]
        negative_actions = torch.cat((random_negatives, in_batch), dim=1)
        negative_actions = torch.where(
            negative_actions.eq(positive_actions[:, None]),
            short_action_tensor[
                torch.remainder(random_positions[:, :1] + 1, len(short_action_tensor))
            ],
            negative_actions,
        )
        negative_effects = effects[negative_actions]
        negative_children = states[:, None].expand(-1, args.negative_actions, -1, -1).gather(
            -1, negative_effects.long()
        )
        target_tokens = torch.from_numpy(action_tokens[chosen[selected]]).to(device).long()
        prefixes = torch.full_like(target_tokens, config.bos_token)
        prefixes[:, 1:] = target_tokens[:, :-1].clamp_min(0)
        state_targets = torch.from_numpy(all_values[selected]).to(device)
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
            logits, state_scaled = train_model(states, prefixes)
            positive_hidden = train_model.encode(positive_children)
            positive_scaled = train_model.value_head(positive_hidden).squeeze(1)
            negative_hidden = train_model.encode(negative_children.flatten(0, 1))
            negative_scaled = train_model.value_head(negative_hidden).squeeze(1).reshape(
                args.batch_size, args.negative_actions
            )
            token_loss = F.cross_entropy(
                logits.float().reshape(-1, config.vocabulary_size),
                target_tokens.reshape(-1),
                ignore_index=-100,
            )
            state_value_loss = F.smooth_l1_loss(
                state_scaled.float(), state_targets / config.value_scale, beta=0.05
            )
            child_value_loss = F.smooth_l1_loss(
                positive_scaled.float(), child_targets / config.value_scale, beta=0.05
            )
            candidate_moves = torch.cat(
                (
                    positive_scaled.float()[:, None],
                    negative_scaled.float(),
                ),
                dim=1,
            ) * config.value_scale
            ranking_logits = -candidate_moves / args.ranking_temperature
            ranking_loss = F.cross_entropy(
                ranking_logits,
                torch.zeros(args.batch_size, dtype=torch.long, device=device),
            )
            nonnegative = F.relu(-candidate_moves).mean()
            loss = (
                args.token_weight * token_loss
                + args.state_value_weight * state_value_loss
                + args.child_value_weight * child_value_loss
                + args.ranking_weight * ranking_loss
                + 0.05 * nonnegative
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                top1 = candidate_moves.argmin(dim=1).eq(0).float().mean()
                top8 = candidate_moves.topk(8, dim=1, largest=False).indices.eq(0).any(dim=1).float().mean()
            row = {
                "child_value_loss": float(child_value_loss.detach()),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "loss": float(loss.detach()),
                "ranking_loss": float(ranking_loss.detach()),
                "state_value_loss": float(state_value_loss.detach()),
                "step": step,
                "token_loss": float(token_loss.detach()),
                "train_rank_top1": float(top1),
                "train_rank_top8": float(top8),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "batch_size": args.batch_size,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "history": history,
        "init_checkpoint": str(args.init_checkpoint),
        "negative_actions": args.negative_actions,
        "steps": args.steps,
        "train_samples": int(len(train_indices)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_macro_autoregressive_localrank_v1",
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
