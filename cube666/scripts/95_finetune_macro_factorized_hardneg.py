"""Fine-tune factorized primitive value on harvested proposer hard negatives."""

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
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hard-actions", type=int, default=48)
    parser.add_argument("--random-actions", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--state-value-weight", type=float, default=10.0)
    parser.add_argument("--cluster-value-weight", type=float, default=1.0)
    parser.add_argument("--child-value-weight", type=float, default=5.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-temperature", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=95666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def choose_labels(labels: np.ndarray, counts: np.ndarray, costs: np.ndarray) -> np.ndarray:
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
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(np.int32)
    effects = torch.from_numpy(
        np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(np.uint8)
    ).to(device)
    short_actions = torch.from_numpy(np.flatnonzero(costs <= 14).astype(np.int32)).to(device)
    chosen = choose_labels(labels, counts, costs)
    with np.load(args.hardneg, allow_pickle=False) as hard:
        row_indices = hard["row_indices"].astype(np.int64, copy=False)
        proposals = hard["proposals"].astype(np.int32, copy=False)
    if np.any(chosen[row_indices] < 0) or np.any(proposals < 0):
        raise ValueError("hard-negative rows/proposals are incomplete")
    negative_count = args.hard_actions + args.random_actions
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        positions = rng.integers(len(row_indices), size=args.batch_size)
        selected = row_indices[positions]
        states = torch.from_numpy(all_states[selected]).to(device)
        positive_actions = torch.from_numpy(chosen[selected]).to(device)
        hard_columns = rng.integers(proposals.shape[1], size=(args.batch_size, args.hard_actions))
        hard_ids = proposals[positions[:, None], hard_columns]
        hard_actions = torch.from_numpy(hard_ids).to(device)
        random_positions = torch.randint(
            len(short_actions), (args.batch_size, args.random_actions), device=device
        )
        random_ids = short_actions[random_positions]
        negatives = torch.cat((hard_actions, random_ids), dim=1)
        replacement = short_actions[
            torch.remainder(random_positions[:, :1] + 1, len(short_actions))
        ]
        negatives = torch.where(
            negatives.eq(positive_actions[:, None]), replacement, negatives
        )
        positive_children = states.gather(-1, effects[positive_actions].long())
        negative_children = states[:, None].expand(-1, negative_count, -1, -1).gather(
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
            negative_scaled = negative_scaled.reshape(args.batch_size, negative_count)
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
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "hard_actions": args.hard_actions,
        "hard_rows": len(row_indices),
        "history": history,
        "random_actions": args.random_actions,
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_macro_factorized_primitive_value_hardneg_v1",
            "model_state_dict": model.state_dict(),
            "hardneg_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
