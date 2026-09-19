"""Fine-tune the direct short-macro policy on fitted Bellman action targets."""

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
    MacroActionPolicyNet,
    load_macro_action_policy_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=134666)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@torch.inference_mode()
def evaluate(
    model: MacroActionPolicyNet,
    states: np.ndarray,
    actions: np.ndarray,
    depths: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float | int]:
    model.eval()
    ranks: list[np.ndarray] = []
    losses: list[float] = []
    for start in range(0, len(indices), 1024):
        rows = indices[start : start + 1024]
        batch_states = torch.from_numpy(states[rows]).cuda()
        batch_actions = torch.from_numpy(actions[rows].astype(np.int64)).cuda()
        batch_depths = torch.from_numpy(depths[rows].astype(np.int64)).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(batch_states, remaining_depth=batch_depths)
        target_scores = logits.gather(1, batch_actions[:, None]).squeeze(1)
        ranks.append(((logits > target_scores[:, None]).sum(dim=1) + 1).cpu().numpy())
        losses.append(float(F.cross_entropy(logits.float(), batch_actions)))
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
    with np.load(args.targets, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        actions = payload["bellman_actions"].astype(np.int32, copy=False)
        depths = payload["remaining_depths"].astype(np.int16, copy=False)
        groups = payload["source_state_ids"]
    train_indices = np.flatnonzero(np.mod(groups, 10) != 0)
    heldout_indices = np.flatnonzero(np.mod(groups, 10) == 0)
    eval_indices = rng.choice(
        heldout_indices, size=min(8192, len(heldout_indices)), replace=False
    )
    model, checkpoint = load_macro_action_policy_checkpoint(
        str(args.init_checkpoint), device="cuda"
    )
    before = evaluate(model, states, actions, depths, eval_indices)
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
        batch_states = torch.from_numpy(states[rows]).cuda()
        batch_actions = torch.from_numpy(actions[rows].astype(np.int64)).cuda()
        batch_depths = torch.from_numpy(depths[rows].astype(np.int64)).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = train_model(batch_states, remaining_depth=batch_depths)
            loss = F.cross_entropy(
                logits.float(), batch_actions, label_smoothing=args.label_smoothing
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach())
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"elapsed_seconds": round(time.perf_counter() - started, 3), "loss": last_loss, "step": step}), flush=True)
    after = evaluate(model, states, actions, depths, eval_indices)
    report = {
        "after": after,
        "before": before,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "label_smoothing": args.label_smoothing,
        "last_loss": last_loss,
        "model_config": model.config.to_dict(),
        "source_checkpoint": str(args.init_checkpoint),
        "steps": args.steps,
        "targets": str(args.targets),
        "train_samples": int(len(train_indices)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": checkpoint.get("kind", "cube666_macro_direct_action_policy_v1"),
            "model_config": model.config.to_dict(),
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
