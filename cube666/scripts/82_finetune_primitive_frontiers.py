"""Fine-tune a primitive value model on replay-recoverable beam hard frontiers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.primitive_policy import load_primitive_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument(
        "--memory-model",
        type=Path,
        default=PROJECT / "models/cube666_trajectory_memory_portfolio_v1/model.npz",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--frontier-fraction", type=float, default=0.5)
    parser.add_argument("--frontier-target-cap", type=float, default=120.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=82666)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    states: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, scaled = model(batch)
        output.append((scaled.float() * model.config.value_scale).cpu().numpy())
    return np.concatenate(output)


def value_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    return {
        "bias": float(errors.mean()),
        "correlation": float(np.corrcoef(predictions, targets)[0, 1]),
        "mae": float(np.abs(errors).mean()),
        "prediction_mean": float(predictions.mean()),
        "target_mean": float(targets.mean()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 < args.frontier_fraction < 1:
        raise ValueError("frontier-fraction must be between zero and one")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_primitive_checkpoint(args.init_checkpoint, device)
    for parameter in model.policy_head.parameters():
        parameter.requires_grad_(False)
    orbit_positions = np.asarray(checkpoint["orbit_positions"], dtype=np.int64)
    global_to_local = np.asarray(checkpoint["global_to_local"], dtype=np.uint8)
    frontier_states: list[np.ndarray] = []
    frontier_targets: list[np.ndarray] = []
    for path in sorted(args.frontier_dir.glob("pid_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            frontier_states.append(payload["states"].astype(np.uint8, copy=False))
            frontier_targets.append(
                np.minimum(
                    payload["recovery_targets"].astype(np.float32, copy=False),
                    args.frontier_target_cap,
                )
            )
    if not frontier_states:
        raise ValueError("frontier directory has no pid_*.npz files")
    hard_states = np.concatenate(frontier_states)
    hard_targets = np.concatenate(frontier_targets)
    with np.load(args.memory_model, allow_pickle=False) as memory:
        raw_states = memory["states"].astype(np.uint8, copy=False)
        memory_targets = memory["distances"].astype(np.float32, copy=False)
        pids = memory["pids"].astype(np.int64, copy=False)
    memory_states = global_to_local[raw_states[:, orbit_positions]]
    train_memory = np.flatnonzero(np.mod(pids, 5) != 0)
    heldout_memory = np.flatnonzero(np.mod(pids, 5) == 0)
    eval_memory = rng.choice(
        heldout_memory, size=min(8192, len(heldout_memory)), replace=False
    )
    before_frontier = value_metrics(
        predict(model, hard_states, args.eval_batch_size, device), hard_targets
    )
    before_memory = value_metrics(
        predict(model, memory_states[eval_memory], args.eval_batch_size, device),
        memory_targets[eval_memory],
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=1e-4,
        fused=True,
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    frontier_count = int(round(args.batch_size * args.frontier_fraction))
    memory_count = args.batch_size - frontier_count
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        hard_indices = rng.integers(len(hard_states), size=frontier_count)
        memory_indices = train_memory[rng.integers(len(train_memory), size=memory_count)]
        batch_states = np.concatenate(
            (hard_states[hard_indices], memory_states[memory_indices])
        )
        batch_targets = np.concatenate(
            (hard_targets[hard_indices], memory_targets[memory_indices])
        )
        order = rng.permutation(args.batch_size)
        states = torch.from_numpy(batch_states[order]).to(device)
        targets = torch.from_numpy(batch_targets[order]).to(device) / model.config.value_scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, predictions = train_model(states)
            value_loss = F.smooth_l1_loss(
                predictions.float(), targets.float(), beta=0.05
            )
            nonnegative = F.relu(-predictions.float()).mean()
            loss = value_loss + 0.1 * nonnegative
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "loss": float(loss.detach()),
                        "step": step,
                    }
                ),
                flush=True,
            )
    after_frontier = value_metrics(
        predict(model, hard_states, args.eval_batch_size, device), hard_targets
    )
    after_memory = value_metrics(
        predict(model, memory_states[eval_memory], args.eval_batch_size, device),
        memory_targets[eval_memory],
    )
    report = {
        "after_frontier": after_frontier,
        "after_heldout_memory": after_memory,
        "before_frontier": before_frontier,
        "before_heldout_memory": before_memory,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "frontier_states": len(hard_states),
        "frontier_target_cap": args.frontier_target_cap,
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_primitive_frontier_value_v1",
            "model_state_dict": model.state_dict(),
            "frontier_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
