"""Train a full-cube primitive value model, then refine it with exhaustive Bellman targets."""

from __future__ import annotations

import argparse
import copy
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

from cube666.primitive_policy import load_primitive_checkpoint  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--pool-size", type=int, default=131072)
    parser.add_argument("--eval-size", type=int, default=16384)
    parser.add_argument("--maximum-depth", type=int, default=200)
    parser.add_argument("--pretrain-steps", type=int, default=5000)
    parser.add_argument("--bellman-steps", type=int, default=1500)
    parser.add_argument("--pretrain-batch-size", type=int, default=4096)
    parser.add_argument("--bellman-batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-update-every", type=int, default=50)
    parser.add_argument("--minimum-target-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=80666)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def local_effects(
    puzzle: Cube666Puzzle,
    orbit_positions: np.ndarray,
    global_to_local: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            [
                global_to_local[np.asarray(puzzle.generators[name])[orbit]]
                for orbit in orbit_positions
            ]
            for name in puzzle.move_names
        ],
        dtype=np.uint8,
    )


@torch.inference_mode()
def generate_pool(
    size: int,
    maximum_depth: int,
    effects: torch.Tensor,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=effects.device)
    generator.manual_seed(seed)
    depths = torch.randint(
        1,
        maximum_depth + 1,
        (size,),
        generator=generator,
        device=effects.device,
    )
    identity = torch.arange(24, dtype=torch.uint8, device=effects.device)
    states = identity[None, None].expand(size, 9, -1).clone()
    last = torch.full((size,), -1, dtype=torch.long, device=effects.device)
    for step in range(maximum_depth):
        actions = torch.randint(
            0,
            len(effects),
            (size,),
            generator=generator,
            device=effects.device,
        )
        if step:
            backtrack = actions.eq(torch.bitwise_xor(last, 1))
            actions = torch.where(
                backtrack,
                torch.remainder(actions + 2, len(effects)),
                actions,
            )
        active = step < depths
        moved = states.gather(-1, effects[actions].long())
        states = torch.where(active[:, None, None], moved, states)
        last = torch.where(active, actions, last)
    return states, depths.float()


@torch.inference_mode()
def predict_values(
    model: torch.nn.Module,
    states: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    values: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, scaled = model(states[start : start + batch_size])
        values.append(scaled.float() * model.config.value_scale)
    return torch.cat(values)


@torch.inference_mode()
def metrics(
    model: torch.nn.Module,
    states: torch.Tensor,
    depths: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    predictions = predict_values(model, states, batch_size)
    errors = predictions - depths
    correlation = float(
        np.corrcoef(predictions.cpu().numpy(), depths.cpu().numpy())[0, 1]
    )
    return {
        "bias": float(errors.mean()),
        "correlation": correlation,
        "mae": float(errors.abs().mean()),
        "prediction_maximum": float(predictions.max()),
        "prediction_minimum": float(predictions.min()),
        "rmse": float((errors.square().mean()).sqrt()),
    }


def cosine_lr(step: int, total: int, warmup: int = 100) -> float:
    if step < warmup:
        return max((step + 1) / warmup, 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 <= args.minimum_target_fraction <= 1.0:
        raise ValueError("minimum-target-fraction must be in [0, 1]")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model, checkpoint = load_primitive_checkpoint(args.init_checkpoint, device)
    if model.config.encoding != "orbit_one_hot":
        raise ValueError("random-walk Bellman training requires orbit-one-hot encoding")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    orbit_positions = np.asarray(checkpoint["orbit_positions"], dtype=np.int64)
    global_to_local = np.asarray(checkpoint["global_to_local"], dtype=np.uint8)
    effects = torch.from_numpy(
        local_effects(puzzle, orbit_positions, global_to_local)
    ).to(device)
    pool_states, pool_depths = generate_pool(
        args.pool_size, args.maximum_depth, effects, seed=args.seed
    )
    eval_states, eval_depths = generate_pool(
        args.eval_size, args.maximum_depth, effects, seed=args.seed + 1
    )
    before = metrics(model, eval_states, eval_depths, args.inference_batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    started = time.perf_counter()
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 2)
    for step in range(1, args.pretrain_steps + 1):
        factor = cosine_lr(step, args.pretrain_steps)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * factor
        indices = torch.randint(
            0,
            len(pool_states),
            (args.pretrain_batch_size,),
            generator=generator,
            device=device,
        )
        states = pool_states[indices]
        targets = pool_depths[indices] / model.config.value_scale
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
        if step == 1 or step % args.log_every == 0 or step == args.pretrain_steps:
            print(
                json.dumps(
                    {
                        "phase": "pretrain",
                        "step": step,
                        "loss": float(loss.detach()),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    after_pretrain = metrics(
        model, eval_states, eval_depths, args.inference_batch_size
    )

    target_model = copy.deepcopy(model).eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate * 0.5, weight_decay=1e-4, fused=True
    )
    for step in range(1, args.bellman_steps + 1):
        indices = torch.randint(
            0,
            len(pool_states),
            (args.bellman_batch_size,),
            generator=generator,
            device=device,
        )
        states = pool_states[indices]
        upper_bounds = pool_depths[indices]
        neighbors = states[:, None].expand(-1, len(effects), -1, -1).gather(
            -1, effects[None].expand(len(states), -1, -1, -1).long()
        )
        flat_neighbors = neighbors.flatten(0, 1)
        with torch.no_grad():
            neighbor_values = predict_values(
                target_model, flat_neighbors, args.inference_batch_size
            ).reshape(len(states), len(effects))
            solved = neighbors.eq(
                torch.arange(24, dtype=torch.uint8, device=device)[None, None, None]
            ).all(dim=(2, 3))
            neighbor_values = torch.where(
                solved, torch.zeros_like(neighbor_values), neighbor_values.clamp_min(0)
            )
            targets_moves = torch.minimum(
                upper_bounds, 1.0 + neighbor_values.min(dim=1).values
            )
            targets_moves = torch.maximum(
                targets_moves, args.minimum_target_fraction * upper_bounds
            )
            targets = targets_moves / model.config.value_scale
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, predictions = train_model(states)
            bellman_loss = F.smooth_l1_loss(
                predictions.float(), targets.float(), beta=0.02
            )
            upper_penalty = F.relu(
                predictions.float() - upper_bounds / model.config.value_scale
            ).mean()
            nonnegative = F.relu(-predictions.float()).mean()
            loss = bellman_loss + 0.1 * upper_penalty + 0.1 * nonnegative
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if args.target_update_every > 0 and step % args.target_update_every == 0:
            target_model.load_state_dict(model.state_dict())
        if step == 1 or step % args.log_every == 0 or step == args.bellman_steps:
            print(
                json.dumps(
                    {
                        "phase": "bellman",
                        "step": step,
                        "loss": float(loss.detach()),
                        "target_mean": float(targets_moves.mean()),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    after_bellman = metrics(
        model, eval_states, eval_depths, args.inference_batch_size
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "after_bellman": after_bellman,
        "after_pretrain": after_pretrain,
        "before": before,
        "bellman_steps": args.bellman_steps,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "eval_size": args.eval_size,
        "maximum_depth": args.maximum_depth,
        "minimum_target_fraction": args.minimum_target_fraction,
        "pool_size": args.pool_size,
        "pretrain_steps": args.pretrain_steps,
        "target_update_every": args.target_update_every,
        "seed": args.seed,
    }
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_primitive_randomwalk_bellman_v1",
            "model_state_dict": model.state_dict(),
            "randomwalk_bellman_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
