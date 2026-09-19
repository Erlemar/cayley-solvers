from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .gflownet import (
    JewelGFNConfig,
    build_gflownet,
    build_jewel_env,
    random_walk_states,
    regularized_tb_loss,
    rollout_solve,
    sample_forward_trajectories,
)
from .official import OfficialPuzzle
from .puzzle import GROUP_ORDER


@torch.no_grad()
def evaluate_rollouts(model, env, evaluation: dict[int, torch.Tensor], max_steps: int) -> dict:
    model.eval()
    result: dict[str, float | int] = {}
    solved_total = 0
    count_total = 0
    solved_lengths: list[float] = []
    for depth, states in evaluation.items():
        solved, lengths = rollout_solve(model, env, states, max_steps=max_steps, greedy=True)
        count = int(len(states))
        n_solved = int(solved.sum())
        result[f"greedy_d{depth}_solved"] = n_solved
        result[f"greedy_d{depth}_count"] = count
        result[f"greedy_d{depth}_mean_length"] = (
            float(lengths[solved].float().mean()) if n_solved else float("nan")
        )
        solved_total += n_solved
        count_total += count
        if n_solved:
            solved_lengths.extend(lengths[solved].float().cpu().tolist())
    result["greedy_solved"] = solved_total
    result["greedy_count"] = count_total
    result["greedy_mean_length"] = float(np.mean(solved_lengths)) if solved_lengths else float("nan")
    model.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--out", default="jewel/models/gflownet_v1")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trajectory-length", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--res-blocks", type=int, default=3)
    parser.add_argument("--encoding", choices=["onehot", "embedding"], default="onehot")
    parser.add_argument("--embed-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--reg-coef", type=float, default=1e-5)
    parser.add_argument("--reg-mode", choices=["flow", "logflow"], default="flow")
    parser.add_argument("--eps-explore", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=1_000)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--eval-per-depth", type=int, default=128)
    parser.add_argument("--eval-depths", default="8,12,16,18")
    parser.add_argument("--eval-max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    official = OfficialPuzzle.load(args.puzzle_info)
    env = build_jewel_env(official, device)
    config = JewelGFNConfig(args.hidden, args.res_blocks, args.encoding, args.embed_dim)
    raw_model = build_gflownet(config).to(device)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    start_step = 0
    history: list[dict] = []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint.get("step", 0))
        history = list(checkpoint.get("history", []))

    model = torch.compile(raw_model, mode="reduce-overhead") if args.compile else raw_model
    depths = [int(value) for value in args.eval_depths.split(",")]
    evaluation = random_walk_states(
        env,
        depths,
        args.eval_per_depth,
        seed=args.seed + 1,
        non_backtracking=True,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    params = sum(parameter.numel() for parameter in raw_model.parameters())
    print(
        json.dumps(
            {
                "device": str(device),
                "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "parameters": params,
                "group_order": GROUP_ORDER,
                "true_log_z": env.true_log_z,
                "config": config.to_dict(),
                "args": vars(args),
            }
        ),
        flush=True,
    )

    started = perf_counter()
    interval_started = started
    interval_loss = 0.0
    interval_count = 0
    best_solved = max(
        (int(row.get("greedy_solved", -1)) for row in history), default=-1
    )
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            states, actions = sample_forward_trajectories(
                model,
                env,
                args.batch_size,
                args.trajectory_length,
                eps_explore=args.eps_explore,
            )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, metrics = regularized_tb_loss(
                model,
                env,
                states,
                actions,
                reg_coef=args.reg_coef,
                reg_mode=args.reg_mode,
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite GFlowNet loss at step {step}: {metrics}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        optimizer.step()
        interval_loss += float(loss.detach())
        interval_count += 1

        if step % args.print_every == 0:
            elapsed = perf_counter() - interval_started
            row = {
                "step": step,
                "loss": interval_loss / max(interval_count, 1),
                **metrics,
                "grad_norm": float(grad_norm),
                "steps_per_second": interval_count / max(elapsed, 1e-9),
                "elapsed_seconds": perf_counter() - started,
            }
            print(json.dumps(row), flush=True)
            interval_started = perf_counter()
            interval_loss = 0.0
            interval_count = 0

        do_eval = step % args.eval_every == 0 or step == args.steps
        if do_eval:
            evaluation_metrics = evaluate_rollouts(raw_model, env, evaluation, args.eval_max_steps)
            row = {"step": step, **evaluation_metrics}
            history.append(row)
            print(json.dumps({"evaluation": row}), flush=True)
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.to_dict(),
                "step": step,
                "history": history,
                "training_args": vars(args),
            }
            torch.save(checkpoint, output / "last.pt")
            if int(evaluation_metrics["greedy_solved"]) > best_solved:
                best_solved = int(evaluation_metrics["greedy_solved"])
                torch.save(checkpoint, output / "best.pt")
        elif step % args.save_every == 0:
            torch.save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config.to_dict(),
                    "step": step,
                    "history": history,
                    "training_args": vars(args),
                },
                output / "last.pt",
            )


if __name__ == "__main__":
    main()
