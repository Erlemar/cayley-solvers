"""Train primitive policy/value with exhaustive local child-ranking supervision."""

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
sys.path.insert(0, str(PROJECT / "cube_nnn" / "src"))

from cube666.primitive_policy import load_primitive_checkpoint  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.symmetry import build_symmetries  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--memory-model",
        type=Path,
        default=PROJECT / "models/cube666_trajectory_memory_portfolio_v1/model.npz",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--state-value-weight", type=float, default=0.5)
    parser.add_argument("--child-value-weight", type=float, default=0.5)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-temperature", type=float, default=8.0)
    parser.add_argument("--symmetry-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=84666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def action_targets(masks: np.ndarray, action_count: int) -> np.ndarray:
    bits = np.arange(action_count, dtype=np.uint64)
    return ((masks[:, None] >> bits[None, :]) & np.uint64(1)).astype(np.bool_)


def symmetry_tables(
    data_dir: Path,
    move_names: tuple[str, ...],
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = build_symmetries(NCube.from_puzzle_info(data_dir / "puzzle_info.json"))
    if len(table) != 48 or not 1 <= count <= len(table):
        raise ValueError("symmetry-count must be in 1..48")
    permutations = np.asarray(table.perms[:count], dtype=np.uint8)
    inverse_positions = np.argsort(permutations, axis=1).astype(np.int64)
    move_to_index = {name: index for index, name in enumerate(move_names)}
    action_maps = np.asarray(
        [
            [move_to_index[relabel[name]] for name in move_names]
            for relabel in table.relabel[:count]
        ],
        dtype=np.int64,
    )
    return permutations, inverse_positions, action_maps


def augment_batch(
    raw_states: np.ndarray,
    action_targets_all: np.ndarray,
    selected: np.ndarray,
    frame_indices: np.ndarray,
    permutations: np.ndarray,
    inverse_positions: np.ndarray,
    action_maps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frames = permutations[frame_indices]
    reordered = np.take_along_axis(
        raw_states[selected], inverse_positions[frame_indices], axis=1
    )
    transformed_states = np.take_along_axis(
        frames, reordered.astype(np.int64, copy=False), axis=1
    )
    transformed_actions = np.zeros_like(action_targets_all[selected])
    np.put_along_axis(
        transformed_actions,
        action_maps[frame_indices],
        action_targets_all[selected],
        axis=1,
    )
    return transformed_states, transformed_actions


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


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.ranking_temperature <= 0:
        raise ValueError("ranking-temperature must be positive")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_primitive_checkpoint(args.init_checkpoint, device)
    if model.config.encoding != "orbit_one_hot":
        raise ValueError("local ranking requires orbit-one-hot encoding")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    orbit_positions = np.asarray(checkpoint["orbit_positions"], dtype=np.int64)
    global_to_local = np.asarray(checkpoint["global_to_local"], dtype=np.uint8)
    effects = torch.from_numpy(
        local_effects(puzzle, orbit_positions, global_to_local)
    ).to(device)
    permutations, inverse_positions, action_maps = symmetry_tables(
        args.data_dir, tuple(puzzle.move_names), args.symmetry_count
    )
    with np.load(args.memory_model, allow_pickle=False) as memory:
        raw_states = memory["states"].astype(np.uint8, copy=False)
        pids = memory["pids"].astype(np.int64, copy=False)
        distances = memory["distances"].astype(np.float32, copy=False)
        masks = memory["action_masks"].astype(np.uint64, copy=False)
        offsets = memory["offsets"].astype(np.int64, copy=False)
    all_actions = action_targets(masks, model.config.action_count)
    train_pids = np.asarray(
        [pid for pid in range(len(offsets) - 1) if pid % args.folds != args.fold],
        dtype=np.int64,
    )
    train_starts = offsets[train_pids]
    train_sizes = offsets[train_pids + 1] - train_starts - 1
    if np.any(train_sizes <= 0):
        raise ValueError("a training PID has no nonterminal state")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        pid_rows = rng.integers(0, len(train_pids), size=args.batch_size)
        selected = train_starts[pid_rows] + (
            rng.random(args.batch_size) * train_sizes[pid_rows]
        ).astype(np.int64)
        frame_indices = rng.integers(0, args.symmetry_count, size=args.batch_size)
        augmented_raw, augmented_actions = augment_batch(
            raw_states,
            all_actions,
            selected,
            frame_indices,
            permutations,
            inverse_positions,
            action_maps,
        )
        states = torch.from_numpy(
            global_to_local[augmented_raw[:, orbit_positions]]
        ).to(device)
        correct = torch.from_numpy(augmented_actions).to(device)
        target_moves = torch.from_numpy(distances[selected]).to(device)
        children = states[:, None].expand(-1, len(effects), -1, -1).gather(
            -1, effects[None].expand(len(states), -1, -1, -1).long()
        )
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(
                args.steps - args.warmup_steps, 1
            )
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, scaled_values = train_model(states)
            _, child_scaled = train_model(children.flatten(0, 1))
            child_scaled = child_scaled.reshape(args.batch_size, len(effects))
            log_probabilities = F.log_softmax(logits.float(), dim=1)
            policy_loss = -torch.logsumexp(
                log_probabilities.masked_fill(~correct, -torch.inf), dim=1
            ).mean()
            targets_scaled = target_moves / model.config.value_scale
            state_value_loss = F.smooth_l1_loss(
                scaled_values.float(), targets_scaled, beta=0.05
            )
            correct_child_values = child_scaled.float().masked_fill(
                ~correct, torch.nan
            )[correct]
            correct_child_targets = (
                (target_moves - 1).clamp_min(0) / model.config.value_scale
            )[:, None].expand_as(correct)[correct]
            child_value_loss = F.smooth_l1_loss(
                correct_child_values, correct_child_targets, beta=0.05
            )
            ranking_logits = (
                -child_scaled.float() * model.config.value_scale
                / args.ranking_temperature
            )
            ranking_log_probabilities = F.log_softmax(ranking_logits, dim=1)
            ranking_loss = -torch.logsumexp(
                ranking_log_probabilities.masked_fill(~correct, -torch.inf), dim=1
            ).mean()
            nonnegative = F.relu(-child_scaled.float()).mean()
            loss = (
                args.policy_weight * policy_loss
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
                order = torch.argsort(child_scaled.float(), dim=1)
                ranks = correct.gather(1, order).float().argmax(dim=1) + 1
            row = {
                "child_value_loss": float(child_value_loss.detach()),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "ranking_loss": float(ranking_loss.detach()),
                "state_value_loss": float(state_value_loss.detach()),
                "step": step,
                "train_value_top1": float((ranks == 1).float().mean()),
                "train_value_top8": float((ranks <= 8).float().mean()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "batch_size": args.batch_size,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "fold": args.fold,
        "folds": args.folds,
        "history": history,
        "init_checkpoint": str(args.init_checkpoint),
        "ranking_temperature": args.ranking_temperature,
        "steps": args.steps,
        "symmetry_count": args.symmetry_count,
        "train_pids": int(len(train_pids)),
    }
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_primitive_local_ranking_v1",
            "model_state_dict": model.state_dict(),
            "local_ranking_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
