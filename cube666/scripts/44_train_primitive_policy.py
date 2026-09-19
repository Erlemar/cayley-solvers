"""Distill exact trajectory memory into a full-state primitive policy/value net."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "cube_nnn" / "src"))

from cube666.primitive_policy import (  # noqa: E402
    PrimitivePolicyConfig,
    PrimitivePolicyValueNetwork,
)
from cube666.classical import position_orbits  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.symmetry import build_symmetries  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-model",
        type=Path,
        default=PROJECT / "models/cube666_trajectory_memory_portfolio_v1/model.npz",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--eval-batch-size", type=int, default=8_192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--value-loss-weight", type=float, default=0.5)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument(
        "--encoding",
        choices=("piece_embedding", "orbit_one_hot"),
        default="orbit_one_hot",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--symmetry-augmentation", action="store_true")
    parser.add_argument("--symmetry-count", type=int, default=48)
    parser.add_argument("--perturbation-fraction", type=float, default=0.0)
    parser.add_argument("--maximum-perturbation-depth", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise ValueError("symmetry-count must be in 1..48 for the verified frame table")
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


def local_generator_effects(
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


def perturb_training_batch(
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    effects: torch.Tensor,
    *,
    fraction: float,
    maximum_depth: int,
    value_scale: float,
) -> None:
    count = int(round(len(states) * fraction))
    if count == 0:
        return
    lengths = torch.randint(1, maximum_depth + 1, (count,), device=states.device)
    walked = states[:count]
    last = torch.full((count,), -1, dtype=torch.long, device=states.device)
    for step in range(maximum_depth):
        proposed = torch.randint(0, len(effects), (count,), device=states.device)
        if step:
            immediate_backtrack = proposed.eq(torch.bitwise_xor(last, 1))
            proposed = torch.where(
                immediate_backtrack,
                torch.remainder(proposed + 2, len(effects)),
                proposed,
            )
        active = step < lengths
        moved = walked.gather(-1, effects[proposed].long())
        walked = torch.where(active[:, None, None], moved, walked)
        last = torch.where(active, proposed, last)
    states[:count] = walked
    actions[:count] = False
    actions[:count].scatter_(1, torch.bitwise_xor(last, 1)[:, None], True)
    values[:count] += lengths.to(values.dtype) / value_scale


@torch.inference_mode()
def evaluate(
    model: PrimitivePolicyValueNetwork,
    states: np.ndarray,
    distances: np.ndarray,
    optimal_actions: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    value_scale = model.config.value_scale
    top_hits = {1: 0, 4: 0, 8: 0}
    total_policy = 0
    absolute_error = 0.0
    squared_error = 0.0
    total_value = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch_states = torch.from_numpy(states[selected]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, scaled_value = model(batch_states)
        prediction = scaled_value.float().cpu().numpy() * value_scale
        target_value = distances[selected].astype(np.float32)
        error = prediction - target_value
        absolute_error += float(np.abs(error).sum())
        squared_error += float(np.square(error).sum())
        total_value += len(selected)

        valid = distances[selected] > 0
        if not np.any(valid):
            continue
        policy_logits = logits[torch.from_numpy(valid).to(device)]
        labels = optimal_actions[selected][valid]
        ranking = torch.topk(policy_logits.float(), k=8, dim=1).indices.cpu().numpy()
        for width in top_hits:
            top_hits[width] += int(
                np.any(np.take_along_axis(labels, ranking[:, :width], axis=1), axis=1).sum()
            )
        total_policy += int(valid.sum())
    return {
        "nodes": int(len(indices)),
        "policy_nodes": total_policy,
        "top1_optimal": top_hits[1] / max(total_policy, 1),
        "top4_optimal": top_hits[4] / max(total_policy, 1),
        "top8_optimal": top_hits[8] / max(total_policy, 1),
        "value_mae": absolute_error / max(total_value, 1),
        "value_rmse": math.sqrt(squared_error / max(total_value, 1)),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.fold < args.folds:
        raise ValueError("fold must satisfy 0 <= fold < folds")
    if not 0.0 <= args.perturbation_fraction <= 1.0:
        raise ValueError("perturbation-fraction must be in [0, 1]")
    if args.maximum_perturbation_depth <= 0:
        raise ValueError("maximum-perturbation-depth must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for this training recipe")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    corpus = np.load(args.memory_model)
    raw_states = corpus["states"]
    pids = corpus["pids"].astype(np.int64)
    distances = corpus["distances"].astype(np.float32)
    masks = corpus["action_masks"]
    offsets = corpus["offsets"].astype(np.int64)
    config = PrimitivePolicyConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
        expansion=args.expansion,
        encoding=args.encoding,
    )
    orbit_positions: np.ndarray | None = None
    global_to_local: np.ndarray | None = None
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    if args.encoding == "orbit_one_hot":
        orbit_positions = np.asarray(position_orbits(puzzle.generators), dtype=np.int64)
        global_to_local = np.empty(config.state_size, dtype=np.uint8)
        for orbit in orbit_positions:
            global_to_local[orbit] = np.arange(config.orbit_size, dtype=np.uint8)
        states = global_to_local[raw_states[:, orbit_positions]]
    else:
        states = raw_states
    optimal_actions = action_targets(masks, config.action_count)
    if args.encoding == "orbit_one_hot":
        if orbit_positions is None or global_to_local is None:
            raise AssertionError("orbit metadata missing")
        training_effects = torch.from_numpy(
            local_generator_effects(puzzle, orbit_positions, global_to_local)
        ).to(device)
    else:
        training_effects = None
    if args.symmetry_augmentation:
        sym_permutations, sym_inverse_positions, sym_action_maps = symmetry_tables(
            args.data_dir,
            tuple(puzzle.move_names),
            args.symmetry_count,
        )
    else:
        sym_permutations = sym_inverse_positions = sym_action_maps = None
    train_pids = np.asarray(
        [pid for pid in range(len(offsets) - 1) if pid % args.folds != args.fold],
        dtype=np.int64,
    )
    validation_pids = np.asarray(
        [pid for pid in range(len(offsets) - 1) if pid % args.folds == args.fold],
        dtype=np.int64,
    )
    train_indices = np.flatnonzero((pids % args.folds) != args.fold)
    validation_indices = np.flatnonzero((pids % args.folds) == args.fold)
    train_starts = offsets[train_pids]
    train_sizes = offsets[train_pids + 1] - train_starts

    model = PrimitivePolicyValueNetwork(config).to(device)
    initialization = "random"
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if initial["model_config"] != config.to_dict():
            raise ValueError("init checkpoint architecture differs from requested config")
        model.load_state_dict(initial["model_state_dict"])
        initialization = str(args.init_checkpoint)
    train_model: torch.nn.Module = model
    if args.compile:
        train_model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    generator = np.random.default_rng(args.seed)
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        selected_pid_rows = generator.integers(0, len(train_pids), size=args.batch_size)
        selected = train_starts[selected_pid_rows] + (
            generator.random(args.batch_size) * train_sizes[selected_pid_rows]
        ).astype(np.int64)
        if args.symmetry_augmentation:
            if (
                sym_permutations is None
                or sym_inverse_positions is None
                or sym_action_maps is None
                or orbit_positions is None
                or global_to_local is None
            ):
                raise AssertionError("symmetry augmentation requires orbit-one-hot encoding")
            frame_indices = generator.integers(
                0, len(sym_permutations), size=args.batch_size
            )
            augmented_raw, augmented_actions = augment_batch(
                raw_states,
                optimal_actions,
                selected,
                frame_indices,
                sym_permutations,
                sym_inverse_positions,
                sym_action_maps,
            )
            augmented_states = global_to_local[augmented_raw[:, orbit_positions]]
            batch_states = torch.from_numpy(augmented_states).to(device)
            batch_actions = torch.from_numpy(augmented_actions).to(device)
        else:
            batch_states = torch.from_numpy(states[selected]).to(device)
            batch_actions = torch.from_numpy(optimal_actions[selected]).to(device)
        batch_values = torch.from_numpy(distances[selected]).to(device) / config.value_scale
        if args.perturbation_fraction:
            if training_effects is None:
                raise ValueError("perturbations require orbit-one-hot encoding")
            perturb_training_batch(
                batch_states,
                batch_actions,
                batch_values,
                training_effects,
                fraction=args.perturbation_fraction,
                maximum_depth=args.maximum_perturbation_depth,
                value_scale=config.value_scale,
            )

        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, scaled_value = train_model(batch_states)
            log_probabilities = F.log_softmax(logits.float(), dim=1)
            selected_log_probabilities = log_probabilities.masked_fill(
                ~batch_actions, float("-inf")
            )
            policy_rows = batch_actions.any(dim=1)
            policy_loss = -torch.logsumexp(
                selected_log_probabilities[policy_rows], dim=1
            ).mean()
            value_loss = F.smooth_l1_loss(
                scaled_value.float(), batch_values.float(), beta=0.1
            )
            loss = policy_loss + args.value_loss_weight * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    metrics = evaluate(
        model,
        states,
        distances,
        optimal_actions,
        validation_indices,
        args.eval_batch_size,
        device,
    )
    train_metrics = evaluate(
        model,
        states,
        distances,
        optimal_actions,
        train_indices,
        args.eval_batch_size,
        device,
    )
    checkpoint = {
        "kind": "cube666_primitive_policy_value_v1",
        "model_config": config.to_dict(),
        "model_state_dict": model.state_dict(),
        "fold": args.fold,
        "folds": args.folds,
        "seed": args.seed,
        "symmetry_augmentation": args.symmetry_augmentation,
        "symmetry_count": args.symmetry_count if args.symmetry_augmentation else 1,
        "memory_model": str(args.memory_model),
        "memory_sha256": sha256_file(args.memory_model),
        "initialization": initialization,
        "perturbation_fraction": args.perturbation_fraction,
        "maximum_perturbation_depth": args.maximum_perturbation_depth,
        "steps": args.steps,
        "orbit_positions": orbit_positions,
        "global_to_local": global_to_local,
    }
    checkpoint_path = args.out_dir / "checkpoint.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    report = {
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
        "fold": args.fold,
        "folds": args.folds,
        "history": history,
        "model_config": config.to_dict(),
        "initialization": initialization,
        "perturbation_fraction": args.perturbation_fraction,
        "maximum_perturbation_depth": args.maximum_perturbation_depth,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "symmetry_augmentation": args.symmetry_augmentation,
        "symmetry_count": args.symmetry_count if args.symmetry_augmentation else 1,
        "train_metrics": train_metrics,
        "train_nodes": int(len(train_indices)),
        "train_pids": int(len(train_pids)),
        "validation_metrics": metrics,
        "validation_nodes": int(len(validation_indices)),
        "validation_pids": int(len(validation_pids)),
    }
    (args.out_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
