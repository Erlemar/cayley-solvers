"""Train a primitive policy/value model conditioned on recent path history."""

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

from cube666.primitive_history_policy import (  # noqa: E402
    PrimitiveHistoryPolicyConfig,
    PrimitiveHistoryPolicyValueNetwork,
)
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
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--history-embedding-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--symmetry-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=85666)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def action_targets(masks: np.ndarray, action_count: int) -> np.ndarray:
    bits = np.arange(action_count, dtype=np.uint64)
    return ((masks[:, None] >> bits[None, :]) & np.uint64(1)).astype(np.bool_)


def build_histories(
    raw_states: np.ndarray,
    offsets: np.ndarray,
    masks: np.ndarray,
    generators: np.ndarray,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    pad = len(generators)
    histories = np.full((len(raw_states), history_length), pad, dtype=np.int16)
    selected_actions = np.full(len(raw_states), pad, dtype=np.int16)
    discontinuities = 0
    for pid in range(len(offsets) - 1):
        start, end = int(offsets[pid]), int(offsets[pid + 1])
        recent: list[int] = []
        for index in range(start, end):
            previous = np.asarray(recent[-history_length:], dtype=np.int16)
            histories[index, history_length - len(previous) :] = previous
            if index == end - 1:
                continue
            candidates = [
                action
                for action in range(pad)
                if int(masks[index]) & (1 << action)
            ]
            matches = [
                action
                for action in candidates
                if np.array_equal(raw_states[index][generators[action]], raw_states[index + 1])
            ]
            if matches:
                selected_actions[index] = matches[0]
                recent.append(matches[0])
            else:
                # The portfolio memory can switch between equally short source
                # trajectories at a shared distance.  No fictitious history is
                # carried across such a splice.
                discontinuities += 1
                recent.clear()
    return histories, selected_actions, discontinuities


def symmetry_tables(
    data_dir: Path,
    move_names: tuple[str, ...],
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = build_symmetries(NCube.from_puzzle_info(data_dir / "puzzle_info.json"))
    if len(table) != 48 or not 1 <= count <= 48:
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
    targets: np.ndarray,
    histories: np.ndarray,
    selected: np.ndarray,
    frames: np.ndarray,
    permutations: np.ndarray,
    inverse_positions: np.ndarray,
    action_maps: np.ndarray,
    pad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reordered = np.take_along_axis(
        raw_states[selected], inverse_positions[frames], axis=1
    )
    augmented_states = np.take_along_axis(
        permutations[frames], reordered.astype(np.int64, copy=False), axis=1
    )
    augmented_targets = np.zeros_like(targets[selected])
    np.put_along_axis(
        augmented_targets,
        action_maps[frames],
        targets[selected],
        axis=1,
    )
    original_history = histories[selected]
    clipped = np.minimum(original_history, pad - 1)
    augmented_history = np.take_along_axis(action_maps[frames], clipped, axis=1)
    augmented_history = np.where(original_history == pad, pad, augmented_history)
    return augmented_states, augmented_targets, augmented_history.astype(np.int64)


@torch.inference_mode()
def evaluate(
    model: PrimitiveHistoryPolicyValueNetwork,
    states: np.ndarray,
    histories: np.ndarray,
    targets: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    ranks: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    eval_distances: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch_states = torch.from_numpy(states[selected]).to(device)
        batch_histories = torch.from_numpy(histories[selected].astype(np.int64)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, scaled = model(batch_states, batch_histories)
        order = torch.argsort(logits.float(), dim=1, descending=True)
        correct = torch.from_numpy(targets[selected]).to(device)
        ranks.append((correct.gather(1, order).float().argmax(dim=1) + 1).cpu().numpy())
        predictions.append((scaled.float() * model.config.value_scale).cpu().numpy())
        eval_distances.append(distances[selected])
    rank = np.concatenate(ranks)
    prediction = np.concatenate(predictions)
    distance = np.concatenate(eval_distances)

    def metrics(selected: np.ndarray) -> dict[str, float | int]:
        return {
            "count": int(selected.sum()),
            "policy_mean_rank": float(rank[selected].mean()),
            "policy_top1": float((rank[selected] <= 1).mean()),
            "policy_top4": float((rank[selected] <= 4).mean()),
            "policy_top8": float((rank[selected] <= 8).mean()),
            "policy_top16": float((rank[selected] <= 16).mean()),
            "value_mae": float(np.abs(prediction[selected] - distance[selected]).mean()),
        }

    report: dict[str, object] = {"overall": metrics(np.ones(len(rank), dtype=bool)), "bins": {}}
    for lower, upper in ((1, 20), (21, 40), (41, 80), (81, 120), (121, 160), (161, 10_000)):
        selected = (distance >= lower) & (distance <= upper)
        if selected.any():
            report["bins"][f"{lower}-{upper}"] = metrics(selected)
    return report


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source_model, source_checkpoint = load_primitive_checkpoint(
        args.init_checkpoint, device
    )
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    generators = np.asarray(
        [puzzle.generators[name] for name in puzzle.move_names], dtype=np.int64
    )
    orbit_positions = np.asarray(source_checkpoint["orbit_positions"], dtype=np.int64)
    global_to_local = np.asarray(source_checkpoint["global_to_local"], dtype=np.uint8)
    with np.load(args.memory_model, allow_pickle=False) as memory:
        raw_states = memory["states"].astype(np.uint8, copy=False)
        pids = memory["pids"].astype(np.int64, copy=False)
        distances = memory["distances"].astype(np.float32, copy=False)
        masks = memory["action_masks"].astype(np.uint64, copy=False)
        offsets = memory["offsets"].astype(np.int64, copy=False)
    targets = action_targets(masks, len(generators))
    histories, selected_actions, discontinuities = build_histories(
        raw_states, offsets, masks, generators, args.history_length
    )
    nonterminal = distances > 0
    states = global_to_local[raw_states[:, orbit_positions]]
    permutations, inverse_positions, action_maps = symmetry_tables(
        args.data_dir, tuple(puzzle.move_names), args.symmetry_count
    )
    train_pids = np.asarray(
        [pid for pid in range(len(offsets) - 1) if pid % args.folds != args.fold],
        dtype=np.int64,
    )
    train_starts = offsets[train_pids]
    train_sizes = offsets[train_pids + 1] - train_starts - 1
    validation_indices = np.flatnonzero(
        (pids % args.folds == args.fold) & nonterminal
    )
    config = PrimitiveHistoryPolicyConfig(
        history_length=args.history_length,
        history_embedding_dim=args.history_embedding_dim,
        hidden_dim=source_model.config.hidden_dim,
        residual_blocks=source_model.config.residual_blocks,
        expansion=source_model.config.expansion,
        value_scale=source_model.config.value_scale,
    )
    model = PrimitiveHistoryPolicyValueNetwork(config).to(device)
    model.initialize_state_path(source_model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    started = time.perf_counter()
    history_rows: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        pid_rows = rng.integers(0, len(train_pids), size=args.batch_size)
        selected = train_starts[pid_rows] + (
            rng.random(args.batch_size) * train_sizes[pid_rows]
        ).astype(np.int64)
        frames = rng.integers(0, args.symmetry_count, size=args.batch_size)
        augmented_raw, augmented_targets, augmented_histories = augment_batch(
            raw_states,
            targets,
            histories,
            selected,
            frames,
            permutations,
            inverse_positions,
            action_maps,
            len(generators),
        )
        batch_states = torch.from_numpy(
            global_to_local[augmented_raw[:, orbit_positions]]
        ).to(device)
        batch_histories = torch.from_numpy(augmented_histories).to(device)
        batch_targets = torch.from_numpy(augmented_targets).to(device)
        batch_values = torch.from_numpy(distances[selected]).to(device) / config.value_scale
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, scaled_values = train_model(batch_states, batch_histories)
            log_probabilities = F.log_softmax(logits.float(), dim=1)
            policy_loss = -torch.logsumexp(
                log_probabilities.masked_fill(~batch_targets, -torch.inf), dim=1
            ).mean()
            value_loss = F.smooth_l1_loss(
                scaled_values.float(), batch_values.float(), beta=0.1
            )
            loss = policy_loss + args.value_weight * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "step": step,
                "value_loss": float(value_loss.detach()),
            }
            history_rows.append(row)
            print(json.dumps(row), flush=True)
    heldout = evaluate(
        model,
        states,
        histories,
        targets,
        distances,
        validation_indices,
        args.eval_batch_size,
        device,
    )
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "heldout": heldout,
        "history": history_rows,
        "history_length": args.history_length,
        "history_discontinuities": discontinuities,
        "init_checkpoint": str(args.init_checkpoint),
        "steps": args.steps,
        "train_pids": int(len(train_pids)),
        "validation_nodes": int(len(validation_indices)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_primitive_history_policy_v1",
            "history_model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "orbit_positions": orbit_positions,
            "global_to_local": global_to_local,
            "fold": args.fold,
            "folds": args.folds,
            "history_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
