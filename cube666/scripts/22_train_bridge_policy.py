"""Train a PID-held-out goal-conditioned cube666 local bridge policy."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.bridge import (  # noqa: E402
    BridgeTrajectoryDataset,
    OrbitBridgeConfig,
    OrbitBridgePolicyValueNet,
    bridge_policy_value_loss,
    checkpoint_manifest,
    global_to_local_lookup,
    manifest_digest,
    relative_orbit_states_torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=PROJECT / "cube666" / "training" / "bridge_v1" / "trajectories.npz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "models" / "cube666_bridge_v1",
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--maximum-horizon", type=int, default=32)
    parser.add_argument("--anchor-fraction", type=float, default=0.25)
    parser.add_argument("--orbit-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=0.2)
    parser.add_argument(
        "--random-walk-fraction",
        type=float,
        default=0.0,
        help="fraction of every batch generated as fresh non-backtracking local walks",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="optional compatible bridge checkpoint used to initialize the model",
    )
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-samples", type=int, default=8_192)
    parser.add_argument("--eval-batch-size", type=int, default=2_048)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def learning_rate_factor(step: int, *, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return max((step + 1) / warmup, 1e-3)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def sample_windows(
    dataset: BridgeTrajectoryDataset,
    eligible_indices: np.ndarray,
    sample_count: int,
    *,
    maximum_horizon: int,
    anchor_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = rng.integers(0, len(eligible_indices), size=sample_count)
    current_indices = eligible_indices[positions]
    maximum = np.minimum(
        dataset.remaining_lengths[current_indices].astype(np.int64),
        maximum_horizon,
    )
    if np.any(maximum < 1):
        raise AssertionError("eligible sample contains a terminal state")
    anchor = (rng.random(sample_count) < anchor_fraction) | (maximum < 4)
    horizons = np.empty(sample_count, dtype=np.int64)
    anchor_maximum = np.minimum(maximum[anchor], 3)
    horizons[anchor] = 1 + np.floor(rng.random(np.sum(anchor)) * anchor_maximum).astype(
        np.int64
    )
    far = ~anchor
    if np.any(far):
        far_choices = maximum[far] - 3
        horizons[far] = 4 + np.floor(rng.random(np.sum(far)) * far_choices).astype(
            np.int64
        )
    goal_indices = current_indices + horizons
    return (
        dataset.states[current_indices],
        dataset.states[goal_indices],
        dataset.next_actions[current_indices],
        horizons.astype(np.uint8),
    )


def sample_random_walks(
    sample_count: int,
    *,
    maximum_horizon: int,
    generators: torch.Tensor,
    inverse_actions: torch.Tensor,
    orbit_positions: torch.Tensor,
    local_lookup: torch.Tensor,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate fresh local problems with a known valid undo action."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    device = generators.device
    action_count, state_size = generators.shape
    depths = torch.randint(
        1,
        maximum_horizon + 1,
        (sample_count,),
        generator=rng,
        device=device,
    )
    states = torch.arange(state_size, device=device).expand(sample_count, -1).clone()
    previous = torch.full((sample_count,), -1, dtype=torch.long, device=device)
    labels = torch.empty(sample_count, dtype=torch.long, device=device)
    allowed = torch.empty(
        (action_count, action_count - 1),
        dtype=torch.long,
        device=device,
    )
    all_actions = torch.arange(action_count, device=device)
    for action in range(action_count):
        allowed[action] = all_actions[all_actions != inverse_actions[action]]
    for step in range(1, maximum_horizon + 1):
        if step == 1:
            actions = torch.randint(
                action_count,
                (sample_count,),
                generator=rng,
                device=device,
            )
        else:
            choices = torch.randint(
                action_count - 1,
                (sample_count,),
                generator=rng,
                device=device,
            )
            actions = allowed[previous, choices]
        active = depths >= step
        moved = states.gather(1, generators[actions])
        states = torch.where(active[:, None], moved, states)
        labels = torch.where(depths == step, inverse_actions[actions], labels)
        previous = actions
    local_states = local_lookup[states[:, orbit_positions].long()]
    return local_states, labels, depths


@torch.inference_mode()
def evaluate(
    model: OrbitBridgePolicyValueNet,
    samples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    orbit_positions: torch.Tensor,
    local_lookup: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    current, goals, actions, horizons = samples
    correct = {1: 0, 4: 0, 8: 0}
    value_error = 0.0
    probability_sum = 0.0
    seen = 0
    for start in range(0, len(current), batch_size):
        stop = min(start + batch_size, len(current))
        current_batch = torch.from_numpy(current[start:stop]).to(device)
        goal_batch = torch.from_numpy(goals[start:stop]).to(device)
        action_batch = torch.from_numpy(actions[start:stop]).long().to(device)
        local = relative_orbit_states_torch(
            current_batch,
            goal_batch,
            orbit_positions,
            local_lookup,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, values = model(local)
        logits = logits.float()
        proposed = logits.topk(8, dim=1).indices
        for k in correct:
            correct[k] += int(proposed[:, :k].eq(action_batch[:, None]).any(dim=1).sum())
        value_error += float(
            (values.float() - torch.from_numpy(horizons[start:stop]).to(device)).abs().sum()
        )
        probability_sum += float(
            torch.softmax(logits, dim=1).gather(1, action_batch[:, None]).sum()
        )
        seen += stop - start
    return {
        "policy_probability": probability_sum / seen,
        "recall_at_1": correct[1] / seen,
        "recall_at_4": correct[4] / seen,
        "recall_at_8": correct[8] / seen,
        "value_mae": value_error / seen,
    }


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.maximum_horizon <= 0:
        raise ValueError("steps, batch-size, and maximum-horizon must be positive")
    if not 0.0 <= args.anchor_fraction <= 1.0:
        raise ValueError("anchor-fraction must be between zero and one")
    if not 0.0 <= args.random_walk_fraction < 1.0:
        raise ValueError("random-walk-fraction must be in [0, 1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.compile and device.type != "cuda":
        raise ValueError("--compile requires CUDA")

    dataset = BridgeTrajectoryDataset.load(args.teacher)
    training_indices = dataset.indices_for_split(0)
    validation_indices = dataset.indices_for_split(1)
    if len(training_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("training and validation splits must both be nonempty")
    training_rng = np.random.default_rng(args.seed)
    evaluation_samples = sample_windows(
        dataset,
        validation_indices,
        args.eval_samples,
        maximum_horizon=args.maximum_horizon,
        anchor_fraction=args.anchor_fraction,
        rng=np.random.default_rng(args.seed + 1),
    )

    config = OrbitBridgeConfig(
        action_count=len(dataset.move_names),
        orbit_dim=args.orbit_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
        dropout=args.dropout,
        maximum_horizon=args.maximum_horizon,
    )
    model = OrbitBridgePolicyValueNet(config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        initial_manifest = initial["manifest"]
        if initial_manifest["kind"] != "cube666_goal_conditioned_bridge_v1":
            raise ValueError("initial checkpoint is not a bridge model")
        if initial_manifest["model_config"] != config.as_dict():
            raise ValueError("initial checkpoint model configuration differs")
        if initial_manifest["source_digest"] != dataset.source_digest:
            raise ValueError("initial checkpoint and teacher source digests differ")
        model.load_state_dict(initial["model_state_dict"])
    train_model = model
    if args.compile:
        train_model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_factor(step, warmup=args.warmup_steps, total=args.steps),
    )
    orbit_positions = torch.from_numpy(dataset.orbit_positions).long().to(device)
    local_lookup = torch.from_numpy(global_to_local_lookup(dataset.orbit_positions)).long().to(
        device
    )
    generators = torch.from_numpy(dataset.generator_permutations.astype(np.int64)).to(device)
    inverse_actions = torch.from_numpy(dataset.inverse_actions.astype(np.int64)).to(device)
    walk_rng = torch.Generator(device=device).manual_seed(args.seed + 2)

    started = time.perf_counter()
    last_losses: dict[str, float] = {}
    train_model.train()
    for step in range(1, args.steps + 1):
        random_walk_count = round(args.batch_size * args.random_walk_fraction)
        path_count = args.batch_size - random_walk_count
        relative_parts: list[torch.Tensor] = []
        action_parts: list[torch.Tensor] = []
        horizon_parts: list[torch.Tensor] = []
        if path_count:
            current, goals, actions, horizons = sample_windows(
                dataset,
                training_indices,
                path_count,
                maximum_horizon=args.maximum_horizon,
                anchor_fraction=args.anchor_fraction,
                rng=training_rng,
            )
            current_tensor = torch.from_numpy(current).to(device)
            goal_tensor = torch.from_numpy(goals).to(device)
            relative_parts.append(
                relative_orbit_states_torch(
                    current_tensor,
                    goal_tensor,
                    orbit_positions,
                    local_lookup,
                )
            )
            action_parts.append(torch.from_numpy(actions).long().to(device))
            horizon_parts.append(torch.from_numpy(horizons).to(device))
        if random_walk_count:
            walk_states, walk_actions, walk_horizons = sample_random_walks(
                random_walk_count,
                maximum_horizon=args.maximum_horizon,
                generators=generators,
                inverse_actions=inverse_actions,
                orbit_positions=orbit_positions,
                local_lookup=local_lookup,
                rng=walk_rng,
            )
            relative_parts.append(walk_states)
            action_parts.append(walk_actions)
            horizon_parts.append(walk_horizons)
        relative = torch.cat(relative_parts)
        action_tensor = torch.cat(action_parts)
        horizon_tensor = torch.cat(horizon_parts)
        batch_permutation = torch.randperm(args.batch_size, generator=walk_rng, device=device)
        relative = relative[batch_permutation]
        action_tensor = action_tensor[batch_permutation]
        horizon_tensor = horizon_tensor[batch_permutation]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = train_model(relative)
            loss, loss_parts = bridge_policy_value_loss(
                outputs,
                action_tensor,
                horizon_tensor,
                maximum_horizon=args.maximum_horizon,
                value_weight=args.value_weight,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_losses = {name: float(value) for name, value in loss_parts.items()}
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "step": step,
                        **last_losses,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    validation = evaluate(
        model,
        evaluation_samples,
        orbit_positions=orbit_positions,
        local_lookup=local_lookup,
        device=device,
        batch_size=args.eval_batch_size,
    )
    report: dict[str, object] = {
        "anchor_fraction": args.anchor_fraction,
        "batch_size": args.batch_size,
        "compiled": args.compile,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "last_train_losses": last_losses,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "model_config": config.as_dict(),
        "source_digest": dataset.source_digest,
        "random_walk_fraction": args.random_walk_fraction,
        "steps": args.steps,
        "training_paths": int(np.sum(dataset.path_splits == 0)),
        "training_states": len(training_indices),
        "validation": validation,
        "validation_paths": int(np.sum(dataset.path_splits == 1)),
        "validation_samples": args.eval_samples,
        "value_weight": args.value_weight,
    }
    manifest = checkpoint_manifest(config, dataset, report)
    checkpoint = {
        "manifest": manifest,
        "manifest_digest": manifest_digest(manifest),
        "model_state_dict": model.state_dict(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(checkpoint, args.out_dir / "checkpoint.pt")
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
