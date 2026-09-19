"""Train an autoregressive decoder over verified short primitive macros."""

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

from cube666.macro_autoregressive import (  # noqa: E402
    MacroAutoregressiveConfig,
    MacroAutoregressivePolicyValueNet,
    MacroFactorizedAutoregressiveConfig,
    MacroFactorizedAutoregressivePolicyValueNet,
)
from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--factorized-init-checkpoint", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--maximum-action-cost", type=int, default=14)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--token-embedding-dim", type=int, default=64)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=88666)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def build_action_tokens(
    library_path: Path,
    move_names: tuple[str, ...],
    maximum_cost: int,
) -> np.ndarray:
    paths = json.loads(library_path.read_text(encoding="utf-8"))["paths"]
    move_to_token = {name: index for index, name in enumerate(move_names)}
    tokens = np.full((len(paths), maximum_cost + 1), -100, dtype=np.int16)
    eos = len(move_names)
    for action, path in enumerate(paths):
        if len(path) > maximum_cost:
            continue
        encoded = [move_to_token[name] for name in path]
        tokens[action, : len(encoded)] = encoded
        tokens[action, len(encoded)] = eos
    return tokens


@torch.inference_mode()
def evaluate(
    model: MacroAutoregressivePolicyValueNet,
    states: np.ndarray,
    targets: np.ndarray,
    values: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    token_hits = 0
    token_top5_hits = 0
    token_total = 0
    sequence_hits = 0
    absolute_error = 0.0
    for start in range(0, len(states), batch_size):
        batch_states = torch.from_numpy(states[start : start + batch_size]).to(device)
        target = torch.from_numpy(targets[start : start + batch_size]).to(device).long()
        prefixes = torch.full_like(target, model.config.bos_token)
        prefixes[:, 1:] = target[:, :-1].clamp_min(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, scaled_values = model(batch_states, prefixes)
        valid = target >= 0
        prediction = logits.argmax(dim=2)
        token_hits += int((prediction.eq(target) & valid).sum())
        top5 = logits.topk(5, dim=2).indices
        token_top5_hits += int((top5.eq(target[:, :, None]) & valid[:, :, None]).any(dim=2).sum())
        token_total += int(valid.sum())
        sequence_hits += int(((prediction.eq(target) | ~valid).all(dim=1)).sum())
        predicted_values = scaled_values.float().cpu().numpy() * model.config.value_scale
        absolute_error += float(
            np.abs(predicted_values - values[start : start + len(batch_states)]).sum()
        )
    return {
        "samples": len(states),
        "sequence_exact": sequence_hits / max(len(states), 1),
        "token_accuracy": token_hits / max(token_total, 1),
        "token_top5": token_top5_hits / max(token_total, 1),
        "value_mae": absolute_error / max(len(states), 1),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0 <= args.high_value_fraction <= 1:
        raise ValueError("high-value-fraction must be in [0, 1]")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        all_states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        all_values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False)
    action_tokens = build_action_tokens(
        args.teacher_dir / "action_library.json",
        tuple(puzzle.move_names),
        args.maximum_action_cost,
    )
    chosen = np.full(len(labels), -1, dtype=np.int32)
    for row in range(len(labels)):
        valid = [
            int(action)
            for action in labels[row, : int(counts[row])]
            if action >= 0 and costs[int(action)] <= args.maximum_action_cost
        ]
        if valid:
            chosen[row] = valid[0]
    eligible = np.flatnonzero(chosen >= 0)
    states = all_states[eligible]
    targets = action_tokens[chosen[eligible]]
    values = all_values[eligible]
    eligible_groups = groups[eligible]
    if np.any(targets[:, 0] < 0):
        raise AssertionError("an eligible short action has no token sequence")
    train_indices = np.flatnonzero(np.mod(eligible_groups, args.folds) != args.fold)
    heldout_indices = np.flatnonzero(np.mod(eligible_groups, args.folds) == args.fold)
    if args.factorized_init_checkpoint is None:
        config = MacroAutoregressiveConfig(
            maximum_tokens=args.maximum_action_cost + 1,
            hidden_dim=args.hidden_dim,
            residual_blocks=args.residual_blocks,
            token_embedding_dim=args.token_embedding_dim,
            decoder_layers=args.decoder_layers,
        )
        model = MacroAutoregressivePolicyValueNet(config).to(device)
        architecture = "monolithic"
    else:
        value_model, _ = load_macro_factorized_value_checkpoint(
            args.factorized_init_checkpoint, device
        )
        config = MacroFactorizedAutoregressiveConfig(
            maximum_tokens=args.maximum_action_cost + 1,
            local_dim=value_model.config.local_dim,
            local_blocks=value_model.config.local_blocks,
            hidden_dim=value_model.config.global_dim,
            global_blocks=value_model.config.global_blocks,
            token_embedding_dim=args.token_embedding_dim,
            decoder_layers=args.decoder_layers,
        )
        model = MacroFactorizedAutoregressivePolicyValueNet(config).to(device)
        for name in (
            "local_stem",
            "local_blocks",
            "local_norm",
            "global_stem",
            "global_blocks",
            "global_norm",
        ):
            getattr(model, name).load_state_dict(getattr(value_model, name).state_dict())
        model.cluster_embedding.data.copy_(value_model.cluster_embedding.data)
        model.value_head.load_state_dict(value_model.total_head.state_dict())
        architecture = "factorized_value_initialized"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    training_values = values[train_indices]
    high_weights = np.sqrt(training_values / max(float(training_values.mean()), 1e-6))
    high_cdf = np.cumsum(high_weights, dtype=np.float64)
    high_cdf /= high_cdf[-1]
    high_count = int(round(args.batch_size * args.high_value_fraction))
    uniform_count = args.batch_size - high_count
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
        batch_states = torch.from_numpy(states[selected]).to(device)
        target = torch.from_numpy(targets[selected]).to(device).long()
        prefixes = torch.full_like(target, config.bos_token)
        prefixes[:, 1:] = target[:, :-1].clamp_min(0)
        value_targets = torch.from_numpy(values[selected]).to(device) / config.value_scale
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, scaled_values = train_model(batch_states, prefixes)
            token_loss = F.cross_entropy(
                logits.float().reshape(-1, config.vocabulary_size),
                target.reshape(-1),
                ignore_index=-100,
            )
            value_loss = F.smooth_l1_loss(
                scaled_values.float(), value_targets.float(), beta=0.05
            )
            loss = token_loss + args.value_weight * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "step": step,
                "token_loss": float(token_loss.detach()),
                "value_loss": float(value_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    eval_count = min(8192, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    heldout = evaluate(
        model,
        states[eval_indices],
        targets[eval_indices],
        values[eval_indices],
        args.eval_batch_size,
        device,
    )
    report = {
        "architecture": architecture,
        "eligible_samples": int(len(eligible)),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "heldout": heldout,
        "heldout_samples": int(len(heldout_indices)),
        "history": history,
        "maximum_action_cost": args.maximum_action_cost,
        "model_config": config.to_dict(),
        "steps": args.steps,
        "train_samples": int(len(train_indices)),
        "unique_target_actions": int(len(np.unique(chosen[eligible]))),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_macro_autoregressive_v1",
            **(
                {"model_config": config.to_dict()}
                if isinstance(config, MacroAutoregressiveConfig)
                else {"factorized_autoregressive_config": config.to_dict()}
            ),
            "model_state_dict": model.state_dict(),
            "action_tokens": action_tokens,
            "action_costs": costs,
            "action_library": str(args.teacher_dir / "action_library.json"),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
