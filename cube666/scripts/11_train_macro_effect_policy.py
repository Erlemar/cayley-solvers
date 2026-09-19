"""Train the compositional 6x6x6 macro-effect shortlist policy."""

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

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    MacroActionTable,
    MacroTeacherDataset,
    load_macro_action_library,
)
from cube666.macro_policy import (  # noqa: E402
    MacroEffectPolicyConfig,
    MacroEffectPolicyValueNet,
    macro_effect_policy_loss,
)
from cube666.macros import (  # noqa: E402
    enumerate_basic_corner_fixing_commutators,
    enumerate_conjugated_macros,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--action-library", type=Path)
    parser.add_argument(
        "--group-ids",
        type=Path,
        help="optional source PID per sample for leakage-safe validation exclusion",
    )
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--validation-folds", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="optional compatible macro-effect checkpoint used to initialize weights",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--negative-actions", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
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


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.negative_actions <= 0:
        raise ValueError("steps, batch-size, and negative-actions must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    if args.action_library:
        _, table = load_macro_action_library(
            args.action_library,
            puzzle.generators,
            decomposition,
        )
    else:
        macros = enumerate_conjugated_macros(
            enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition),
            puzzle.generators,
            decomposition,
            max_conjugator_depth=1,
        )
        table = MacroActionTable.from_macros(macros)
    dataset = MacroTeacherDataset.load(args.teacher)
    dataset.validate(table.action_count)
    if dataset.action_digest != table.digest:
        raise ValueError("teacher action digest does not match the current macro enumeration")
    if args.group_ids:
        groups = np.load(args.group_ids, allow_pickle=False)
        if groups.shape != (dataset.sample_count,):
            raise ValueError("group-ids must contain one value per teacher sample")
        if args.validation_folds <= 1:
            raise ValueError("validation-folds must exceed one")
        validation_fold = args.validation_fold % args.validation_folds
        training_indices = np.flatnonzero(
            np.mod(groups, args.validation_folds) != validation_fold
        )
        split = {
            "kind": "group_modulo_exclusion",
            "validation_fold": validation_fold,
            "validation_folds": args.validation_folds,
        }
    else:
        training_indices = np.arange(dataset.sample_count)
        split = {"kind": "all_samples"}
    if len(training_indices) == 0:
        raise ValueError("training split is empty")

    config = MacroEffectPolicyConfig(
        action_count=table.action_count,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
        dropout=args.dropout,
    )
    model = MacroEffectPolicyValueNet(config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        initial_config = dict(initial["model_config"])
        if initial_config.pop("architecture", None) != "effect":
            raise ValueError("initial checkpoint is not a macro-effect model")
        initial_config.pop("action_count", None)
        target_config = config.as_dict()
        target_config.pop("architecture", None)
        target_config.pop("action_count", None)
        if initial_config != target_config:
            raise ValueError("initial macro-effect architecture differs")
        model.load_state_dict(initial["model_state_dict"])
    train_model = model
    if args.compile:
        if device.type != "cuda":
            raise ValueError("--compile requires CUDA for this training recipe")
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
    action_effects = torch.from_numpy(table.effects).to(device)
    sample_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    started = time.perf_counter()
    last_losses: dict[str, float] = {}
    train_model.train()
    for step in range(1, args.steps + 1):
        positions = torch.randint(
            len(training_indices),
            (args.batch_size,),
            generator=sample_generator,
        ).numpy()
        selected = training_indices[positions]
        states = torch.from_numpy(dataset.states[selected]).to(device)
        labels = torch.from_numpy(dataset.teacher_actions[selected]).to(device)
        counts = torch.from_numpy(dataset.teacher_action_counts[selected]).to(device)
        targets = torch.from_numpy(dataset.cluster_cost_targets[selected]).to(device)
        negatives = torch.randint(
            table.action_count,
            (args.batch_size, args.negative_actions),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = train_model(states)
            loss, loss_parts = macro_effect_policy_loss(
                outputs,
                labels,
                counts,
                targets,
                action_effects,
                negatives,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "architecture": "effect",
        "batch_size": args.batch_size,
        "compiled": args.compile,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "last_train_losses": last_losses,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "model_config": config.as_dict(),
        "negative_actions": args.negative_actions,
        "split": split,
        "steps": args.steps,
        "teacher_samples": dataset.sample_count,
        "training_samples": len(training_indices),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "action_digest": table.digest,
            "model_config": config.as_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
