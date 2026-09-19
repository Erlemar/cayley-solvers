"""Train the 6x6x6 bulk-macro shortlist policy on an exact teacher dataset."""

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
    validate_factorized_action_table,
)
from cube666.macro_policy import (  # noqa: E402
    FactorizedMacroPolicyConfig,
    FactorizedMacroPolicyValueNet,
    MacroPolicyConfig,
    MacroPolicyValueNet,
    macro_policy_loss,
    policy_recall_at_k,
)
from cube666.macros import (  # noqa: E402
    enumerate_basic_corner_fixing_commutators,
    enumerate_conjugated_macros,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument(
        "--teacher",
        type=Path,
        default=PROJECT / "cube666" / "training" / "macro_teacher_v1.npz",
    )
    parser.add_argument(
        "--action-library",
        type=Path,
        help="optional custom action_library.json; defaults to the 19,680 commutator table",
    )
    parser.add_argument(
        "--group-ids",
        type=Path,
        help="optional .npy group ID per sample for leakage-safe validation",
    )
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--validation-folds", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "models" / "cube666_macro_policy_v1",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="optional compatible checkpoint used to initialize model weights",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--architecture", choices=("direct", "factorized"), default="direct")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=0.2)
    parser.add_argument("--cluster-value-weight", type=float, default=0.2)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=256)
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


@torch.no_grad()
def evaluate(
    model: MacroPolicyValueNet,
    dataset: MacroTeacherDataset,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    recall_sums = {1: 0.0, 16: 0.0, 64: 0.0}
    total_value_error = 0.0
    seen = 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        states = torch.from_numpy(dataset.states[selected]).to(device)
        labels = torch.from_numpy(dataset.teacher_actions[selected]).to(device)
        counts = torch.from_numpy(dataset.teacher_action_counts[selected]).to(device)
        targets = torch.from_numpy(dataset.cluster_cost_targets[selected]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, value_prediction, _ = model(states)
        recalls = policy_recall_at_k(logits, labels, counts)
        count = len(selected)
        for k, value in recalls.items():
            recall_sums[k] += value * count
        target_total = targets.float().sum(dim=1)
        total_value_error += float((value_prediction.float() * 72.0 - target_total).abs().sum())
        seen += count
    return {
        "recall_at_1": recall_sums[1] / seen,
        "recall_at_16": recall_sums[16] / seen,
        "recall_at_64": recall_sums[64] / seen,
        "value_mae": total_value_error / seen,
    }


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")

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
    if args.architecture == "factorized":
        validate_factorized_action_table(table)

    split_description: dict[str, object]
    if args.group_ids:
        groups = np.load(args.group_ids, allow_pickle=False)
        if groups.shape != (dataset.sample_count,):
            raise ValueError("group-ids must contain one value per teacher sample")
        if args.validation_folds <= 1:
            raise ValueError("validation-folds must exceed one")
        fold = args.validation_fold % args.validation_folds
        validation_mask = np.mod(groups, args.validation_folds) == fold
        validation_indices = np.flatnonzero(validation_mask)
        training_indices = np.flatnonzero(~validation_mask)
        split_description = {
            "kind": "group_modulo",
            "validation_fold": fold,
            "validation_folds": args.validation_folds,
            "validation_groups": sorted({int(value) for value in groups[validation_mask]}),
        }
    else:
        permutation = np.random.default_rng(args.seed).permutation(dataset.sample_count)
        validation_size = max(1, round(dataset.sample_count * args.validation_fraction))
        validation_indices = permutation[:validation_size]
        training_indices = permutation[validation_size:]
        split_description = {"kind": "random_samples"}
    if len(training_indices) == 0:
        raise ValueError("teacher dataset is too small for the requested validation split")
    if len(validation_indices) == 0:
        raise ValueError("validation split is empty")

    config_class = (
        FactorizedMacroPolicyConfig
        if args.architecture == "factorized"
        else MacroPolicyConfig
    )
    config = config_class(
        action_count=table.action_count,
        hidden_dim=args.hidden_dim,
        residual_blocks=args.residual_blocks,
        dropout=args.dropout,
    )
    model_class = (
        FactorizedMacroPolicyValueNet
        if args.architecture == "factorized"
        else MacroPolicyValueNet
    )
    model = model_class(config).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if initial["action_digest"] != table.digest:
            raise ValueError("initial checkpoint action digest does not match the action table")
        if initial["model_config"] != config.as_dict():
            raise ValueError("initial checkpoint model configuration does not match")
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

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = train_model(states)
            loss, loss_parts = macro_policy_loss(
                outputs,
                labels,
                counts,
                targets,
                value_weight=args.value_weight,
                cluster_value_weight=args.cluster_value_weight,
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

    metrics = evaluate(
        model,
        dataset,
        validation_indices,
        device=device,
        batch_size=args.eval_batch_size,
    )
    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "architecture": args.architecture,
        "batch_size": args.batch_size,
        "compiled": args.compile,
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "last_train_losses": last_losses,
        "model_config": config.as_dict(),
        "steps": args.steps,
        "value_weight": args.value_weight,
        "cluster_value_weight": args.cluster_value_weight,
        "split": split_description,
        "teacher_samples": dataset.sample_count,
        "training_samples": len(training_indices),
        "validation": metrics,
        "validation_samples": len(validation_indices),
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
    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
