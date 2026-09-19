"""Fine-tune frozen value features on grouped off-policy KMC completion cost."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_factorized_value import (  # noqa: E402
    load_macro_factorized_value_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--rank-weight", type=float, default=4.0)
    parser.add_argument("--ranking-temperature", type=float, default=8.0)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--validation-pids", default="876,903")
    parser.add_argument("--seed", type=int, default=114)
    return parser.parse_args()


def predict_moves(model: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
    total, clusters = model(states)
    return 0.5 * (
        total.float() * model.config.total_scale
        + clusters.float().sum(dim=1) * model.config.cluster_scale
    )


def grouped_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    costs: torch.Tensor,
    pids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses = []
    for pid in pids.unique():
        group = torch.nonzero(pids == pid, as_tuple=False).flatten()
        actual_q = targets[group] + costs[group]
        predicted_q = predictions[group] + costs[group]
        target_distribution = torch.softmax(-actual_q / temperature, dim=0)
        losses.append(
            -(target_distribution * torch.log_softmax(-predicted_q / temperature, dim=0)).sum()
        )
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
    costs: torch.Tensor,
    pids: torch.Tensor,
    indices: torch.Tensor,
) -> dict[str, object]:
    if not len(indices):
        return {
            "mae": None,
            "mean_pid_correlation": None,
            "pids": [],
            "selected_regret": None,
            "winner_hits": None,
        }
    predictions = predict_moves(model, states[indices]).cpu().numpy()
    actual = targets[indices].cpu().numpy()
    action_costs = costs[indices].cpu().numpy()
    group_ids = pids[indices].cpu().numpy()
    rows = []
    correlations = []
    selected_regret = 0.0
    winner_hits = 0
    for pid in sorted(np.unique(group_ids)):
        group = np.flatnonzero(group_ids == pid)
        actual_q = actual[group] + action_costs[group]
        predicted_q = predictions[group] + action_costs[group]
        selected = int(np.argmin(predicted_q))
        oracle = float(actual_q.min())
        regret = float(actual_q[selected] - oracle)
        correlation = (
            float(np.corrcoef(predicted_q, actual_q)[0, 1])
            if np.std(predicted_q) > 1e-8 and np.std(actual_q) > 1e-8
            else math.nan
        )
        if math.isfinite(correlation):
            correlations.append(correlation)
        selected_regret += regret
        winner_hits += int(regret == 0)
        rows.append(
            {
                "correlation": correlation,
                "oracle_q": oracle,
                "pid": int(pid),
                "regret": regret,
                "selected_q": float(actual_q[selected]),
            }
        )
    return {
        "mae": float(np.mean(np.abs(predictions - actual))),
        "mean_pid_correlation": float(np.mean(correlations)) if correlations else None,
        "pids": rows,
        "selected_regret": selected_regret,
        "winner_hits": winner_hits,
    }


def configure_trainable(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = (model.cluster_head, model.global_norm, model.total_head)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def fit(
    *,
    base_model: torch.nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
    costs: torch.Tensor,
    pids: torch.Tensor,
    train_indices: torch.Tensor,
    steps: int,
    learning_rate: float,
    rank_weight: float,
    temperature: float,
    eval_every: int,
    validation_indices: torch.Tensor | None,
) -> tuple[torch.nn.Module, int, list[dict[str, object]]]:
    model = copy.deepcopy(base_model).train()
    parameters = configure_trainable(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, object]] = []
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_step = steps
    for step in range(1, steps + 1):
        prediction = predict_moves(model, states[train_indices])
        train_targets = targets[train_indices]
        train_costs = costs[train_indices]
        train_pids = pids[train_indices]
        regression = F.smooth_l1_loss(prediction, train_targets, beta=8.0)
        ranking = grouped_ranking_loss(
            prediction,
            train_targets,
            train_costs,
            train_pids,
            temperature,
        )
        loss = regression + rank_weight * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if validation_indices is not None and (
            step == 1 or step % eval_every == 0 or step == steps
        ):
            model.eval()
            validation = evaluate(
                model, states, targets, costs, pids, validation_indices
            )
            key = (
                float(validation["selected_regret"]),
                float(validation["mae"]),
                step,
            )
            history.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "ranking_loss": float(ranking.detach().cpu()),
                    "regression_loss": float(regression.detach().cpu()),
                    "step": step,
                    "validation": validation,
                }
            )
            if best_key is None or key < best_key:
                best_key = key
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    if validation_indices is not None:
        if best_state is None:
            raise AssertionError("validation never produced a checkpoint")
        model.load_state_dict(best_state)
    model.eval()
    return model, best_step, history


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.steps, args.eval_every, args.holdout_modulus) <= 0:
        raise ValueError("steps, eval-every, and holdout-modulus must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    base_model, payload = load_macro_factorized_value_checkpoint(
        args.checkpoint, device
    )
    with np.load(args.dataset / "completion.npz", allow_pickle=False) as dataset:
        states = torch.from_numpy(dataset["states"].astype(np.uint8, copy=False)).to(device)
        targets = torch.from_numpy(dataset["targets"].astype(np.float32, copy=False)).to(device)
        costs = torch.from_numpy(dataset["action_costs"].astype(np.float32, copy=False)).to(device)
        pids = torch.from_numpy(dataset["pids"].astype(np.int64, copy=False)).to(device)
    holdout_mask = pids.remainder(args.holdout_modulus) == args.holdout_remainder
    validation_pids = {
        int(token)
        for token in args.validation_pids.replace("_", ",").split(",")
        if token.strip()
    }
    validation_mask = torch.zeros_like(holdout_mask)
    for pid in validation_pids:
        validation_mask |= pids == pid
    if torch.any(validation_mask & holdout_mask):
        raise ValueError("validation and holdout PID sets overlap")
    tuning_train = torch.nonzero(
        ~holdout_mask & ~validation_mask, as_tuple=False
    ).flatten()
    validation = torch.nonzero(validation_mask, as_tuple=False).flatten()
    holdout = torch.nonzero(holdout_mask, as_tuple=False).flatten()
    production_train = torch.nonzero(~holdout_mask, as_tuple=False).flatten()
    if min(len(tuning_train), len(validation), len(holdout)) == 0:
        raise ValueError("train, validation, and holdout splits must all be non-empty")

    base_metrics = {
        "holdout": evaluate(base_model, states, targets, costs, pids, holdout),
        "validation": evaluate(base_model, states, targets, costs, pids, validation),
    }
    _, best_step, tuning_history = fit(
        base_model=base_model,
        states=states,
        targets=targets,
        costs=costs,
        pids=pids,
        train_indices=tuning_train,
        steps=args.steps,
        learning_rate=args.learning_rate,
        rank_weight=args.rank_weight,
        temperature=args.ranking_temperature,
        eval_every=args.eval_every,
        validation_indices=validation,
    )
    model, _, _ = fit(
        base_model=base_model,
        states=states,
        targets=targets,
        costs=costs,
        pids=pids,
        train_indices=production_train,
        steps=best_step,
        learning_rate=args.learning_rate,
        rank_weight=args.rank_weight,
        temperature=args.ranking_temperature,
        eval_every=args.eval_every,
        validation_indices=None,
    )
    final_metrics = {
        "holdout": evaluate(model, states, targets, costs, pids, holdout),
        "production_train": evaluate(
            model, states, targets, costs, pids, production_train
        ),
        "validation": evaluate(model, states, targets, costs, pids, validation),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = dict(payload)
    checkpoint_payload["model_state_dict"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    checkpoint_payload["completion_finetune"] = {
        "best_step": best_step,
        "dataset": str(args.dataset),
        "holdout_modulus": args.holdout_modulus,
        "holdout_remainder": args.holdout_remainder,
        "learning_rate": args.learning_rate,
        "rank_weight": args.rank_weight,
        "ranking_temperature": args.ranking_temperature,
        "validation_pids": sorted(validation_pids),
    }
    torch.save(checkpoint_payload, args.out_dir / "checkpoint.pt")
    report = {
        "base": base_metrics,
        "best_step": best_step,
        "checkpoint": str(args.out_dir / "checkpoint.pt"),
        "dataset": str(args.dataset),
        "final": final_metrics,
        "holdout_pids": sorted(int(pid) for pid in pids[holdout].unique().cpu()),
        "tuning_history": tuning_history,
        "tuning_train_pids": sorted(
            int(pid) for pid in pids[tuning_train].unique().cpu()
        ),
        "validation_pids": sorted(validation_pids),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "base": base_metrics,
                "best_step": best_step,
                "final": final_metrics,
                "holdout_pids": report["holdout_pids"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
