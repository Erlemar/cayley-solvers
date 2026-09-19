"""Train a PID-held-out transformer ranker for complete KMC rough trajectories.

The model predicts a correction to the deterministic rough-length/residual proxy.
It consumes the ordered rough move sequence, the exact six-cluster endpoint, and
the insertion/configuration descriptors.  Nested PID folds keep early stopping
separate from the fold used for the reported out-of-fold score.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_uniform72_v3/rough_rows_uniform_b20000.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "models/cube666_path_context_transformer_v1",
    )
    parser.add_argument("--target-field", default="uniform_final_moves")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--patience", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--rank-weight", type=float, default=4.0)
    parser.add_argument("--ranking-temperature", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=117)
    return parser.parse_args()


@dataclass(frozen=True)
class ModelConfig:
    move_dim: int = 96
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ffn: int = 256
    state_local_dim: int = 64
    state_dim: int = 128
    numeric_dim: int = 96
    hidden_dim: int = 192
    dropout: float = 0.1


class PathContextTransformer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        move_count: int,
        maximum_length: int,
        numeric_features: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.move_embedding = nn.Embedding(move_count + 1, config.move_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, maximum_length + 1, config.move_dim)
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, config.move_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.move_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers, enable_nested_tensor=False
        )
        self.path_norm = nn.LayerNorm(config.move_dim)

        self.state_local = nn.Sequential(
            nn.Linear(24 * 24, config.state_local_dim),
            nn.LayerNorm(config.state_local_dim),
            nn.SiLU(),
        )
        self.cluster_embedding = nn.Parameter(torch.zeros(6, config.state_local_dim))
        self.state_global = nn.Sequential(
            nn.Linear(6 * config.state_local_dim, config.state_dim),
            nn.LayerNorm(config.state_dim),
            nn.SiLU(),
            nn.Linear(config.state_dim, config.state_dim),
            nn.SiLU(),
        )
        self.numeric = nn.Sequential(
            nn.Linear(numeric_features, config.numeric_dim),
            nn.LayerNorm(config.numeric_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.numeric_dim, config.numeric_dim),
            nn.SiLU(),
        )
        joint_dim = config.move_dim + config.state_dim + config.numeric_dim
        self.head = nn.Sequential(
            nn.Linear(joint_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.cluster_embedding, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        states: torch.Tensor,
        numeric: torch.Tensor,
    ) -> torch.Tensor:
        batch = tokens.shape[0]
        embedded = self.move_embedding(tokens)
        cls = self.cls.expand(batch, -1, -1)
        path = torch.cat((cls, embedded), dim=1)
        path = path + self.position_embedding[:, : path.shape[1]]
        padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=tokens.device),
                tokens == 0,
            ),
            dim=1,
        )
        path_hidden = self.transformer(path, src_key_padding_mask=padding)
        path_hidden = self.path_norm(path_hidden[:, 0])

        one_hot = F.one_hot(states.long(), num_classes=24).flatten(start_dim=2)
        local = self.state_local(one_hot.to(self.state_local[0].weight.dtype))
        state_hidden = self.state_global(
            (local + self.cluster_embedding[None]).flatten(start_dim=1)
        )
        numeric_hidden = self.numeric(numeric)
        return self.head(
            torch.cat((path_hidden, state_hidden, numeric_hidden), dim=1)
        ).squeeze(1)


def load_feature_module():
    source = Path(__file__).with_name("50_train_path_context_ranker.py")
    spec = importlib.util.spec_from_file_location("path_context_features_50", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proxy(row: dict[str, object]) -> float:
    return float(row["rough_reduced_moves"]) + 4.34 * float(
        row["residual_three_cycles"]
    )


def score_predictions(
    predictions: np.ndarray, targets: np.ndarray, pids: np.ndarray
) -> dict[str, float | int]:
    selected_total = 0
    oracle_total = 0
    winner_hits = 0
    pair_correct = 0
    pair_total = 0
    for pid in np.unique(pids):
        group = np.flatnonzero(pids == pid)
        selected = group[int(np.argmin(predictions[group]))]
        oracle = float(targets[group].min())
        selected_total += float(targets[selected])
        oracle_total += oracle
        winner_hits += int(targets[selected] == oracle)
        for left, right in itertools.combinations(group, 2):
            actual = float(targets[left] - targets[right])
            if actual == 0:
                continue
            predicted = float(predictions[left] - predictions[right])
            pair_correct += int(actual * predicted > 0)
            pair_total += 1
    unique = len(np.unique(pids))
    return {
        "oracle_total": int(round(oracle_total)),
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "selected_regret": int(round(selected_total - oracle_total)),
        "selected_total": int(round(selected_total)),
        "winner_hits": winner_hits,
        "winner_hit_fraction": winner_hits / unique,
    }


def grouped_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    losses = []
    for pid in pids.unique():
        group = torch.nonzero(pids == pid, as_tuple=False).flatten()
        target_distribution = torch.softmax(-targets[group] / temperature, dim=0)
        losses.append(
            -(
                target_distribution
                * torch.log_softmax(-predictions[group] / temperature, dim=0)
            ).sum()
        )
    return torch.stack(losses).mean()


def standardize(
    raw: torch.Tensor, train_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw[train_indices].mean(dim=0)
    scale = raw[train_indices].std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return (raw - mean) / scale, mean, scale


@torch.inference_mode()
def predict(
    model: nn.Module,
    tokens: torch.Tensor,
    states: torch.Tensor,
    numeric: torch.Tensor,
    base_proxy: torch.Tensor,
    indices: torch.Tensor,
) -> np.ndarray:
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        correction = model(tokens[indices], states[indices], numeric[indices])
    return (base_proxy[indices] + correction.float()).cpu().numpy()


def train_model(
    *,
    config: ModelConfig,
    tokens: torch.Tensor,
    states: torch.Tensor,
    numeric: torch.Tensor,
    base_proxy: torch.Tensor,
    targets: torch.Tensor,
    pids: torch.Tensor,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor | None,
    maximum_steps: int,
    eval_every: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    rank_weight: float,
    ranking_temperature: float,
    seed: int,
    move_count: int,
) -> tuple[nn.Module, int, list[dict[str, object]], torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    standardized, mean, scale = standardize(numeric, train_indices)
    model = PathContextTransformer(
        config,
        move_count=move_count,
        maximum_length=tokens.shape[1],
        numeric_features=numeric.shape[1],
    ).to(tokens.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=True,
    )
    history: list[dict[str, object]] = []
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_step = maximum_steps
    for step in range(1, maximum_steps + 1):
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            correction = model(
                tokens[train_indices],
                states[train_indices],
                standardized[train_indices],
            )
        predictions = base_proxy[train_indices] + correction.float()
        train_targets = targets[train_indices]
        train_pids = pids[train_indices]
        regression = F.smooth_l1_loss(predictions, train_targets, beta=4.0)
        ranking = grouped_ranking_loss(
            predictions, train_targets, train_pids, ranking_temperature
        )
        loss = regression + rank_weight * ranking
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if validation_indices is not None and (
            step == 1 or step % eval_every == 0 or step == maximum_steps
        ):
            validation_predictions = predict(
                model,
                tokens,
                states,
                standardized,
                base_proxy,
                validation_indices,
            )
            validation_targets = targets[validation_indices].cpu().numpy()
            validation_pids = pids[validation_indices].cpu().numpy()
            metrics = score_predictions(
                validation_predictions, validation_targets, validation_pids
            )
            mae = float(np.mean(np.abs(validation_predictions - validation_targets)))
            key = (float(metrics["selected_regret"]), mae, step)
            history.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "mae": mae,
                    "metrics": metrics,
                    "step": step,
                }
            )
            if best_key is None or key < best_key:
                best_key = key
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if step - best_step >= patience:
                break
    if validation_indices is not None:
        if best_state is None:
            raise AssertionError("validation never produced a checkpoint")
        model.load_state_dict(best_state)
    model.eval()
    return model, best_step, history, mean, scale


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.folds, args.steps, args.eval_every, args.patience) <= 1:
        raise ValueError("folds, steps, eval-every and patience must exceed one")
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    missing = [index for index, row in enumerate(rows) if args.target_field not in row]
    if missing:
        raise KeyError(f"target field is missing from {len(missing)} rows")
    pids_np = np.asarray([int(row["pid"]) for row in rows], dtype=np.int64)
    unique_pids = np.asarray(sorted(np.unique(pids_np)))
    if len(unique_pids) < args.folds * 2:
        raise ValueError("nested folds require at least two PID groups per fold")

    moves = sorted({str(move) for row in rows for move in row["rough_path"]})
    move_to_id = {move: index + 1 for index, move in enumerate(moves)}
    maximum_length = max(len(row["rough_path"]) for row in rows)
    tokens_np = np.zeros((len(rows), maximum_length), dtype=np.int64)
    for row_index, row in enumerate(rows):
        path = [move_to_id[str(move)] for move in row["rough_path"]]
        tokens_np[row_index, : len(path)] = path
    states_np = np.asarray(
        [row["cluster_permutations"] for row in rows], dtype=np.uint8
    )

    features = load_feature_module()
    numeric_np = np.stack(
        [
            np.concatenate(
                (
                    features.compact_frontier_features(row),
                    features.cluster_structure_features(row),
                    features.configuration_features(row),
                )
            )
            for row in rows
        ]
    ).astype(np.float32)
    targets_np = np.asarray(
        [float(row[args.target_field]) for row in rows], dtype=np.float32
    )
    proxy_np = np.asarray([proxy(row) for row in rows], dtype=np.float32)

    device = torch.device("cuda")
    tokens = torch.from_numpy(tokens_np).to(device)
    states = torch.from_numpy(states_np).to(device)
    numeric = torch.from_numpy(numeric_np).to(device)
    targets = torch.from_numpy(targets_np).to(device)
    base_proxy = torch.from_numpy(proxy_np).to(device)
    pids = torch.from_numpy(pids_np).to(device)
    config = ModelConfig()

    oof_predictions = np.full(len(rows), np.nan, dtype=np.float64)
    folds = []
    best_steps = []
    for fold in range(args.folds):
        test_pids = unique_pids[fold :: args.folds]
        validation_pids = unique_pids[(fold + 1) % args.folds :: args.folds]
        test_mask = np.isin(pids_np, test_pids)
        validation_mask = np.isin(pids_np, validation_pids)
        train_mask = ~(test_mask | validation_mask)
        train_indices = torch.from_numpy(np.flatnonzero(train_mask)).to(device)
        validation_indices = torch.from_numpy(
            np.flatnonzero(validation_mask)
        ).to(device)
        test_indices = torch.from_numpy(np.flatnonzero(test_mask)).to(device)
        model, best_step, history, mean, scale = train_model(
            config=config,
            tokens=tokens,
            states=states,
            numeric=numeric,
            base_proxy=base_proxy,
            targets=targets,
            pids=pids,
            train_indices=train_indices,
            validation_indices=validation_indices,
            maximum_steps=args.steps,
            eval_every=args.eval_every,
            patience=args.patience,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            rank_weight=args.rank_weight,
            ranking_temperature=args.ranking_temperature,
            seed=args.seed + fold,
            move_count=len(moves),
        )
        standardized = (numeric - mean) / scale
        fold_predictions = predict(
            model, tokens, states, standardized, base_proxy, test_indices
        )
        test_positions = test_indices.cpu().numpy()
        oof_predictions[test_positions] = fold_predictions
        metrics = score_predictions(
            fold_predictions, targets_np[test_positions], pids_np[test_positions]
        )
        best_steps.append(best_step)
        folds.append(
            {
                "best_step": best_step,
                "history": history,
                "metrics": metrics,
                "test_pids": [int(pid) for pid in test_pids],
                "validation_pids": [int(pid) for pid in validation_pids],
            }
        )
        print(
            f"fold={fold} best_step={best_step} regret={metrics['selected_regret']} "
            f"hits={metrics['winner_hits']}/{len(test_pids)}",
            flush=True,
        )
    if np.isnan(oof_predictions).any():
        raise AssertionError("nested cross-validation missed rows")

    oof_metrics = score_predictions(oof_predictions, targets_np, pids_np)
    proxy_metrics = score_predictions(proxy_np, targets_np, pids_np)
    final_steps = int(round(statistics.median(best_steps)))
    all_indices = torch.arange(len(rows), device=device)
    final_model, _, _, mean, scale = train_model(
        config=config,
        tokens=tokens,
        states=states,
        numeric=numeric,
        base_proxy=base_proxy,
        targets=targets,
        pids=pids,
        train_indices=all_indices,
        validation_indices=None,
        maximum_steps=final_steps,
        eval_every=args.eval_every,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank_weight=args.rank_weight,
        ranking_temperature=args.ranking_temperature,
        seed=args.seed + args.folds,
        move_count=len(moves),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "config": asdict(config),
        "dataset": str(args.dataset),
        "final_steps": final_steps,
        "maximum_length": maximum_length,
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in final_model.state_dict().items()
        },
        "move_to_id": move_to_id,
        "numeric_mean": mean.detach().cpu(),
        "numeric_scale": scale.detach().cpu(),
        "target_field": args.target_field,
    }
    torch.save(checkpoint, args.out_dir / "checkpoint.pt")
    report = {
        "dataset": str(args.dataset),
        "final_steps": final_steps,
        "folds": folds,
        "model_config": asdict(config),
        "oof": oof_metrics,
        "proxy": proxy_metrics,
        "rows": len(rows),
        "unique_pids": len(unique_pids),
    }
    atomic_json(args.out_dir / "report.json", report)
    np.savez_compressed(
        args.out_dir / "oof_predictions.npz",
        pids=pids_np,
        predictions=oof_predictions,
        proxy=proxy_np,
        targets=targets_np,
    )
    print(json.dumps({"final_steps": final_steps, "oof": oof_metrics, "proxy": proxy_metrics}, indent=2))


if __name__ == "__main__":
    main()
