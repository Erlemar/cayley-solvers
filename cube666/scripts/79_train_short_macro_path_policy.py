"""Train a compositional primitive-path policy for the short cube666 macro table."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.puzzle import Cube666Puzzle  # noqa: E402


@dataclasses.dataclass(frozen=True)
class Config:
    action_count: int
    maximum_tokens: int = 15
    vocabulary_size: int = 37
    hidden_dim: int = 512
    residual_blocks: int = 4
    dropout: float = 0.0

    def as_dict(self) -> dict[str, int | float | str]:
        return {"architecture": "path_tokens", **dataclasses.asdict(self)}


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.linear_in(self.norm(inputs)))
        return inputs + self.dropout(self.linear_out(hidden))


class MacroPathPolicyValueNet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Sequential(
            nn.Linear(6 * 24 * 24, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden_dim, config.dropout)
            for _ in range(config.residual_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.path_head = nn.Linear(
            config.hidden_dim, config.maximum_tokens * config.vocabulary_size
        )
        self.total_cost_head = nn.Linear(config.hidden_dim, 1)
        self.cluster_cost_head = nn.Linear(config.hidden_dim, 6)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = F.one_hot(states.long(), num_classes=24)
        hidden = self.stem(encoded.flatten(start_dim=1).to(self.stem[0].weight.dtype))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return (
            self.path_head(hidden).reshape(
                -1, self.config.maximum_tokens, self.config.vocabulary_size
            ),
            self.total_cost_head(hidden).squeeze(1),
            self.cluster_cost_head(hidden),
        )


def score_candidates(logits: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    candidate_count = action_tokens.shape[1]
    expanded = log_probs[:, None].expand(-1, candidate_count, -1, -1)
    valid = action_tokens >= 0
    selected = expanded.gather(
        -1, action_tokens.clamp_min(0).long().unsqueeze(-1)
    ).squeeze(-1)
    selected = selected.masked_fill(~valid, 0)
    return selected.sum(dim=2) / valid.sum(dim=2).clamp_min(1).sqrt()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--group-ids", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--action-digest", required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negative-actions", type=int, default=512)
    parser.add_argument("--hard-negative-actions", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--cluster-value-weight", type=float, default=1.0)
    parser.add_argument("--policy-eval-samples", type=int, default=256)
    parser.add_argument("--action-chunk-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=79666)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def build_action_tokens(
    action_library: Path,
    move_names: tuple[str, ...],
) -> np.ndarray:
    paths = json.loads(action_library.read_text(encoding="utf-8"))["paths"]
    maximum = max(len(path) for path in paths) + 1
    if maximum > 15:
        raise ValueError("short action path exceeds 14 primitive moves")
    move_to_token = {name: index for index, name in enumerate(move_names)}
    tokens = np.full((len(paths), 15), -1, dtype=np.int16)
    for action, path in enumerate(paths):
        encoded = [move_to_token[name] for name in path]
        tokens[action, : len(encoded)] = encoded
        tokens[action, len(encoded)] = len(move_names)
    return tokens


@torch.inference_mode()
def topk_action_ids(
    logits: torch.Tensor,
    action_tokens: torch.Tensor,
    *,
    topk: int,
    chunk_size: int,
) -> torch.Tensor:
    best_scores = torch.empty((len(logits), 0), device=logits.device)
    best_actions = torch.empty(
        (len(logits), 0), dtype=torch.long, device=logits.device
    )
    for start in range(0, len(action_tokens), chunk_size):
        stop = min(start + chunk_size, len(action_tokens))
        candidates = action_tokens[start:stop][None].expand(len(logits), -1, -1)
        scores = score_candidates(logits, candidates)
        ids = torch.arange(start, stop, device=logits.device)[None].expand(
            len(logits), -1
        )
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, ids), dim=1)
        keep = min(topk, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    return best_actions


@torch.inference_mode()
def metrics(
    model: MacroPathPolicyValueNet,
    states: np.ndarray,
    labels: np.ndarray,
    values: np.ndarray,
    tokens: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, float | int]:
    state_tensor = torch.from_numpy(states).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, total, clusters = model(state_tensor)
    predictions = (
        total.float() * 72.0 + clusters.float().sum(dim=1) * 12.0
    ) * 0.5
    label_tensor = torch.from_numpy(labels[:, 0]).long().to(device)
    positive = score_candidates(logits, tokens[label_tensor][:, None]).squeeze(1)
    better = torch.zeros(len(states), dtype=torch.long, device=device)
    for start in range(0, len(tokens), chunk_size):
        stop = min(start + chunk_size, len(tokens))
        candidate = tokens[start:stop][None].expand(len(states), -1, -1)
        better += (score_candidates(logits, candidate) > positive[:, None]).sum(dim=1)
    ranks = (better + 1).cpu().numpy()
    errors = predictions.cpu().numpy() - values
    return {
        "median_teacher_rank": float(np.median(ranks)),
        "p90_teacher_rank": float(np.quantile(ranks, 0.9)),
        "samples": len(states),
        "top1_recall": float(np.mean(ranks <= 1)),
        "top16_recall": float(np.mean(ranks <= 16)),
        "top128_recall": float(np.mean(ranks <= 128)),
        "top512_recall": float(np.mean(ranks <= 512)),
        "value_mae": float(np.abs(errors).mean()),
        "value_correlation": float(np.corrcoef(predictions.cpu().numpy(), values)[0, 1]),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    tokens_np = build_action_tokens(args.action_library, tuple(puzzle.move_names))
    action_tokens = torch.from_numpy(tokens_np).to(device)
    with np.load(args.teacher, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        labels = payload["teacher_actions"].astype(np.int32, copy=False)
        counts = payload["teacher_action_counts"].astype(np.int16, copy=False)
        values = payload["search_value_targets"].astype(np.float32, copy=False)
        clusters = payload["search_cluster_targets"].astype(np.float32, copy=False)
    if np.any(counts != 1):
        raise ValueError("path-token trainer currently requires one label per state")
    groups = np.load(args.group_ids, allow_pickle=False)
    train_indices = np.flatnonzero(np.mod(groups, 10) != 0)
    heldout_indices = np.flatnonzero(np.mod(groups, 10) == 0)
    config = Config(action_count=len(tokens_np))
    model = MacroPathPolicyValueNet(config).to(device)
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    compatible = {
        key: value
        for key, value in initial["model_state_dict"].items()
        if key in model.state_dict() and model.state_dict()[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected or set(missing) != {"path_head.weight", "path_head.bias"}:
        raise ValueError(f"unexpected warm-start mismatch: {missing=} {unexpected=}")
    eval_count = min(args.policy_eval_samples, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    before = metrics(
        model,
        states[eval_indices],
        labels[eval_indices],
        values[eval_indices],
        action_tokens,
        device=device,
        chunk_size=args.action_chunk_size,
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / args.warmup_steps, 1e-3)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    started = time.perf_counter()
    last: dict[str, float] = {}
    for step in range(1, args.steps + 1):
        selected = train_indices[rng.integers(len(train_indices), size=args.batch_size)]
        batch_states = torch.from_numpy(states[selected]).to(device)
        batch_labels = torch.from_numpy(labels[selected, 0]).long().to(device)
        batch_values = torch.from_numpy(values[selected]).to(device)
        batch_clusters = torch.from_numpy(clusters[selected]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, total, cluster_prediction = train_model(batch_states)
            negatives: list[torch.Tensor] = []
            if args.hard_negative_actions:
                negatives.append(
                    topk_action_ids(
                        logits.detach(),
                        action_tokens,
                        topk=args.hard_negative_actions,
                        chunk_size=args.action_chunk_size,
                    )
                )
            if args.negative_actions:
                negatives.append(
                    torch.randint(
                        len(tokens_np),
                        (args.batch_size, args.negative_actions),
                        device=device,
                    )
                )
            negative_ids = torch.cat(negatives, dim=1)
            candidate_ids = torch.cat((batch_labels[:, None], negative_ids), dim=1)
            scores = score_candidates(logits, action_tokens[candidate_ids])
            duplicate = negative_ids.eq(batch_labels[:, None])
            scores[:, 1:] = scores[:, 1:].masked_fill(duplicate, -torch.inf)
            contrastive = torch.logsumexp(scores, dim=1).sub(scores[:, 0]).mean()
            token_targets = action_tokens[batch_labels]
            token_loss = F.cross_entropy(
                logits.float().reshape(-1, config.vocabulary_size),
                token_targets.reshape(-1).long(),
                ignore_index=-1,
            )
            value_loss = F.smooth_l1_loss(total.float(), batch_values / 72.0)
            cluster_loss = F.smooth_l1_loss(
                cluster_prediction.float(), batch_clusters / 12.0
            )
            loss = (
                contrastive
                + 0.2 * token_loss
                + args.value_weight * value_loss
                + args.cluster_value_weight * cluster_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "cluster_loss": float(cluster_loss.detach()),
            "contrastive_loss": float(contrastive.detach()),
            "loss": float(loss.detach()),
            "token_loss": float(token_loss.detach()),
            "value_loss": float(value_loss.detach()),
        }
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step": step, **last}), flush=True)
    after = metrics(
        model,
        states[eval_indices],
        labels[eval_indices],
        values[eval_indices],
        action_tokens,
        device=device,
        chunk_size=args.action_chunk_size,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "action_tokens.npy", tokens_np)
    report = {
        "action_count": len(tokens_np),
        "action_digest": args.action_digest,
        "after": after,
        "before": before,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "last_train_losses": last,
        "model_config": config.as_dict(),
        "steps": args.steps,
        "teacher_samples": len(states),
        "token_names": list(puzzle.move_names) + ["<eos>"],
    }
    torch.save(
        {
            "action_digest": args.action_digest,
            "model_config": config.as_dict(),
            "model_state_dict": model.state_dict(),
            "report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
