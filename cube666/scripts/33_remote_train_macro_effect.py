"""Self-contained fixed-shape macro-effect fine-tuner for remote marimo GPUs."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CLUSTER_COUNT = 6
CLUSTER_SIZE = 24


@dataclass(frozen=True)
class Config:
    action_count: int
    hidden_dim: int = 512
    residual_blocks: int = 4
    dropout: float = 0.0

    def as_dict(self) -> dict[str, int | float | str]:
        return {"architecture": "effect", **asdict(self)}


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


class MacroEffectPolicyValueNet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        input_dim = CLUSTER_COUNT * CLUSTER_SIZE * CLUSTER_SIZE
        self.stem = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden_dim, config.dropout)
            for _ in range(config.residual_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.effect_head = nn.Linear(
            config.hidden_dim, CLUSTER_COUNT * CLUSTER_SIZE * CLUSTER_SIZE
        )
        self.total_cost_head = nn.Linear(config.hidden_dim, 1)
        self.cluster_cost_head = nn.Linear(config.hidden_dim, CLUSTER_COUNT)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = F.one_hot(states.long(), num_classes=CLUSTER_SIZE)
        hidden = self.stem(encoded.flatten(start_dim=1).to(self.stem[0].weight.dtype))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return (
            self.effect_head(hidden).reshape(
                -1, CLUSTER_COUNT, CLUSTER_SIZE, CLUSTER_SIZE
            ),
            self.total_cost_head(hidden).squeeze(-1),
            self.cluster_cost_head(hidden),
        )


def score_candidates(logits: torch.Tensor, effects: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    candidate_count = effects.shape[1]
    expanded = log_probs[:, None].expand(-1, candidate_count, -1, -1, -1)
    selected = expanded.gather(-1, effects.long().unsqueeze(-1)).squeeze(-1)
    identity = torch.arange(CLUSTER_SIZE, device=logits.device)
    identity = identity.view(1, 1, 1, CLUSTER_SIZE, 1).expand(
        logits.shape[0], candidate_count, CLUSTER_COUNT, -1, -1
    )
    baseline = expanded.gather(-1, identity).squeeze(-1)
    active = effects.ne(
        torch.arange(CLUSTER_SIZE, device=logits.device).view(1, 1, 1, -1)
    )
    return ((selected - baseline) * active).sum(dim=(2, 3)) / active.sum(
        dim=(2, 3)
    ).clamp_min(1).sqrt()


def loss_function(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    counts: torch.Tensor,
    value_targets: torch.Tensor,
    cluster_targets: torch.Tensor,
    action_effects: torch.Tensor,
    negatives: torch.Tensor,
    value_weight: float,
    cluster_value_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits, total_prediction, cluster_prediction = outputs
    batch_size, positive_slots = labels.shape
    valid = torch.arange(positive_slots, device=labels.device)[None] < counts[:, None]
    positive_ids = labels.clamp_min(0)
    candidate_ids = torch.cat((positive_ids, negatives), dim=1)
    scores = score_candidates(logits, action_effects[candidate_ids])
    scores[:, :positive_slots] = scores[:, :positive_slots].masked_fill(~valid, -torch.inf)
    duplicates = negatives[:, :, None].eq(positive_ids[:, None, :])
    duplicates &= valid[:, None, :]
    scores[:, positive_slots:] = scores[:, positive_slots:].masked_fill(
        duplicates.any(dim=2), -torch.inf
    )
    contrastive = (
        torch.logsumexp(scores, dim=1)
        - torch.logsumexp(scores[:, :positive_slots], dim=1)
    ).mean()
    target_effects = action_effects[positive_ids[:, 0]]
    flat_logits = logits.reshape(-1, CLUSTER_SIZE)
    flat_targets = target_effects.reshape(-1).long()
    token_all = F.cross_entropy(flat_logits.float(), flat_targets)
    identity = torch.arange(CLUSTER_SIZE, device=logits.device).view(1, 1, -1)
    active = target_effects.ne(identity).reshape(-1)
    token = token_all + 2.0 * F.cross_entropy(
        flat_logits[active].float(), flat_targets[active]
    )
    value = F.smooth_l1_loss(total_prediction.float(), value_targets.float() / 72.0)
    cluster_value = F.smooth_l1_loss(
        cluster_prediction.float(), cluster_targets.float() / 12.0
    )
    total = (
        contrastive
        + 0.2 * token
        + value_weight * value
        + cluster_value_weight * cluster_value
    )
    return total, {
        "loss": float(total.detach()),
        "contrastive_loss": float(contrastive.detach()),
        "token_loss": float(token.detach()),
        "value_loss": float(value.detach()),
        "cluster_value_loss": float(cluster_value.detach()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--group-ids", type=Path, required=True)
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--action-digest", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--negative-actions", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=32666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--value-weight", type=float, default=0.2)
    parser.add_argument("--cluster-value-weight", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("remote training requires CUDA")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    with np.load(args.teacher, allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        labels = payload["teacher_actions"].astype(np.int32, copy=False)
        counts = payload["teacher_action_counts"].astype(np.int16, copy=False)
        if "search_value_targets" in payload:
            search_values = payload["search_value_targets"].astype(
                np.float32, copy=False
            )
            search_clusters = payload["search_cluster_targets"].astype(
                np.float32, copy=False
            )
        else:
            search_clusters = payload["cluster_cost_targets"].astype(
                np.float32, copy=False
            )
            search_values = search_clusters.sum(axis=1)
    groups = np.load(args.group_ids, allow_pickle=False)
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(np.uint8, copy=False)
    if groups.shape != (len(states),) or effects_np.shape[1:] != (6, 24):
        raise ValueError("training artifact shapes disagree")
    training_indices = np.flatnonzero(np.mod(groups, 10) != 0)
    config = Config(action_count=len(effects_np))
    model = MacroEffectPolicyValueNet(config).to(device)
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(initial["model_state_dict"])
    train_model = torch.compile(model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / args.warmup_steps, 1e-3)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    action_effects = torch.from_numpy(effects_np).to(device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    started = time.perf_counter()
    last_losses: dict[str, float] = {}
    train_model.train()
    for step in range(1, args.steps + 1):
        positions = torch.randint(
            len(training_indices), (args.batch_size,), generator=generator
        ).numpy()
        selected = training_indices[positions]
        batch_states = torch.from_numpy(states[selected]).to(device)
        batch_labels = torch.from_numpy(labels[selected]).to(device)
        batch_counts = torch.from_numpy(counts[selected]).to(device)
        batch_values = torch.from_numpy(search_values[selected]).to(device)
        batch_cluster_values = torch.from_numpy(search_clusters[selected]).to(device)
        negatives = torch.randint(
            len(effects_np),
            (args.batch_size, args.negative_actions),
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, last_losses = loss_function(
                train_model(batch_states),
                batch_labels,
                batch_counts,
                batch_values,
                batch_cluster_values,
                action_effects,
                negatives,
                args.value_weight,
                args.cluster_value_weight,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "step": step,
                        **last_losses,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    report = {
        "action_count": len(effects_np),
        "action_digest": args.action_digest,
        "architecture": "effect",
        "batch_size": args.batch_size,
        "compiled": True,
        "device": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "last_train_losses": last_losses,
        "model_config": config.as_dict(),
        "negative_actions": args.negative_actions,
        "seed": args.seed,
        "steps": args.steps,
        "teacher_samples": len(states),
        "training_samples": len(training_indices),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
