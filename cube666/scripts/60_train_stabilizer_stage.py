"""Train and gate one projected stabilizer-ladder stage on a GPU.

Each stage is a 24-slot permutation problem with 8--30 verified macro actions.
Training states are fresh non-backtracking walks in the exact projected action
group.  The policy target is a verified predecessor action and the value target
is remaining length on that generated trajectory.  Evaluation is autonomous:
fresh mixed states, exact identity goal checks, and learned policy/value beam
ranking.  The held-out walk itself is never exposed to the beam.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    action_count: int
    embedding_dim: int = 32
    hidden_dim: int = 1024
    residual_blocks: int = 6


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.linear1(F.silu(self.norm(inputs)))
        hidden = self.linear2(F.silu(hidden))
        return inputs + hidden


class StagePolicyValue(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.piece_embedding = nn.Embedding(24, config.embedding_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(24, config.embedding_dim)
        )
        self.input = nn.Linear(24 * config.embedding_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualBlock(config.hidden_dim) for _ in range(config.residual_blocks)
        )
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.policy = nn.Linear(config.hidden_dim, config.action_count)
        self.value = nn.Linear(config.hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.piece_embedding(states.long()) + self.position_embedding[None]
        hidden = self.input(embedded.flatten(1))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = F.silu(self.final_norm(hidden))
        return self.policy(hidden), self.value(hidden).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--max-train-depth", type=int, default=36)
    parser.add_argument("--eval-depth", type=int, default=96)
    parser.add_argument("--eval-trials", type=int, default=24)
    parser.add_argument("--beam-width", type=int, default=8_192)
    parser.add_argument("--beam-steps", type=int, default=60)
    parser.add_argument("--pool-factor", type=int, default=3)
    parser.add_argument("--policy-weight", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def inverse_indices(effects: np.ndarray) -> np.ndarray:
    by_effect = {row.tobytes(): index for index, row in enumerate(effects)}
    result = np.empty(len(effects), dtype=np.int64)
    for action, effect in enumerate(effects):
        inverse = np.empty_like(effect)
        inverse[effect] = np.arange(24, dtype=effect.dtype)
        if inverse.tobytes() not in by_effect:
            raise ValueError(f"action {action} has no inverse")
        result[action] = by_effect[inverse.tobytes()]
    return result


def random_walk_batch(
    effects: torch.Tensor,
    inverse: torch.Tensor,
    depths: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(depths)
    action_count = len(effects)
    states = torch.arange(24, device=effects.device, dtype=torch.uint8)[None].repeat(
        batch, 1
    )
    last = torch.full((batch,), -1, device=effects.device, dtype=torch.long)
    rows = torch.arange(batch, device=effects.device)
    for walk_step in range(int(depths.max().item())):
        active = walk_step < depths
        actions = torch.randint(
            action_count,
            (batch,),
            device=effects.device,
            generator=generator,
        )
        has_last = last >= 0
        forbidden = torch.zeros_like(last)
        forbidden[has_last] = inverse[last[has_last]]
        actions = torch.where(
            has_last & (actions == forbidden),
            (actions + 1) % action_count,
            actions,
        )
        moved = states.gather(1, effects[actions].long())
        states = torch.where(active[:, None], moved, states)
        last = torch.where(active, actions, last)
    labels = inverse[last]
    return states, labels


@torch.no_grad()
def predict_in_chunks(
    model: nn.Module,
    states: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    policies = []
    values = []
    for start in range(0, len(states), chunk_size):
        logits, value = model(states[start : start + chunk_size])
        policies.append(logits.float())
        values.append(value.float())
    return torch.cat(policies), torch.cat(values)


@torch.no_grad()
def learned_beam(
    initial: torch.Tensor,
    model: nn.Module,
    effects: torch.Tensor,
    inverse: torch.Tensor,
    *,
    beam_width: int,
    max_steps: int,
    pool_factor: int,
    policy_weight: float,
    max_train_depth: int,
    inference_batch: int,
) -> dict[str, int | bool | float]:
    device = initial.device
    identity = torch.arange(24, device=device, dtype=torch.uint8)
    states = initial[None]
    last_actions = torch.full((1,), -1, device=device, dtype=torch.long)
    generated = 0
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        parent_logits, _ = predict_in_chunks(
            model,
            states,
            chunk_size=inference_batch,
        )
        parent_nll = -torch.log_softmax(parent_logits, dim=1)
        if torch.any(last_actions >= 0):
            rows = torch.nonzero(last_actions >= 0, as_tuple=False).flatten()
            parent_nll[rows, inverse[last_actions[rows]]] = torch.inf

        parent_count = len(states)
        action_count = len(effects)
        children = states[:, None, :].expand(-1, action_count, -1).gather(
            2,
            effects[None].expand(parent_count, -1, -1).long(),
        ).reshape(-1, 24)
        generated += len(children)
        solved = torch.all(children == identity[None], dim=1)
        if torch.any(solved):
            return {
                "beam": parent_count,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "generated": generated,
                "solved": True,
                "steps": step,
            }

        _, child_values = predict_in_chunks(
            model,
            children,
            chunk_size=inference_batch,
        )
        ranks = (
            child_values * float(max_train_depth)
            + policy_weight * parent_nll.reshape(-1)
        )
        finite = torch.isfinite(ranks)
        finite_count = int(finite.sum().item())
        keep = min(finite_count, beam_width * pool_factor)
        candidate_ids = torch.topk(
            ranks,
            keep,
            largest=False,
            sorted=True,
        ).indices
        candidate_states = children[candidate_ids].cpu().numpy()
        packed = np.ascontiguousarray(candidate_states).view(
            np.dtype((np.void, candidate_states.shape[1]))
        ).reshape(-1)
        _, first = np.unique(packed, return_index=True)
        first.sort()
        first = first[:beam_width]
        selected = candidate_ids[torch.from_numpy(first).to(device)]
        states = children[selected]
        last_actions = selected % action_count

    return {
        "beam": len(states),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "generated": generated,
        "solved": False,
        "steps": max_steps,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(
        args.steps,
        args.batch_size,
        args.max_train_depth,
        args.eval_depth,
        args.eval_trials,
        args.beam_width,
        args.beam_steps,
        args.pool_factor,
    ) <= 0:
        raise ValueError("training and search dimensions must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    payload = json.loads(args.ladder.read_text(encoding="utf-8"))
    stage = payload["stages"][args.stage]
    effects_np = np.asarray(
        [entry["active_effect"] for entry in stage["basis"]],
        dtype=np.uint8,
    )
    inverse_np = inverse_indices(effects_np)
    effects = torch.from_numpy(effects_np).to(device)
    inverse = torch.from_numpy(inverse_np).to(device)

    config = ModelConfig(action_count=len(effects_np))
    eager_model = StagePolicyValue(config).to(device)
    model: nn.Module = eager_model
    if args.compile:
        model = torch.compile(eager_model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(
        eager_model.parameters(),
        lr=3.0e-4,
        weight_decay=1.0e-4,
        fused=True,
    )
    scaler_enabled = False
    walk_generator = torch.Generator(device=device).manual_seed(args.seed + 10_000)
    history = []
    started = time.perf_counter()
    eager_model.train()

    for update in range(1, args.steps + 1):
        depths = torch.randint(
            1,
            args.max_train_depth + 1,
            (args.batch_size,),
            device=device,
            generator=walk_generator,
        )
        states, labels = random_walk_batch(
            effects,
            inverse,
            depths,
            generator=walk_generator,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, values = model(states)
            policy_loss = F.cross_entropy(logits.float(), labels)
            value_targets = depths.float() / float(args.max_train_depth)
            value_loss = F.smooth_l1_loss(values.float(), value_targets)
            loss = policy_loss + 0.5 * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eager_model.parameters(), 1.0)
        optimizer.step()

        if update == 1 or update % max(1, args.steps // 20) == 0:
            with torch.no_grad():
                top1 = float((logits.argmax(dim=1) == labels).float().mean().item())
                value_mae = float(
                    torch.mean(
                        torch.abs(values.float() * args.max_train_depth - depths.float())
                    ).item()
                )
            row = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "loss": round(float(loss.item()), 6),
                "policy_loss": round(float(policy_loss.item()), 6),
                "top1": round(top1, 6),
                "update": update,
                "value_mae_walk_steps": round(value_mae, 6),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    eager_model.eval()
    eval_generator = torch.Generator(device=device).manual_seed(args.seed + 20_000)
    eval_depths = torch.full(
        (args.eval_trials,),
        args.eval_depth,
        device=device,
        dtype=torch.long,
    )
    eval_states, _ = random_walk_batch(
        effects,
        inverse,
        eval_depths,
        generator=eval_generator,
    )
    results = []
    for trial, initial in enumerate(eval_states):
        result = learned_beam(
            initial,
            eager_model,
            effects,
            inverse,
            beam_width=args.beam_width,
            max_steps=args.beam_steps,
            pool_factor=args.pool_factor,
            policy_weight=args.policy_weight,
            max_train_depth=args.max_train_depth,
            inference_batch=args.batch_size,
        )
        result["trial"] = trial
        results.append(result)
        print(json.dumps(result), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "config": asdict(config),
            "effects": effects_np,
            "inverse_indices": inverse_np,
            "model": eager_model.state_dict(),
            "stage": stage,
        },
        checkpoint_path,
    )
    solved = sum(bool(result["solved"]) for result in results)
    report = {
        "beam_steps": args.beam_steps,
        "beam_width": args.beam_width,
        "checkpoint": str(checkpoint_path),
        "config": asdict(config),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "eval_depth": args.eval_depth,
        "eval_trials": args.eval_trials,
        "gate_passed": solved >= math.ceil(0.9 * args.eval_trials),
        "history": history,
        "ladder": str(args.ladder),
        "results": results,
        "seed": args.seed,
        "solved": solved,
        "stage_index": args.stage,
        "steps": args.steps,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"history", "results"}}, indent=2))


if __name__ == "__main__":
    main()
