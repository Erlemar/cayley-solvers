"""Train a policy/value model on exact Schreier stage factors and gate its beam.

The teacher paths may be long, but every label is a verified action on an
arbitrary reachable state.  Held-out splitting is by complete factorized
solution, not by row, and the deployment gate starts from held-out initial
states with no access to their teacher path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.stabilizer_policy import (  # noqa: E402
    StabilizerModelConfig as ModelConfig,
    StabilizerPolicyValue as FactorPolicyValue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--heldout-mod", type=int, default=5)
    parser.add_argument("--heldout-remainder", type=int, default=0)
    parser.add_argument("--eval-trials", type=int, default=24)
    parser.add_argument("--beam-width", type=int, default=8_192)
    parser.add_argument("--branch-width", type=int, default=32)
    parser.add_argument("--beam-steps", type=int, default=200)
    parser.add_argument("--pool-factor", type=int, default=3)
    parser.add_argument("--policy-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=63666)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def predict_chunks(
    model: nn.Module,
    states: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = []
    values = []
    for start in range(0, len(states), batch_size):
        batch_logits, batch_values = model(states[start : start + batch_size])
        logits.append(batch_logits.float())
        values.append(batch_values.float())
    return torch.cat(logits), torch.cat(values)


@torch.no_grad()
def learned_beam(
    initial: torch.Tensor,
    model: nn.Module,
    effects: torch.Tensor,
    inverse: torch.Tensor,
    *,
    beam_width: int,
    branch_width: int,
    max_steps: int,
    pool_factor: int,
    policy_weight: float,
    value_scale: float,
    inference_batch: int,
) -> dict[str, int | bool | float | list[int]]:
    device = initial.device
    identity = torch.arange(24, device=device, dtype=torch.uint8)
    states = initial[None]
    last_actions = torch.full((1,), -1, dtype=torch.long, device=device)
    path_nll = torch.zeros(1, dtype=torch.float32, device=device)
    paths = torch.empty((1, 0), dtype=torch.long, device=device)
    generated = 0
    started = time.perf_counter()

    for step in range(1, max_steps + 1):
        logits, _ = predict_chunks(model, states, batch_size=inference_batch)
        if torch.any(last_actions >= 0):
            rows = torch.nonzero(last_actions >= 0, as_tuple=False).flatten()
            logits[rows, inverse[last_actions[rows]]] = -torch.inf
        log_probabilities = torch.log_softmax(logits, dim=1)
        proposal_logp, actions = torch.topk(
            log_probabilities,
            min(branch_width, logits.shape[1]),
            dim=1,
        )
        parent_count, actual_branch = actions.shape
        parent_ids = torch.arange(parent_count, device=device)[:, None].expand(
            -1, actual_branch
        ).reshape(-1)
        flat_actions = actions.reshape(-1)
        candidate_path_nll = path_nll[parent_ids] - proposal_logp.reshape(-1)
        children = states[parent_ids].gather(1, effects[flat_actions].long())
        child_paths = torch.cat(
            (paths[parent_ids], flat_actions[:, None]),
            dim=1,
        )
        generated += len(children)
        solved = torch.all(children == identity[None], dim=1)
        if torch.any(solved):
            solved_index = int(torch.nonzero(solved, as_tuple=False)[0].item())
            return {
                "action_path": child_paths[solved_index].cpu().tolist(),
                "beam": parent_count,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "generated": generated,
                "solved": True,
                "steps": step,
            }

        _, child_values = predict_chunks(
            model,
            children,
            batch_size=inference_batch,
        )
        ranks = child_values * value_scale + policy_weight * candidate_path_nll
        keep = min(len(ranks), beam_width * pool_factor)
        candidate_ids = torch.topk(ranks, keep, largest=False, sorted=True).indices
        candidate_states = children[candidate_ids].cpu().numpy()
        packed = np.ascontiguousarray(candidate_states).view(
            np.dtype((np.void, candidate_states.shape[1]))
        ).reshape(-1)
        _, first = np.unique(packed, return_index=True)
        first.sort()
        first = first[:beam_width]
        selected = candidate_ids[torch.from_numpy(first).to(device)]
        states = children[selected]
        last_actions = flat_actions[selected]
        path_nll = candidate_path_nll[selected]
        paths = child_paths[selected]

    return {
        "action_path": [],
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
        args.batch_size,
        args.heldout_mod,
        args.eval_trials,
        args.beam_width,
        args.branch_width,
        args.beam_steps,
        args.pool_factor,
    ) <= 0:
        raise ValueError("training and search dimensions must be positive")
    if args.steps < 0 or (args.steps == 0 and args.init_checkpoint is None):
        raise ValueError("zero training steps require --init-checkpoint")
    if not 0 <= args.heldout_remainder < args.heldout_mod:
        raise ValueError("heldout remainder is outside heldout modulus")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    data = np.load(args.teacher)
    states_np = data["states"].astype(np.uint8, copy=False)
    actions_np = data["teacher_actions"].astype(np.int64, copy=False)
    remaining_np = data["remaining_macro_steps"].astype(np.float32, copy=False)
    sample_ids = data["source_sample_ids"].astype(np.int64, copy=False)
    effects_np = data["effects"].astype(np.uint8, copy=False)
    inverse_np = data["inverse_indices"].astype(np.int64, copy=False)
    value_scale = float(max(1.0, remaining_np.max()))
    heldout = sample_ids % args.heldout_mod == args.heldout_remainder
    train_indices_np = np.flatnonzero(~heldout)
    if not len(train_indices_np) or not np.any(heldout):
        raise ValueError("teacher split produced an empty train or held-out set")

    states = torch.from_numpy(states_np).to(device)
    actions = torch.from_numpy(actions_np).to(device)
    remaining = torch.from_numpy(remaining_np).to(device)
    effects = torch.from_numpy(effects_np).to(device)
    inverse = torch.from_numpy(inverse_np).to(device)
    train_indices = torch.from_numpy(train_indices_np).to(device)

    config = ModelConfig(action_count=len(effects_np))
    eager_model = FactorPolicyValue(config).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["config"] != asdict(config):
            raise ValueError("initial checkpoint model config does not match teacher")
        if not np.array_equal(checkpoint["effects"], effects_np):
            raise ValueError("initial checkpoint action effects do not match teacher")
        eager_model.load_state_dict(checkpoint["model"])
    model: nn.Module = eager_model
    if args.compile:
        model = torch.compile(eager_model, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(
        eager_model.parameters(),
        lr=args.learning_rate,
        weight_decay=1.0e-4,
        fused=True,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1000)
    history = []
    started = time.perf_counter()
    eager_model.train()

    for update in range(1, args.steps + 1):
        positions = torch.randint(
            len(train_indices),
            (args.batch_size,),
            device=device,
            generator=generator,
        )
        indices = train_indices[positions]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, values = model(states[indices])
            policy_loss = F.cross_entropy(logits.float(), actions[indices])
            targets = remaining[indices] / value_scale
            value_loss = F.smooth_l1_loss(values.float(), targets)
            loss = policy_loss + 0.5 * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eager_model.parameters(), 1.0)
        optimizer.step()

        if update == 1 or update % max(1, args.steps // 20) == 0:
            with torch.no_grad():
                labels = actions[indices]
                top1 = float((logits.argmax(dim=1) == labels).float().mean().item())
                top32 = float(
                    torch.any(
                        torch.topk(logits, min(32, logits.shape[1]), dim=1).indices
                        == labels[:, None],
                        dim=1,
                    ).float().mean().item()
                )
                value_mae = float(
                    torch.mean(torch.abs(values.float() * value_scale - remaining[indices])).item()
                )
            row = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "loss": round(float(loss.item()), 6),
                "policy_loss": round(float(policy_loss.item()), 6),
                "top1": round(top1, 6),
                "top32": round(top32, 6),
                "update": update,
                "value_mae_macro_steps": round(value_mae, 6),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

    eager_model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "config": asdict(config),
            "effects": effects_np,
            "inverse_indices": inverse_np,
            "model": eager_model.state_dict(),
            "value_scale": value_scale,
        },
        checkpoint_path,
    )

    heldout_rows_np = np.flatnonzero(heldout)
    heldout_rng = np.random.default_rng(args.seed + 2000)
    if len(heldout_rows_np) > 8192:
        heldout_rows_np = heldout_rng.choice(
            heldout_rows_np,
            size=8192,
            replace=False,
        )
    heldout_rows = torch.from_numpy(heldout_rows_np).to(device)
    heldout_logits, heldout_values = predict_chunks(
        eager_model,
        states[heldout_rows],
        batch_size=args.batch_size,
    )
    heldout_labels = actions[heldout_rows]
    heldout_metrics = {
        "examples": len(heldout_rows_np),
        "top1": round(
            float((heldout_logits.argmax(dim=1) == heldout_labels).float().mean().item()),
            6,
        ),
        "top32": round(
            float(
                torch.any(
                    torch.topk(
                        heldout_logits,
                        min(32, heldout_logits.shape[1]),
                        dim=1,
                    ).indices
                    == heldout_labels[:, None],
                    dim=1,
                ).float().mean().item()
            ),
            6,
        ),
        "value_mae_macro_steps": round(
            float(
                torch.mean(
                    torch.abs(
                        heldout_values * value_scale - remaining[heldout_rows]
                    )
                ).item()
            ),
            6,
        ),
    }
    print(json.dumps({"heldout": heldout_metrics}), flush=True)

    heldout_ids = np.unique(sample_ids[heldout])
    initial_rows = []
    for sample_id in heldout_ids:
        rows = np.flatnonzero(sample_ids == sample_id)
        initial_rows.append(int(rows[np.argmax(remaining_np[rows])]))
    initial_rows = initial_rows[: args.eval_trials]
    results = []
    for trial, row in enumerate(initial_rows):
        result = learned_beam(
            states[row],
            eager_model,
            effects,
            inverse,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            max_steps=args.beam_steps,
            pool_factor=args.pool_factor,
            policy_weight=args.policy_weight,
            value_scale=value_scale,
            inference_batch=args.batch_size,
        )
        result["teacher_macro_steps"] = int(remaining_np[row])
        result["trial"] = trial
        results.append(result)
        print(json.dumps(result), flush=True)

    solved = sum(bool(result["solved"]) for result in results)
    report = {
        "beam_steps": args.beam_steps,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "checkpoint": str(checkpoint_path.resolve()),
        "config": asdict(config),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "eval_trials": len(results),
        "gate_passed": bool(results) and solved >= math.ceil(0.9 * len(results)),
        "heldout_mod": args.heldout_mod,
        "heldout_metrics": heldout_metrics,
        "heldout_remainder": args.heldout_remainder,
        "history": history,
        "results": results,
        "seed": args.seed,
        "solved": solved,
        "steps": args.steps,
        "teacher": str(args.teacher.resolve()),
        "train_examples": len(train_indices_np),
        "value_scale": value_scale,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key not in {"history", "results"}}, indent=2))


if __name__ == "__main__":
    main()
