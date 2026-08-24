"""Distill dense cost-plus-child-value rankings into the macro dual retriever."""

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

from cube666.macro_dual_policy import load_macro_dual_policy_checkpoint  # noqa: E402
from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hard-actions", type=int, default=64)
    parser.add_argument("--random-actions", type=int, default=64)
    parser.add_argument("--teacher-temperature", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=101666)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


@torch.inference_mode()
def child_values(
    model: torch.nn.Module, states: torch.Tensor, batch_size: int
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, clusters = model(states[start : start + batch_size])
        moves = 0.5 * (
            total.float() * model.config.total_scale
            + clusters.float().sum(dim=1) * model.config.cluster_scale
        )
        output.append(moves.clamp_min(0))
    return torch.cat(output)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_macro_dual_policy_checkpoint(args.init_checkpoint, device)
    value_model, _ = load_macro_factorized_value_checkpoint(args.value_checkpoint, device)
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    with np.load(args.hardneg, allow_pickle=False) as hard:
        row_indices = hard["row_indices"].astype(np.int64, copy=False)
        proposals = hard["proposals"].astype(np.int32, copy=False)
    trainable_prefixes = ("state_projection.", "action_", "logit_scale")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(trainable_prefixes))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    effects_tensor = torch.from_numpy(effects).to(device)
    costs_tensor = torch.from_numpy(costs).to(device)
    candidate_count = 1 + args.hard_actions + args.random_actions
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        positions = rng.integers(len(row_indices), size=args.batch_size)
        selected_rows = row_indices[positions]
        hard_columns = rng.integers(
            proposals.shape[1], size=(args.batch_size, args.hard_actions)
        )
        hard_ids = proposals[positions[:, None], hard_columns]
        random_ids = rng.integers(
            len(effects), size=(args.batch_size, args.random_actions), dtype=np.int32
        )
        teacher_ids = labels[selected_rows, 0, None]
        candidate_ids_np = np.concatenate((teacher_ids, hard_ids, random_ids), axis=1)
        candidate_ids = torch.from_numpy(candidate_ids_np).to(device)
        batch_states = torch.from_numpy(states[selected_rows]).to(device)
        candidate_effects = effects_tensor[candidate_ids]
        children = batch_states[:, None].expand(-1, candidate_count, -1, -1).gather(
            -1, candidate_effects.long()
        )
        with torch.inference_mode():
            values = child_values(
                value_model, children.flatten(0, 1), args.inference_batch_size
            ).reshape(args.batch_size, candidate_count)
            q_targets = costs_tensor[candidate_ids] + values
            target_probabilities = torch.softmax(
                -q_targets / args.teacher_temperature, dim=1
            )
        target_probabilities = target_probabilities.clone()
        if step <= args.warmup_steps:
            learning_rate = args.learning_rate * step / max(args.warmup_steps, 1)
        else:
            progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
            learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            queries = model.encode_states(batch_states)
            keys = model.encode_actions(
                candidate_effects.flatten(0, 1), costs_tensor[candidate_ids].flatten()
            ).reshape(args.batch_size, candidate_count, -1)
            logits = (queries[:, None] * keys).sum(dim=2)
            logits = logits * model.logit_scale.exp().clamp_max(100.0)
            distillation_loss = -(
                target_probabilities * F.log_softmax(logits.float(), dim=1)
            ).sum(dim=1).mean()
        distillation_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                agreement = logits.argmax(dim=1).eq(q_targets.argmin(dim=1)).float().mean()
                target_rank = 1 + (
                    q_targets < q_targets[:, :1]
                ).sum(dim=1).float()
            row = {
                "distillation_loss": float(distillation_loss.detach()),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "sampled_top1_agreement": float(agreement),
                "step": step,
                "teacher_action_mean_q_rank": float(target_rank.mean()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "candidate_count": candidate_count,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "hard_rows": len(row_indices),
        "history": history,
        "steps": args.steps,
        "teacher_temperature": args.teacher_temperature,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_macro_dual_policy_qdistill_v1",
            "model_state_dict": model.state_dict(),
            "qdistill_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
