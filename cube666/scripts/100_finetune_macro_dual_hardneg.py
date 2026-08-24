"""Fine-tune macro dual retrieval on its own full-library false positives."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_dual_policy import load_macro_dual_policy_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hard-actions", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=100666)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--freeze-state-trunk", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def load_eval_module() -> object:
    path = Path(__file__).with_name("98_train_macro_dual_policy.py")
    spec = importlib.util.spec_from_file_location("cube666_dual_policy_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load evaluator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, checkpoint = load_macro_dual_policy_checkpoint(args.init_checkpoint, device)
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
        clusters = teacher["search_cluster_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    with np.load(args.hardneg, allow_pickle=False) as hard:
        row_indices = hard["row_indices"].astype(np.int64, copy=False)
        proposals = hard["proposals"].astype(np.int32, copy=False)
    if args.freeze_state_trunk:
        trainable_prefixes = (
            "state_projection.",
            "action_",
            "logit_scale",
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable_prefixes))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=1e-4, fused=True
    )
    started = time.perf_counter()
    history: list[dict[str, float | int]] = []
    model.train()
    for step in range(1, args.steps + 1):
        positions = rng.integers(len(row_indices), size=args.batch_size)
        selected_rows = row_indices[positions]
        columns = rng.integers(
            proposals.shape[1], size=(args.batch_size, args.hard_actions)
        )
        positive_ids = labels[selected_rows, 0]
        hard_ids = proposals[positions[:, None], columns]
        candidate_ids = np.concatenate((positive_ids[:, None], hard_ids), axis=1)
        batch_states = torch.from_numpy(states[selected_rows]).to(device)
        candidate_effects = torch.from_numpy(effects[candidate_ids].reshape(-1, 6, 24)).to(
            device
        )
        candidate_costs = torch.from_numpy(costs[candidate_ids].reshape(-1)).to(device)
        candidate_id_tensor = torch.from_numpy(candidate_ids).to(device)
        positive_tensor = torch.from_numpy(positive_ids).to(device)
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
            keys = model.encode_actions(candidate_effects, candidate_costs).reshape(
                args.batch_size, 1 + args.hard_actions, -1
            )
            logits = (queries[:, None] * keys).sum(dim=2)
            logits = logits * model.logit_scale.exp().clamp_max(100.0)
            positive_mask = candidate_id_tensor.eq(positive_tensor[:, None])
            policy_loss = (
                torch.logsumexp(logits.float(), dim=1)
                - torch.logsumexp(
                    logits.float().masked_fill(~positive_mask, -torch.inf), dim=1
                )
            ).mean()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            row = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "logit_scale": float(model.logit_scale.exp().detach()),
                "policy_loss": float(policy_loss.detach()),
                "step": step,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    evaluator = load_eval_module()
    heldout_indices = np.flatnonzero(np.mod(groups, 10) == 0)
    eval_count = min(args.eval_samples, len(heldout_indices))
    eval_indices = rng.choice(heldout_indices, size=eval_count, replace=False)
    ranks = evaluator.policy_ranks(
        model,
        states[eval_indices],
        labels[eval_indices],
        counts[eval_indices],
        effects,
        costs,
        args.eval_batch_size,
        device,
    )
    heldout_policy = evaluator.summarize_ranks(ranks)
    heldout_values = evaluator.value_metrics(
        model,
        states[eval_indices],
        values[eval_indices],
        clusters[eval_indices],
        args.eval_batch_size,
        device,
    )
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "freeze_state_trunk": args.freeze_state_trunk,
        "hard_actions": args.hard_actions,
        "hard_rows": len(row_indices),
        "heldout_policy": heldout_policy,
        "heldout_samples": eval_count,
        "heldout_values": heldout_values,
        "history": history,
        "steps": args.steps,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint,
            "kind": "cube666_macro_dual_policy_hardneg_v1",
            "model_state_dict": model.state_dict(),
            "hardneg_report": report,
        },
        args.out_dir / "checkpoint.pt",
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
