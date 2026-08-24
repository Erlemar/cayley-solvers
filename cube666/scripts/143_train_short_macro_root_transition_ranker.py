"""Train a root-local child-state ranker on verified short-macro transitions."""

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

from cube666.macro_action_policy import load_macro_action_policy_checkpoint  # noqa: E402
from cube666.macro_factorized_value import (  # noqa: E402
    MacroFactorizedPrimitiveValueNet,
    load_macro_factorized_value_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--proposal-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--negative-count", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--ranking-weight", type=float, default=8.0)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--ranking-temperature", type=float, default=2.0)
    parser.add_argument("--inverse-probability", type=float, default=0.5)
    parser.add_argument("--evaluation-samples", type=int, default=2048)
    parser.add_argument("--minimum-depth", type=int, default=5)
    parser.add_argument("--maximum-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=143666)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if not 0.0 <= args.inverse_probability <= 1.0:
        parser.error("inverse probability must be in [0, 1]")
    return args


def combined_value(
    model: MacroFactorizedPrimitiveValueNet,
    outputs: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    total, clusters = outputs
    return 0.5 * (
        total.float() * model.config.total_scale
        + clusters.float().sum(dim=1) * model.config.cluster_scale
    )


def make_root_batch(
    states: np.ndarray,
    actions: np.ndarray,
    inverse_actions: np.ndarray,
    depths: np.ndarray,
    rows: np.ndarray,
    inverse_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_states = states[rows].copy()
    labels = actions[rows, 0].astype(np.int64, copy=True)
    if np.any(inverse_mask):
        inverse_rows = rows[inverse_mask]
        batch_states[inverse_mask] = np.argsort(
            states[inverse_rows], axis=2
        ).astype(np.uint8)
        final_actions = actions[
            inverse_rows, depths[inverse_rows].astype(np.int64) - 1
        ].astype(np.int64, copy=False)
        labels[inverse_mask] = inverse_actions[final_actions]
    return batch_states, labels


@torch.inference_mode()
def evaluate_candidates(
    model: MacroFactorizedPrimitiveValueNet,
    proposal: torch.nn.Module,
    states: np.ndarray,
    labels: np.ndarray,
    remaining_depths: np.ndarray,
    effects: torch.Tensor,
    costs: torch.Tensor,
    negative_count: int,
    batch_size: int,
) -> dict[str, float | int]:
    model.eval()
    ranks: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch_states = torch.from_numpy(states[start : start + batch_size]).cuda()
        batch_labels = torch.from_numpy(labels[start : start + batch_size]).cuda()
        batch_depths = torch.from_numpy(
            remaining_depths[start : start + batch_size]
        ).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = proposal(batch_states, remaining_depth=batch_depths)
        negatives = logits.topk(negative_count, dim=1).indices
        candidate_actions = torch.cat((batch_labels[:, None], negatives), dim=1)
        candidate_effects = effects[candidate_actions]
        children = batch_states[:, None].expand(
            -1, candidate_actions.shape[1], -1, -1
        ).gather(-1, candidate_effects.long())
        flat_children = children.flatten(0, 1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            values = combined_value(model, model(flat_children)).view(
                len(batch_states), -1
            )
        q = costs[candidate_actions] + values.clamp_min(0)
        positive_q = q[:, 0]
        ranks.append(((q[:, 1:] < positive_q[:, None]).sum(1) + 1).cpu().numpy())
    rank = np.concatenate(ranks)
    return {
        "candidate_count": int(negative_count),
        "median_rank": float(np.median(rank)),
        "p90_rank": float(np.quantile(rank, 0.9)),
        "samples": int(len(rank)),
        "top1_recall": float(np.mean(rank <= 1)),
        "top8_recall": float(np.mean(rank <= 8)),
        "top32_recall": float(np.mean(rank <= 32)),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(args.seed)
    dataset = args.dataset_dir
    with np.load(dataset / "teacher.npz", allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        values = payload["search_value_targets"].astype(np.float32, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        actions = payload["teacher_solution_actions"].astype(np.int64, copy=False)
    groups = np.load(dataset / "source_state_ids.npy", allow_pickle=False)
    effects_np = np.load(dataset / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs_np = np.load(dataset / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    inverse_actions = np.load(
        dataset / "inverse_actions.npy", allow_pickle=False
    ).astype(np.int64, copy=False)
    depth_mask = (depths >= args.minimum_depth) & (depths <= args.maximum_depth)
    train_indices = np.flatnonzero((groups % 10 != 0) & depth_mask)
    heldout_indices = np.flatnonzero((groups % 10 == 0) & depth_mask)
    evaluation_rows = rng.choice(
        heldout_indices,
        size=min(args.evaluation_samples, len(heldout_indices)),
        replace=False,
    )
    evaluation_inverse = rng.random(len(evaluation_rows)) < args.inverse_probability
    evaluation_states, evaluation_labels = make_root_batch(
        states,
        actions,
        inverse_actions,
        depths,
        evaluation_rows,
        evaluation_inverse,
    )
    evaluation_depths = depths[evaluation_rows].astype(np.int64, copy=False)

    model, source_checkpoint = load_macro_factorized_value_checkpoint(
        str(args.init_checkpoint), device="cuda"
    )
    proposal, _ = load_macro_action_policy_checkpoint(
        str(args.proposal_checkpoint), device="cuda"
    )
    proposal.eval()
    for parameter in proposal.parameters():
        parameter.requires_grad_(False)
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).cuda()
    before = evaluate_candidates(
        model,
        proposal,
        evaluation_states,
        evaluation_labels,
        evaluation_depths,
        effects,
        costs,
        args.negative_count,
        args.batch_size,
    )
    train_model = torch.compile(model, mode="reduce-overhead") if args.compile else model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4, fused=True
    )

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return max((step + 1) / max(args.warmup_steps, 1), 1e-3)
        progress = (step - args.warmup_steps) / max(
            args.steps - args.warmup_steps, 1
        )
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    started = time.perf_counter()
    last: dict[str, float] = {}
    model.train()
    for step in range(1, args.steps + 1):
        rows = rng.choice(train_indices, size=args.batch_size, replace=True)
        inverse_mask = rng.random(args.batch_size) < args.inverse_probability
        batch_states_np, labels_np = make_root_batch(
            states, actions, inverse_actions, depths, rows, inverse_mask
        )
        batch_states = torch.from_numpy(batch_states_np).cuda()
        labels = torch.from_numpy(labels_np).cuda()
        remaining_depths = torch.from_numpy(
            depths[rows].astype(np.int64, copy=False)
        ).cuda()
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            logits = proposal(batch_states, remaining_depth=remaining_depths)
            negative_actions = logits.topk(args.negative_count, dim=1).indices
        positive_children = batch_states.gather(-1, effects[labels].long())
        negative_effects = effects[negative_actions]
        negative_children = batch_states[:, None].expand(
            -1, args.negative_count, -1, -1
        ).gather(-1, negative_effects.long())
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            positive_value = combined_value(
                model, train_model(positive_children)
            )
            negative_value = combined_value(
                model, train_model(negative_children.flatten(0, 1))
            ).view(args.batch_size, args.negative_count)
            positive_q = costs[labels] + positive_value
            negative_q = costs[negative_actions] + negative_value
            valid_negative = negative_actions.ne(labels[:, None])
            pairwise = F.softplus(
                (
                    positive_q[:, None]
                    - negative_q
                    + args.ranking_margin
                )
                / args.ranking_temperature
            ) * args.ranking_temperature
            ranking_loss = pairwise[valid_negative].mean() / 72.0
            positive_targets = torch.from_numpy(
                values[rows] - costs_np[labels_np]
            ).cuda()
            anchor_loss = F.smooth_l1_loss(
                positive_value, positive_targets, beta=2.0
            ) / 72.0
            loss = (
                args.ranking_weight * ranking_loss
                + args.anchor_weight * anchor_loss
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last = {
            "anchor_loss": float(anchor_loss.detach()),
            "loss": float(loss.detach()),
            "ranking_loss": float(ranking_loss.detach()),
        }
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "step": step,
                        **last,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    after = evaluate_candidates(
        model,
        proposal,
        evaluation_states,
        evaluation_labels,
        evaluation_depths,
        effects,
        costs,
        args.negative_count,
        args.batch_size,
    )
    report = {
        "after": after,
        "anchor_weight": args.anchor_weight,
        "before": before,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "inverse_probability": args.inverse_probability,
        "last_train_losses": last,
        "model_config": model.config.to_dict(),
        "negative_count": args.negative_count,
        "ranking_margin": args.ranking_margin,
        "ranking_temperature": args.ranking_temperature,
        "ranking_weight": args.ranking_weight,
        "source_checkpoint": str(args.init_checkpoint),
        "proposal_checkpoint": str(args.proposal_checkpoint),
        "source_payload_kind": source_checkpoint.get("kind"),
        "steps": args.steps,
        "train_samples": int(len(train_indices)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "cube666_short_macro_root_transition_ranker_v1",
            "factorized_value_config": model.config.to_dict(),
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
