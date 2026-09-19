"""Audit certified action-ranking pairs for the short cube666 macro graph.

For a constructive trace action, its recorded continuation is an upper bound on
the best achievable Q value.  For every competing action, a portfolio of dual
cycle metrics gives a rigorous lower bound.  We call a comparison certified only
when the constructive upper bound is strictly cheaper than the competing lower
bound.  No failed or unsearched child is treated as a negative.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_action_policy import (  # noqa: E402
    MacroActionPolicyNet,
    load_macro_action_policy_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--direct-action-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--proposal-branch", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--minimum-depth", type=int, default=2)
    parser.add_argument("--maximum-depth", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=135666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def cycle_distance_vectors(states: np.ndarray) -> np.ndarray:
    """Return half the transposition distance for each cluster.

    ``(24 - cycle_count) / 2`` is conjugation invariant and subadditive.  It
    may be weaker than the true 3-cycle distance, which is safe for a lower
    bound.
    """

    states = np.asarray(states, dtype=np.uint8)
    flat = states.reshape(-1, 6, 24)
    output = np.empty((len(flat), 6), dtype=np.float32)
    for row in range(len(flat)):
        for cluster in range(6):
            permutation = flat[row, cluster]
            seen = np.zeros(24, dtype=np.bool_)
            cycles = 0
            for start in range(24):
                if seen[start]:
                    continue
                cycles += 1
                cursor = start
                while not seen[cursor]:
                    seen[cursor] = True
                    cursor = int(permutation[cursor])
            output[row, cluster] = (24 - cycles) / 2.0
    return output.reshape(states.shape[:-2] + (6,))


def build_subset_dual_weights(
    action_effects: np.ndarray, action_costs: np.ndarray
) -> np.ndarray:
    """Build 63 feasible dual metrics for the weighted action graph.

    For subset S, choose the largest shared coordinate weight r such that
    ``r * sum(D_j(action), j in S) <= cost(action)`` for every action.  Then
    ``r * sum(D_j(state), j in S)`` lower-bounds the cost of every route from
    that state to the identity.  Taking the maximum over feasible duals stays
    a valid lower bound.
    """

    action_vectors = cycle_distance_vectors(action_effects)
    weights: list[np.ndarray] = []
    for mask in range(1, 1 << 6):
        selected = np.asarray([(mask >> bit) & 1 for bit in range(6)], dtype=bool)
        distance = action_vectors[:, selected].sum(axis=1)
        active = distance > 0
        if not np.any(active):
            continue
        ratio = float(np.min(action_costs[active] / distance[active]))
        weight = np.zeros(6, dtype=np.float32)
        weight[selected] = ratio
        weights.append(weight)
    output = np.stack(weights)
    lhs = action_vectors @ output.T
    if np.any(lhs > action_costs[:, None] + 1e-5):
        raise AssertionError("constructed an infeasible cycle dual")
    return output


def cycle_cost_lower_bounds(
    states: np.ndarray, dual_weights: np.ndarray
) -> np.ndarray:
    vectors = cycle_distance_vectors(states)
    return np.max(vectors @ dual_weights.T, axis=-1)


@torch.inference_mode()
def cycle_cost_lower_bounds_torch(
    states: torch.Tensor, dual_weights: torch.Tensor
) -> torch.Tensor:
    """Vectorized GPU form used for millions of proposed children."""

    flat = states.reshape(-1, 6, 24)
    starts = torch.arange(24, device=states.device).view(1, 1, 24)
    starts = starts.expand(len(flat), 6, -1)
    cursor = starts
    minimum = starts
    for _ in range(24):
        cursor = flat.gather(-1, cursor.long()).long()
        minimum = torch.minimum(minimum, cursor)
    cycles = minimum.eq(starts).sum(dim=2)
    vectors = (24 - cycles).float() * 0.5
    return (vectors @ dual_weights.T).max(dim=1).values.reshape(states.shape[:-2])


def reconstruct_parents(
    roots: np.ndarray,
    solution_actions: np.ndarray,
    effects: np.ndarray,
    rows: np.ndarray,
    prefixes: np.ndarray,
) -> np.ndarray:
    parents = roots[rows].copy()
    for step in range(int(prefixes.max(initial=0))):
        active = prefixes > step
        actions = solution_actions[rows[active], step].astype(np.int64, copy=False)
        parents[active] = np.take_along_axis(
            parents[active], effects[actions], axis=-1
        )
    return parents


@torch.inference_mode()
def propose_actions(
    models: list[MacroActionPolicyNet],
    states: torch.Tensor,
    remaining_depths: torch.Tensor,
    branch: int,
) -> torch.Tensor:
    proposals: list[torch.Tensor] = []
    for model in models:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(states, remaining_depth=remaining_depths).float()
        proposals.append(logits.topk(min(branch, logits.shape[1]), dim=1).indices)
    return torch.cat(proposals, dim=1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.samples <= 0 or args.proposal_branch <= 0 or args.batch_size <= 0:
        raise ValueError("samples, proposal-branch, and batch-size must be positive")
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    root = args.dataset_dir
    with np.load(root / "teacher.npz", allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        solution_actions = payload["teacher_solution_actions"].astype(
            np.int32, copy=False
        )
    groups = np.load(root / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(root / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(root / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    inverse = np.load(root / "inverse_actions.npy", allow_pickle=False).astype(
        np.int64, copy=False
    )
    eligible = (groups % 10 != 0) & (depths >= args.minimum_depth)
    if args.maximum_depth:
        eligible &= depths <= args.maximum_depth
    eligible_rows = np.flatnonzero(eligible)
    if not len(eligible_rows):
        raise ValueError("depth filters left no training rows")
    rows = rng.choice(
        eligible_rows, size=min(args.samples, len(eligible_rows)), replace=False
    )
    prefixes = np.asarray(
        [rng.integers(0, int(depths[row])) for row in rows], dtype=np.int64
    )
    parents = reconstruct_parents(states, solution_actions, effects, rows, prefixes)
    teacher_actions = solution_actions[rows, prefixes].astype(np.int64, copy=False)
    previous_actions = np.full(len(rows), -1, dtype=np.int64)
    has_previous = prefixes > 0
    previous_actions[has_previous] = solution_actions[
        rows[has_previous], prefixes[has_previous] - 1
    ]
    forbidden = np.full(len(rows), -1, dtype=np.int64)
    forbidden[has_previous] = inverse[previous_actions[has_previous]]
    teacher_q_upper = np.asarray(
        [
            costs[
                solution_actions[row, prefix : int(depths[row])].astype(
                    np.int64, copy=False
                )
            ].sum(dtype=np.float64)
            for row, prefix in zip(rows, prefixes, strict=True)
        ],
        dtype=np.float32,
    )

    dual_weights = build_subset_dual_weights(effects, costs)
    effects_gpu = torch.from_numpy(effects).cuda()
    costs_gpu = torch.from_numpy(costs).cuda()
    dual_weights_gpu = torch.from_numpy(dual_weights).cuda()
    models: list[MacroActionPolicyNet] = []
    checkpoint_reports: list[dict[str, object]] = []
    for checkpoint_path in args.direct_action_checkpoint:
        model, payload = load_macro_action_policy_checkpoint(
            str(checkpoint_path), device="cuda"
        )
        if model.config.action_count != len(effects):
            raise ValueError("checkpoint and action table disagree")
        models.append(model)
        checkpoint_reports.append(
            {
                "path": str(checkpoint_path),
                "depth_conditioned": model.config.depth_conditioned,
                "kind": payload.get("kind"),
            }
        )

    certified_counts: list[int] = []
    proposed_teacher: list[bool] = []
    candidate_counts: list[int] = []
    gaps: list[float] = []
    prefix_certified = np.zeros(int(depths.max()) + 1, dtype=np.int64)
    prefix_total = np.zeros_like(prefix_certified)
    for start in range(0, len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        parent_gpu = torch.from_numpy(parents[start:stop]).cuda()
        remaining_gpu = torch.from_numpy(
            (depths[rows[start:stop]] - prefixes[start:stop]).astype(np.int64)
        ).cuda()
        proposal_gpu = propose_actions(
            models, parent_gpu, remaining_gpu, args.proposal_branch
        )
        proposal_gpu = proposal_gpu.sort(dim=1).values
        unique = torch.ones_like(proposal_gpu, dtype=torch.bool)
        unique[:, 1:] = proposal_gpu[:, 1:].ne(proposal_gpu[:, :-1])
        teacher_gpu = torch.from_numpy(teacher_actions[start:stop]).cuda()
        forbidden_gpu = torch.from_numpy(forbidden[start:stop]).cuda()
        valid = unique & proposal_gpu.ne(forbidden_gpu[:, None])
        proposed_teacher_gpu = (valid & proposal_gpu.eq(teacher_gpu[:, None])).any(dim=1)
        selected_effects = effects_gpu[proposal_gpu]
        children = parent_gpu[:, None].expand_as(selected_effects).gather(
            -1, selected_effects.long()
        )
        child_lower = cycle_cost_lower_bounds_torch(children, dual_weights_gpu)
        candidate_q_lower = costs_gpu[proposal_gpu] + child_lower
        threshold = torch.from_numpy(
            teacher_q_upper[start:stop] + args.margin
        ).cuda()
        certified = (
            valid
            & proposal_gpu.ne(teacher_gpu[:, None])
            & candidate_q_lower.gt(threshold[:, None])
        )
        counts = certified.sum(dim=1).cpu().numpy()
        valid_counts = valid.sum(dim=1).cpu().numpy()
        proposed_teacher_batch = proposed_teacher_gpu.cpu().numpy()
        certified_counts.extend(int(value) for value in counts)
        candidate_counts.extend(int(value) for value in valid_counts)
        proposed_teacher.extend(bool(value) for value in proposed_teacher_batch)
        batch_prefixes = prefixes[start:stop]
        for local, count in enumerate(counts):
            prefix = int(batch_prefixes[local])
            prefix_certified[prefix] += int(count)
            prefix_total[prefix] += int(valid_counts[local]) - int(
                proposed_teacher_batch[local]
            )
        if certified.any():
            gaps.extend(
                (
                    candidate_q_lower[certified]
                    - threshold[:, None].expand_as(candidate_q_lower)[certified]
                    + args.margin
                )
                .cpu()
                .tolist()
            )
        print(
            json.dumps(
                {
                    "audited": stop,
                    "certified_pairs": int(sum(certified_counts)),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            ),
            flush=True,
        )

    certified_array = np.asarray(certified_counts)
    gap_array = np.asarray(gaps, dtype=np.float32)
    report = {
        "action_count": int(len(effects)),
        "checkpoints": checkpoint_reports,
        "constructive_action_proposal_recall": float(np.mean(proposed_teacher)),
        "dual_count": int(len(dual_weights)),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "margin": args.margin,
        "mean_certified_pairs_per_parent": float(certified_array.mean()),
        "median_certified_pairs_per_parent": float(np.median(certified_array)),
        "parents_with_certified_pair": int(np.count_nonzero(certified_array)),
        "parents_with_certified_pair_rate": float(np.mean(certified_array > 0)),
        "prefix_certified_pairs": prefix_certified.tolist(),
        "prefix_candidate_pairs": prefix_total.tolist(),
        "proposal_branch_per_model": args.proposal_branch,
        "proposal_unique_mean": float(np.mean(candidate_counts)),
        "samples": int(len(rows)),
        "total_certified_pairs": int(certified_array.sum()),
        "certified_gap_median": (
            float(np.median(gap_array)) if len(gap_array) else None
        ),
        "certified_gap_p90": (
            float(np.percentile(gap_array, 90)) if len(gap_array) else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
