"""Audit certified cross-parent ranks on two-layer short-macro frontiers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--direct-action-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--utilities-script", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--root-branch", type=int, default=128)
    parser.add_argument("--second-branch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=138666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("certified_value_utilities", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def unique_action_mask(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = actions.sort(dim=-1).values
    unique = torch.ones_like(ordered, dtype=torch.bool)
    unique[..., 1:] = ordered[..., 1:].ne(ordered[..., :-1])
    return ordered, unique


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    utilities = load_module(args.utilities_script)
    rng = np.random.default_rng(args.seed)
    root = args.dataset_dir
    with np.load(root / "teacher.npz", allow_pickle=False) as payload:
        states = payload["states"].astype(np.uint8, copy=False)
        depths = payload["walk_depths"].astype(np.int16, copy=False)
        solution_actions = payload["teacher_solution_actions"].astype(
            np.int32, copy=False
        )
        teacher_upper = payload["search_value_targets"].astype(
            np.float32, copy=False
        )
    groups = np.load(root / "source_state_ids.npy", allow_pickle=False)
    effects_np = np.load(root / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs_np = np.load(root / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    inverse_np = np.load(root / "inverse_actions.npy", allow_pickle=False).astype(
        np.int64, copy=False
    )
    eligible = np.flatnonzero((groups % 10 != 0) & (depths == 6))
    rows = rng.choice(
        eligible, size=min(args.samples, len(eligible)), replace=False
    )
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).cuda()
    inverse = torch.from_numpy(inverse_np).cuda()
    dual_weights = torch.from_numpy(
        utilities.build_subset_dual_weights(effects_np, costs_np)
    ).cuda()
    models = [
        utilities.load_macro_action_policy_checkpoint(str(path), device="cuda")[0]
        for path in args.direct_action_checkpoint
    ]

    total_candidates = 0
    total_certified = 0
    roots_with_certified = 0
    root_teacher_proposed = 0
    depth2_teacher_proposed = 0
    gaps: list[torch.Tensor] = []
    per_root_counts: list[int] = []
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        stop = min(start + args.batch_size, len(rows))
        batch_rows = rows[start:stop]
        batch = len(batch_rows)
        roots = torch.from_numpy(states[batch_rows]).cuda()
        remaining6 = torch.full((batch,), 6, dtype=torch.long, device="cuda")
        root_actions, root_unique = unique_action_mask(
            utilities.propose_actions(
                models, roots, remaining6, args.root_branch
            )
        )
        teacher0 = torch.from_numpy(solution_actions[batch_rows, 0].astype(np.int64)).cuda()
        root_teacher_proposed += int(
            (root_unique & root_actions.eq(teacher0[:, None])).any(dim=1).sum()
        )
        root_effects = effects[root_actions]
        parents = roots[:, None].expand_as(root_effects).gather(
            -1, root_effects.long()
        )
        root_width = root_actions.shape[1]
        flat_parents = parents.flatten(0, 1)
        remaining5 = torch.full(
            (len(flat_parents),), 5, dtype=torch.long, device="cuda"
        )
        second_actions, second_unique = unique_action_mask(
            utilities.propose_actions(
                models, flat_parents, remaining5, args.second_branch
            )
        )
        second_width = second_actions.shape[1]
        second_actions = second_actions.reshape(batch, root_width, second_width)
        second_unique = second_unique.reshape(batch, root_width, second_width)
        forbidden = inverse[root_actions]
        valid = (
            root_unique[:, :, None]
            & second_unique
            & second_actions.ne(forbidden[:, :, None])
        )
        teacher1 = torch.from_numpy(solution_actions[batch_rows, 1].astype(np.int64)).cuda()
        teacher_root_positions = root_actions.eq(teacher0[:, None]) & root_unique
        teacher_second = second_actions.eq(teacher1[:, None, None]) & valid
        depth2_teacher_proposed += int(
            (teacher_root_positions[:, :, None] & teacher_second)
            .any(dim=(1, 2))
            .sum()
        )
        selected_effects = effects[second_actions]
        children = parents[:, :, None].expand_as(selected_effects).gather(
            -1, selected_effects.long()
        )
        lower = utilities.cycle_cost_lower_bounds(children, dual_weights)
        path_cost = costs[root_actions][:, :, None] + costs[second_actions]
        q_lower = path_cost + lower
        upper = torch.from_numpy(teacher_upper[batch_rows]).cuda()
        certified = valid & q_lower.gt(upper[:, None, None])
        counts = certified.sum(dim=(1, 2))
        candidates = valid.sum(dim=(1, 2))
        total_certified += int(counts.sum())
        total_candidates += int(candidates.sum())
        roots_with_certified += int(counts.gt(0).sum())
        per_root_counts.extend(int(value) for value in counts.cpu().tolist())
        if certified.any():
            gaps.append(
                (q_lower - upper[:, None, None])[certified].detach().cpu()
            )
        print(
            json.dumps(
                {
                    "audited": stop,
                    "certified_pairs": total_certified,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                }
            ),
            flush=True,
        )
    counts_np = np.asarray(per_root_counts)
    gap_np = torch.cat(gaps).numpy() if gaps else np.empty(0, dtype=np.float32)
    report = {
        "candidate_pairs": total_candidates,
        "certified_gap_median": float(np.median(gap_np)) if len(gap_np) else None,
        "certified_gap_p90": (
            float(np.percentile(gap_np, 90)) if len(gap_np) else None
        ),
        "certified_pair_rate": total_certified / max(total_candidates, 1),
        "depth2_teacher_proposal_recall": depth2_teacher_proposed / len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "mean_certified_pairs_per_root": float(counts_np.mean()),
        "median_certified_pairs_per_root": float(np.median(counts_np)),
        "root_branch_per_model": args.root_branch,
        "root_teacher_proposal_recall": root_teacher_proposed / len(rows),
        "roots_with_certified_pair": roots_with_certified,
        "roots_with_certified_pair_rate": roots_with_certified / len(rows),
        "samples": int(len(rows)),
        "second_branch_per_model": args.second_branch,
        "total_certified_pairs": total_certified,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
