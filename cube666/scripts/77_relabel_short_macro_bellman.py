"""Conservatively relabel short-macro walks with one-step learned Bellman improvements."""

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
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--group-ids", type=Path, required=True)
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--action-costs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-module", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=25_000)
    parser.add_argument("--candidate-actions", type=int, default=256)
    parser.add_argument("--state-batch-size", type=int, default=32)
    parser.add_argument("--action-chunk-size", type=int, default=1024)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--maximum-improvement", type=float, default=14.0)
    parser.add_argument("--seed", type=int, default=77666)
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("cube666_bellman_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def predict_values(
    model: torch.nn.Module,
    states: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, total, clusters = model(states[start : start + batch_size])
        output.append(
            (total.float() * 72.0 + clusters.float().sum(dim=1) * 12.0) * 0.5
        )
    return torch.cat(output)


@torch.inference_mode()
def topk_actions(
    module: ModuleType,
    logits: torch.Tensor,
    effects: torch.Tensor,
    *,
    topk: int,
    chunk_size: int,
) -> torch.Tensor:
    best_scores = torch.empty((len(logits), 0), device=logits.device)
    best_actions = torch.empty(
        (len(logits), 0), dtype=torch.long, device=logits.device
    )
    for start in range(0, len(effects), chunk_size):
        stop = min(start + chunk_size, len(effects))
        candidates = effects[start:stop][None].expand(len(logits), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        ids = torch.arange(start, stop, device=logits.device)[None].expand(
            len(logits), -1
        )
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_actions = torch.cat((best_actions, ids), dim=1)
        keep = min(topk, merged_scores.shape[1])
        best_scores, positions = merged_scores.topk(keep, dim=1)
        best_actions = merged_actions.gather(1, positions)
    return best_actions


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    module = load_module(args.train_module)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = module.Config(action_count=checkpoint["model_config"]["action_count"])
    model = module.MacroEffectPolicyValueNet(config).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with np.load(args.teacher, allow_pickle=False) as payload:
        all_states = payload["states"].astype(np.uint8, copy=False)
        all_labels = payload["teacher_actions"].astype(np.int32, copy=False)
        all_values = payload["search_value_targets"].astype(np.float32, copy=False)
        all_clusters = payload["search_cluster_targets"].astype(np.float32, copy=False)
    all_groups = np.load(args.group_ids, allow_pickle=False)
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs_np = np.load(args.action_costs, allow_pickle=False).astype(
        np.float32, copy=False
    )
    if len(effects_np) != config.action_count:
        raise ValueError("checkpoint and action table disagree")
    count = min(args.samples, len(all_states))
    selected = rng.choice(len(all_states), size=count, replace=False)
    states = all_states[selected]
    labels = all_labels[selected, :1].copy()
    values = all_values[selected].copy()
    clusters = all_clusters[selected].copy()
    groups = all_groups[selected].copy()
    effects = torch.from_numpy(effects_np).cuda()
    costs = torch.from_numpy(costs_np).cuda()
    identity = torch.arange(24, dtype=torch.uint8, device="cuda")[None].expand(6, -1)
    improved = 0
    improvement_sum = 0.0
    predicted_candidates = 0
    for start in range(0, count, args.state_batch_size):
        stop = min(start + args.state_batch_size, count)
        batch_states = torch.from_numpy(states[start:stop]).cuda()
        batch_labels = torch.from_numpy(labels[start:stop, 0]).long().cuda()
        batch_values = torch.from_numpy(values[start:stop]).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _, _ = model(batch_states)
        proposed = topk_actions(
            module,
            logits,
            effects,
            topk=args.candidate_actions,
            chunk_size=args.action_chunk_size,
        )
        candidates = torch.cat((batch_labels[:, None], proposed), dim=1)
        selected_effects = effects[candidates]
        children = batch_states[:, None].expand(
            -1, candidates.shape[1], -1, -1
        ).gather(-1, selected_effects.long())
        flat_children = children.flatten(0, 1)
        child_values = predict_values(
            model, flat_children, args.inference_batch_size
        ).reshape(children.shape[:2])
        solved = children.eq(identity).all(dim=(2, 3))
        child_values = torch.where(solved, torch.zeros_like(child_values), child_values)
        q_values = costs[candidates] + child_values
        q_values[:, 0] = batch_values
        lower_target = (batch_values - args.maximum_improvement).clamp_min(0)
        calibrated = (q_values < batch_values[:, None] - 0.5) & (
            q_values >= lower_target[:, None]
        ) & (q_values >= 0)
        q_values = q_values.masked_fill(~calibrated, torch.inf)
        q_values[:, 0] = batch_values
        best_q, best_positions = q_values.min(dim=1)
        best_actions = candidates.gather(1, best_positions[:, None]).squeeze(1)
        batch_improvement = batch_values - best_q
        accepted = batch_improvement > 0
        accepted_np = accepted.cpu().numpy()
        old_values = values[start:stop].copy()
        labels[start:stop, 0] = best_actions.cpu().numpy()
        values[start:stop] = best_q.cpu().numpy()
        ratios = values[start:stop] / np.maximum(old_values, 1.0e-6)
        clusters[start:stop] *= ratios[:, None]
        improved += int(accepted.sum())
        improvement_sum += float(batch_improvement[accepted].sum())
        predicted_candidates += int(children.shape[0] * children.shape[1])
        if stop == count or stop % (args.state_batch_size * 25) == 0:
            print(
                json.dumps(
                    {
                        "improved": improved,
                        "processed": stop,
                        "samples": count,
                    }
                ),
                flush=True,
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if np.any(values < 0) or np.any(clusters < 0):
        raise AssertionError("Bellman relabel produced a negative cost target")
    if not np.allclose(clusters.sum(axis=1), values, atol=1.0e-4):
        raise AssertionError("Bellman cluster targets do not sum to total target")
    np.savez_compressed(
        args.out_dir / "teacher.npz",
        states=states,
        teacher_actions=labels,
        teacher_action_counts=np.ones(count, dtype=np.int16),
        search_value_targets=values,
        search_cluster_targets=clusters,
    )
    np.save(args.out_dir / "source_state_ids.npy", groups)
    report = {
        "accepted_improvements": improved,
        "candidate_actions": args.candidate_actions,
        "checkpoint": str(args.checkpoint),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "mean_accepted_improvement": improvement_sum / max(improved, 1),
        "mean_target_after": float(values.mean()),
        "mean_target_before": float(all_values[selected].mean()),
        "predicted_children": predicted_candidates,
        "samples": count,
        "seed": args.seed,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
