"""Audit valid inverse-action rank and value progress on curriculum states."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-module", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--action-effects", type=Path, required=True)
    parser.add_argument("--action-costs", type=Path, required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("short_macro_audit_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@torch.inference_mode()
def values(model: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, total, clusters = model(states)
    return 0.5 * (total.float() * 72.0 + clusters.float().sum(dim=1) * 12.0)


def main() -> None:
    args = parse_args()
    module = load_module(args.train_module)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = module.Config(action_count=checkpoint["model_config"]["action_count"])
    model = module.MacroEffectPolicyValueNet(config).cuda()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    indices = [
        int(token)
        for token in args.indices.replace("_", ",").split(",")
        if token.strip()
    ]
    with np.load(args.teacher, allow_pickle=False) as payload:
        states_np = payload["states"][indices].astype(np.uint8, copy=False)
        labels_np = payload["teacher_actions"][indices, 0].astype(np.int64, copy=False)
        targets_np = payload["search_value_targets"][indices].astype(np.float32, copy=False)
    effects_np = np.load(args.action_effects, allow_pickle=False).astype(np.uint8, copy=False)
    costs_np = np.load(args.action_costs, allow_pickle=False).astype(np.float32, copy=False)
    states = torch.from_numpy(states_np).cuda()
    effects = torch.from_numpy(effects_np).cuda()
    labels = torch.from_numpy(labels_np).cuda()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _, _ = model(states)
    label_effects = effects[labels][:, None]
    label_scores = module.score_candidates(logits, label_effects)[:, 0]
    higher = torch.zeros(len(indices), dtype=torch.int64, device="cuda")
    for start in range(0, len(effects), args.chunk_size):
        stop = min(start + args.chunk_size, len(effects))
        candidates = effects[start:stop][None].expand(len(indices), -1, -1, -1)
        scores = module.score_candidates(logits, candidates)
        higher += (scores > label_scores[:, None]).sum(dim=1)
    children = states.gather(-1, effects[labels].long())
    parent_values = values(model, states).cpu().numpy()
    child_values = values(model, children).cpu().numpy()
    rows = []
    for position, index in enumerate(indices):
        label = int(labels_np[position])
        rows.append(
            {
                "child_predicted_value": float(child_values[position]),
                "expected_return_upper_bound": float(targets_np[position] - costs_np[label]),
                "index": index,
                "label_action": label,
                "label_action_cost": float(costs_np[label]),
                "label_rank": int(higher[position]) + 1,
                "parent_predicted_value": float(parent_values[position]),
                "target_upper_bound": float(targets_np[position]),
            }
        )
    report = {
        "checkpoint": str(args.checkpoint),
        "median_label_rank": float(np.median([row["label_rank"] for row in rows])),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
