"""Harvest full-library top retrieval confusions for the macro dual encoder."""

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

from cube666.macro_dual_policy import load_macro_dual_policy_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--action-batch-size", type=int, default=4096)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=99666)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)
    model, _ = load_macro_dual_policy_checkpoint(args.checkpoint, device)
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.float32, copy=False
    )
    eligible = np.flatnonzero(np.mod(groups, args.folds) != args.fold)
    high_count = int(round(args.samples * args.high_value_fraction))
    uniform_count = args.samples - high_count
    parts: list[np.ndarray] = []
    if uniform_count:
        parts.append(rng.choice(eligible, size=uniform_count, replace=False))
    if high_count:
        weights = np.sqrt(values[eligible] / max(float(values[eligible].mean()), 1e-6))
        weights /= weights.sum()
        parts.append(rng.choice(eligible, size=high_count, replace=False, p=weights))
    selected = np.unique(np.concatenate(parts))
    if len(selected) < args.samples:
        remaining = np.setdiff1d(eligible, selected, assume_unique=False)
        selected = np.concatenate(
            (selected, rng.choice(remaining, size=args.samples - len(selected), replace=False))
        )
    rng.shuffle(selected)

    key_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(effects), args.action_batch_size):
            stop = min(start + args.action_batch_size, len(effects))
            effect_batch = torch.from_numpy(effects[start:stop]).to(device)
            cost_batch = torch.from_numpy(costs[start:stop]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                key_parts.append(model.encode_actions(effect_batch, cost_batch))
        action_keys = torch.cat(key_parts)
        proposals = np.empty((len(selected), args.topk), dtype=np.int32)
        started = time.perf_counter()
        for start in range(0, len(selected), args.batch_size):
            stop = min(start + args.batch_size, len(selected))
            batch = torch.from_numpy(states[selected[start:stop]]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                queries = model.encode_states(batch)
            scores = queries @ action_keys.transpose(0, 1)
            top = scores.topk(args.topk, dim=1).indices
            proposals[start:stop] = top.cpu().numpy().astype(np.int32, copy=False)
            if stop == args.batch_size or stop % 2048 == 0 or stop == len(selected):
                print(
                    json.dumps(
                        {
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                            "samples": stop,
                        }
                    ),
                    flush=True,
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, row_indices=selected, proposals=proposals)
    report = {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "samples": len(selected),
        "topk": args.topk,
        "value_maximum": float(values[selected].max()),
        "value_mean": float(values[selected].mean()),
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
