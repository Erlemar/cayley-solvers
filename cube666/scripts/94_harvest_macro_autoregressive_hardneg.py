"""Harvest autoregressive top-macro hard negatives on training-PID teacher states."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--high-value-fraction", type=float, default=0.5)
    parser.add_argument("--decode-beam", type=int, default=512)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=94666)
    return parser.parse_args()


def load_decoder_module() -> object:
    path = Path(__file__).with_name("90_eval_macro_autoregressive_beam.py")
    spec = importlib.util.spec_from_file_location("cube666_ar_trie_decoder", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load trie decoder {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    module = load_decoder_module()
    device = torch.device("cuda")
    model, checkpoint = module.load_macro_autoregressive_checkpoint(
        args.checkpoint, device
    )
    action_tokens = np.asarray(checkpoint["action_tokens"], dtype=np.int16)
    action_costs = np.asarray(checkpoint["action_costs"])
    trie = module.build_trie(action_tokens, model.config.eos_token)
    transitions, terminals = module.trie_tensors(
        trie, model.config.primitive_action_count, device
    )
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    eligible: list[int] = []
    for row in range(len(states)):
        has_short = any(
            action >= 0 and action_costs[int(action)] <= 14
            for action in labels[row, : int(counts[row])]
        )
        if has_short and int(groups[row]) % args.folds != args.fold:
            eligible.append(row)
    eligible_np = np.asarray(eligible, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    high_count = int(round(args.samples * args.high_value_fraction))
    uniform_count = args.samples - high_count
    selected_parts: list[np.ndarray] = []
    if uniform_count:
        selected_parts.append(rng.choice(eligible_np, size=uniform_count, replace=False))
    if high_count:
        weights = np.square(values[eligible_np] / max(float(values[eligible_np].mean()), 1e-6))
        weights /= weights.sum()
        selected_parts.append(
            rng.choice(eligible_np, size=high_count, replace=False, p=weights)
        )
    selected = np.unique(np.concatenate(selected_parts))
    if len(selected) < args.samples:
        missing_pool = np.setdiff1d(eligible_np, selected, assume_unique=False)
        selected = np.concatenate(
            (selected, rng.choice(missing_pool, size=args.samples - len(selected), replace=False))
        )
    rng.shuffle(selected)
    proposals = np.full((len(selected), args.topk), -1, dtype=np.int32)
    started = time.perf_counter()
    for start in range(0, len(selected), args.decode_batch_size):
        stop = min(start + args.decode_batch_size, len(selected))
        decoded, _ = module.decode_batch(
            model,
            states[selected[start:stop]],
            transitions,
            terminals,
            beam_width=args.decode_beam,
            topk=args.topk,
            device=device,
        )
        if np.any(decoded < 0):
            raise RuntimeError("batched trie decoder returned incomplete proposals")
        proposals[start:stop] = decoded
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
    np.savez_compressed(
        args.out,
        row_indices=selected,
        proposals=proposals,
        values=values[selected],
    )
    report = {
        "decode_beam": args.decode_beam,
        "decode_batch_size": args.decode_batch_size,
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
