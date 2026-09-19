"""Measure dual-retriever overlap with exhaustive cost-plus-value action ranks."""

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
from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=102666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def predict_values(
    model: torch.nn.Module, states: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            total, clusters = model(batch)
        moves = 0.5 * (
            total.float() * model.config.total_scale
            + clusters.float().sum(dim=1) * model.config.cluster_scale
        )
        output.append(moves.clamp_min(0).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dual, _ = load_macro_dual_policy_checkpoint(args.checkpoint, device)
    value_model, _ = load_macro_factorized_value_checkpoint(args.value_checkpoint, device)
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
    eligible = np.flatnonzero(np.mod(groups, args.folds) == args.fold)
    order = eligible[np.argsort(values[eligible])]
    positions = np.linspace(0, len(order) - 1, min(args.samples, len(order))).round().astype(
        np.int64
    )
    selected = order[positions]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(selected)
    key_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(effects), args.inference_batch_size):
            stop = min(start + args.inference_batch_size, len(effects))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                key_parts.append(
                    dual.encode_actions(
                        torch.from_numpy(effects[start:stop]).to(device),
                        torch.from_numpy(costs[start:stop]).to(device),
                    )
                )
        action_keys = torch.cat(key_parts)
    widths = (16, 64, 256, 1024)
    regrets = {width: [] for width in widths}
    best_ranks: list[int] = []
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for number, row in enumerate(selected, 1):
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            query = dual.encode_states(torch.from_numpy(states[row][None]).to(device))
        retrieval = (query @ action_keys.transpose(0, 1))[0].float().cpu().numpy()
        retrieval_order = np.argsort(-retrieval)
        children = np.take_along_axis(
            np.broadcast_to(states[row], (len(effects),) + states[row].shape),
            effects,
            axis=-1,
        )
        q_values = costs + predict_values(
            value_model, children, args.inference_batch_size, device
        )
        best_action = int(np.argmin(q_values))
        inverse_order = np.empty(len(retrieval_order), dtype=np.int32)
        inverse_order[retrieval_order] = np.arange(len(retrieval_order), dtype=np.int32)
        best_rank = 1 + int(inverse_order[best_action])
        best_ranks.append(best_rank)
        row_regrets: dict[str, float] = {}
        for width in widths:
            regret = float(q_values[retrieval_order[:width]].min() - q_values[best_action])
            regrets[width].append(regret)
            row_regrets[str(width)] = regret
        rows.append(
            {
                "best_action": best_action,
                "best_dual_rank": best_rank,
                "pid": int(groups[row]),
                "q_regret": row_regrets,
                "row": int(row),
                "teacher_value": float(values[row]),
            }
        )
        if number == 1 or number % 16 == 0 or number == len(selected):
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "samples": number,
                    }
                ),
                flush=True,
            )
    report = {
        "checkpoint": str(args.checkpoint),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "best_q_action_rank": {
            "median": float(np.median(best_ranks)),
            "p90": float(np.quantile(best_ranks, 0.9)),
            "top16": float(np.mean(np.asarray(best_ranks) <= 16)),
            "top64": float(np.mean(np.asarray(best_ranks) <= 64)),
            "top256": float(np.mean(np.asarray(best_ranks) <= 256)),
            "top1024": float(np.mean(np.asarray(best_ranks) <= 1024)),
        },
        "q_regret": {
            str(width): {
                "mean": float(np.mean(regrets[width])),
                "median": float(np.median(regrets[width])),
                "p90": float(np.quantile(regrets[width], 0.9)),
            }
            for width in widths
        },
        "rows": rows,
        "samples": len(selected),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
