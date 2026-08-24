"""Audit primitive policy/value child ordering on PID-disjoint teacher paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.primitive_policy import load_primitive_checkpoint  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--memory-model",
        type=Path,
        default=PROJECT / "models/cube666_trajectory_memory_portfolio_v1/model.npz",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--maximum-states", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=83666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def local_effects(
    puzzle: Cube666Puzzle,
    orbit_positions: np.ndarray,
    global_to_local: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            [
                global_to_local[np.asarray(puzzle.generators[name])[orbit]]
                for orbit in orbit_positions
            ]
            for name in puzzle.move_names
        ],
        dtype=np.uint8,
    )


@torch.inference_mode()
def predict(
    model: torch.nn.Module, states: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch_logits, batch_values = model(states[start : start + batch_size])
        logits.append(batch_logits.float())
        values.append(batch_values.float() * model.config.value_scale)
    return torch.cat(logits), torch.cat(values)


def summarize(
    distances: np.ndarray,
    state_values: np.ndarray,
    policy_ranks: np.ndarray,
    value_ranks: np.ndarray,
    correct_values: np.ndarray,
    minimum_values: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {
        "count": int(len(distances)),
        "distance_mean": float(distances.mean()),
        "state_value_mae": float(np.abs(state_values - distances).mean()),
        "state_value_correlation": float(np.corrcoef(state_values, distances)[0, 1]),
        "correct_child_value_delta_from_target": float(
            (correct_values - (distances - 1)).mean()
        ),
        "correct_child_value_regret": float((correct_values - minimum_values).mean()),
        "policy_mean_rank": float(policy_ranks.mean()),
        "value_mean_rank": float(value_ranks.mean()),
    }
    for k in (1, 2, 4, 8, 16, 32):
        result[f"policy_top{k}"] = float((policy_ranks <= k).mean())
        result[f"value_top{k}"] = float((value_ranks <= k).mean())
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model, checkpoint = load_primitive_checkpoint(args.checkpoint, device)
    if model.config.encoding != "orbit_one_hot":
        raise ValueError("rank audit requires orbit-one-hot encoding")
    orbit_positions = np.asarray(checkpoint["orbit_positions"], dtype=np.int64)
    global_to_local = np.asarray(checkpoint["global_to_local"], dtype=np.uint8)
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    effects = torch.from_numpy(
        local_effects(puzzle, orbit_positions, global_to_local)
    ).to(device)
    with np.load(args.memory_model, allow_pickle=False) as memory:
        raw_states = memory["states"].astype(np.uint8, copy=False)
        pids = memory["pids"].astype(np.int64, copy=False)
        distances = memory["distances"].astype(np.int64, copy=False)
        action_masks = memory["action_masks"].astype(np.uint64, copy=False)
    eligible = np.flatnonzero(
        (np.mod(pids, args.folds) == args.fold) & (distances > 0)
    )
    rng = np.random.default_rng(args.seed)
    if len(eligible) > args.maximum_states:
        eligible = np.sort(
            rng.choice(eligible, size=args.maximum_states, replace=False)
        )
    states_np = global_to_local[raw_states[eligible][:, orbit_positions]]
    states = torch.from_numpy(states_np).to(device)
    state_logits, state_values = predict(model, states, args.batch_size)
    children = states[:, None].expand(-1, len(effects), -1, -1).gather(
        -1, effects[None].expand(len(states), -1, -1, -1).long()
    )
    _, flat_child_values = predict(model, children.flatten(0, 1), args.batch_size)
    child_values = flat_child_values.reshape(len(states), len(effects))
    masks = torch.from_numpy(action_masks[eligible].view(np.int64)).to(device)
    actions = torch.arange(len(effects), device=device)
    correct = torch.bitwise_and(masks[:, None], torch.bitwise_left_shift(1, actions)) != 0
    if not bool(correct.any(dim=1).all()):
        raise AssertionError("a nonterminal teacher row has no correct action")
    policy_order = torch.argsort(state_logits, dim=1, descending=True)
    policy_correct = correct.gather(1, policy_order)
    policy_ranks = policy_correct.float().argmax(dim=1) + 1
    value_order = torch.argsort(child_values, dim=1)
    value_correct = correct.gather(1, value_order)
    value_ranks = value_correct.float().argmax(dim=1) + 1
    correct_values = child_values.masked_fill(~correct, torch.inf).min(dim=1).values
    minimum_values = child_values.min(dim=1).values
    arrays = {
        "distances": distances[eligible],
        "state_values": state_values.cpu().numpy(),
        "policy_ranks": policy_ranks.cpu().numpy(),
        "value_ranks": value_ranks.cpu().numpy(),
        "correct_values": correct_values.cpu().numpy(),
        "minimum_values": minimum_values.cpu().numpy(),
    }
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "fold": args.fold,
        "folds": args.folds,
        "overall": summarize(**arrays),
        "bins": {},
        "local_combined": {},
    }
    local_nll = -torch.log_softmax(state_logits, dim=1)
    for weight in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        combined_order = torch.argsort(child_values + weight * local_nll, dim=1)
        combined_correct = correct.gather(1, combined_order)
        combined_ranks = combined_correct.float().argmax(dim=1) + 1
        report["local_combined"][str(weight)] = {
            "mean_rank": float(combined_ranks.float().mean()),
            **{
                f"top{k}": float((combined_ranks <= k).float().mean())
                for k in (1, 2, 4, 8, 16, 32)
            },
        }
    for lower, upper in ((1, 20), (21, 40), (41, 80), (81, 120), (121, 160), (161, 10_000)):
        selected = (arrays["distances"] >= lower) & (arrays["distances"] <= upper)
        if selected.any():
            report["bins"][f"{lower}-{upper}"] = summarize(
                **{key: value[selected] for key, value in arrays.items()}
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
