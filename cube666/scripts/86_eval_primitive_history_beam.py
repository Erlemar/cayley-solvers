"""Replay-verified beam search with the history-conditioned primitive model."""

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

from cube666.primitive_history_policy import load_primitive_history_checkpoint  # noqa: E402
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
    parser.add_argument("--pids", default="200")
    parser.add_argument("--beam", type=int, default=4096)
    parser.add_argument("--branch", type=int, default=16)
    parser.add_argument("--maximum-depth", type=int, default=260)
    parser.add_argument("--policy-weight", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--policy-mode", choices=("cumulative", "local"), default="cumulative")
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--allow-backtrack", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=86666)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def generator_effects(
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
    model: torch.nn.Module,
    states: torch.Tensor,
    histories: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for start in range(0, len(states), batch_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            batch_logits, batch_values = model(
                states[start : start + batch_size],
                histories[start : start + batch_size],
            )
        logits.append(batch_logits.float())
        values.append(batch_values.float() * model.config.value_scale)
    return torch.cat(logits), torch.cat(values)


@torch.inference_mode()
def solve(
    model: torch.nn.Module,
    initial: np.ndarray,
    effects: torch.Tensor,
    identity: torch.Tensor,
    *,
    beam_width: int,
    branch: int,
    maximum_depth: int,
    policy_weight: float,
    value_weight: float,
    policy_mode: str,
    exhaustive: bool,
    allow_backtrack: bool,
    inference_batch_size: int,
    state_zobrist: torch.Tensor,
    history_zobrist: torch.Tensor,
) -> tuple[list[int] | None, dict[str, float | int]]:
    device = effects.device
    history_length = model.config.history_length
    pad = model.config.action_count
    states = torch.from_numpy(initial[None]).to(device)
    histories = torch.full(
        (1, history_length), pad, dtype=torch.int16, device=device
    )
    logits, values = predict(model, states, histories, inference_batch_size)
    cumulative_nll = torch.zeros(1, device=device)
    last_actions = torch.full((1,), -1, dtype=torch.long, device=device)
    paths = torch.full((1, maximum_depth), -1, dtype=torch.int8, device=device)
    positions = torch.arange(initial.size, device=device)
    history_positions = torch.arange(history_length, device=device)
    expanded = 0
    started = time.perf_counter()
    for depth in range(1, maximum_depth + 1):
        log_probabilities = torch.log_softmax(logits, dim=1)
        if exhaustive:
            proposed = torch.arange(len(effects), device=device)[None].expand(
                len(states), -1
            )
            proposed_logp = log_probabilities
            effective_branch = len(effects)
        else:
            proposal_logits = log_probabilities.clone()
            if depth > 1 and not allow_backtrack:
                proposal_logits[
                    torch.arange(len(states), device=device),
                    torch.bitwise_xor(last_actions, 1),
                ] = -torch.inf
            proposed_logp, proposed = proposal_logits.topk(branch, dim=1)
            effective_branch = branch
        selected_effects = effects[proposed]
        children = states[:, None].expand(-1, effective_branch, -1, -1).gather(
            -1, selected_effects.long()
        )
        child_histories = histories[:, None].expand(-1, effective_branch, -1).clone()
        child_histories[..., :-1] = child_histories[..., 1:].clone()
        child_histories[..., -1] = proposed.to(torch.int16)
        flat_children = children.flatten(0, 1)
        flat_histories = child_histories.flatten(0, 1)
        parent_indices = torch.arange(len(states), device=device).repeat_interleave(
            effective_branch
        )
        flat_actions = proposed.flatten()
        local_nll = -proposed_logp.flatten()
        child_nll = cumulative_nll[parent_indices] + local_nll
        expanded += len(flat_children)
        solved = flat_children.eq(identity).all(dim=(1, 2))
        if solved.any():
            solved_indices = torch.flatnonzero(solved)
            best = solved_indices[child_nll[solved_indices].argmin()]
            parent = int(parent_indices[best])
            result = paths[parent, : depth - 1].long().cpu().tolist()
            result.append(int(flat_actions[best]))
            return result, {
                "depth": depth,
                "elapsed_seconds": time.perf_counter() - started,
                "expanded_children": expanded,
            }
        child_logits, child_values = predict(
            model, flat_children, flat_histories, inference_batch_size
        )
        policy_cost = child_nll if policy_mode == "cumulative" else local_nll
        ranks = value_weight * child_values.clamp_min(0) + policy_weight * policy_cost
        if depth > 1 and exhaustive and not allow_backtrack:
            inverse = torch.bitwise_xor(last_actions[parent_indices], 1)
            ranks = ranks.masked_fill(flat_actions.eq(inverse), torch.inf)
        flat = flat_children.reshape(len(flat_children), -1).long()
        hashes = state_zobrist[positions, flat].sum(dim=1)
        hashes = hashes + history_zobrist[
            history_positions[None], flat_histories.long()
        ].sum(dim=1)
        order = torch.argsort(ranks)
        ordered_hashes = hashes[order].cpu().numpy()
        keep_cpu: list[int] = []
        seen: set[int] = set()
        for ordered_position, node_hash in enumerate(ordered_hashes):
            key = int(node_hash)
            if key in seen:
                continue
            seen.add(key)
            keep_cpu.append(ordered_position)
            if len(keep_cpu) == beam_width:
                break
        keep = order[torch.as_tensor(keep_cpu, device=device)]
        selected_parents = parent_indices[keep]
        states = flat_children[keep]
        histories = flat_histories[keep]
        logits = child_logits[keep]
        values = child_values[keep]
        cumulative_nll = child_nll[keep]
        last_actions = flat_actions[keep]
        next_paths = paths[selected_parents].clone()
        next_paths[:, depth - 1] = last_actions.to(torch.int8)
        paths = next_paths
        if depth == 1 or depth % 20 == 0:
            print(
                json.dumps(
                    {
                        "beam": len(states),
                        "depth": depth,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "minimum_value": round(float(values.min()), 3),
                        "minimum_rank": round(float(ranks[keep].min()), 3),
                    }
                ),
                flush=True,
            )
    return None, {
        "depth": maximum_depth,
        "elapsed_seconds": time.perf_counter() - started,
        "expanded_children": expanded,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.exhaustive and not 1 <= args.branch <= 36:
        raise ValueError("branch must be in 1..36")
    device = torch.device("cuda")
    model, checkpoint = load_primitive_history_checkpoint(args.checkpoint, device)
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    orbit_positions = np.asarray(checkpoint["orbit_positions"], dtype=np.int64)
    global_to_local = np.asarray(checkpoint["global_to_local"], dtype=np.uint8)
    effects = torch.from_numpy(
        generator_effects(puzzle, orbit_positions, global_to_local)
    ).to(device)
    identity_np = np.broadcast_to(
        np.arange(model.config.orbit_size, dtype=np.uint8),
        (model.config.orbit_count, model.config.orbit_size),
    ).copy()
    identity = torch.from_numpy(identity_np).to(device)
    rng = np.random.default_rng(args.seed)
    state_zobrist = torch.from_numpy(
        rng.integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=(identity_np.size, model.config.orbit_size),
            dtype=np.int64,
        )
    ).to(device)
    history_zobrist = torch.from_numpy(
        rng.integers(
            np.iinfo(np.int64).min,
            np.iinfo(np.int64).max,
            size=(model.config.history_length, model.config.action_count + 1),
            dtype=np.int64,
        )
    ).to(device)
    with np.load(args.memory_model, allow_pickle=False) as memory:
        raw_states = memory["states"].astype(np.uint8, copy=False)
        offsets = memory["offsets"].astype(np.int64, copy=False)
        distances = memory["distances"].astype(np.int64, copy=False)
    rows: list[dict[str, object]] = []
    for pid in [int(value) for value in args.pids.split(",") if value.strip()]:
        index = int(offsets[pid])
        raw_initial = raw_states[index]
        initial = global_to_local[raw_initial[orbit_positions]]
        path_ids, stats = solve(
            model,
            initial,
            effects,
            identity,
            beam_width=args.beam,
            branch=args.branch,
            maximum_depth=args.maximum_depth,
            policy_weight=args.policy_weight,
            value_weight=args.value_weight,
            policy_mode=args.policy_mode,
            exhaustive=args.exhaustive,
            allow_backtrack=args.allow_backtrack,
            inference_batch_size=args.inference_batch_size,
            state_zobrist=state_zobrist,
            history_zobrist=history_zobrist,
        )
        path = None if path_ids is None else [puzzle.move_names[action] for action in path_ids]
        replay_ok = path is not None and puzzle.apply_path(tuple(raw_initial), path) == puzzle.solved_state
        if path is not None and not replay_ok:
            raise AssertionError(f"PID {pid}: history beam path failed replay")
        row = {
            **stats,
            "pid": pid,
            "replay_verified": replay_ok,
            "solved": path is not None,
            "teacher_moves": int(distances[index]),
            "moves": None if path is None else len(path),
            "path": None if path is None else ".".join(path),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {
        "beam": args.beam,
        "branch": 36 if args.exhaustive else args.branch,
        "checkpoint": str(args.checkpoint),
        "maximum_depth": args.maximum_depth,
        "policy_mode": args.policy_mode,
        "policy_weight": args.policy_weight,
        "rows": rows,
        "solved": sum(bool(row["solved"]) for row in rows),
        "value_weight": args.value_weight,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
