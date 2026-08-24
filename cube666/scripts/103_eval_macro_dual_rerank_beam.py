"""Two-stage full-library macro beam: dual retrieval then exact child-value rerank."""

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
from cube666.macro_data import _three_cycle_units_batch_unchecked  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--pids", required=True)
    parser.add_argument("--beam", type=int, default=128)
    parser.add_argument("--shortlist", type=int, default=1024)
    parser.add_argument("--maximum-steps", type=int, default=20)
    parser.add_argument("--exact-cost-weight", type=float, default=0.0)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--state-query-batch-size", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def predict_values(
    model: torch.nn.Module,
    states: np.ndarray,
    batch_size: int,
    device: torch.device,
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


@torch.inference_mode()
def retrieve(
    model: torch.nn.Module,
    states: np.ndarray,
    action_keys: torch.Tensor,
    topk: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        batch = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            queries = model.encode_states(batch)
        scores = queries @ action_keys.transpose(0, 1)
        output.append(scores.topk(topk, dim=1).indices.cpu().numpy())
    return np.concatenate(output).astype(np.int32, copy=False)


def solve(
    policy: torch.nn.Module,
    value_model: torch.nn.Module,
    initial: np.ndarray,
    action_keys: torch.Tensor,
    effects: np.ndarray,
    costs: np.ndarray,
    *,
    beam_width: int,
    shortlist: int,
    maximum_steps: int,
    inference_batch_size: int,
    state_query_batch_size: int,
    exact_cost_weight: float,
    device: torch.device,
) -> dict[str, object]:
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24))
    states = initial[None].copy()
    paths: list[tuple[int, ...]] = [()]
    path_costs = np.zeros(1, dtype=np.int32)
    best_g = {initial.tobytes(): 0}
    best_solution: tuple[int, tuple[int, ...]] | None = None
    best_exact: tuple[int, int, tuple[int, ...]] | None = None
    generated = 0
    started = time.perf_counter()
    for step in range(1, maximum_steps + 1):
        proposed = retrieve(
            policy,
            states,
            action_keys,
            shortlist,
            state_query_batch_size,
            device,
        )
        parents = np.repeat(np.arange(len(states), dtype=np.int32), shortlist)
        actions = proposed.reshape(-1)
        children = np.take_along_axis(states[parents], effects[actions], axis=-1)
        child_costs = path_costs[parents] + costs[actions]
        generated += len(children)
        solved = np.all(children == identity, axis=(1, 2))
        for position in np.flatnonzero(solved):
            parent = int(parents[position])
            candidate = (
                int(child_costs[position]),
                paths[parent] + (int(actions[position]),),
            )
            if best_solution is None or candidate < best_solution:
                best_solution = candidate
        predicted = predict_values(
            value_model, children, inference_batch_size, device
        )
        if exact_cost_weight:
            exact_costs = _three_cycle_units_batch_unchecked(children).sum(axis=1)
            exact_position = int(np.argmin(exact_costs))
            exact_parent = int(parents[exact_position])
            exact_candidate = (
                int(exact_costs[exact_position]),
                int(child_costs[exact_position]),
                paths[exact_parent] + (int(actions[exact_position]),),
            )
            if best_exact is None or exact_candidate < best_exact:
                best_exact = exact_candidate
        else:
            exact_costs = np.zeros(len(children), dtype=np.int16)
        ranks = child_costs + predicted + exact_cost_weight * exact_costs
        order = np.lexsort((child_costs, ranks))
        next_states: list[np.ndarray] = []
        next_paths: list[tuple[int, ...]] = []
        next_costs: list[int] = []
        for position in order:
            if solved[position]:
                continue
            cost = int(child_costs[position])
            if best_solution is not None and cost >= best_solution[0]:
                continue
            key = children[position].tobytes()
            old = best_g.get(key)
            if old is not None and old <= cost:
                continue
            best_g[key] = cost
            parent = int(parents[position])
            next_states.append(children[position])
            next_paths.append(paths[parent] + (int(actions[position]),))
            next_costs.append(cost)
            if len(next_states) == beam_width:
                break
        if step == 1 or step % 5 == 0 or best_solution is not None:
            print(
                json.dumps(
                    {
                        "beam": len(next_states),
                        "best_solution": None
                        if best_solution is None
                        else best_solution[0],
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "generated": generated,
                        "minimum_exact_cost": None
                        if not exact_cost_weight
                        else int(exact_costs.min()),
                        "minimum_rank": None
                        if not len(order)
                        else round(float(ranks[order[0]]), 3),
                        "step": step,
                    }
                ),
                flush=True,
            )
        if best_solution is not None:
            break
        if not next_states:
            break
        states = np.stack(next_states)
        paths = next_paths
        path_costs = np.asarray(next_costs, dtype=np.int32)
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "generated": generated,
        "best_exact_residual": None if best_exact is None else best_exact[0],
        "best_exact_primitive_cost": None if best_exact is None else best_exact[1],
        "best_exact_path": None if best_exact is None else list(best_exact[2]),
        "macro_steps": None if best_solution is None else len(best_solution[1]),
        "path": None if best_solution is None else list(best_solution[1]),
        "primitive_cost": None if best_solution is None else best_solution[0],
        "solved": best_solution is not None,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    policy, _ = load_macro_dual_policy_checkpoint(args.checkpoint, device)
    value_model, _ = load_macro_factorized_value_checkpoint(
        args.value_checkpoint, device
    )
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.int32, copy=False
    )
    key_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(effects), args.inference_batch_size):
            stop = min(start + args.inference_batch_size, len(effects))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                key_parts.append(
                    policy.encode_actions(
                        torch.from_numpy(effects[start:stop]).to(device),
                        torch.from_numpy(costs[start:stop]).to(device),
                    )
                )
    action_keys = torch.cat(key_parts)
    rows: list[dict[str, object]] = []
    for pid in (int(token) for token in args.pids.split(",") if token.strip()):
        candidates = np.flatnonzero(groups == pid)
        if not len(candidates):
            raise ValueError(f"PID {pid} is absent from the teacher")
        row = int(candidates[np.argmax(values[candidates])])
        result = solve(
            policy,
            value_model,
            states[row],
            action_keys,
            effects,
            costs,
            beam_width=args.beam,
            shortlist=args.shortlist,
            maximum_steps=args.maximum_steps,
            inference_batch_size=args.inference_batch_size,
            state_query_batch_size=args.state_query_batch_size,
            exact_cost_weight=args.exact_cost_weight,
            device=device,
        )
        result.update(
            {
                "pid": pid,
                "row": row,
                "teacher_upper_bound": float(values[row]),
            }
        )
        rows.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    report = {
        "beam": args.beam,
        "checkpoint": str(args.checkpoint),
        "exact_cost_weight": args.exact_cost_weight,
        "rows": rows,
        "shortlist": args.shortlist,
        "solved": sum(bool(row["solved"]) for row in rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
