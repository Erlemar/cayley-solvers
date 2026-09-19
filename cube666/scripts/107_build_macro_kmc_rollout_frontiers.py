"""Build top one-step macro children for exact KMC completion-cost labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macro_dual_policy import load_macro_dual_policy_checkpoint  # noqa: E402
from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402
from cube666.macros import state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--shortlist", type=int, default=4096)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def full_state_from_clusters(
    clusters: np.ndarray, puzzle: Cube666Puzzle, decomposition: object
) -> tuple[int, ...]:
    state = list(puzzle.solved_state)
    for permutation, orbit in zip(
        clusters[: len(decomposition.center_orbits)],
        decomposition.center_orbits,
        strict=True,
    ):
        for destination, source in enumerate(permutation):
            state[orbit[destination]] = puzzle.solved_state[orbit[int(source)]]
    offset = len(decomposition.center_orbits)
    for permutation, pair in zip(
        clusters[offset:], decomposition.wing_pairs, strict=True
    ):
        for destination, source in enumerate(permutation):
            left_destination = pair.left[destination]
            left_source = pair.left[int(source)]
            right_destination = pair.right_position_by_left_index[destination]
            right_source = pair.right_position_by_left_index[int(source)]
            state[left_destination] = puzzle.solved_state[left_source]
            state[right_destination] = puzzle.solved_state[right_source]
    return tuple(state)


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
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    policy, _ = load_macro_dual_policy_checkpoint(args.checkpoint, device)
    value_model, _ = load_macro_factorized_value_checkpoint(
        args.value_checkpoint, device
    )
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(
        np.uint8, copy=False
    )
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(
        np.int32, copy=False
    )
    library = json.loads(
        (args.teacher_dir / "action_library.json").read_text(encoding="utf-8")
    )
    paths = library["paths"]
    pid_rows = np.flatnonzero(groups == args.pid)
    if not len(pid_rows):
        raise ValueError(f"PID {args.pid} is absent from the teacher")
    row_index = int(pid_rows[np.argmax(values[pid_rows])])
    state = states[row_index]
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            query = policy.encode_states(torch.from_numpy(state[None]).to(device))
        retrieved = (
            (query @ action_keys.transpose(0, 1))
            .topk(args.shortlist, dim=1)
            .indices[0]
            .cpu()
            .numpy()
        )
    children = np.take_along_axis(
        np.broadcast_to(state, (len(retrieved),) + state.shape),
        effects[retrieved],
        axis=-1,
    )
    predicted = predict_values(
        value_model, children, args.inference_batch_size, device
    )
    q_scores = costs[retrieved] + predicted
    selected = retrieved[np.argsort(q_scores)[: args.candidates]]
    teacher_actions = labels[row_index, : int(counts[row_index])]
    selected = np.unique(np.concatenate((selected, teacher_actions))).astype(np.int32)
    full_state = full_state_from_clusters(state, puzzle, decomposition)
    projected = np.asarray(
        state_cluster_permutations(full_state, puzzle.solved_state, decomposition),
        dtype=np.uint8,
    )
    if not np.array_equal(projected, state):
        raise AssertionError("synthetic full state does not match the source clusters")
    full_frontiers: list[np.ndarray] = []
    cluster_frontiers: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for number, action in enumerate(selected):
        macro_path = tuple(paths[int(action)])
        frontier = puzzle.apply_path(full_state, macro_path)
        clusters = np.asarray(
            state_cluster_permutations(frontier, puzzle.solved_state, decomposition),
            dtype=np.uint8,
        )
        expected = np.take_along_axis(state, effects[action], axis=-1)
        if not np.array_equal(clusters, expected):
            raise AssertionError(f"action {action}: full/cluster replay mismatch")
        exact = residual_report(frontier, puzzle.solved_state, decomposition)
        if not exact.all_even:
            raise AssertionError(f"action {action}: frontier parity is not even")
        query_id = f"pid{args.pid:04d}_row{row_index:05d}_a{int(action):05d}"
        full_frontiers.append(np.asarray(frontier, dtype=np.uint8))
        cluster_frontiers.append(clusters)
        rows.append(
            {
                "action_id": int(action),
                "candidate_number": number,
                "frontier_residual": int(exact.unrestricted_three_cycles),
                "macro_primitive_moves": int(costs[action]),
                "pid": args.pid,
                "predicted_child_value": float(
                    predict_values(
                        value_model, clusters[None], args.inference_batch_size, device
                    )[0]
                ),
                "query_id": query_id,
                "replay_verified": True,
                "teacher_action": bool(action in set(teacher_actions.tolist())),
            }
        )
    initial_residual = int(
        residual_report(full_state, puzzle.solved_state, decomposition).unrestricted_three_cycles
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "frontiers.npz",
        full_states=np.stack(full_frontiers),
        cluster_states=np.stack(cluster_frontiers),
        source_pids=np.full(len(rows), args.pid, dtype=np.int32),
        initial_residuals=np.full(len(rows), initial_residual, dtype=np.int16),
        frontier_residuals=np.asarray(
            [int(row["frontier_residual"]) for row in rows], dtype=np.int16
        ),
    )
    report = {
        "rows": rows,
        "source_row": row_index,
        "teacher_upper_bound": float(values[row_index]),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "candidates": len(rows),
                "source_row": row_index,
                "teacher_actions": teacher_actions.tolist(),
                "teacher_upper_bound": float(values[row_index]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
