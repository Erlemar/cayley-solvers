"""Materialize a cluster-beam action path as a full replay-verified cube frontier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macros import state_cluster_permutations  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--beam-report", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def full_state_from_clusters(
    clusters: np.ndarray,
    puzzle: Cube666Puzzle,
    decomposition: object,
) -> tuple[int, ...]:
    state = list(puzzle.solved_state)
    for permutation, orbit in zip(
        clusters[: len(decomposition.center_orbits)],
        decomposition.center_orbits,
        strict=True,
    ):
        for destination, source in enumerate(permutation):
            state[orbit[destination]] = puzzle.solved_state[orbit[int(source)]]
    wing_offset = len(decomposition.center_orbits)
    for permutation, pair in zip(
        clusters[wing_offset:], decomposition.wing_pairs, strict=True
    ):
        for destination, source in enumerate(permutation):
            left_destination = pair.left[destination]
            left_source = pair.left[int(source)]
            right_destination = pair.right_position_by_left_index[destination]
            right_source = pair.right_position_by_left_index[int(source)]
            state[left_destination] = puzzle.solved_state[left_source]
            state[right_destination] = puzzle.solved_state[right_source]
    return tuple(state)


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    report = json.loads(args.beam_report.read_text(encoding="utf-8"))
    matching = [row for row in report["rows"] if int(row["pid"]) == args.pid]
    if len(matching) != 1:
        raise ValueError(f"expected one beam row for PID {args.pid}")
    beam_row = matching[0]
    row_index = int(beam_row["row"])
    action_ids = tuple(int(action) for action in beam_row["best_exact_path"])
    if not action_ids:
        raise ValueError("beam report has no exact frontier path")
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        cluster_state = teacher["states"][row_index].astype(np.uint8, copy=False)
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
    full_state = full_state_from_clusters(cluster_state, puzzle, decomposition)
    projected = np.asarray(
        state_cluster_permutations(full_state, puzzle.solved_state, decomposition),
        dtype=np.uint8,
    )
    if not np.array_equal(projected, cluster_state):
        raise AssertionError("synthetic full state does not match the cluster projection")
    macro_path = tuple(move for action in action_ids for move in paths[action])
    frontier = puzzle.apply_path(full_state, macro_path)
    expected = cluster_state.copy()
    for action in action_ids:
        expected = np.take_along_axis(expected, effects[action], axis=-1)
    frontier_clusters = np.asarray(
        state_cluster_permutations(frontier, puzzle.solved_state, decomposition),
        dtype=np.uint8,
    )
    if not np.array_equal(frontier_clusters, expected):
        raise AssertionError("full-state macro replay disagrees with cluster replay")
    frontier_report = residual_report(frontier, puzzle.solved_state, decomposition)
    if not frontier_report.all_even:
        raise AssertionError("frontier cluster parity is not even")
    query_id = f"pid{args.pid:04d}_modelres{frontier_report.unrestricted_three_cycles:02d}"
    row = {
        "beam_action_ids": list(action_ids),
        "frontier_residual": int(frontier_report.unrestricted_three_cycles),
        "macro_primitive_moves": len(macro_path),
        "pid": args.pid,
        "query_id": query_id,
        "replay_verified": True,
        "row": row_index,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "frontiers.npz",
        full_states=np.asarray([frontier], dtype=np.uint8),
        cluster_states=frontier_clusters[None],
        source_pids=np.asarray([args.pid], dtype=np.int32),
        initial_residuals=np.asarray(
            [int(residual_report(full_state, puzzle.solved_state, decomposition).unrestricted_three_cycles)],
            dtype=np.int16,
        ),
        frontier_residuals=np.asarray(
            [int(frontier_report.unrestricted_three_cycles)], dtype=np.int16
        ),
    )
    output = {"rows": [row], "source_beam_report": str(args.beam_report)}
    (args.out_dir / "report.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
