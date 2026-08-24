"""Extract on-policy KMC DAgger labels, including residual valley crossings."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "cube_nnn" / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.frames import FrameTransform  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    MacroTeacherDataset,
    cluster_costs,
    save_macro_action_library,
)
from cube666.macros import (  # noqa: E402
    MacroEffect,
    analyze_corner_fixing_macro,
    invert_path,
    state_cluster_permutations,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.symmetry import build_symmetries  # noqa: E402


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    current_cluster_costs: np.ndarray
    next_cost: int
    effect_key: bytes
    source_pid: int
    source: str
    prefix_depth: int
    remaining_moves: int
    segment_moves: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT / "cayley-py-666-cube",
    )
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--maximum-segment-moves", type=int, default=64)
    parser.add_argument(
        "--search-target",
        choices=("macro-steps", "primitive-moves"),
        default="macro-steps",
    )
    parser.add_argument(
        "--symmetry-indices",
        default="3,18,35,1,40,25,7",
        help="comma-separated verified spatial frames added to identity",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.maximum_segment_moves <= 0:
        raise ValueError("maximum-segment-moves must be positive")
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    with np.load(args.frontier_dir / "frontiers.npz", allow_pickle=False) as payload:
        full_states = payload["full_states"].astype(np.uint8, copy=False)
        source_pids = payload["source_pids"].astype(np.int32, copy=False)
    frontier_report = json.loads((args.frontier_dir / "report.json").read_text(encoding="utf-8"))
    frontier_rows = frontier_report["rows"]
    if full_states.shape != (len(frontier_rows), puzzle.size):
        raise ValueError("frontier states do not match report")

    symmetry_indices = tuple(
        int(token) for token in args.symmetry_indices.split(",") if token.strip()
    )
    frames: list[FrameTransform | None] = [None]
    if symmetry_indices:
        geometric_cube = NCube.from_puzzle_info(args.data_dir / "puzzle_info.json")
        symmetry_table = build_symmetries(geometric_cube)
        if any(index < 0 or index >= len(symmetry_table.perms) for index in symmetry_indices):
            raise ValueError("symmetry index is outside the verified frame table")
        for index in symmetry_indices:
            frames.append(
                FrameTransform(
                    label=f"sym{index:02d}",
                    move_names=puzzle.move_names,
                    sticker_permutation=symmetry_table.perms[index],
                    move_to_transformed=symmetry_table.relabel[index],
                )
            )

    known_moves = set(puzzle.generators)
    macro_by_effect: dict[bytes, MacroEffect] = {}
    transitions: list[Transition] = []
    skipped: Counter[str] = Counter()
    path_lengths: list[int] = []
    boundary_counts: list[int] = []
    replayed_queries = 0
    for index, frontier_row in enumerate(frontier_rows):
        query_id = str(frontier_row["query_id"])
        path_file = args.query_dir / "paths" / f"{query_id}.path.txt"
        if not path_file.exists():
            skipped["missing_query_path"] += 1
            continue
        path = tuple(
            token
            for token in path_file.read_text(encoding="utf-8").strip().split(".")
            if token
        )
        unknown = sorted(set(path).difference(known_moves))
        if unknown:
            raise ValueError(f"{query_id}: unknown moves {unknown}")
        start_state = tuple(int(value) for value in full_states[index])
        if puzzle.apply_path(start_state, path) != puzzle.solved_state:
            raise ValueError(f"{query_id}: KMC teacher path failed full replay")
        path_lengths.append(len(path))
        replayed_queries += 1

        for frame in frames:
            frame_label = "identity" if frame is None else frame.label
            state = start_state if frame is None else frame.transform_state(start_state)
            framed_path = path if frame is None else frame.transform_path(path)
            if puzzle.apply_path(state, framed_path) != puzzle.solved_state:
                raise AssertionError(f"{query_id}:{frame_label}: symmetry replay failed")
            boundaries: list[tuple[int, tuple[int, ...], int]] = []
            current = state
            for prefix in range(len(framed_path) + 1):
                corners_solved = all(
                    current[position] == puzzle.solved_state[position]
                    for position in decomposition.corner_orbit
                )
                if corners_solved:
                    report = residual_report(current, puzzle.solved_state, decomposition)
                    if report.all_even:
                        if report.unrestricted_three_cycles is None:
                            raise AssertionError("even normalized boundary lacks an exact cost")
                        boundaries.append((prefix, current, int(report.unrestricted_three_cycles)))
                if prefix < len(framed_path):
                    current = puzzle.apply_move(current, framed_path[prefix])
            if not boundaries or boundaries[0][0] != 0 or boundaries[-1][0] != len(framed_path):
                raise AssertionError(f"{query_id}:{frame_label}: missing endpoint boundary")
            boundary_counts.append(len(boundaries))

            for before, after in zip(boundaries, boundaries[1:], strict=False):
                before_prefix, before_state, before_cost = before
                after_prefix, after_state, after_cost = after
                segment = framed_path[before_prefix:after_prefix]
                if not segment or len(segment) > args.maximum_segment_moves:
                    skipped["segment_too_long"] += 1
                    continue
                if before_state == after_state:
                    skipped["identity_transition"] += 1
                    continue
                try:
                    macro = analyze_corner_fixing_macro(segment, puzzle.generators, decomposition)
                except ValueError:
                    skipped["invalid_boundary_macro"] += 1
                    continue
                effect = np.asarray(macro.cluster_permutations, dtype=np.uint8)
                identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24))
                if np.array_equal(effect, identity):
                    skipped["identity_effect"] += 1
                    continue
                effect_key = effect.tobytes()
                old = macro_by_effect.get(effect_key)
                if old is None or (len(macro.path), macro.path) < (len(old.path), old.path):
                    macro_by_effect[effect_key] = macro
                cluster_state = np.asarray(
                    state_cluster_permutations(before_state, puzzle.solved_state, decomposition),
                    dtype=np.uint8,
                )
                exact_costs = cluster_costs(cluster_state).astype(np.int16)
                if int(exact_costs.sum()) != before_cost:
                    raise AssertionError("cluster projection cost disagrees with residual report")
                transitions.append(
                    Transition(
                        state=cluster_state,
                        current_cluster_costs=exact_costs,
                        next_cost=after_cost,
                        effect_key=effect_key,
                        source_pid=int(source_pids[index]),
                        source=f"{query_id}:{frame_label}",
                        prefix_depth=before_prefix,
                        remaining_moves=len(framed_path) - before_prefix,
                        segment_moves=len(segment),
                    )
                )

    if not transitions:
        raise RuntimeError("no normalized KMC transitions were extracted")
    for macro in tuple(macro_by_effect.values()):
        inverse = analyze_corner_fixing_macro(
            invert_path(macro.path),
            puzzle.generators,
            decomposition,
        )
        inverse_key = np.asarray(inverse.cluster_permutations, dtype=np.uint8).tobytes()
        old = macro_by_effect.get(inverse_key)
        if old is None or (len(inverse.path), inverse.path) < (len(old.path), old.path):
            macro_by_effect[inverse_key] = inverse
    macros = tuple(
        sorted(
            macro_by_effect.values(),
            key=lambda macro: (len(macro.path), macro.path, macro.cluster_permutations),
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = save_macro_action_library(args.out_dir / "action_library.json", macros)
    action_by_effect = {
        effect.tobytes(): action for action, effect in enumerate(table.effects)
    }

    by_source: defaultdict[str, list[Transition]] = defaultdict(list)
    for transition in transitions:
        by_source[transition.source].append(transition)
    remaining_macro_steps: dict[tuple[str, int], int] = {}
    for source, source_rows in by_source.items():
        ordered = sorted(source_rows, key=lambda transition: transition.prefix_depth)
        for position, transition in enumerate(ordered):
            remaining_macro_steps[(source, transition.prefix_depth)] = (
                len(ordered) - position
            )

    grouped: defaultdict[tuple[int, bytes], list[Transition]] = defaultdict(list)
    for transition in transitions:
        grouped[(transition.source_pid, transition.state.tobytes())].append(transition)
    teacher_rows: list[tuple[Transition, list[int]]] = []
    for group in grouped.values():
        best_remaining = min(transition.remaining_moves for transition in group)
        best = [transition for transition in group if transition.remaining_moves == best_remaining]
        best_next = min(transition.next_cost for transition in best)
        best = [transition for transition in best if transition.next_cost == best_next]
        actions = sorted({action_by_effect[transition.effect_key] for transition in best})
        teacher_rows.append((best[0], actions))
    teacher_rows.sort(
        key=lambda item: (
            item[0].source_pid,
            item[0].source,
            item[0].prefix_depth,
            item[0].state.tobytes(),
        )
    )
    max_labels = max(len(actions) for _, actions in teacher_rows)
    states = np.stack([transition.state for transition, _ in teacher_rows])
    labels = np.full((len(teacher_rows), max_labels), -1, dtype=np.int32)
    counts = np.empty(len(teacher_rows), dtype=np.int16)
    for index, (_, actions) in enumerate(teacher_rows):
        labels[index, : len(actions)] = actions
        counts[index] = len(actions)
    search_values = np.asarray(
        [
            (
                transition.remaining_moves
                if args.search_target == "primitive-moves"
                else remaining_macro_steps[(transition.source, transition.prefix_depth)]
            )
            for transition, _ in teacher_rows
        ],
        dtype=np.float32,
    )
    dataset = MacroTeacherDataset(
        states=states,
        teacher_actions=labels,
        teacher_action_counts=counts,
        cluster_cost_targets=np.stack(
            [transition.current_cluster_costs for transition, _ in teacher_rows]
        ),
        teacher_next_costs=np.asarray(
            [transition.next_cost for transition, _ in teacher_rows],
            dtype=np.int16,
        ),
        walk_depths=np.asarray(
            [transition.prefix_depth for transition, _ in teacher_rows],
            dtype=np.int16,
        ),
        action_digest=table.digest,
        allow_non_improving=True,
        search_value_targets=search_values,
        search_cluster_targets=np.broadcast_to(
            (search_values / 6.0)[:, None],
            (len(search_values), 6),
        ).copy(),
    )
    dataset.validate(table.action_count)
    dataset.save(args.out_dir / "teacher.npz")
    atomic_npy(
        args.out_dir / "source_state_ids.npy",
        np.asarray(
            [transition.source_pid for transition, _ in teacher_rows],
            dtype=np.int32,
        ),
    )
    current_costs = dataset.cluster_cost_targets.sum(axis=1)
    deltas = current_costs - dataset.teacher_next_costs
    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "improving_samples": int(np.count_nonzero(deltas > 0)),
        "maximum_boundary_count": max(boundary_counts),
        "maximum_path_moves": max(path_lengths),
        "maximum_segment_moves": args.maximum_segment_moves,
        "mean_boundary_count": float(np.mean(boundary_counts)),
        "mean_path_moves": float(np.mean(path_lengths)),
        "non_improving_samples": int(np.count_nonzero(deltas <= 0)),
        "plateau_samples": int(np.count_nonzero(deltas == 0)),
        "raw_transitions": len(transitions),
        "replayed_queries": replayed_queries,
        "search_target": args.search_target,
        "samples": dataset.sample_count,
        "maximum_search_value": float(search_values.max()),
        "mean_search_value": float(search_values.mean()),
        "skipped": dict(sorted(skipped.items())),
        "source_pids": sorted(set(int(value) for value in source_pids)),
        "symmetry_indices": list(symmetry_indices),
        "total_frames": len(frames),
        "uphill_samples": int(np.count_nonzero(deltas < 0)),
    }
    temporary = (args.out_dir / "report.json").with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out_dir / "report.json")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
