"""Extract a PID-grouped macro teacher from a replay-verified cube666 submission."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
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
    source_state_id: int
    prefix_depth: int
    remaining_steps: int
    remaining_moves: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT / "cayley-py-666-cube",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument(
        "--indices",
        default="",
        help="optional comma-separated initial_state_id subset",
    )
    parser.add_argument("--minimum-improvement", type=int, default=2)
    parser.add_argument("--maximum-segment-moves", type=int, default=32)
    parser.add_argument(
        "--allow-non-improving",
        action="store_true",
        help="retain plateau and uphill normalized transitions from the exact path",
    )
    parser.add_argument(
        "--search-target",
        choices=("macro-steps", "primitive-moves"),
        default="macro-steps",
    )
    parser.add_argument(
        "--symmetry-indices",
        default="",
        help="optional comma-separated spatial symmetry indices added to the identity frame",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "training" / "submission_macro_teacher_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        (not args.allow_non_improving and args.minimum_improvement <= 0)
        or args.maximum_segment_moves <= 0
    ):
        raise ValueError("invalid improvement or segment-length threshold")
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    states_by_id = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
    known_moves = set(puzzle.generators)
    symmetry_indices = tuple(
        int(token) for token in args.symmetry_indices.split(",") if token.strip()
    )
    selected_indices = {
        int(token) for token in args.indices.split(",") if token.strip()
    }
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
    macro_by_effect: dict[bytes, MacroEffect] = {}
    transitions: list[Transition] = []
    failure_counts: defaultdict[str, int] = defaultdict(int)
    replayed_paths = 0

    with open(args.submission, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"initial_state_id", "path"}:
        raise ValueError("submission must contain initial_state_id,path")

    for row in rows:
        state_id = str(int(row["initial_state_id"]))
        if selected_indices and int(state_id) not in selected_indices:
            continue
        state = states_by_id.get(state_id)
        if state is None:
            failure_counts["unknown_state_id"] += 1
            continue
        original_path = tuple(token for token in row["path"].split(".") if token)
        if set(original_path).difference(known_moves):
            failure_counts["unknown_move"] += 1
            continue
        for frame in frames:
            framed_state = state if frame is None else frame.transform_state(state)
            path = original_path if frame is None else frame.transform_path(original_path)
            boundary_states: list[tuple[int, tuple[int, ...], int]] = []
            current = framed_state
            for prefix in range(len(path) + 1):
                corners_solved = all(
                    current[position] == puzzle.solved_state[position]
                    for position in decomposition.corner_orbit
                )
                if corners_solved:
                    report = residual_report(current, puzzle.solved_state, decomposition)
                    if report.all_even:
                        if report.unrestricted_three_cycles is None:
                            raise AssertionError("even boundary has no exact residual")
                        boundary_states.append(
                            (prefix, current, report.unrestricted_three_cycles)
                        )
                if prefix < len(path):
                    current = puzzle.apply_move(current, path[prefix])
            if current != puzzle.solved_state:
                failure_counts["replay_failed"] += 1
                continue
            replayed_paths += 1

            frame_rows: list[
                tuple[int, np.ndarray, int, int, bytes]
            ] = []
            for before, after in zip(boundary_states, boundary_states[1:], strict=False):
                before_prefix, before_state, before_cost = before
                after_prefix, _, after_cost = after
                segment = path[before_prefix:after_prefix]
                improvement = before_cost - after_cost
                if (
                    not args.allow_non_improving
                    and improvement < args.minimum_improvement
                ):
                    failure_counts["insufficient_improvement"] += 1
                    continue
                if not segment or len(segment) > args.maximum_segment_moves:
                    failure_counts["segment_too_long"] += 1
                    continue
                try:
                    macro = analyze_corner_fixing_macro(
                        segment,
                        puzzle.generators,
                        decomposition,
                    )
                except ValueError:
                    failure_counts["invalid_boundary_macro"] += 1
                    continue
                effect_key = np.asarray(macro.cluster_permutations, dtype=np.uint8).tobytes()
                old = macro_by_effect.get(effect_key)
                if old is None or (len(macro.path), macro.path) < (len(old.path), old.path):
                    macro_by_effect[effect_key] = macro
                cluster_state = np.asarray(
                    state_cluster_permutations(
                        before_state,
                        puzzle.solved_state,
                        decomposition,
                    ),
                    dtype=np.uint8,
                )
                frame_rows.append(
                    (
                        before_prefix,
                        cluster_state,
                        after_cost,
                        before_cost,
                        effect_key,
                    )
                )
            for position, (
                before_prefix,
                cluster_state,
                after_cost,
                before_cost,
                effect_key,
            ) in enumerate(frame_rows):
                exact_costs = cluster_costs(cluster_state).astype(np.int16)
                if int(exact_costs.sum()) != before_cost:
                    raise AssertionError("cluster projection cost disagrees with residual")
                transitions.append(
                    Transition(
                        state=cluster_state,
                        current_cluster_costs=exact_costs,
                        next_cost=after_cost,
                        effect_key=effect_key,
                        source_state_id=int(state_id),
                        prefix_depth=before_prefix,
                        remaining_steps=len(frame_rows) - position,
                        remaining_moves=len(path) - before_prefix,
                    )
                )

    if not transitions:
        raise RuntimeError("no improving normalized transitions were found")
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

    # Keep identical states from different source puzzles in separate groups so
    # a PID-held-out fold cannot inherit a validation label through deduplication.
    grouped: defaultdict[tuple[int, bytes], list[Transition]] = defaultdict(list)
    for transition in transitions:
        grouped[(transition.source_state_id, transition.state.tobytes())].append(transition)
    teacher_rows: list[tuple[Transition, list[int]]] = []
    for group in grouped.values():
        best_remaining = min(transition.remaining_steps for transition in group)
        best = [
            transition
            for transition in group
            if transition.remaining_steps == best_remaining
        ]
        best_next = min(transition.next_cost for transition in best)
        best = [transition for transition in best if transition.next_cost == best_next]
        actions = sorted({action_by_effect[transition.effect_key] for transition in best})
        teacher_rows.append((best[0], actions))
    teacher_rows.sort(
        key=lambda item: (item[0].source_state_id, item[0].prefix_depth, item[0].state.tobytes())
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
                else transition.remaining_steps
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
        allow_non_improving=args.allow_non_improving,
        search_value_targets=search_values,
        search_cluster_targets=np.broadcast_to(
            (search_values / 6.0)[:, None],
            (len(search_values), 6),
        ).copy(),
    )
    dataset.validate(table.action_count)
    dataset.save(args.out_dir / "teacher.npz")

    group_path = args.out_dir / "source_state_ids.npy"
    group_temporary = group_path.with_suffix(group_path.suffix + ".tmp")
    with open(group_temporary, "wb") as handle:
        np.save(
            handle,
            np.asarray(
                [transition.source_state_id for transition, _ in teacher_rows],
                dtype=np.int32,
            ),
        )
    group_temporary.replace(group_path)

    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "failure_counts": dict(sorted(failure_counts.items())),
        "allow_non_improving": args.allow_non_improving,
        "maximum_search_value": float(search_values.max()),
        "mean_search_value": float(search_values.mean()),
        "maximum_segment_moves": args.maximum_segment_moves,
        "minimum_improvement": args.minimum_improvement,
        "raw_transitions": len(transitions),
        "replayed_paths": replayed_paths,
        "search_target": args.search_target,
        "selected_indices": sorted(selected_indices),
        "samples": dataset.sample_count,
        "source_pids": len({transition.source_state_id for transition in transitions}),
        "submission": str(args.submission),
        "symmetry_indices": list(symmetry_indices),
        "total_frames": len(frames),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
