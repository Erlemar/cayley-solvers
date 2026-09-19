"""Extract strong corner/parity-preserving macro transitions from KMC paths."""

from __future__ import annotations

import argparse
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
    source: str
    source_state_id: int
    prefix_depth: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument("--results-root", type=Path, default=PROJECT / "cube666" / "results")
    parser.add_argument("--minimum-improvement", type=int, default=2)
    parser.add_argument("--maximum-segment-moves", type=int, default=20)
    parser.add_argument(
        "--symmetry-indices",
        default="",
        help="optional comma-separated spatial symmetry indices added to identity",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "training" / "kmc_macro_teacher_v1",
    )
    return parser.parse_args()


def path_from_log(path: Path, known_moves: set[str]) -> tuple[str, ...]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("empty stdout log")
    moves = tuple(token for token in lines[-1].split(".") if token)
    unknown = set(moves).difference(known_moves)
    if unknown:
        raise ValueError(f"unknown moves: {sorted(unknown)}")
    return moves


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    states_by_id = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))
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
    macro_by_effect: dict[bytes, MacroEffect] = {}
    transitions: list[Transition] = []
    failure_counts: defaultdict[str, int] = defaultdict(int)
    scanned_logs = 0
    replayed_logs = 0

    for log_path in sorted(args.results_root.rglob("*.stdout.log")):
        scanned_logs += 1
        state_id = str(int(log_path.name.split(".", 1)[0]))
        state = states_by_id.get(state_id)
        if state is None:
            failure_counts["unknown_state_id"] += 1
            continue
        try:
            path = path_from_log(log_path, set(puzzle.generators))
        except (OSError, ValueError):
            failure_counts["invalid_log"] += 1
            continue

        for frame in frames:
            framed_state = state if frame is None else frame.transform_state(state)
            framed_path = path if frame is None else frame.transform_path(path)
            frame_label = "identity" if frame is None else frame.label
            boundary_states: list[tuple[int, tuple[int, ...], int]] = []
            current = framed_state
            initial_report = residual_report(current, puzzle.solved_state, decomposition)
            initial_corners = all(
                current[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            )
            if initial_corners and initial_report.all_even:
                if initial_report.unrestricted_three_cycles is None:
                    raise AssertionError("even boundary has no exact residual")
                boundary_states.append((0, current, initial_report.unrestricted_three_cycles))
            for prefix, move in enumerate(framed_path, start=1):
                current = puzzle.apply_move(current, move)
                corners_solved = all(
                    current[position] == puzzle.solved_state[position]
                    for position in decomposition.corner_orbit
                )
                if not corners_solved:
                    continue
                report = residual_report(current, puzzle.solved_state, decomposition)
                if report.all_even:
                    if report.unrestricted_three_cycles is None:
                        raise AssertionError("even boundary has no exact residual")
                    boundary_states.append((prefix, current, report.unrestricted_three_cycles))
            if not boundary_states:
                failure_counts["no_normalized_boundary"] += 1
                continue
            replayed_logs += 1

            for before, after in zip(boundary_states, boundary_states[1:], strict=False):
                before_prefix, before_state, before_cost = before
                after_prefix, _, after_cost = after
                segment = framed_path[before_prefix:after_prefix]
                improvement = before_cost - after_cost
                if improvement < args.minimum_improvement:
                    continue
                if not segment or len(segment) > args.maximum_segment_moves:
                    continue
                try:
                    macro = analyze_corner_fixing_macro(segment, puzzle.generators, decomposition)
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
                transitions.append(
                    Transition(
                        state=cluster_state,
                        current_cluster_costs=cluster_costs(cluster_state).astype(np.int16),
                        next_cost=after_cost,
                        effect_key=effect_key,
                        source=f"{log_path.relative_to(args.results_root)}:{frame_label}",
                        source_state_id=int(state_id),
                        prefix_depth=before_prefix,
                    )
                )

    if not transitions:
        raise RuntimeError("no improving KMC boundary transitions were found")
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
        effect.tobytes(): action
        for action, effect in enumerate(table.effects)
    }

    grouped: defaultdict[tuple[int, bytes], list[Transition]] = defaultdict(list)
    for transition in transitions:
        grouped[(transition.source_state_id, transition.state.tobytes())].append(transition)
    rows: list[tuple[Transition, list[int]]] = []
    for group in grouped.values():
        best_next = min(transition.next_cost for transition in group)
        best = [transition for transition in group if transition.next_cost == best_next]
        actions = sorted({action_by_effect[transition.effect_key] for transition in best})
        rows.append((best[0], actions))
    rows.sort(key=lambda row: row[0].state.tobytes())
    max_labels = max(len(actions) for _, actions in rows)
    states = np.stack([transition.state for transition, _ in rows])
    labels = np.full((len(rows), max_labels), -1, dtype=np.int32)
    counts = np.empty(len(rows), dtype=np.int16)
    for index, (_, actions) in enumerate(rows):
        labels[index, : len(actions)] = actions
        counts[index] = len(actions)
    dataset = MacroTeacherDataset(
        states=states,
        teacher_actions=labels,
        teacher_action_counts=counts,
        cluster_cost_targets=np.stack(
            [transition.current_cluster_costs for transition, _ in rows]
        ),
        teacher_next_costs=np.asarray(
            [transition.next_cost for transition, _ in rows],
            dtype=np.int16,
        ),
        walk_depths=np.asarray(
            [transition.prefix_depth for transition, _ in rows],
            dtype=np.int16,
        ),
        action_digest=table.digest,
    )
    dataset.validate(table.action_count)
    dataset.save(args.out_dir / "teacher.npz")
    group_path = args.out_dir / "source_state_ids.npy"
    group_temporary = group_path.with_suffix(group_path.suffix + ".tmp")
    with open(group_temporary, "wb") as handle:
        np.save(
            handle,
            np.asarray(
                [transition.source_state_id for transition, _ in rows],
                dtype=np.int32,
            ),
        )
    group_temporary.replace(group_path)
    report = {
        "action_count": table.action_count,
        "action_digest": table.digest,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "failure_counts": dict(sorted(failure_counts.items())),
        "raw_transitions": len(transitions),
        "replayed_logs": replayed_logs,
        "samples": dataset.sample_count,
        "scanned_logs": scanned_logs,
        "sources": len({transition.source for transition in transitions}),
        "symmetry_indices": list(symmetry_indices),
        "total_frames": len(frames),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
