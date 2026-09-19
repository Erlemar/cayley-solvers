"""Replace classical boundary macros with shorter exact words from a learned library."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macros import analyze_corner_fixing_macro, reduce_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=PROJECT / "submissions" / "cube666_classical_merged.csv",
    )
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument(
        "--max-components",
        type=int,
        choices=(1, 2),
        default=2,
        help="maximum learned macros composed to reproduce one classical boundary effect",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "submissions" / "cube666_macro_effect_rewritten.csv",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "macro_effect_rewrite.json",
    )
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library, puzzle.generators, decomposition
    )
    action_by_effect = {
        effect.tobytes(): action for action, effect in enumerate(table.effects)
    }
    action_lengths = np.asarray([len(path) for path in table.paths], dtype=np.int16)
    minimum_action_length = int(action_lengths.min())
    states = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))

    with open(args.submission, newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows or set(source_rows[0]) != {"initial_state_id", "path"}:
        raise ValueError("submission must contain initial_state_id,path")

    output_rows: list[dict[str, str]] = []
    report_rows: list[dict[str, int]] = []
    source_total = 0
    output_total = 0
    replay_verified = 0
    candidate_replacements = 0
    accepted_replacements = 0
    rejected_full_effect = 0
    two_component_candidates = 0
    two_component_replacements = 0

    for row in source_rows:
        state_id = str(int(row["initial_state_id"]))
        initial_state = states[state_id]
        source_path = tuple(token for token in row["path"].split(".") if token)
        boundary_states: list[tuple[int, tuple[int, ...]]] = []
        current = initial_state
        for prefix in range(len(source_path) + 1):
            corners_solved = all(
                current[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            )
            if corners_solved:
                report = residual_report(current, puzzle.solved_state, decomposition)
                if report.all_even:
                    boundary_states.append((prefix, current))
            if prefix < len(source_path):
                current = puzzle.apply_move(current, source_path[prefix])
        if current != puzzle.solved_state:
            raise AssertionError(f"PID {state_id}: source path failed replay")

        replacements: dict[tuple[int, int], tuple[str, ...]] = {}
        for before, after in zip(boundary_states, boundary_states[1:], strict=False):
            before_prefix, before_state = before
            after_prefix, after_state = after
            segment = source_path[before_prefix:after_prefix]
            if not segment:
                continue
            macro = analyze_corner_fixing_macro(
                segment, puzzle.generators, decomposition
            )
            key = np.asarray(macro.cluster_permutations, dtype=np.uint8).tobytes()
            action = action_by_effect.get(key)
            best_candidate = segment
            best_components = 0
            if action is not None and len(table.paths[action]) < len(best_candidate):
                best_candidate = table.paths[action]
                best_components = 1

            if args.max_components >= 2 and len(segment) > 2 * minimum_action_length:
                target_effect = np.asarray(macro.cluster_permutations, dtype=np.uint8)
                first_actions = np.flatnonzero(
                    action_lengths + minimum_action_length < len(best_candidate)
                )
                if first_actions.size:
                    inverse_first = table.effects[table.inverse_indices[first_actions]]
                    required_seconds = np.take_along_axis(
                        inverse_first,
                        np.broadcast_to(target_effect, inverse_first.shape),
                        axis=-1,
                    )
                    for first, required in zip(
                        first_actions, required_seconds, strict=True
                    ):
                        second = action_by_effect.get(required.tobytes())
                        if second is None:
                            continue
                        candidate = reduce_quarter_turn_path(
                            table.paths[int(first)] + table.paths[second]
                        )
                        if (len(candidate), candidate) < (
                            len(best_candidate),
                            best_candidate,
                        ):
                            best_candidate = candidate
                            best_components = 2

            if len(best_candidate) >= len(segment):
                continue
            candidate_replacements += 1
            if best_components == 2:
                two_component_candidates += 1
            if puzzle.apply_path(before_state, best_candidate) != after_state:
                rejected_full_effect += 1
                continue
            replacements[(before_prefix, after_prefix)] = best_candidate
            if best_components == 2:
                two_component_replacements += 1

        pieces: list[str] = []
        cursor = 0
        for (before_prefix, after_prefix), candidate in sorted(replacements.items()):
            if before_prefix < cursor:
                raise AssertionError("overlapping adjacent boundary replacements")
            pieces.extend(source_path[cursor:before_prefix])
            pieces.extend(candidate)
            cursor = after_prefix
            accepted_replacements += 1
        pieces.extend(source_path[cursor:])
        output_path = reduce_quarter_turn_path(tuple(pieces))
        if puzzle.apply_path(initial_state, output_path) != puzzle.solved_state:
            raise AssertionError(f"PID {state_id}: rewritten path failed replay")
        replay_verified += 1
        source_total += len(source_path)
        output_total += len(output_path)
        if len(output_path) < len(source_path):
            report_rows.append(
                {
                    "pid": int(state_id),
                    "source_moves": len(source_path),
                    "output_moves": len(output_path),
                    "saving": len(source_path) - len(output_path),
                    "replacements": len(replacements),
                }
            )
        output_rows.append(
            {"initial_state_id": state_id, "path": ".".join(output_path)}
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(args.out)

    report = {
        "accepted_replacements": accepted_replacements,
        "action_count": table.action_count,
        "action_digest": table.digest,
        "candidate_replacements": candidate_replacements,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "improved_pids": len(report_rows),
        "maximum_pid_saving": max((row["saving"] for row in report_rows), default=0),
        "max_components": args.max_components,
        "mean_pid_saving": (
            statistics.fmean(row["saving"] for row in report_rows)
            if report_rows
            else 0.0
        ),
        "output": str(args.out),
        "output_total": output_total,
        "rejected_full_effect": rejected_full_effect,
        "replay_verified": replay_verified,
        "rows": report_rows,
        "source": str(args.submission),
        "source_total": source_total,
        "total_saving": source_total - output_total,
        "two_component_candidates": two_component_candidates,
        "two_component_replacements": two_component_replacements,
    }
    atomic_write(args.report_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
