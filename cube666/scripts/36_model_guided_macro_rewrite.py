"""Use learned anytime macro beam to shorten exact classical phase windows."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402
from cube666.macros import (  # noqa: E402
    analyze_corner_fixing_macro,
    inverse_permutation,
    reduce_quarter_turn_path,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


@dataclass(frozen=True)
class Window:
    pid: int
    before_prefix: int
    after_prefix: int
    before_state: tuple[int, ...]
    after_state: tuple[int, ...]
    source_path: tuple[str, ...]
    inverse_effect: np.ndarray
    target_actions: tuple[int, ...]
    boundary_span: int

    @property
    def source_moves(self) -> int:
        return self.after_prefix - self.before_prefix


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--action-library", type=Path, required=True)
    parser.add_argument("--top-windows", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--branch-width", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-boundary-span", type=int, default=1)
    parser.add_argument("--min-boundary-span", type=int, default=1)
    parser.add_argument(
        "--exact-close",
        action="store_true",
        help="Algebraically test every model-proposed child for a final library action.",
    )
    parser.add_argument("--policy-nll-weight", type=float, default=1.0)
    parser.add_argument("--move-cost-weight", type=float, default=0.1)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "submissions" / "cube666_model_macro_rewritten.csv",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "model_macro_rewrite.json",
    )
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(
        args.top_windows,
        args.beam_width,
        args.branch_width,
        args.max_steps,
        args.max_boundary_span,
        args.min_boundary_span,
    ) <= 0:
        raise ValueError("window and search dimensions must be positive")
    if args.min_boundary_span > args.max_boundary_span:
        raise ValueError("min-boundary-span exceeds max-boundary-span")
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library, puzzle.generators, decomposition
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["action_digest"] != table.digest:
        raise ValueError("checkpoint and action library digests differ")
    model = build_macro_policy_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    action_by_effect = {
        effect.tobytes(): action for action, effect in enumerate(table.effects)
    }
    states = dict(puzzle.iter_test_states(args.data_dir / "test.csv"))

    with open(args.submission, newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    paths_by_pid = {
        int(row["initial_state_id"]): tuple(
            token for token in row["path"].split(".") if token
        )
        for row in source_rows
    }
    windows: list[Window] = []
    for row in source_rows:
        state_id = str(int(row["initial_state_id"]))
        pid = int(state_id)
        path = paths_by_pid[pid]
        initial_state = states[state_id]
        boundaries: list[tuple[int, tuple[int, ...]]] = []
        current = initial_state
        for prefix in range(len(path) + 1):
            if all(
                current[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            ):
                report = residual_report(current, puzzle.solved_state, decomposition)
                if report.all_even:
                    boundaries.append((prefix, current))
            if prefix < len(path):
                current = puzzle.apply_move(current, path[prefix])
        if current != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: source path failed replay")
        for before_index in range(len(boundaries) - 1):
            before_prefix, before_state = boundaries[before_index]
            target_actions: list[int] = []
            maximum_after = min(
                len(boundaries), before_index + args.max_boundary_span + 1
            )
            for after_index in range(before_index + 1, maximum_after):
                adjacent_before = boundaries[after_index - 1][0]
                after_prefix, after_state = boundaries[after_index]
                adjacent_segment = path[adjacent_before:after_prefix]
                if not adjacent_segment:
                    break
                adjacent_macro = analyze_corner_fixing_macro(
                    adjacent_segment, puzzle.generators, decomposition
                )
                adjacent_key = np.asarray(
                    adjacent_macro.cluster_permutations, dtype=np.uint8
                ).tobytes()
                target_action = action_by_effect.get(adjacent_key)
                if target_action is None:
                    break
                target_actions.append(target_action)
                segment = path[before_prefix:after_prefix]
                macro = analyze_corner_fixing_macro(
                    segment, puzzle.generators, decomposition
                )
                inverse_effect = np.asarray(
                    [
                        inverse_permutation(permutation)
                        for permutation in macro.cluster_permutations
                    ],
                    dtype=np.uint8,
                )
                boundary_span = after_index - before_index
                if boundary_span >= args.min_boundary_span:
                    windows.append(
                        Window(
                            pid,
                            before_prefix,
                            after_prefix,
                            before_state,
                            after_state,
                            path,
                            inverse_effect,
                            tuple(target_actions),
                            boundary_span,
                        )
                    )

    selected = sorted(
        windows,
        key=lambda window: (-window.source_moves, window.pid, window.before_prefix),
    )[: args.top_windows]
    replacements_by_pid: dict[int, dict[tuple[int, int], tuple[str, ...]]] = {}
    attempts: list[dict[str, object]] = []
    for window in selected:
        case_started = time.perf_counter()
        result = learned_macro_beam_search(
            window.inverse_effect,
            model,
            table,
            beam_width=args.beam_width,
            branch_width=args.branch_width,
            max_steps=args.max_steps,
            policy_nll_weight=args.policy_nll_weight,
            exact_cost_weight=None,
            move_cost_weight=args.move_cost_weight,
            model_batch_size=2048,
            extra_action_indices=tuple(dict.fromkeys(window.target_actions)),
            initial_solution_actions=window.target_actions,
            exact_close_actions=action_by_effect if args.exact_close else None,
            fallback_exact_cost=False,
            stop_on_first_solution=False,
        )
        candidate = reduce_quarter_turn_path(
            tuple(move for action in result.actions for move in table.paths[action])
        )
        full_effect_ok = (
            result.solved
            and puzzle.apply_path(window.before_state, candidate) == window.after_state
        )
        accepted = full_effect_ok and len(candidate) < window.source_moves
        if accepted:
            existing = replacements_by_pid.get(window.pid, {})
            overlaps = any(
                window.before_prefix < after and before < window.after_prefix
                for before, after in existing
            )
            if overlaps:
                accepted = False
        if accepted:
            replacements_by_pid.setdefault(window.pid, {})[
                (window.before_prefix, window.after_prefix)
            ] = candidate
        attempt = {
            "accepted": accepted,
            "beam_actions": len(result.actions),
            "boundary_span": window.boundary_span,
            "candidate_moves": len(candidate),
            "elapsed_seconds": round(time.perf_counter() - case_started, 4),
            "expanded_states": result.expanded_states,
            "full_effect_ok": full_effect_ok,
            "generated_states": result.generated_states,
            "pid": window.pid,
            "prefix": window.before_prefix,
            "saving": window.source_moves - len(candidate) if accepted else 0,
            "solved_effect": result.solved,
            "source_moves": window.source_moves,
            "upper_bound_actions": len(window.target_actions),
        }
        attempts.append(attempt)
        print(json.dumps(attempt, sort_keys=True), flush=True)

    output_rows: list[dict[str, str]] = []
    source_total = 0
    output_total = 0
    improved_rows: list[dict[str, int]] = []
    for row in source_rows:
        pid = int(row["initial_state_id"])
        source_path = paths_by_pid[pid]
        pieces: list[str] = []
        cursor = 0
        replacements = replacements_by_pid.get(pid, {})
        for (before_prefix, after_prefix), candidate in sorted(replacements.items()):
            if before_prefix < cursor:
                raise AssertionError("model rewrite windows overlap")
            pieces.extend(source_path[cursor:before_prefix])
            pieces.extend(candidate)
            cursor = after_prefix
        pieces.extend(source_path[cursor:])
        output_path = reduce_quarter_turn_path(tuple(pieces))
        state_id = str(pid)
        if puzzle.apply_path(states[state_id], output_path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: model-rewritten path failed replay")
        source_total += len(source_path)
        output_total += len(output_path)
        if len(output_path) < len(source_path):
            improved_rows.append(
                {
                    "pid": pid,
                    "source_moves": len(source_path),
                    "output_moves": len(output_path),
                    "saving": len(source_path) - len(output_path),
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
        "action_count": table.action_count,
        "action_digest": table.digest,
        "attempts": attempts,
        "beam_width": args.beam_width,
        "branch_width": args.branch_width,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "exact_close": args.exact_close,
        "improved_pids": len(improved_rows),
        "max_steps": args.max_steps,
        "max_boundary_span": args.max_boundary_span,
        "min_boundary_span": args.min_boundary_span,
        "mean_saving": (
            statistics.fmean(row["saving"] for row in improved_rows)
            if improved_rows
            else 0.0
        ),
        "move_cost_weight": args.move_cost_weight,
        "output": str(args.out),
        "output_total": output_total,
        "policy_nll_weight": args.policy_nll_weight,
        "replay_verified": len(output_rows),
        "rows": improved_rows,
        "source": str(args.submission),
        "source_total": source_total,
        "top_windows": args.top_windows,
        "total_saving": source_total - output_total,
    }
    atomic_write(args.report_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
