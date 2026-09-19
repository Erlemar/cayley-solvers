"""Find exact intermediate states shared by two replay-verified submissions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_submission(path: Path) -> dict[int, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"initial_state_id", "path"}:
        raise ValueError(f"{path}: expected initial_state_id,path")
    return {
        int(row["initial_state_id"]): tuple(
            token for token in row["path"].split(".") if token
        )
        for row in rows
    }


def replay_states(
    puzzle: Cube666Puzzle,
    initial_state: tuple[int, ...],
    path: tuple[str, ...],
) -> list[tuple[int, ...]]:
    states = [initial_state]
    current = initial_state
    for move in path:
        current = puzzle.apply_move(current, move)
        states.append(current)
    if current != puzzle.solved_state:
        raise AssertionError("submission path failed exact replay")
    return states


def main() -> None:
    args = parse_args()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    initial_states = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    left = load_submission(args.left)
    right = load_submission(args.right)
    if set(left) != set(right):
        raise ValueError("submissions have different PID coverage")

    rows: list[dict[str, object]] = []
    for pid in sorted(left):
        if len(right[pid]) >= len(left[pid]):
            continue
        left_states = replay_states(puzzle, initial_states[pid], left[pid])
        right_states = replay_states(puzzle, initial_states[pid], right[pid])
        right_positions = {state: position for position, state in enumerate(right_states)}
        splices: list[dict[str, int]] = []
        intersections: list[tuple[int, int, tuple[int, ...], bool]] = []
        for left_prefix, state in enumerate(left_states):
            right_prefix = right_positions.get(state)
            if right_prefix is None or left_prefix in (0, len(left[pid])):
                continue
            corners_solved = all(
                state[position] == puzzle.solved_state[position]
                for position in decomposition.corner_orbit
            )
            normalized = corners_solved and residual_report(
                state, puzzle.solved_state, decomposition
            ).all_even
            intersections.append((left_prefix, right_prefix, state, normalized))
            candidate = reduce_commuting_quarter_turn_path(
                left[pid][:left_prefix] + right[pid][right_prefix:]
            )
            if puzzle.apply_path(initial_states[pid], candidate) != puzzle.solved_state:
                raise AssertionError(f"PID {pid}: exact shared-state splice failed")
            splices.append(
                {
                    "left_prefix": left_prefix,
                    "right_prefix": right_prefix,
                    "moves": len(candidate),
                    "saving": len(left[pid]) - len(candidate),
                }
            )
        normalized_replacements: list[dict[str, int]] = []
        endpoints = [
            (0, 0, initial_states[pid], False),
            *intersections,
            (len(left[pid]), len(right[pid]), puzzle.solved_state, True),
        ]
        for start_index, start in enumerate(endpoints):
            left_start, right_start, _, start_normalized = start
            if not start_normalized:
                continue
            for end in endpoints[start_index + 1 :]:
                left_end, right_end, _, end_normalized = end
                if (
                    not end_normalized
                    or left_end <= left_start
                    or right_end <= right_start
                ):
                    continue
                left_segment = left_end - left_start
                right_segment = right_end - right_start
                if right_segment >= left_segment:
                    continue
                normalized_replacements.append(
                    {
                        "left_start": left_start,
                        "left_end": left_end,
                        "left_moves": left_segment,
                        "right_start": right_start,
                        "right_end": right_end,
                        "right_moves": right_segment,
                        "saving": left_segment - right_segment,
                    }
                )
        rows.append(
            {
                "pid": pid,
                "left_moves": len(left[pid]),
                "right_moves": len(right[pid]),
                "shared_nonterminal_states": len(splices),
                "best_splice": min(splices, key=lambda row: row["moves"])
                if splices
                else None,
                "best_normalized_replacement": max(
                    normalized_replacements,
                    key=lambda row: (row["saving"], -row["right_moves"]),
                )
                if normalized_replacements
                else None,
            }
        )

    report = {
        "left": str(args.left),
        "right": str(args.right),
        "improved_pids": len(rows),
        "pids_with_nonterminal_intersection": sum(
            int(row["shared_nonterminal_states"] > 0) for row in rows
        ),
        "pids_with_normalized_replacement": sum(
            int(row["best_normalized_replacement"] is not None) for row in rows
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
