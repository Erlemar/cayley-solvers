"""Verify that a path solves a scrambled state.

A path is valid iff every move name exists in the puzzle's generator dict AND
applying all moves in order to the initial state yields the solved identity.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from cayley.puzzle import PictureCube


@dataclass
class VerifyResult:
    ok: bool
    reason: str
    moves: int


def verify_path(puzzle: PictureCube, initial_state: tuple[int, ...], path: list[str]) -> VerifyResult:
    cur = initial_state
    for i, move in enumerate(path):
        if move not in puzzle.generators:
            return VerifyResult(False, f"unknown move {move!r} at index {i}", len(path))
        cur = puzzle.apply_move(cur, move)
    if puzzle.is_solved(cur):
        return VerifyResult(True, "solved", len(path))
    # Report Hamming distance to help debugging.
    hamming = sum(1 for a, b in zip(cur, puzzle.solved_state) if a != b)
    return VerifyResult(False, f"not solved (hamming={hamming})", len(path))


def load_test_states(test_csv: str | Path) -> dict[int, tuple[int, ...]]:
    """Read test.csv -> {id: state_tuple}."""
    out: dict[int, tuple[int, ...]] = {}
    with open(test_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["initial_state_id"])
            state = tuple(int(x) for x in row["initial_state"].split(","))
            out[pid] = state
    return out


def load_submission(submission_csv: str | Path) -> dict[int, list[str]]:
    """Read a submission CSV -> {id: path_list}. Empty paths decode to []."""
    out: dict[int, list[str]] = {}
    with open(submission_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["initial_state_id"])
            path_str = row["path"].strip()
            out[pid] = path_str.split(".") if path_str else []
    return out


@dataclass
class SubmissionReport:
    n_total: int
    n_valid: int
    total_moves: int
    failures: list[tuple[int, str]]  # (puzzle_id, reason)

    @property
    def all_valid(self) -> bool:
        return self.n_valid == self.n_total


def verify_submission(
    puzzle: PictureCube,
    test_csv: str | Path,
    submission_csv: str | Path,
) -> SubmissionReport:
    states = load_test_states(test_csv)
    paths = load_submission(submission_csv)

    missing = set(states) - set(paths)
    if missing:
        raise ValueError(f"submission missing {len(missing)} puzzle ids: {sorted(missing)[:5]}...")

    n_valid = 0
    total_moves = 0
    failures: list[tuple[int, str]] = []
    for pid, state in states.items():
        path = paths[pid]
        result = verify_path(puzzle, state, path)
        total_moves += result.moves
        if result.ok:
            n_valid += 1
        else:
            failures.append((pid, result.reason))

    return SubmissionReport(
        n_total=len(states),
        n_valid=n_valid,
        total_moves=total_moves,
        failures=failures,
    )
