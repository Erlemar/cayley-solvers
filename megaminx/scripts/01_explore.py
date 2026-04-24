"""Sanity-check Megaminx data files and the sample submission.

Verifies:
 - Megaminx puzzle loads, inverse pairs are consistent
 - test.csv has 1001 puzzles, each a 120-element permutation
 - sample_submission.csv has a valid path that solves each test puzzle
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
# Put the megaminx source on the path, plus the parent cayley/src for shared modules.
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def load_test(path: Path) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            pid = int(row[0])
            state = tuple(int(x) for x in row[1].split(","))
            out[pid] = state
    return out


def load_paths(path: Path) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            pid = int(row[0])
            p = row[1].split(".") if row[1] else []
            out[pid] = p
    return out


def main() -> int:
    data = PROJECT / "data"
    puzzle = Megaminx.load(data / "puzzle_info.json")
    print(f"puzzle loaded: {len(puzzle.move_names)} moves, state_size={len(puzzle.solved_state)}")
    print(f"moves: {list(puzzle.move_names)}")
    puzzle.verify_inverse_pairs()
    print("inverse pairs OK")

    states = load_test(data / "test.csv")
    print(f"test.csv: {len(states)} puzzles")
    state_sizes = {len(s) for s in states.values()}
    print(f"state sizes: {state_sizes}")
    assert state_sizes == {120}, state_sizes

    paths = load_paths(data / "sample_submission.csv")
    print(f"sample_submission.csv: {len(paths)} paths")

    # Verify sample submission solves every puzzle.
    n_ok = 0
    n_bad = 0
    total_moves = 0
    lengths: list[int] = []
    for pid, state in states.items():
        path = paths[pid]
        cur = state
        try:
            final = puzzle.apply_path(cur, path)
        except KeyError as e:
            print(f"  pid={pid}: bad move {e}")
            n_bad += 1
            continue
        if puzzle.is_solved(final):
            n_ok += 1
            total_moves += len(path)
            lengths.append(len(path))
        else:
            n_bad += 1
            if n_bad <= 3:
                print(f"  pid={pid}: NOT SOLVED after {len(path)} moves")

    print()
    print(f"sample submission: {n_ok} solved, {n_bad} failed")
    print(f"total moves: {total_moves:,}")
    if lengths:
        lengths.sort()
        print(f"per-puzzle: min={lengths[0]} median={lengths[len(lengths) // 2]} "
              f"mean={total_moves / len(lengths):.1f} max={lengths[-1]}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
