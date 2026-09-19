"""Round-trip tests for symmetry and inverse 666 solver formulations."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "cube_nnn" / "src"))

from cube666.frames import FrameTransform, invert_path, invert_state  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.symmetry import build_symmetries  # noqa: E402


DATA = PROJECT / "cayley-py-666-cube"


def test_inverse_state_and_path_round_trip() -> None:
    state = (2, 0, 3, 1)
    path = ("f0", "-r2", "d5")
    assert invert_state(invert_state(state)) == state
    assert invert_path(invert_path(path)) == path


def test_selected_spatial_and_inverse_frames_preserve_solutions() -> None:
    puzzle = Cube666Puzzle.load(DATA / "puzzle_info.json")
    geometric_cube = NCube.from_puzzle_info(DATA / "puzzle_info.json")
    table = build_symmetries(geometric_cube)
    scramble = ("f1", "r4", "-d2", "f5", "r0", "d3", "-f1")
    state = puzzle.apply_path(puzzle.solved_state, scramble)
    solution = invert_path(scramble)

    formulations = (
        (None, True),
        (3, False),
        (18, False),
        (35, False),
        (1, False),
        (40, False),
        (25, True),
    )
    for symmetry_index, use_inverse in formulations:
        frame = FrameTransform(
            label="test",
            move_names=puzzle.move_names,
            sticker_permutation=(
                None if symmetry_index is None else table.perms[symmetry_index]
            ),
            move_to_transformed=(
                None if symmetry_index is None else table.relabel[symmetry_index]
            ),
            use_inverse=use_inverse,
        )
        transformed_state = frame.transform_state(state)
        transformed_solution = frame.transform_path(solution)
        assert puzzle.apply_path(transformed_state, transformed_solution) == puzzle.solved_state
        assert frame.restore_path(transformed_solution) == solution

