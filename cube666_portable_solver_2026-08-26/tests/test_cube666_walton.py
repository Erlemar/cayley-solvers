from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.walton import (
    NATIVE_FACE_ORDER,
    WALTON_FACE_ORDER,
    exact_state_is_colour_solved,
    project_exact_state_to_walton,
    split_marked_solution,
    translate_walton_move,
    translate_walton_path,
)
from cube666.classical import build_decomposition
from cube666.macro_data import load_macro_action_library
from cube666.puzzle import Cube666Puzzle


def test_identity_projects_to_walton_face_order() -> None:
    target = tuple(range(216))
    projected = project_exact_state_to_walton(target, target)
    assert projected == "".join(face * 36 for face in WALTON_FACE_ORDER)
    assert NATIVE_FACE_ORDER == ("U", "F", "R", "B", "L", "D")
    assert exact_state_is_colour_solved(target, target)


def test_colour_solved_ignores_within_face_permutations() -> None:
    target = tuple(range(216))
    state = list(target)
    state[0], state[35] = state[35], state[0]
    assert exact_state_is_colour_solved(state, target)
    state[35], state[36] = state[36], state[35]
    assert not exact_state_is_colour_solved(state, target)


def test_walton_move_translation_matches_notebook_convention() -> None:
    assert translate_walton_move("U") == ("-d5",)
    assert translate_walton_move("U'") == ("d5",)
    assert translate_walton_move("3Rw") == ("r0", "r1", "r2")
    assert translate_walton_move("Bw") == ("-f4", "-f5")
    assert translate_walton_move("Lw'") == ("r4", "r5")
    assert translate_walton_move("3Dw2") == (
        "d0", "d1", "d2", "d0", "d1", "d2",
    )
    assert translate_walton_path(("R", "U2")) == ("r0", "-d5", "-d5")


def test_marked_solution_phase_split() -> None:
    phases = split_marked_solution(
        ("Uw", "R", "COMMENT_centers_(2_steps)", "U", "COMMENT_edges_(1_steps)")
    )
    assert [(phase.label, phase.standard_moves) for phase in phases] == [
        ("centers_(2_steps)", ("Uw", "R")),
        ("edges_(1_steps)", ("U",)),
    ]


def test_four_pid_walton_artifact_replays_to_colour_goal() -> None:
    data = PROJECT / "cayley-py-666-cube"
    raw = PROJECT / "cube666" / "kaggle_walton_macro_mining" / "output" / "walton_runs.json"
    puzzle = Cube666Puzzle.load(data / "puzzle_info.json")
    states = {
        int(pid): state
        for pid, state in puzzle.iter_test_states(data / "test.csv")
    }
    runs = json.loads(raw.read_text(encoding="utf-8"))["runs"]
    assert set(runs) == {"597", "808", "854", "906"}
    for pid_text, run in runs.items():
        phases = split_marked_solution(run["marked_solution"])
        reconstructed = tuple(
            move
            for phase in phases
            for move in phase.standard_moves
        )
        assert reconstructed == tuple(run["solution"])
        final_state = puzzle.apply_path(
            states[int(pid_text)],
            translate_walton_path(reconstructed),
        )
        assert exact_state_is_colour_solved(final_state, puzzle.solved_state)
        assert final_state != puzzle.solved_state


def test_kmc_walton_library_preserves_model_indices_and_adds_25_effects() -> None:
    data = PROJECT / "cayley-py-666-cube"
    puzzle = Cube666Puzzle.load(data / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, kmc = load_macro_action_library(
        PROJECT / "cube666" / "training" / "kmc_macro_teacher_v1" / "action_library.json",
        puzzle.generators,
        decomposition,
    )
    _, combined = load_macro_action_library(
        PROJECT / "cube666" / "training" / "kmc_walton_macro4" / "action_library.json",
        puzzle.generators,
        decomposition,
    )
    assert kmc.action_count == 1070
    assert combined.action_count == 1095
    assert np.array_equal(combined.effects[: kmc.action_count], kmc.effects)
    assert combined.paths[: kmc.action_count] == kmc.paths
