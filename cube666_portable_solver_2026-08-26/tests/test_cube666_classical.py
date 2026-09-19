"""Structural tests for the exact-sticker classical 6x6x6 solver core."""

from __future__ import annotations

import random
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import (  # noqa: E402
    apply_path,
    build_decomposition,
    minimum_unrestricted_three_cycles,
    parity_repair_path,
    permutation_parity,
    position_orbits,
    residual_report,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402
from cube666.macros import (  # noqa: E402
    commutator,
    build_isolated_three_cycle_library,
    compose_pullbacks,
    enumerate_basic_corner_fixing_commutators,
    enumerate_basic_inner_commutators,
    enumerate_conjugated_macros,
    enumerate_nested_corner_fixing_commutators,
    finish_with_inserted_three_cycles,
    greedy_macro_reduce,
    state_cluster_permutations,
    invert_path,
    path_effect,
)


DATA = PROJECT / "cayley-py-666-cube"


def load_puzzle_and_decomposition():
    puzzle = Cube666Puzzle.load(DATA / "puzzle_info.json")
    return puzzle, build_decomposition(puzzle.generators)


def test_real_definition_has_nine_24_sticker_orbits():
    puzzle = Cube666Puzzle.load(DATA / "puzzle_info.json")
    puzzle.verify_inverse_pairs()
    assert puzzle.size == 216
    assert len(puzzle.generators) == 36
    assert [len(orbit) for orbit in position_orbits(puzzle.generators)] == [24] * 9


def test_discovers_six_physical_clusters():
    _, decomposition = load_puzzle_and_decomposition()
    assert len(decomposition.corner_orbit) == 24
    assert len(decomposition.center_orbits) == 4
    assert len(decomposition.wing_pairs) == 2
    assert len(decomposition.physical_clusters) == 6
    assert all(len(orbit) == 24 for orbit in decomposition.physical_clusters)
    assert len(decomposition.inner_move_names) == 24


def test_all_real_states_respect_cluster_and_wing_invariants():
    puzzle, decomposition = load_puzzle_and_decomposition()
    count = 0
    for _, state in puzzle.iter_test_states(DATA / "test.csv"):
        report = residual_report(state, puzzle.solved_state, decomposition)
        assert len(report.parity_vector) == 6
        count += 1
    assert count == 1012


def test_shortest_parity_repair_makes_inner_scrambles_even():
    puzzle, decomposition = load_puzzle_and_decomposition()
    rng = random.Random(666)
    for depth in range(9):
        state = puzzle.apply_path(
            puzzle.solved_state,
            (rng.choice(decomposition.inner_move_names) for _ in range(depth)),
        )
        before = residual_report(state, puzzle.solved_state, decomposition)
        repair = parity_repair_path(before.parity_vector, decomposition)
        repaired = apply_path(state, puzzle.generators, repair)
        after = residual_report(repaired, puzzle.solved_state, decomposition)
        assert after.corner_parity == 0
        assert after.all_even
        assert after.unrestricted_three_cycles is not None


def test_move_effect_columns_match_observed_parity_changes():
    puzzle, decomposition = load_puzzle_and_decomposition()
    state = puzzle.solved_state
    rng = random.Random(666)
    state = puzzle.apply_path(state, (rng.choice(puzzle.move_names) for _ in range(40)))
    before = residual_report(state, puzzle.solved_state, decomposition).parity_vector
    effect_by_move = decomposition.parity_effect_dict()
    for name in decomposition.inner_move_names:
        moved = puzzle.apply_move(state, name)
        after = residual_report(moved, puzzle.solved_state, decomposition).parity_vector
        observed = tuple(left ^ right for left, right in zip(before, after, strict=True))
        assert observed == effect_by_move[name]


def test_minimum_unrestricted_three_cycle_count_examples():
    assert minimum_unrestricted_three_cycles((0, 1, 2, 3)) == 0
    assert minimum_unrestricted_three_cycles((1, 2, 0, 3)) == 1
    assert minimum_unrestricted_three_cycles((1, 0, 3, 2)) == 2
    assert minimum_unrestricted_three_cycles((1, 2, 3, 4, 0)) == 2
    assert permutation_parity((1, 0, 2)) == 1


def test_exact_corner_solver_on_real_states():
    puzzle, decomposition = load_puzzle_and_decomposition()
    coordinates = CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    solver = ExactCornerSolver.build(coordinates)
    states = list(puzzle.iter_test_states(DATA / "test.csv"))
    for index in (0, 100, 500, 1011):
        state_id, state = states[index]
        corner_path = solver.solve(state, puzzle.solved_state)
        solved_corners = apply_path(state, puzzle.generators, corner_path)
        assert all(
            solved_corners[position] == puzzle.solved_state[position]
            for position in decomposition.corner_orbit
        ), state_id
        post_corner = residual_report(solved_corners, puzzle.solved_state, decomposition)
        repair = parity_repair_path(post_corner.parity_vector, decomposition)
        repaired = apply_path(solved_corners, puzzle.generators, repair)
        assert residual_report(repaired, puzzle.solved_state, decomposition).all_even


def test_basic_commutator_effects_fix_corners_and_are_three_cycles():
    puzzle, decomposition = load_puzzle_and_decomposition()
    macros = enumerate_basic_inner_commutators(puzzle.generators, decomposition)
    assert macros
    assert all(
        macro.effect[position] == position
        for macro in macros
        for position in decomposition.corner_orbit
    )
    assert any(macro.pure_three_cycles for macro in macros)

    full_family = enumerate_basic_corner_fixing_commutators(
        puzzle.generators,
        decomposition,
    )
    assert len(full_family) > len(macros)
    assert any(macro.cluster_cycles[4] or macro.cluster_cycles[5] for macro in full_family)
    assert {macro.unrestricted_three_cycle_units for macro in full_family} == {2, 6}
    expanded = enumerate_conjugated_macros(
        full_family,
        puzzle.generators,
        decomposition,
        max_conjugator_depth=1,
    )
    assert len(expanded) > len(full_family)
    assert all(
        macro.effect[position] == position
        for macro in expanded
        for position in decomposition.corner_orbit
    )


def test_path_effect_composition_and_inverse():
    puzzle, _ = load_puzzle_and_decomposition()
    path = commutator("f1", "r1")
    effect = path_effect(puzzle.generators, path)
    inverse_effect = path_effect(puzzle.generators, invert_path(path))
    identity = tuple(range(puzzle.size))
    assert compose_pullbacks(effect, inverse_effect) == identity
    assert puzzle.apply_path(puzzle.solved_state, path) == effect


def test_greedy_macro_reduction_replays_exactly():
    puzzle, decomposition = load_puzzle_and_decomposition()
    macros = enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition)
    macro = macros[0]
    scrambled = puzzle.apply_path(puzzle.solved_state, invert_path(macro.path))
    clusters = state_cluster_permutations(scrambled, puzzle.solved_state, decomposition)
    result = greedy_macro_reduce(clusters, macros, max_macros=2)
    assert result.final_cost == 0
    replayed = puzzle.apply_path(scrambled, result.path)
    assert replayed == puzzle.solved_state


def test_nested_commutators_fix_corners():
    puzzle, decomposition = load_puzzle_and_decomposition()
    macros = enumerate_nested_corner_fixing_commutators(
        puzzle.generators,
        decomposition,
        inner_conjugator_depth=1,
    )
    assert macros
    assert all(
        macro.effect[position] == position
        for macro in macros
        for position in decomposition.corner_orbit
    )
    library = build_isolated_three_cycle_library(
        macros,
        puzzle.generators,
        decomposition,
        max_conjugator_depth=1,
    )
    assert len(library) > 4032
    assert all(macro.total_nontrivial_cycles == 1 for macro in library.values())


def test_inserted_finisher_solves_a_known_macro_effect():
    puzzle, decomposition = load_puzzle_and_decomposition()
    nested = enumerate_nested_corner_fixing_commutators(
        puzzle.generators,
        decomposition,
        inner_conjugator_depth=1,
    )
    isolated = [
        macro
        for macro in nested
        if macro.total_nontrivial_cycles == 1 and macro.total_three_cycles == 1
    ]
    library = build_isolated_three_cycle_library(
        isolated,
        puzzle.generators,
        decomposition,
        max_conjugator_depth=0,
    )
    macro = isolated[0]
    scrambled = puzzle.apply_path(puzzle.solved_state, invert_path(macro.path))
    clusters = state_cluster_permutations(scrambled, puzzle.solved_state, decomposition)
    result = finish_with_inserted_three_cycles(
        clusters,
        (),
        library,
        puzzle.generators,
        decomposition,
    )
    assert result.final_cost == 0
    assert puzzle.apply_path(scrambled, result.path) == puzzle.solved_state
