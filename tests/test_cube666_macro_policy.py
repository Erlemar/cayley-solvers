"""Tests for exact macro teacher data and the neural shortlist policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    MacroActionTable,
    MacroTeacherDataset,
    cluster_costs,
    generate_geodesic_macro_teacher,
    generate_random_walk_teacher,
    load_macro_action_library,
    save_macro_action_library,
    three_cycle_units_batch,
)
from cube666.macro_policy import (  # noqa: E402
    FactorizedMacroPolicyConfig,
    FactorizedMacroPolicyValueNet,
    MacroEffectPolicyConfig,
    MacroEffectPolicyValueNet,
    MacroPolicyConfig,
    MacroPolicyValueNet,
    macro_effect_policy_loss,
    macro_policy_loss,
    policy_recall_at_k,
    score_macro_effect_candidates,
)
from cube666.macro_beam import learned_macro_beam_search  # noqa: E402
from cube666.macros import (  # noqa: E402
    apply_macro_to_clusters,
    enumerate_basic_corner_fixing_commutators,
    inverse_permutation,
    load_three_cycle_library,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DATA = PROJECT / "cayley-py-666-cube"


def small_action_table() -> MacroActionTable:
    puzzle = Cube666Puzzle.load(DATA / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    macros = enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition)
    return MacroActionTable.from_macros(macros)


def test_vectorized_cycle_cost_matches_known_permutations():
    permutations = np.asarray(
        [
            tuple(range(24)),
            (1, 2, 0) + tuple(range(3, 24)),
            (1, 0, 3, 2) + tuple(range(4, 24)),
        ],
        dtype=np.uint8,
    )
    assert three_cycle_units_batch(permutations).tolist() == [0, 1, 2]


def test_action_table_has_exact_inverses_and_matches_reference_composition():
    table = small_action_table()
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()
    action = 3
    moved = table.apply(identity, action)
    restored = table.apply(moved, int(table.inverse_indices[action]))
    assert np.array_equal(restored, identity)

    puzzle = Cube666Puzzle.load(DATA / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    macros = enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition)
    reference_state = tuple(inverse_permutation(permutation) for permutation in macros[5].cluster_permutations)
    assert np.array_equal(
        table.apply(np.asarray(reference_state, dtype=np.uint8), action),
        np.asarray(apply_macro_to_clusters(reference_state, macros[action]), dtype=np.uint8),
    )


def test_macro_action_library_round_trip(tmp_path: Path):
    puzzle = Cube666Puzzle.load(DATA / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    macros = enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition)
    output = tmp_path / "actions.json"
    saved = save_macro_action_library(output, macros)
    loaded_macros, loaded = load_macro_action_library(
        output,
        puzzle.generators,
        decomposition,
    )
    assert loaded.digest == saved.digest
    assert tuple(macro.path for macro in loaded_macros) == tuple(macro.path for macro in macros)


def test_teacher_labels_are_exact_minima_and_round_trip(tmp_path: Path):
    table = small_action_table()
    dataset = generate_random_walk_teacher(
        table,
        sample_count=8,
        max_walk_depth=4,
        max_labels=8,
        seed=666,
        score_chunk_size=64,
    )
    dataset.validate(table.action_count)
    for index in range(dataset.sample_count):
        costs = table.action_costs(dataset.states[index], chunk_size=64)
        labels = dataset.teacher_actions[index, : dataset.teacher_action_counts[index]]
        assert np.all(costs[labels] == costs.min())
        assert dataset.teacher_next_costs[index] == costs.min()
        assert dataset.cluster_cost_targets[index].tolist() == cluster_costs(dataset.states[index]).tolist()

    output = tmp_path / "teacher.npz"
    dataset.save(output)
    loaded = type(dataset).load(output)
    assert loaded.action_digest == dataset.action_digest
    assert np.array_equal(loaded.states, dataset.states)
    assert np.array_equal(loaded.teacher_actions, dataset.teacher_actions)


def test_valley_teacher_round_trip_requires_explicit_opt_in(tmp_path: Path):
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (1, 6, 24)).copy()
    fields = {
        "states": identity,
        "teacher_actions": np.asarray([[0]], dtype=np.int32),
        "teacher_action_counts": np.asarray([1], dtype=np.int16),
        "cluster_cost_targets": np.zeros((1, 6), dtype=np.int16),
        "teacher_next_costs": np.asarray([1], dtype=np.int16),
        "walk_depths": np.asarray([0], dtype=np.int16),
        "action_digest": "test",
    }
    strict = MacroTeacherDataset(**fields)
    try:
        strict.validate(action_count=1)
    except ValueError as exc:
        assert "strictly reduce" in str(exc)
    else:
        raise AssertionError("non-improving labels must require explicit opt-in")

    valley = MacroTeacherDataset(**fields, allow_non_improving=True)
    output = tmp_path / "valley_teacher.npz"
    valley.save(output)
    loaded = MacroTeacherDataset.load(output)
    assert loaded.allow_non_improving
    loaded.validate(action_count=1)


def test_geodesic_teacher_inverse_actions_reduce_exact_cost():
    puzzle = Cube666Puzzle.load(DATA / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    library = load_three_cycle_library(
        PROJECT / "cube666" / "artifacts" / "three_cycle_library.json",
        puzzle.generators,
        decomposition,
    )
    table = MacroActionTable.from_macros(tuple(library.values()))
    dataset = generate_geodesic_macro_teacher(
        table,
        sample_count=12,
        max_depth=4,
        max_labels=4,
        label_candidates=256,
        seed=666,
    )
    assert dataset.walk_depths.tolist() == [1, 2, 3, 4] * 3
    for index, state in enumerate(dataset.states):
        actions = dataset.teacher_actions[index, : dataset.teacher_action_counts[index]]
        for action in actions:
            child = table.apply(state, int(action))
            assert int(cluster_costs(child).sum()) == int(cluster_costs(state).sum()) - 1


def test_policy_forward_loss_and_recall():
    config = MacroPolicyConfig(action_count=32, hidden_dim=32, residual_blocks=1)
    model = MacroPolicyValueNet(config)
    states = torch.arange(24, dtype=torch.uint8).repeat(4, 6, 1)
    teacher_actions = torch.tensor([[1, -1], [2, 3], [4, -1], [5, 6]])
    counts = torch.tensor([1, 2, 1, 2])
    targets = torch.zeros((4, 6), dtype=torch.int16)
    outputs = model(states)
    assert outputs[0].shape == (4, 32)
    loss, parts = macro_policy_loss(outputs, teacher_actions, counts, targets)
    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "policy_loss", "value_loss", "cluster_value_loss"}

    logits = torch.full((4, 32), -10.0)
    logits[0, 1] = 10.0
    logits[1, 3] = 10.0
    logits[2, 4] = 10.0
    logits[3, 6] = 10.0
    recalls = policy_recall_at_k(logits, teacher_actions, counts, ks=(1, 4))
    assert recalls == {1: 1.0, 4: 1.0}


def test_factorized_policy_shares_cluster_solver_and_has_compatible_outputs():
    config = FactorizedMacroPolicyConfig(
        action_count=6 * 20,
        hidden_dim=32,
        residual_blocks=1,
    )
    model = FactorizedMacroPolicyValueNet(config)
    states = torch.arange(24, dtype=torch.uint8).repeat(4, 6, 1)
    logits, totals, clusters = model(states)
    assert logits.shape == (4, 120)
    assert totals.shape == (4,)
    assert clusters.shape == (4, 6)
    assert torch.allclose(totals, clusters.sum(dim=1) / 6)
    assert torch.allclose(logits[:, :20], logits[:, 20:40])


def test_effect_policy_scores_and_loss():
    table = small_action_table()
    config = MacroEffectPolicyConfig(
        action_count=table.action_count,
        hidden_dim=32,
        residual_blocks=1,
    )
    model = MacroEffectPolicyValueNet(config)
    states = torch.arange(24, dtype=torch.uint8).repeat(4, 6, 1)
    outputs = model(states)
    assert outputs[0].shape == (4, 6, 24, 24)

    effects = torch.from_numpy(table.effects)
    candidate_ids = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    scores = score_macro_effect_candidates(outputs[0], effects[candidate_ids])
    assert scores.shape == (4, 3)
    labels = candidate_ids[:, :2]
    counts = torch.tensor([1, 2, 1, 2])
    labels[0, 1] = -1
    labels[2, 1] = -1
    targets = torch.zeros((4, 6), dtype=torch.int16)
    negatives = candidate_ids[:, 2:]
    loss, parts = macro_effect_policy_loss(
        outputs,
        labels,
        counts,
        targets,
        effects,
        negatives,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {
        "loss",
        "contrastive_loss",
        "token_loss",
        "value_loss",
        "cluster_value_loss",
    }


def test_learned_macro_beam_solves_a_one_action_state():
    table = small_action_table()
    action = 3
    inverse = int(table.inverse_indices[action])
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()
    state = table.apply(identity, action)
    model = MacroPolicyValueNet(
        MacroPolicyConfig(action_count=table.action_count, hidden_dim=16, residual_blocks=0)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_head.bias[inverse] = 10.0
    result = learned_macro_beam_search(
        state,
        model,
        table,
        beam_width=4,
        branch_width=2,
        max_steps=2,
        model_batch_size=8,
    )
    assert result.solved
    assert result.actions == (inverse,)


def test_learned_macro_beam_exact_cost_ranking_solves_a_one_action_state():
    table = small_action_table()
    action = 3
    inverse = int(table.inverse_indices[action])
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()
    state = table.apply(identity, action)
    model = MacroPolicyValueNet(
        MacroPolicyConfig(action_count=table.action_count, hidden_dim=16, residual_blocks=0)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_head.bias[inverse] = 10.0
    result = learned_macro_beam_search(
        state,
        model,
        table,
        beam_width=4,
        branch_width=2,
        max_steps=2,
        exact_cost_weight=3.0,
        model_batch_size=8,
    )
    assert result.solved
    assert result.actions == (inverse,)


def test_macro_effect_beam_solves_a_one_action_state():
    table = small_action_table()
    action = 3
    inverse = int(table.inverse_indices[action])
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()
    state = table.apply(identity, action)
    model = MacroEffectPolicyValueNet(
        MacroEffectPolicyConfig(
            action_count=table.action_count,
            hidden_dim=16,
            residual_blocks=0,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        target = table.effects[inverse]
        for cluster in range(6):
            for position in range(24):
                model.effect_head.bias[
                    cluster * 24 * 24 + position * 24 + int(target[cluster, position])
                ] = 10.0
    result = learned_macro_beam_search(
        state,
        model,
        table,
        beam_width=4,
        branch_width=2,
        max_steps=2,
        exact_cost_weight=3.0,
        model_batch_size=8,
    )
    assert result.solved
    assert result.actions == (inverse,)


def test_learned_macro_beam_can_append_unscored_extra_actions():
    puzzle = Cube666Puzzle.load(DATA / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    macros = enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition)
    full = MacroActionTable.from_macros(macros)
    inverse_pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for action, inverse in enumerate(full.inverse_indices):
        pair = (action, int(inverse))
        if pair[0] == pair[1] or pair[0] in used or pair[1] in used:
            continue
        inverse_pairs.append(pair)
        used.update(pair)
        if len(inverse_pairs) == 2:
            break
    assert len(inverse_pairs) == 2
    selected = inverse_pairs[0] + inverse_pairs[1]
    table = MacroActionTable.from_macros(tuple(macros[index] for index in selected))
    model = MacroPolicyValueNet(
        MacroPolicyConfig(action_count=2, hidden_dim=16, residual_blocks=0)
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24)).copy()
    state = table.apply(identity, int(table.inverse_indices[2]))
    result = learned_macro_beam_search(
        state,
        model,
        table,
        beam_width=4,
        branch_width=2,
        max_steps=1,
        model_batch_size=8,
        extra_action_indices=(2, 3),
    )
    assert result.solved
    assert result.actions == (2,)
    assert result.final_exact_cost == 0
