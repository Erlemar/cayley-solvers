from __future__ import annotations

from pathlib import Path

import numpy as np

from jewel.ball import ExactBall, build_exact_ball
from jewel.gfn_beam import (
    gfn_adaptive_beam_search,
    gfn_batched_best_first_search,
    gfn_hybrid_search,
)
from jewel.puzzle import SOLVED, apply_action, apply_path, inverse_path


class KnownPathPolicy:
    def __init__(self, start, path):
        self.actions: dict[int, int] = {}
        state = start
        for action in path:
            self.actions[state.rank()] = action
            state = apply_action(state, action)

    def batch(self, states):
        probabilities = np.full((len(states), 12), 1e-6, dtype=np.float64)
        for index, state in enumerate(states):
            probabilities[index, self.actions.get(state.rank(), 0)] = 1.0
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities


def test_gfn_beam_best_first_and_hybrid_return_valid_paths(tmp_path: Path) -> None:
    root = tmp_path / "ball"
    build_exact_ball(root, 2)
    ball = ExactBall.load(root)
    scramble = [0, 2, 4, 6]
    state = apply_path(SOLVED, scramble)
    solution = inverse_path(scramble)
    policy = KnownPathPolicy(state, solution)

    beam = gfn_adaptive_beam_search(
        state,
        policy,
        ball,
        [],
        widths=(1, 4),
        max_learned_steps=4,
    )
    assert beam.path is not None
    assert apply_path(state, beam.path).rank() == 0

    best_first = gfn_batched_best_first_search(
        state,
        policy,
        ball,
        [],
        batch_size=8,
        node_limit=100,
        time_limit=5.0,
        max_depth=6,
        branch_actions=2,
    )
    assert best_first.path is not None
    assert apply_path(state, best_first.path).rank() == 0

    hybrid = gfn_hybrid_search(
        state,
        policy,
        ball,
        [],
        widths=(1,),
        max_learned_steps=4,
        best_first_node_limit=100,
        best_first_time_limit=5.0,
    )
    assert hybrid.path is not None
    assert apply_path(state, hybrid.path).rank() == 0
