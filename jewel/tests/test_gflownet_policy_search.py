from __future__ import annotations

import numpy as np
import torch

from jewel.ball import ExactBall, build_exact_ball
from jewel.gflownet import (
    JewelGFNConfig,
    apply_generators,
    build_gflownet,
    build_jewel_env,
    regularized_tb_loss,
    sample_forward_trajectories,
    source_entry_mask,
)
from jewel.official import OfficialPuzzle
from jewel.pdb import EdgePatternDatabase, build_edge_pdb
from jewel.policy_search import (
    greedy_policy_path,
    phs_search,
    phs_then_exact,
    policy_ida_star,
    uniform_policy,
)
from jewel.puzzle import INVERSE_ACTION, SOLVED, apply_action, apply_path, inverse_path


def test_gflownet_action_alignment_and_source_mask() -> None:
    official = OfficialPuzzle.load("jewel/data/puzzle_info.json")
    env = build_jewel_env(official)
    for action in range(12):
        preimage = env.source_preimages[action : action + 1]
        mask = source_entry_mask(preimage, env)
        assert bool(mask[0, action])
        returned = apply_generators(preimage, env.generators[action : action + 1])
        assert torch.equal(returned[0], env.solved)

        child = apply_generators(env.solved[None], env.generators[action : action + 1])
        parent = apply_generators(
            child,
            env.generators[int(INVERSE_ACTION[action]) : int(INVERSE_ACTION[action]) + 1],
        )
        assert torch.equal(parent[0], env.solved)


def test_prefix_tb_loss_is_finite_and_sampling_never_reenters_source() -> None:
    official = OfficialPuzzle.load("jewel/data/puzzle_info.json")
    env = build_jewel_env(official)
    model = build_gflownet(JewelGFNConfig(hidden=32, num_res_blocks=1, encoding="embedding", embed_dim=4))
    states, actions = sample_forward_trajectories(model, env, batch_size=8, nmax=4)
    assert states.shape == (5, 8, 48)
    assert actions.shape == (4, 8)
    assert not bool((states[1:] == env.solved[None, None]).all(dim=-1).any())
    loss, metrics = regularized_tb_loss(model, env, states, actions, reg_coef=1e-8)
    assert bool(torch.isfinite(loss))
    assert metrics["residual_rms"] > 0.0


def test_policy_search_validity_and_exact_certificate(tmp_path) -> None:
    ball_root = tmp_path / "ball"
    build_exact_ball(ball_root, 4)
    ball = ExactBall.load(ball_root)
    pdb_path = tmp_path / "pdb_012.npy"
    build_edge_pdb(pdb_path, [0, 1, 2])
    pdb = EdgePatternDatabase.load(pdb_path)

    scramble = [0, 2, 4, 6, 8, 10]
    state = apply_path(SOLVED, scramble)
    incumbent = inverse_path(scramble)

    policy_actions: dict[int, int] = {}
    cursor = state
    for action in incumbent:
        policy_actions[cursor.rank()] = action
        cursor = apply_action(cursor, action)

    def known_path_policy(query_state):
        probabilities = np.full(12, 1e-6, dtype=np.float64)
        probabilities[policy_actions.get(query_state.rank(), 0)] = 1.0
        return probabilities

    greedy = greedy_policy_path(state, ball, policy=known_path_policy)
    assert greedy is not None and apply_path(state, greedy).rank() == 0
    assert len(greedy) <= len(incumbent)

    ida = policy_ida_star(
        state,
        ball,
        [pdb],
        policy=uniform_policy,
        ordering="policy",
        incumbent_path=incumbent,
        max_depth=len(incumbent),
        node_limit=500_000,
        time_limit=30.0,
    )
    assert ida.path is not None and ida.optimal
    assert apply_path(state, ida.path).rank() == 0
    assert len(ida.path) <= len(incumbent)

    phs = phs_search(
        state,
        ball,
        [pdb],
        policy=uniform_policy,
        variant="phsh",
        node_limit=100_000,
        time_limit=30.0,
    )
    assert phs.path is not None
    assert apply_path(state, phs.path).rank() == 0

    class BatchedUniform:
        def __call__(self, _state):
            return uniform_policy(_state)

        def batch(self, states):
            return np.full((len(states), 12), 1.0 / 12.0, dtype=np.float64)

    phs_batched = phs_search(
        state,
        ball,
        [pdb],
        policy=BatchedUniform(),
        variant="phsh",
        policy_batch_size=32,
        node_limit=100_000,
        time_limit=30.0,
    )
    assert phs_batched.path == phs.path
    assert phs_batched.expanded == phs.expanded
    assert phs_batched.generated == phs.generated

    combined = phs_then_exact(
        state,
        ball,
        [pdb],
        policy=uniform_policy,
        phs_node_limit=100_000,
        phs_time_limit=30.0,
        ida_node_limit=500_000,
        ida_time_limit=30.0,
        max_depth=len(incumbent),
    )
    assert combined.path is not None and combined.optimal
    assert apply_path(state, combined.path).rank() == 0
    assert len(combined.path) == len(ida.path)
