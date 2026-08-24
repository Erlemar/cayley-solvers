from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.bridge import (  # noqa: E402
    OrbitBridgeConfig,
    OrbitBridgePolicyValueNet,
    bridge_beam_search,
    relative_global_state,
    relative_orbit_states_numpy,
)
from cube666.classical import position_orbits  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


class UniformBridgeModel(nn.Module):
    def __init__(self, action_count: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.action_count = action_count

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = torch.zeros(
            (len(states), self.action_count),
            dtype=torch.float32,
            device=states.device,
        )
        values = torch.zeros(len(states), dtype=torch.float32, device=states.device)
        return logits + self.anchor, values


def load_puzzle() -> Cube666Puzzle:
    return Cube666Puzzle.load(PROJECT / "cayley-py-666-cube" / "puzzle_info.json")


def test_relative_state_uses_the_original_generator_dynamics() -> None:
    puzzle = load_puzzle()
    goal = np.asarray(puzzle.apply_path(puzzle.solved_state, ("r0", "f3")))
    current = np.asarray(puzzle.apply_path(goal, ("d2", "-r5")))
    relative = relative_global_state(current, goal)
    moved_relative = relative[np.asarray(puzzle.generators["f1"])]
    moved_current = np.asarray(puzzle.apply_move(current, "f1"))
    assert np.array_equal(moved_relative, relative_global_state(moved_current, goal))


def test_relative_orbit_encoding_is_identity_at_waypoint() -> None:
    puzzle = load_puzzle()
    state = np.asarray(puzzle.apply_path(puzzle.solved_state, ("r2", "d4", "-f1")))
    orbits = np.asarray(position_orbits(puzzle.generators), dtype=np.uint16)
    encoded = relative_orbit_states_numpy(state, state, orbits)
    expected = np.broadcast_to(np.arange(24, dtype=np.uint8), (1, 9, 24))
    assert np.array_equal(encoded, expected)


def test_orbit_bridge_model_shapes() -> None:
    model = OrbitBridgePolicyValueNet(
        OrbitBridgeConfig(
            action_count=36,
            orbit_dim=32,
            transformer_layers=1,
            attention_heads=4,
            hidden_dim=64,
            residual_blocks=1,
        )
    )
    states = torch.arange(24).view(1, 1, 24).expand(3, 9, -1)
    logits, values = model(states)
    assert logits.shape == (3, 36)
    assert values.shape == (3,)
    assert torch.all(values >= 0)


def test_full_branch_one_step_search_replays_exactly() -> None:
    puzzle = load_puzzle()
    move_names = puzzle.move_names
    move_to_action = {name: index for index, name in enumerate(move_names)}
    inverse_actions = np.asarray(
        [move_to_action[puzzle.inverse_name(name)] for name in move_names],
        dtype=np.uint8,
    )
    generators = np.asarray([puzzle.generators[name] for name in move_names], dtype=np.uint16)
    orbits = np.asarray(position_orbits(puzzle.generators), dtype=np.uint16)
    goal = np.asarray(puzzle.apply_path(puzzle.solved_state, ("r1", "f4")), dtype=np.uint16)
    current = np.asarray(puzzle.apply_move(goal, "d3"), dtype=np.uint16)
    model = UniformBridgeModel(len(move_names))
    result = bridge_beam_search(
        current,
        goal,
        model,
        orbits,
        generators,
        inverse_actions,
        beam_width=36,
        branch_width=36,
        maximum_steps=1,
        model_batch_size=64,
    )
    assert result.solved
    assert len(result.actions) == 1
    replayed = current[generators[result.actions[0]]]
    assert np.array_equal(replayed, goal)
