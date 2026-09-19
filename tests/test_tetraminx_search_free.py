from pathlib import Path
import sys

import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.khoruzhii_search import KhoruzhiiSolver
from tetraminx.puzzle import Tetraminx
from tetraminx.search_free import (
    GreedyRecoveryConfig,
    greedy_recovery_solve,
    lookahead_root_costs,
)


class StaticQ(nn.Module):
    output_dim = 24
    has_value_head = False

    def forward(self, states):
        return torch.arange(24, device=states.device, dtype=torch.float32).expand(
            states.size(0), -1)


class OneStepOracleQ(nn.Module):
    output_dim = 24
    has_value_head = False

    def __init__(self, generators, solved):
        super().__init__()
        self.register_buffer("generators", generators)
        self.register_buffer("solved", solved)

    def forward(self, states):
        children = torch.gather(
            states.unsqueeze(1).expand(-1, 24, -1), 2,
            self.generators.unsqueeze(0).expand(states.size(0), -1, -1))
        solved = (children == self.solved.view(1, 1, -1)).all(dim=2)
        return torch.where(solved, torch.zeros_like(solved, dtype=torch.float32),
                           torch.full_like(solved, 10.0, dtype=torch.float32))


def load_puzzle():
    return Tetraminx.load(PROJECT / "tetraminx" / "data" / "puzzle_info.json")


def test_depth_one_lookahead_is_one_plus_q_and_respects_forbidden_action():
    puzzle = load_puzzle()
    solver = KhoruzhiiSolver(
        puzzle, StaticQ(), device="cpu", internal_batch_size=64,
        use_q_function=True)
    state = torch.tensor(puzzle.solved_state, dtype=solver.state_dtype)
    costs, diag = lookahead_root_costs(
        solver, state, beam_width=24, depth=1,
        forbidden_root_actions=torch.tensor([0]))
    assert torch.isinf(costs[0])
    assert torch.equal(costs[1:], torch.arange(2, 25, dtype=torch.float32))
    assert diag["expanded_layers"] == 1


def test_greedy_policy_reaches_exact_goal_and_returns_replayable_path():
    puzzle = load_puzzle()
    generators = torch.tensor(
        [puzzle.generators[name] for name in puzzle.move_names], dtype=torch.int64)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int8)
    model = OneStepOracleQ(generators, solved)
    solver = KhoruzhiiSolver(
        puzzle, model, device="cpu", internal_batch_size=64,
        use_q_function=True)
    initial = torch.tensor(
        puzzle.apply_move(puzzle.solved_state, puzzle.move_names[0]), dtype=torch.int8)
    found, length, path = greedy_recovery_solve(
        initial, solver,
        GreedyRecoveryConfig(max_steps=3, recovery_beam=1))
    assert found
    assert length == 1
    assert puzzle.is_solved(puzzle.apply_path(initial.tolist(), path))
