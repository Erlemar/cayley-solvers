from pathlib import Path

from jewel.ball import ExactBall, build_exact_ball
from jewel.interactive import interactive_solve
from jewel.model import JewelTransformer, TransformerConfig
from jewel.puzzle import SOLVED, apply_path


def test_interactive_exact_ball_short_circuit(tmp_path: Path) -> None:
    root = tmp_path / "ball"
    build_exact_ball(root, 4)
    ball = ExactBall.load(root)
    state = apply_path(SOLVED, [0, 2, 4])
    model = JewelTransformer(TransformerConfig(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64))
    result = interactive_solve(state, model, ball, [], widths=(1,))
    assert result.path is not None
    assert apply_path(state, result.path).rank() == 0
    assert result.expanded_states == 0
