from pathlib import Path

from jewel.ball import ExactBall, build_exact_ball
from jewel.model import JewelTransformer, TransformerConfig
from jewel.official import OfficialPuzzle
from jewel.puzzle import SOLVED, apply_action, apply_path
from jewel.query import JewelQueryEngine


def test_query_returns_official_exact_descending_move(tmp_path: Path) -> None:
    root = tmp_path / "ball"
    build_exact_ball(root, 4)
    ball = ExactBall.load(root)
    official = OfficialPuzzle.load("jewel/data/puzzle_info.json")
    model = JewelTransformer(
        TransformerConfig(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)
    )
    engine = JewelQueryEngine.from_components(model, official, ball=ball, device="cpu")

    state = apply_path(SOLVED, [0, 2, 4])
    result = engine.query(official.from_structured(state), top_k=4)

    assert result["status"] == "move"
    assert result["source"] == "exact_ball"
    assert result["exact_distance"] == 3
    selected = official.parse_path(result["next_move"])[0]
    assert ball.distance(apply_action(state, selected)) == 2
    assert len(result["alternatives"]) == 4

    searched = engine.solve_query(official.from_structured(state), [], top_k=4, widths=(1,))
    assert searched["verified"] is True
    assert searched["source"] == "exact_ball"
    assert searched["solution_length"] == 3
    assert searched["next_move"] is not None


def test_query_solved_state_needs_no_move() -> None:
    official = OfficialPuzzle.load("jewel/data/puzzle_info.json")
    model = JewelTransformer(
        TransformerConfig(d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)
    )
    engine = JewelQueryEngine.from_components(model, official, device="cpu")

    result = engine.query(official.central_state)

    assert result["status"] == "solved"
    assert result["next_move"] is None
