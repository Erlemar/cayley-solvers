from pathlib import Path

from jewel.ball import ExactBall, build_exact_ball
from jewel.puzzle import SOLVED, apply_action, apply_path


def test_small_ball(tmp_path: Path) -> None:
    root = tmp_path / "ball"
    meta = build_exact_ball(root, 4)
    assert meta["sizes"][0] == 1
    assert meta["sizes"][1] == 12
    ball = ExactBall.load(root)
    assert ball.distance(SOLVED) == 0
    for action in range(12):
        state = apply_action(SOLVED, action)
        assert ball.distance(state) == 1
        assert ball.exact_path(state) is not None
        assert apply_path(state, ball.exact_path(state)).rank() == 0

    state = apply_path(SOLVED, [0, 2, 4, 6])
    d = ball.distance(state)
    assert d is not None and d <= 4
    path = ball.exact_path(state)
    assert path is not None and len(path) == d
    assert apply_path(state, path).rank() == 0

    extended = build_exact_ball(root, 5)
    assert extended["depth"] == 5
    assert len(extended["sizes"]) == 6
