from pathlib import Path

from jewel.ball import ExactBall, build_exact_ball
from jewel.exact_search import solve_ida_star
from jewel.pdb import EdgePatternDatabase, build_edge_pdb, rank_pattern, rank_patterns
from jewel.puzzle import SOLVED, apply_path

import numpy as np


def test_pattern_rank_vectorized() -> None:
    pos = np.asarray([[0, 1, 2], [5, 2, 11], [11, 10, 9]], dtype=np.uint8)
    ori = np.asarray([[0, 0, 0], [1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    assert rank_patterns(pos, ori).tolist() == [rank_pattern(p, o) for p, o in zip(pos, ori)]


def test_pdb_and_exact_search(tmp_path: Path) -> None:
    ball_root = tmp_path / "ball"
    build_exact_ball(ball_root, 4)
    ball = ExactBall.load(ball_root)
    pdb_path = tmp_path / "pdb_012.npy"
    meta = build_edge_pdb(pdb_path, [0, 1, 2])
    assert meta["reachable"] == meta["size"]
    pdb = EdgePatternDatabase.load(pdb_path)

    scramble = [0, 2, 4, 6, 8, 10, 0, 4]
    state = apply_path(SOLVED, scramble)
    result = solve_ida_star(state, ball, [pdb], max_depth=12, node_limit=1_000_000)
    assert result.optimal and result.path is not None
    assert apply_path(state, result.path).rank() == 0
