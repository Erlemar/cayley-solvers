"""Tests for megaminx.post_process. Run: python -m pytest megaminx/src/megaminx/post_process_test.py"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.post_process import (
    cancel_adjacent_inverses,
    full_post_process,
    reduce_same_face_runs,
)
from megaminx.puzzle import Megaminx


def _p() -> Megaminx:
    return Megaminx.load(PROJECT / "data" / "puzzle_info.json")


def test_inverse_pair_cancels():
    assert cancel_adjacent_inverses(["U", "-U"]) == []
    assert cancel_adjacent_inverses(["F", "U", "-U", "F"]) == ["F", "F"]


def test_five_same_face_eliminates():
    assert reduce_same_face_runs(["U"] * 5) == []
    assert reduce_same_face_runs(["-U"] * 5) == []


def test_four_same_becomes_one_inverse():
    assert reduce_same_face_runs(["U"] * 4) == ["-U"]
    assert reduce_same_face_runs(["-U"] * 4) == ["U"]


def test_three_same_becomes_two_inverse():
    assert reduce_same_face_runs(["U"] * 3) == ["-U", "-U"]


def test_mixed_same_face():
    # U + U + U + -U = U + U = net 2 CW → "U U"
    assert reduce_same_face_runs(["U", "U", "U", "-U"]) == ["U", "U"]
    # U + -U + -U + -U + -U = net -3 ≡ 2 (mod 5), so 2 CW
    assert reduce_same_face_runs(["U", "-U", "-U", "-U", "-U"]) == ["U", "U"]


def test_preserves_net_rotation():
    puzzle = _p()
    # Any run of same-face moves, post-processed, should produce the same state.
    for original in [
        ["U", "U", "U", "U"],
        ["U", "-U", "U", "U"],
        ["F", "F", "F"],
        ["-DR", "-DR", "-DR", "-DR"],
    ]:
        reduced = reduce_same_face_runs(original)
        s_orig = puzzle.apply_path(puzzle.solved_state, original)
        s_red = puzzle.apply_path(puzzle.solved_state, reduced)
        assert s_orig == s_red, f"mismatch on {original}: {s_orig[:8]} vs {s_red[:8]}"


def test_full_pipeline_preserves_state():
    puzzle = _p()
    for path in [
        ["U", "F", "-F", "U", "U", "U", "U"],  # -F.F cancel, then 5 U cancel
        ["L", "R", "-R", "L", "L", "L", "L", "L"],  # 5 L cancel
        ["U", "U", "U", "-U", "-U"],  # net 1 CW, should end as ["U"] -> no wait: 3-2=1
    ]:
        reduced = full_post_process(path)
        s_orig = puzzle.apply_path(puzzle.solved_state, path)
        s_red = puzzle.apply_path(puzzle.solved_state, reduced)
        assert s_orig == s_red, f"mismatch on {path}: reduced={reduced}"
        assert len(reduced) <= len(path), f"got longer: {len(path)} -> {len(reduced)}"


if __name__ == "__main__":
    test_inverse_pair_cancels()
    test_five_same_face_eliminates()
    test_four_same_becomes_one_inverse()
    test_three_same_becomes_two_inverse()
    test_mixed_same_face()
    test_preserves_net_rotation()
    test_full_pipeline_preserves_state()
    print("ok")
