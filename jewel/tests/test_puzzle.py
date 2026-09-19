import numpy as np

from jewel.official import OfficialPuzzle

from jewel.puzzle import (
    ACTION_EDGE_FLIP,
    ACTION_EDGE_SRC,
    ACTION_RING_DELTA,
    GROUP_ORDER,
    INVERSE_ACTION,
    JewelState,
    SOLVED,
    apply_action,
    apply_path,
    rank_states,
)


def test_group_order_and_solved_rank() -> None:
    assert GROUP_ORDER == 2_009_078_326_886_400
    assert SOLVED.rank() == 0


def test_action_inverses_and_order_four() -> None:
    for action in range(12):
        moved = apply_action(SOLVED, action)
        assert apply_action(moved, int(INVERSE_ACTION[action])).rank() == 0
        assert apply_path(SOLVED, [action] * 4).rank() == 0


def test_rank_roundtrip_random_walks() -> None:
    rng = np.random.default_rng(7)
    state = SOLVED
    for _ in range(1_000):
        state = apply_action(state, int(rng.integers(12)))
        rank = state.rank()
        restored = JewelState.unrank(rank)
        assert np.array_equal(restored.edge_perm, state.edge_perm)
        assert np.array_equal(restored.edge_ori, state.edge_ori)
        assert np.array_equal(restored.ring_ori, state.ring_ori)


def test_vectorized_rank_matches_scalar() -> None:
    states = []
    state = SOLVED
    for action in [0, 2, 4, 6, 8, 10, 1, 9, 5]:
        state = apply_action(state, action)
        states.append(state)
    ranks = rank_states(
        np.stack([s.edge_perm for s in states]),
        np.stack([s.edge_ori for s in states]),
        np.stack([s.ring_ori for s in states]),
    )
    assert ranks.tolist() == [s.rank() for s in states]


def test_action_tables_are_valid() -> None:
    assert ACTION_EDGE_SRC.shape == (12, 12)
    assert ACTION_EDGE_FLIP.shape == (12, 12)
    assert ACTION_RING_DELTA.shape == (12, 6)
    for action in range(12):
        assert sorted(ACTION_EDGE_SRC[action].tolist()) == list(range(12))
        assert int(ACTION_EDGE_FLIP[action].sum()) % 2 == 0


def test_official_action_isomorphism_and_random_roundtrip() -> None:
    official = OfficialPuzzle.load("jewel/data/puzzle_info.json")
    rng = np.random.default_rng(19)
    structured = SOLVED
    stickers = official.central_state
    for _ in range(200):
        action = int(rng.integers(12))
        structured = apply_action(structured, action)
        stickers = official.apply_action(stickers, action)
        converted = official.to_structured(stickers)
        assert converted.rank() == structured.rank()
        assert np.array_equal(official.from_structured(structured), stickers)
