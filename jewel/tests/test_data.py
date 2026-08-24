from pathlib import Path

import numpy as np

from jewel.ball import ExactBall, build_exact_ball
from jewel.data import build_mixed_dataset, exact_action_masks
from jewel.puzzle import JewelState, apply_action


def test_exact_masks_descend(tmp_path: Path) -> None:
    root = tmp_path / "ball"
    build_exact_ball(root, 4)
    ball = ExactBall.load(root)
    ep = np.load(root / "edge_perm_d4.npy")[:100]
    eo = np.load(root / "edge_ori_d4.npy")[:100]
    ro = np.load(root / "ring_ori_d4.npy")[:100]
    d = np.full(len(ep), 4, dtype=np.uint8)
    masks = exact_action_masks(ep, eo, ro, d, ball)
    for i, mask in enumerate(masks):
        state = JewelState(ep[i], eo[i], ro[i])
        for action in range(12):
            if int(mask) & (1 << action):
                assert ball.distance(apply_action(state, action)) == 3


def test_build_tiny_mixture(tmp_path: Path) -> None:
    root = tmp_path / "ball"
    build_exact_ball(root, 4)
    out = tmp_path / "mixed.npz"
    meta = build_mixed_dataset(root, out, n_exact_balanced=100, n_exact_uniform=100, n_demo=100)
    assert meta["count"] == 300
    data = np.load(out)
    assert data["edge_perm"].shape == (300, 12)
    assert np.all(data["action_mask"] > 0)
