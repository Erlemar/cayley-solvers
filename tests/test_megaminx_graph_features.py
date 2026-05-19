"""Unit tests for megaminx graph_features.pt and action_relabel.pt.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_megaminx_graph_features.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
MEGAMINX = PROJECT / "megaminx"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(MEGAMINX / "src"))

from megaminx.puzzle import Megaminx


PUZZLE_INFO = MEGAMINX / "data" / "puzzle_info.json"
GRAPH_FEATURES = MEGAMINX / "data" / "graph_features.pt"
ACTION_RELABEL = MEGAMINX / "data" / "action_relabel.pt"
ROTATIONS = MEGAMINX / "data" / "rotations.npy"


def test_graph_features_exist():
    assert GRAPH_FEATURES.exists(), (
        f"missing {GRAPH_FEATURES}. Run "
        f"`.venv/Scripts/python.exe megaminx/scripts/72_build_graph_features.py` first."
    )


def test_action_relabel_exist():
    assert ACTION_RELABEL.exists(), (
        f"missing {ACTION_RELABEL}. Run "
        f"`.venv/Scripts/python.exe megaminx/scripts/73_build_action_relabel.py` first."
    )


def _load_features() -> dict:
    return torch.load(GRAPH_FEATURES, map_location="cpu", weights_only=False)


def test_features_shapes_and_dtypes():
    gf = _load_features()
    assert gf["state_size"] == 120
    assert gf["n_faces"] == 12
    assert gf["n_pieces"] == 50
    assert gf["n_relations"] == 6
    assert gf["n_dist_buckets"] == 9
    assert gf["piece_type"].shape == (120,)
    assert gf["piece_id"].shape == (120,)
    assert gf["face_set"].shape == (120, 3)
    assert gf["relation_id"].shape == (120, 120)
    assert gf["dist_bucket"].shape == (120, 120)
    assert gf["relation_id"].dtype == torch.long
    assert gf["dist_bucket"].dtype == torch.long


def test_piece_type_counts():
    gf = _load_features()
    pt = gf["piece_type"]
    assert (pt == 0).sum().item() == 60, "expected 60 corner stickers"
    assert (pt == 1).sum().item() == 60, "expected 60 edge stickers"


def test_piece_id_consistency():
    gf = _load_features()
    pid = gf["piece_id"]
    assert pid.min().item() == 0
    assert pid.max().item() == 49
    # Corners (piece_type=0) -> piece_id in [0, 20)
    # Edges  (piece_type=1) -> piece_id in [20, 50)
    pt = gf["piece_type"]
    assert pid[pt == 0].max().item() < 20
    assert pid[pt == 1].min().item() >= 20
    # Each corner piece has 3 stickers, each edge piece has 2
    from collections import Counter
    counts = Counter(int(p) for p in pid.tolist())
    for cid in range(20):
        assert counts[cid] == 3, f"corner piece {cid} has {counts[cid]} stickers"
    for eid in range(20, 50):
        assert counts[eid] == 2, f"edge piece {eid} has {counts[eid]} stickers"


def test_relation_id_symmetric():
    """Relation IDs are symmetric for our defined relations."""
    gf = _load_features()
    rel = gf["relation_id"]
    assert torch.equal(rel, rel.t()), "relation_id is not symmetric"


def test_relation_id_diagonal():
    gf = _load_features()
    rel = gf["relation_id"]
    diag = torch.diagonal(rel)
    assert (diag == 1).all(), "diagonal must be relation 1 (same_position)"


def test_generator_edges_in_relation():
    """Every (i, gen[i]) pair from any forward generator should have relation_id == 2."""
    puzzle = Megaminx.load(PUZZLE_INFO)
    gf = _load_features()
    rel = gf["relation_id"]
    forward_faces = [n for n in puzzle.move_names if not n.startswith("-")]
    for name in forward_faces:
        g = puzzle.generators[name]
        for i in range(120):
            j = g[i]
            if j != i:
                assert rel[i, j].item() == 2, (
                    f"({i}, {j}) via {name}: expected relation 2, got {rel[i, j].item()}"
                )


def test_dist_bucket_properties():
    gf = _load_features()
    db = gf["dist_bucket"]
    # Diagonal is 0
    diag = torch.diagonal(db)
    assert (diag == 0).all(), "diag of dist_bucket must be 0"
    # Symmetric (undirected graph)
    assert torch.equal(db, db.t()), "dist_bucket is not symmetric"
    # Max bucket index = 8 (cap); the actual max distance in the megaminx generator
    # graph is 6, so all values should fit comfortably below the cap.
    assert db.max().item() <= 8
    # Generator-edge pairs have distance 1
    rel = gf["relation_id"]
    assert ((rel == 2) == (db == 1)).all(), (
        "generator-edge relation should exactly match dist_bucket == 1"
    )


def test_graph_connected():
    gf = _load_features()
    db = gf["dist_bucket"]
    # Reachable from 0: bucket value < 8 OR == 8 (capped). Just check no INF placeholder.
    assert db.max().item() < 9, "graph appears disconnected (bucket overflow)"


def test_edge_index_consistency():
    gf = _load_features()
    edge_index = gf["edge_index"]
    edge_type = gf["edge_type"]
    rel = gf["relation_id"]
    # All edges have relation_id == 2
    src = edge_index[0]
    dst = edge_index[1]
    assert (rel[src, dst] == 2).all()
    # Sorted, undirected (src < dst)
    assert (src < dst).all(), "edge_index should be sorted with src < dst"
    # Count: 300 undirected edges (12 faces × 5 cycles × 5 edges-per-cycle)
    assert edge_index.size(1) == 300, f"got {edge_index.size(1)} edges, expected 300"
    # edge_type in valid face range
    assert edge_type.min().item() >= 0
    assert edge_type.max().item() < gf["n_faces"]


def test_action_relabel_shape_and_round_trip():
    """Verify action_relabel[R, a] satisfies the conjugation identity for random samples."""
    puzzle = Megaminx.load(PUZZLE_INFO)
    ar = torch.load(ACTION_RELABEL, map_location="cpu", weights_only=False)
    rotations = np.load(ROTATIONS)
    relabel = ar["relabel"]
    n_rot = rotations.shape[0]
    n_gen = len(puzzle.move_names)
    assert relabel.shape == (n_rot, n_gen)

    rng = np.random.default_rng(1234)
    n_samples = 100
    for _ in range(n_samples):
        # Random walk
        s = puzzle.solved_state
        for _ in range(int(rng.integers(0, 20))):
            mv = puzzle.move_names[int(rng.integers(0, n_gen))]
            s = puzzle.apply_move(s, mv)
        r = int(rng.integers(0, n_rot))
        a = int(rng.integers(0, n_gen))
        R = rotations[r].astype(np.int64)
        R_inv = np.empty_like(R)
        R_inv[R] = np.arange(len(R))
        s_rot = tuple(int(R[s[int(R_inv[i])]]) for i in range(len(s)))
        a_prime = int(relabel[r, a].item())
        lhs = tuple(s_rot[g] for g in puzzle.generators[puzzle.move_names[a_prime]])
        s_after = tuple(s[g] for g in puzzle.generators[puzzle.move_names[a]])
        rhs = tuple(int(R[s_after[int(R_inv[i])]]) for i in range(len(s)))
        assert lhs == rhs, f"round trip failed: rot={r}, action={a}"


def test_action_relabel_identity_rotation():
    """Rotation 0 should be the identity, and identity rotation gives relabel = arange."""
    ar = torch.load(ACTION_RELABEL, map_location="cpu", weights_only=False)
    rotations = np.load(ROTATIONS)
    # Find the identity rotation: R[i] == i for all i
    n_gen = ar["relabel"].shape[1]
    identity_idx = None
    for r in range(rotations.shape[0]):
        if (rotations[r] == np.arange(120)).all():
            identity_idx = r
            break
    assert identity_idx is not None, "no identity rotation in rotations.npy"
    relabel_id = ar["relabel"][identity_idx]
    assert (relabel_id == torch.arange(n_gen)).all(), (
        "identity rotation should map every action to itself"
    )
