"""Piece decomposition for the IHES Picture Cube.

The 72-facelet state decomposes into three piece classes:
  - 8 corner pieces × 3 stickers
  - 12 edge pieces × 2 stickers
  - 6 center pieces × 4 stickers

Picture cube specifics:
  - Corners / edges behave like standard 3×3×3: each piece moves between slots, stickers
    within a piece keep fixed relative positions.
  - Centers move between faces too (empirically verified). Each center is a 4-sticker
    rotational unit; the full group swaps into another face's slot on any turn that
    moves that face.

Encoding (52 small-vocabulary features per state):
  - corner_id[8]   ∈ {0..7}   : which piece is in each corner slot
  - corner_ori[8]  ∈ {0,1,2}  : its rotation relative to slot
  - edge_id[12]    ∈ {0..11}
  - edge_ori[12]   ∈ {0,1}
  - center_id[6]   ∈ {0..5}
  - center_ori[6]  ∈ {0,1,2,3}

Orientation convention: for a piece with canonical sticker tuple (A, B, …), placed in
slot with sticker tuple (a, b, …) reading the state: ori = index of A within the
current slot's sticker tuple. Solved state → all ori = 0.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


CORNERS: tuple[tuple[int, int, int], ...] = (
    (0, 38, 48), (2, 26, 36), (9, 12, 50), (11, 14, 24),
    (21, 59, 60), (23, 33, 62), (35, 45, 71), (47, 57, 69),
)
EDGES: tuple[tuple[int, int], ...] = (
    (1, 37), (3, 49), (8, 25), (10, 13), (15, 56),
    (20, 27), (22, 61), (32, 39), (34, 68), (44, 51), (46, 70), (58, 63),
)
# Each face's 4 center stickers, in canonical read order (ori=0 = identity).
CENTERS: tuple[tuple[int, int, int, int], ...] = tuple(
    (face * 12 + 4, face * 12 + 5, face * 12 + 7, face * 12 + 6)
    for face in range(6)
)

N_CORNERS = len(CORNERS)      # 8
N_EDGES = len(EDGES)          # 12
N_CENTERS = len(CENTERS)      # 6
N_FEATURES = (
    N_CORNERS + N_CORNERS +   # id + ori
    N_EDGES + N_EDGES +
    N_CENTERS + N_CENTERS
)                             # 52

CORNER_ORI_CARD = 3
EDGE_ORI_CARD = 2
CENTER_ORI_CARD = 4


# Lookup maps: the (frozen) sorted tuple of a piece's sticker labels → piece index.
_CORNER_SET_TO_ID = {tuple(sorted(t)): i for i, t in enumerate(CORNERS)}
_EDGE_SET_TO_ID = {tuple(sorted(t)): i for i, t in enumerate(EDGES)}
_CENTER_SET_TO_ID = {tuple(sorted(t)): i for i, t in enumerate(CENTERS)}


def extract_features_numpy(state: Sequence[int]) -> np.ndarray:
    """Return a length-52 int array of piece features for a single state.

    Layout: corner_id (8) | corner_ori (8) | edge_id (12) | edge_ori (12) |
    center_id (6) | center_ori (6).
    """
    out = np.empty(N_FEATURES, dtype=np.int64)
    i = 0

    # Corners
    for slot in CORNERS:
        stickers_here = tuple(state[s] for s in slot)
        piece_id = _CORNER_SET_TO_ID[tuple(sorted(stickers_here))]
        canonical_first = CORNERS[piece_id][0]
        ori = stickers_here.index(canonical_first)
        out[i] = piece_id
        out[N_CORNERS + i] = ori
        i += 1
    i = 2 * N_CORNERS  # skip over both id and ori blocks

    # Edges
    for j, slot in enumerate(EDGES):
        stickers_here = tuple(state[s] for s in slot)
        piece_id = _EDGE_SET_TO_ID[tuple(sorted(stickers_here))]
        canonical_first = EDGES[piece_id][0]
        ori = stickers_here.index(canonical_first)
        out[i + j] = piece_id
        out[i + N_EDGES + j] = ori
    i = 2 * N_CORNERS + 2 * N_EDGES

    # Centers
    for j, slot in enumerate(CENTERS):
        stickers_here = tuple(state[s] for s in slot)
        piece_id = _CENTER_SET_TO_ID[tuple(sorted(stickers_here))]
        canonical_first = CENTERS[piece_id][0]
        ori = stickers_here.index(canonical_first)
        out[i + j] = piece_id
        out[i + N_CENTERS + j] = ori

    return out


def extract_features_batch_numpy(states: np.ndarray) -> np.ndarray:
    """(N, 72) state batch → (N, 52) feature batch. Plain Python loop; fine for eval-time
    one-shot preprocessing. For training we vectorize via `build_gather_indices` below."""
    N = states.shape[0]
    out = np.empty((N, N_FEATURES), dtype=np.int64)
    for i in range(N):
        out[i] = extract_features_numpy(states[i].tolist())
    return out


# ---- Vectorized GPU-friendly extraction ----------------------------------------------
#
# For training we want to compute features from a (B, 72) int64 tensor on GPU.
# We avoid Python loops by precomputing:
#   - corner_positions: (8, 3) int64 — sticker indices per slot
#   - corner_canonical_multiset_hash: (8,) int64 — hash of each piece's sticker multiset
#   - ... etc.
#
# Then at runtime:
#   for each slot, read the 3 stickers (gather) → sorted multiset hash → lookup table.
#   Orientation = index of canonical_first in the slot's sticker tuple (argmax of equality).


def build_gather_indices() -> dict[str, np.ndarray]:
    """Return arrays suitable for a vectorized GPU extractor.

    Keys per piece class (corner / edge / center):
      {cls}_slots            : (n, k) positions to read per slot
      {cls}_sticker_to_piece : (72,) int64 — for any sticker label in class, which piece id;
                               -1 for labels not in this class.
      {cls}_canonical_first  : (n,) — first sticker of each piece (used for orientation).
    """
    sticker_to_corner = -np.ones(72, dtype=np.int64)
    for i, t in enumerate(CORNERS):
        for s in t:
            sticker_to_corner[s] = i
    sticker_to_edge = -np.ones(72, dtype=np.int64)
    for i, t in enumerate(EDGES):
        for s in t:
            sticker_to_edge[s] = i
    sticker_to_center = -np.ones(72, dtype=np.int64)
    for i, t in enumerate(CENTERS):
        for s in t:
            sticker_to_center[s] = i

    return {
        "corner_slots": np.array(CORNERS, dtype=np.int64),
        "edge_slots": np.array(EDGES, dtype=np.int64),
        "center_slots": np.array(CENTERS, dtype=np.int64),
        "corner_sticker_to_piece": sticker_to_corner,
        "edge_sticker_to_piece": sticker_to_edge,
        "center_sticker_to_piece": sticker_to_center,
        "corner_canonical_first": np.array([t[0] for t in CORNERS], dtype=np.int64),
        "edge_canonical_first": np.array([t[0] for t in EDGES], dtype=np.int64),
        "center_canonical_first": np.array([t[0] for t in CENTERS], dtype=np.int64),
    }


class TorchFeatureExtractor:
    """GPU-friendly piece-feature extractor. Build once per device, call many times.

    Input: (B, 72) int64 state tensor.
    Output: (B, 52) int64 feature tensor in the same layout as `extract_features_numpy`.

    Uses the fact that every sticker label in {0..71} belongs to exactly one piece
    (within its class), so `sticker_to_piece[state[:, slot[0]]]` directly yields the
    piece id. Orientation is the argmax of equality between the canonical first sticker
    and each position in the slot.
    """

    def __init__(self, device: str):
        import torch
        gi = build_gather_indices()
        self.device = device

        self.corner_slots = torch.from_numpy(gi["corner_slots"]).to(device)            # (8, 3)
        self.edge_slots = torch.from_numpy(gi["edge_slots"]).to(device)                # (12, 2)
        self.center_slots = torch.from_numpy(gi["center_slots"]).to(device)            # (6, 4)
        self.corner_s2p = torch.from_numpy(gi["corner_sticker_to_piece"]).to(device)   # (72,)
        self.edge_s2p = torch.from_numpy(gi["edge_sticker_to_piece"]).to(device)
        self.center_s2p = torch.from_numpy(gi["center_sticker_to_piece"]).to(device)
        self.corner_first = torch.from_numpy(gi["corner_canonical_first"]).to(device)  # (8,)
        self.edge_first = torch.from_numpy(gi["edge_canonical_first"]).to(device)
        self.center_first = torch.from_numpy(gi["center_canonical_first"]).to(device)

    def __call__(self, state):
        import torch
        B = state.shape[0]

        # Gather piece class stickers at their slots.
        corner_stickers = state[:, self.corner_slots]   # (B, 8, 3)
        edge_stickers = state[:, self.edge_slots]       # (B, 12, 2)
        center_stickers = state[:, self.center_slots]   # (B, 6, 4)

        # Piece id: the sticker-to-piece table applied to any sticker in the slot.
        corner_id = self.corner_s2p[corner_stickers[:, :, 0]]    # (B, 8)
        edge_id = self.edge_s2p[edge_stickers[:, :, 0]]          # (B, 12)
        center_id = self.center_s2p[center_stickers[:, :, 0]]    # (B, 6)

        # Orientation: position of the canonical first sticker within the slot's tuple.
        corner_first_expected = self.corner_first[corner_id]             # (B, 8)
        edge_first_expected = self.edge_first[edge_id]                    # (B, 12)
        center_first_expected = self.center_first[center_id]              # (B, 6)

        corner_ori = (corner_stickers == corner_first_expected.unsqueeze(-1)).to(torch.int64).argmax(dim=-1)   # (B, 8)
        edge_ori = (edge_stickers == edge_first_expected.unsqueeze(-1)).to(torch.int64).argmax(dim=-1)         # (B, 12)
        center_ori = (center_stickers == center_first_expected.unsqueeze(-1)).to(torch.int64).argmax(dim=-1)   # (B, 6)

        return torch.cat([corner_id, corner_ori, edge_id, edge_ori, center_id, center_ori], dim=1)
