"""Megaminx piece-feature extractor for representation-upgraded ResMLP.

Static tables (slot_id → properties)
------------------------------------
The 120 sticker slots split into:
  - 60 corner-slots (3 slots per corner × 20 corners)
  - 60 edge-slots   (2 slots per edge × 30 edges)
  - 0 center-slots  (centers are NOT represented in the state vector)

Built once from `data/corner_tables.pkl` + `data/edge_tables.pkl`:

  SLOT_TO_PIECE  : (120,)  int  — piece id 0..19 for corner-slots, 20..49 for edge-slots
  SLOT_TO_FACE   : (120,)  int  — 12-class face id (alphabetical), via min(face_set)
  SLOT_TO_LPOS   : (120,)  int  — local position 0..N_LPOS-1 within assigned face

Constants:
  N_CORNERS=20  N_EDGES=30  N_PIECES=50  N_FACES=12
  CORNER_ORI_CARD=3  EDGE_ORI_CARD=2  N_ORI_CLASSES=5
  N_LPOS = max slots-per-face (data-dependent, ≤ ~14)

Dynamic features (state → per-piece orientation)
------------------------------------------------
`MegaminxFeatureExtractor(device).extract_orientations(state) → (corner_ori, edge_ori)`
mirrors the pattern in `cayley.piece_features.TorchFeatureExtractor`:
  - gather stickers at home-slot positions per piece
  - identify the piece via the sticker-to-piece lookup on one of the gathered stickers
  - orientation = position of the canonical-first sticker within the gathered tuple

Per-slot orientation class
--------------------------
For slot s in corner c:   ori_class = corner_ori[b, c]            ∈ {0, 1, 2}
For slot s in edge e:     ori_class = 3 + edge_ori[b, e]          ∈ {3, 4}

So ori_class is in {0..4} and indexes the unified e_ori embedding table.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch


_TABLES_DIR = Path(__file__).resolve().parents[2] / "data"

with open(_TABLES_DIR / "corner_tables.pkl", "rb") as f:
    _C = pickle.load(f)
with open(_TABLES_DIR / "edge_tables.pkl", "rb") as f:
    _E = pickle.load(f)


N_CORNERS = int(_C["n_corners"])
N_EDGES = int(_E["n_edges"])
N_PIECES = N_CORNERS + N_EDGES
STATE_SIZE = 120

CORNER_ORI_CARD = 3
EDGE_ORI_CARD = 2
N_ORI_CLASSES = CORNER_ORI_CARD + EDGE_ORI_CARD

CORNER_SLOTS: list[tuple[int, ...]] = list(_C["corner_slots"])
EDGE_SLOTS: list[tuple[int, ...]] = list(_E["edge_slots"])
CORNER_FACE_SET: list[tuple[str, ...]] = list(_C["corner_face_set"])
EDGE_FACE_SET: list[tuple[str, ...]] = list(_E["edge_face_set"])

FACE_NAMES = sorted({f for fs in (CORNER_FACE_SET + EDGE_FACE_SET) for f in fs})
N_FACES = len(FACE_NAMES)
FACE_NAME_TO_ID = {n: i for i, n in enumerate(FACE_NAMES)}


def _build_static_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    slot_to_piece = np.full(STATE_SIZE, -1, dtype=np.int64)
    slot_to_face = np.zeros(STATE_SIZE, dtype=np.int64)
    for c_idx, slots in enumerate(CORNER_SLOTS):
        f_id = min(FACE_NAME_TO_ID[f] for f in CORNER_FACE_SET[c_idx])
        for s in slots:
            slot_to_piece[s] = c_idx
            slot_to_face[s] = f_id
    for e_idx, slots in enumerate(EDGE_SLOTS):
        f_id = min(FACE_NAME_TO_ID[f] for f in EDGE_FACE_SET[e_idx])
        for s in slots:
            slot_to_piece[s] = N_CORNERS + e_idx
            slot_to_face[s] = f_id
    assert (slot_to_piece >= 0).all(), "every slot must belong to some piece"

    slot_to_lpos = np.zeros(STATE_SIZE, dtype=np.int64)
    n_lpos = 0
    for f_id in range(N_FACES):
        slots_on_face = np.where(slot_to_face == f_id)[0]
        slot_to_lpos[slots_on_face] = np.arange(len(slots_on_face))
        n_lpos = max(n_lpos, len(slots_on_face))
    return slot_to_piece, slot_to_face, slot_to_lpos, n_lpos


SLOT_TO_PIECE, SLOT_TO_FACE, SLOT_TO_LPOS, N_LPOS = _build_static_tables()


class MegaminxFeatureExtractor:
    """GPU-batched extractor: state → per-piece orientations.

    Use `extract_orientations(state) -> (corner_ori, edge_ori)` then expand to
    per-slot via SLOT_TO_PIECE (size 120) which indexes a concatenated piece-ori
    tensor (B, 50).

    Build once per device, call many times.
    """

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)

        self.corner_slots = torch.tensor(
            CORNER_SLOTS, dtype=torch.long, device=self.device
        )  # (20, 3)
        self.edge_slots = torch.tensor(
            EDGE_SLOTS, dtype=torch.long, device=self.device
        )  # (30, 2)

        self.corner_first = torch.tensor(
            [s[0] for s in CORNER_SLOTS], dtype=torch.long, device=self.device
        )  # (20,)
        self.edge_first = torch.tensor(
            [s[0] for s in EDGE_SLOTS], dtype=torch.long, device=self.device
        )  # (30,)

        sticker_to_corner = -torch.ones(STATE_SIZE, dtype=torch.long, device=self.device)
        for c_idx, slots in enumerate(CORNER_SLOTS):
            for s in slots:
                sticker_to_corner[s] = c_idx
        self.sticker_to_corner = sticker_to_corner

        sticker_to_edge = -torch.ones(STATE_SIZE, dtype=torch.long, device=self.device)
        for e_idx, slots in enumerate(EDGE_SLOTS):
            for s in slots:
                sticker_to_edge[s] = e_idx
        self.sticker_to_edge = sticker_to_edge

    def extract_orientations(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-state piece orientations.

        state: (B, 120) int — sticker at each slot.
        Returns:
          corner_ori: (B, 20) int in {0, 1, 2}
          edge_ori:   (B, 30) int in {0, 1}
        """
        state_l = state.long()
        corner_stickers = state_l[:, self.corner_slots]   # (B, 20, 3)
        edge_stickers = state_l[:, self.edge_slots]       # (B, 30, 2)

        # Identify which corner is currently at home_slot c by looking at any sticker
        # there (they all belong to the same corner).
        corner_id_at_home = self.sticker_to_corner[corner_stickers[:, :, 0]]   # (B, 20)
        edge_id_at_home = self.sticker_to_edge[edge_stickers[:, :, 0]]         # (B, 30)

        canonical_first_for_corner = self.corner_first[corner_id_at_home]   # (B, 20)
        canonical_first_for_edge = self.edge_first[edge_id_at_home]         # (B, 30)

        corner_ori = (
            corner_stickers == canonical_first_for_corner.unsqueeze(-1)
        ).to(torch.long).argmax(dim=-1)   # (B, 20)
        edge_ori = (
            edge_stickers == canonical_first_for_edge.unsqueeze(-1)
        ).to(torch.long).argmax(dim=-1)   # (B, 30)
        return corner_ori, edge_ori

    def extract_per_slot_ori_class(self, state: torch.Tensor) -> torch.Tensor:
        """Per-slot orientation class in {0..4}.

        For slot s belonging to corner c: ori_class[b, s] = corner_ori[b, c]    (∈ {0, 1, 2})
        For slot s belonging to edge e:   ori_class[b, s] = 3 + edge_ori[b, e]  (∈ {3, 4})
        """
        corner_ori, edge_ori = self.extract_orientations(state)
        piece_ori = torch.cat([corner_ori, edge_ori + CORNER_ORI_CARD], dim=1)   # (B, 50)
        # SLOT_TO_PIECE_T: (120,) int — piece id per slot
        if not hasattr(self, "_slot_to_piece_t"):
            self._slot_to_piece_t = torch.from_numpy(SLOT_TO_PIECE).to(self.device)
        return piece_ori[:, self._slot_to_piece_t]   # (B, 120)
