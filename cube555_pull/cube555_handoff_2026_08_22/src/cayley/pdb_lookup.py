"""Pattern Database admissible heuristic for the picture cube.

Loads the BFS-derived PDB built by `cayley-pattern-database-build` Kaggle kernel and
returns a callable that maps batched states (B, 72) to a per-state admissible lower
bound on true distance, derived from the CORNER-POSITION subspace.

Use case: combine with neural heuristic in beam search:
    score = torch.maximum(neural_value, pdb_value)
Admissibility guarantees pdb_value ≤ true_distance, so max(neural, pdb) is still a
valid lower-bound surrogate and will tie-break in favor of states the neural model
under-estimates.

Encoding: each corner slot holds one of 8 pieces, 8! = 40,320 total configurations.
We hash as `sum(corner_id[i] * 8^(7-i))` → index in [0, 16M). Lookup is an O(1) gather.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import torch

from cayley.piece_features import TorchFeatureExtractor, N_CORNERS


class PDBLookup:
    """Admissible heuristic driven by the corner-position PDB.

    The underlying dict maps (piece_id_0, ..., piece_id_7) → BFS depth from solved.
    We expand into a dense int16 tensor of size 8^8 for GPU-side gather; unused
    slots hold 0 (harmless lower bound).
    """

    def __init__(self, pdb_path: str | Path, device: str = "cuda"):
        with open(pdb_path, "rb") as f:
            data = pickle.load(f)
        corners = data.get("corners")
        if corners is None:
            raise ValueError(f"no 'corners' table in {pdb_path}")
        self.device = device
        self.feature_extractor = TorchFeatureExtractor(device=device)

        n = 8 ** N_CORNERS  # 16,777,216
        table = torch.zeros(n, dtype=torch.int16, device=device)
        # Weight vector for encoding: 8^(n-1), 8^(n-2), ..., 8^0
        self.weights = torch.tensor(
            [8 ** (N_CORNERS - 1 - i) for i in range(N_CORNERS)],
            dtype=torch.int64, device=device,
        )
        for tup, depth in corners.items():
            idx = 0
            for i, v in enumerate(tup):
                idx += v * (8 ** (N_CORNERS - 1 - i))
            table[idx] = depth
        self.table = table
        self.max_depth = int(max(corners.values()))

    @torch.no_grad()
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """(B, 72) int state tensor → (B,) float PDB depth."""
        feats = self.feature_extractor(states.long())       # (B, 52)
        corner_ids = feats[:, : N_CORNERS]                  # (B, 8) int64
        encoded = (corner_ids * self.weights).sum(dim=1)    # (B,) int64
        return self.table[encoded].to(torch.float32)
