"""Group-theoretic decomposition of megaminx for staged solving.

12 faces split into TOP_HALF (U + 5 collar) and BOTTOM_HALF (D + 5 secondary collar).
Pieces are classified by which faces' stickers they have:
  - F2L (15 pieces, 35 stickers): face_set ⊆ TOP_HALF
  - LL  (15 pieces, 35 stickers): face_set ⊆ BOTTOM_HALF
  - Equator (20 pieces, 50 stickers): straddle both halves

For a two-stage solver:
  - Stage 1: drive an arbitrary state to one where F2L stickers are correct (35 conditions).
    Other 85 stickers may be anything.
  - Stage 2: from F2L-correct state, drive to V0 using the existing full-state V model.

For F2L training data: random walks BACKWARD from any F2L-correct state.
Label = walk depth = "min moves to reach F2L-correct".
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


# 6 faces in top half: U + its 5 adjacent (F, L, R, BL, BR)
TOP_HALF_FACES = frozenset(["U", "F", "L", "R", "BL", "BR"])

# 6 faces in bottom half: D + its 5 adjacent (B, DL, DR, FL, FR)
BOTTOM_HALF_FACES = frozenset(["D", "B", "DL", "DR", "FL", "FR"])


def compute_face_sets(generators: dict) -> dict[int, frozenset]:
    """For each sticker position, return the set of face names whose rotation moves it."""
    n = len(next(iter(generators.values())))
    face_set = {}
    for i in range(n):
        faces = set()
        for name, perm in generators.items():
            if perm[i] != i:
                faces.add(name.lstrip("-"))
        face_set[i] = frozenset(faces)
    return face_set


def f2l_sticker_positions(generators: dict) -> list[int]:
    """Return the 35 sticker positions belonging to F2L pieces (face_set ⊆ TOP_HALF)."""
    fs = compute_face_sets(generators)
    return sorted(i for i, faces in fs.items() if faces.issubset(TOP_HALF_FACES))


def ll_sticker_positions(generators: dict) -> list[int]:
    """Return the 35 sticker positions belonging to LL pieces (face_set ⊆ BOTTOM_HALF)."""
    fs = compute_face_sets(generators)
    return sorted(i for i, faces in fs.items() if faces.issubset(BOTTOM_HALF_FACES))


def equator_sticker_positions(generators: dict) -> list[int]:
    """Return the 50 sticker positions belonging to equator pieces (straddle both halves)."""
    fs = compute_face_sets(generators)
    return sorted(i for i, faces in fs.items()
                  if not faces.issubset(TOP_HALF_FACES)
                  and not faces.issubset(BOTTOM_HALF_FACES))


def f2l_correct(state: torch.Tensor, f2l_positions: torch.Tensor,
                solved_state: torch.Tensor) -> torch.Tensor:
    """Per-state boolean: are all F2L stickers at their solved values?

    Args:
        state: (..., state_size) tensor of sticker values.
        f2l_positions: (n_f2l,) int64 tensor of F2L sticker positions.
        solved_state: (state_size,) int8 tensor of solved values.

    Returns:
        (...,) bool tensor; True where all F2L stickers match.
    """
    f2l_state = state[..., f2l_positions]                      # (..., n_f2l)
    f2l_solved = solved_state[f2l_positions]                   # (n_f2l,)
    return (f2l_state == f2l_solved).all(dim=-1)


# Hard-coded sticker positions (derived from puzzle_info.json) — use these
# constants directly to avoid recomputing each call.
F2L_STICKERS_FROZEN = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 19, 20, 21, 22, 23,
    60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 84, 85, 104, 105,
)

# 35 stickers (35 = 5×3 corners + 5×2 edges + 5×2 edges = 35 ✓)
assert len(F2L_STICKERS_FROZEN) == 35

# Pieces breakdown:
#   5 top corners: F-L-U, F-R-U, BL-L-U, BR-R-U, BL-BR-U
#   5 top edges (with U): F-U, L-U, R-U, BL-U, BR-U
#   5 second-layer edges (no U): F-L, F-R, BL-L, BR-R, BL-BR


if __name__ == "__main__":
    # Self-test: verify against puzzle_info.json
    project = Path(__file__).resolve().parents[2]
    puzzle = json.loads((project / "data" / "puzzle_info.json").read_text())
    gens = puzzle["generators"]

    f2l = f2l_sticker_positions(gens)
    print(f"F2L stickers: {len(f2l)} positions: {f2l}")
    assert tuple(f2l) == F2L_STICKERS_FROZEN, "F2L_STICKERS_FROZEN mismatch"

    ll = ll_sticker_positions(gens)
    print(f"LL stickers:  {len(ll)} positions: {ll}")

    eq = equator_sticker_positions(gens)
    print(f"Equator:      {len(eq)} positions: {eq}")

    assert len(f2l) + len(ll) + len(eq) == 120
    print("\nself-test passed")
