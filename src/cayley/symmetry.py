"""Cube rotational symmetry group (order 24).

The picture cube has 24 rotational symmetries: arbitrary products of whole-cube rotations
around the three face axes. Each symmetry R is a permutation of the 72 facelets that
commutes with "distance to solved" — rotating any scrambled state gives a state with the
same distance.

Training augmentation: for every (state, depth) sample, we can emit (R·state·R⁻¹, depth)
for a random R ∈ G_24. This 24-fold expands training coverage without changing labels.

Derivation: the three "whole-cube turns" are products of our three co-axial generators:
  R_f = f0 · f1 · f2    (all three f-axis layers together)
  R_r = r0 · r1 · r2
  R_d = d0 · d1 · d2
BFS from identity using {R_f, R_r, R_d} and inverses yields all 24 rotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from cayley.puzzle import PictureCube


def compute_rotations(puzzle: PictureCube) -> np.ndarray:
    """Return (24, 72) int64 array of rotation permutations R.

    Each R is represented as a permutation tuple: apply_rotation(state, R)[p] is computed
    elsewhere using R and R⁻¹. The permutations are enumerated by BFS from identity.
    """
    state_size = len(puzzle.solved_state)

    def apply_path(state, path):
        cur = state
        for m in path:
            cur = puzzle.apply_move(cur, m)
        return cur

    def compose(a, b):
        # apply-a-then-b on positions: (b ∘ a)[p] = b[a[p]]
        return tuple(b[a[p]] for p in range(state_size))

    identity = tuple(range(state_size))
    R_f = apply_path(identity, ["f0", "f1", "f2"])
    R_r = apply_path(identity, ["r0", "r1", "r2"])
    R_d = apply_path(identity, ["d0", "d1", "d2"])

    rotations = {identity}
    frontier = [identity]
    while frontier:
        nf = []
        for r in frontier:
            for base in (R_f, R_r, R_d):
                new_r = compose(r, base)
                if new_r not in rotations:
                    rotations.add(new_r)
                    nf.append(new_r)
        frontier = nf

    out = np.array(sorted(rotations), dtype=np.int64)
    assert out.shape == (24, state_size), f"expected 24 rotations, got {out.shape}"
    return out


def invert_permutation_batch(perms: "torch.Tensor") -> "torch.Tensor":
    """Given (N, S) int64 permutations, return (N, S) their inverses."""
    import torch
    N, S = perms.shape
    inv = torch.empty_like(perms)
    idx = torch.arange(S, device=perms.device).expand(N, S)
    inv.scatter_(1, perms, idx)
    return inv


def apply_random_rotation(
    states: "torch.Tensor", rotations: "torch.Tensor", inv_rotations: "torch.Tensor", generator: "torch.Generator"
) -> "torch.Tensor":
    """For each state in `states` (B, 72), pick a random rotation r ∈ {0..23} and return
    R_r · state · R_r⁻¹ as a new (B, 72) tensor.

    Distance-to-solved is invariant under this operation (rotating solved gives solved).

    Formula: new_state[p] = R[state[R⁻¹[p]]].
    Vectorized: for each sample i with rotation index r_i:
      inv = inv_rotations[r_i]     # (72,)
      temp = state[inv]            # (72,) — state read through inverse permutation
      new = rotations[r_i][temp]   # (72,) — relabel with rotation
    """
    import torch
    B = states.shape[0]
    device = states.device
    rot_idx = torch.randint(0, rotations.shape[0], (B,), generator=generator, device=device)

    # Gather the per-sample rotations and inverses: (B, 72)
    R = rotations[rot_idx]       # (B, 72)
    R_inv = inv_rotations[rot_idx]

    # Step 1: read through inverse. out1[i, p] = states[i, R_inv[i, p]]
    step1 = torch.gather(states, 1, R_inv)
    # Step 2: relabel. out[i, p] = R[i, step1[i, p]]
    new_states = torch.gather(R, 1, step1)
    return new_states
