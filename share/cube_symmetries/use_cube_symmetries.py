"""Use the 96 IHES Picture Cube symmetries (built by build_cube_symmetries.py).

Two use cases:

  1. TRAINING AUGMENTATION -- for every (state, depth) sample, emit a random
     distance-preserving relabeling.  Distance is unchanged, so the label is
     reused.  This 96x-expands coverage for free.  See `augment_numpy` /
     `augment_torch`.

  2. SEARCH SYM-ENSEMBLE -- solve several relabeled copies of a scramble and
     keep the shortest, mapping each copy's move sequence back to the original.
     See `move_relabel` and the round-trip demo in __main__.

State convention (matches the competition / puzzle_info.json):
    a state s is a length-72 int permutation; solved = [0,1,...,71];
    applying generator g:  new[i] = s[g[i]]   (i.e. new = s . g).
A symmetry P acts by conjugation:        s -> P . s . P^-1   (new[p]=P[s[P^-1[p]]]).
An inverse antisymmetry acts by:         s -> P . s^-1 . P^-1.

The two .npy / .npz files are produced by build_cube_symmetries.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_group(data_dir="."):
    """Return dict with the 48 symmetries and the full 96-element group.

    keys: sym (48,72), sym_inv (48,72),
          perms (96,72), perms_inv (96,72), is_antisymmetry (96,) bool.
    """
    d = Path(data_dir)
    sym = np.load(d / "cube_symmetries.npy").astype(np.int64)
    sym_inv = np.load(d / "cube_symmetries_inv.npy").astype(np.int64)
    g = np.load(d / "cube_symmetry_group.npz")
    return {
        "sym": sym,
        "sym_inv": sym_inv,
        "perms": g["perms"].astype(np.int64),
        "perms_inv": g["perms_inv"].astype(np.int64),
        "is_antisymmetry": g["is_antisymmetry"].astype(bool),
    }


# --------------------------------------------------------------------------- #
# single-state appliers (numpy)
# --------------------------------------------------------------------------- #
def invert_state(s: np.ndarray) -> np.ndarray:
    """Group inverse of a permutation-state: out[s[i]] = i."""
    out = np.empty_like(s)
    out[s] = np.arange(s.shape[0])
    return out


def apply_symmetry(s, P, P_inv):
    """P . s . P^-1   ->   new[p] = P[s[P^-1[p]]]."""
    return P[s[P_inv]]


def apply_antisymmetry(s, P, P_inv):
    """P . s^-1 . P^-1   (invert the state first, then conjugate)."""
    return P[invert_state(s)[P_inv]]


# --------------------------------------------------------------------------- #
# batched augmentation: pick a random element of the 96-group per state
# --------------------------------------------------------------------------- #
def augment_numpy(states: np.ndarray, group: dict, rng: np.random.Generator) -> np.ndarray:
    """states: (B,72) int.  Returns (B,72) -- one random relabeling per row.

    Half the group is antisymmetric (state inverted first); both halves preserve
    distance-to-solved, so the per-sample depth label is unchanged.
    """
    perms, perms_inv, is_anti = group["perms"], group["perms_inv"], group["is_antisymmetry"]
    B = states.shape[0]
    idx = rng.integers(0, perms.shape[0], size=B)
    P, Pinv, anti = perms[idx], perms_inv[idx], is_anti[idx]      # (B,72),(B,72),(B,)
    base = states.copy()
    if anti.any():                                               # invert the antisymmetric rows
        inv = np.argsort(states[anti], axis=1)                  # permutation inverse, row-wise
        base[anti] = inv
    step1 = np.take_along_axis(base, Pinv, axis=1)              # s . P^-1
    return np.take_along_axis(P, step1, axis=1)                 # P . (that)


def augment_torch(states, group_t, generator=None):
    """torch variant.  states: (B,72) long.  group_t: dict of long tensors
    perms/perms_inv/is_antisymmetry already on the right device.

    Drop-in replacement for src/cayley/symmetry.apply_random_rotation, but over
    the full octahedral group + inverse antisymmetry (96) instead of 24 rotations.
    """
    import torch
    perms, perms_inv = group_t["perms"], group_t["perms_inv"]
    is_anti = group_t["is_antisymmetry"]
    B, dev = states.shape[0], states.device
    idx = torch.randint(0, perms.shape[0], (B,), generator=generator, device=dev)
    P, Pinv, anti = perms[idx], perms_inv[idx], is_anti[idx]
    inv_states = torch.argsort(states, dim=1)                  # row-wise permutation inverse
    base = torch.where(anti.unsqueeze(1), inv_states, states)
    step1 = torch.gather(base, 1, Pinv)
    return torch.gather(P, 1, step1)


# --------------------------------------------------------------------------- #
# search sym-ensemble: relabel a scramble, then map a solution back
# --------------------------------------------------------------------------- #
def move_relabel(P, P_inv, generators: dict) -> dict:
    """Return {move_name: move_name} = sigma, where  P^-1 . g_m . P = g_sigma(m).

    If you transform a scramble  s' = P . s . P^-1,  solve s' to get a move list
    Q', then  [sigma[q] for q in Q']  solves the ORIGINAL scramble s.
    (And conversely [sigma^-1[q] for q in Q] turns a solution of s into one of s'.)
    """
    n = len(P)
    gen_by_perm = {tuple(int(x) for x in g): nm for nm, g in generators.items()}
    Pinv = P_inv
    sigma = {}
    for nm, g in generators.items():
        conj = tuple(Pinv[g[P[k]]] for k in range(n))           # P^-1 . g . P
        sigma[nm] = gen_by_perm[conj]
    return sigma


# --------------------------------------------------------------------------- #
# demo / self-test
# --------------------------------------------------------------------------- #
def _demo(data_dir=".", puzzle_info="puzzle_info.json"):
    import json
    group = load_group(data_dir)
    sym, sym_inv = group["sym"], group["sym_inv"]
    print(f"loaded {sym.shape[0]} symmetries, "
          f"{group['perms'].shape[0]} total (sym + inverse-antisymmetry)")

    gens = {nm: [int(x) for x in p]
            for nm, p in json.load(open(puzzle_info, encoding="utf-8"))["generators"].items()}
    names = list(gens)
    n = 72
    rng = np.random.default_rng(0)

    def scramble(k=300):
        s = np.arange(n)
        for _ in range(k):
            s = s[gens[names[rng.integers(0, 18)]]]              # new = s . g
        return s

    # (a) the 96 act as 96 DISTINCT distance-preserving maps on a real scramble
    s = scramble()
    imgs = {tuple(apply_symmetry(s, sym[i], sym_inv[i]).tolist()) for i in range(48)}
    imgs |= {tuple(apply_antisymmetry(s, sym[i], sym_inv[i]).tolist()) for i in range(48)}
    print(f"distinct images of one scramble under the 96: {len(imgs)}  (expect 96)")

    # (b) batched augmentation shape/identity check
    batch = np.stack([scramble(rng.integers(50, 400)) for _ in range(1000)])
    aug = augment_numpy(batch, group, rng)
    assert aug.shape == batch.shape
    assert all(len(set(row.tolist())) == n for row in aug), "augmented rows must stay permutations"
    print(f"augment_numpy: {batch.shape[0]} states -> shape {aug.shape}, all valid permutations")

    # (c) search round-trip: a known solution, relabeled by sigma, solves the
    #     transformed scramble.  Build s from a known path; its solution is the
    #     reversed-inverted path.  Transform by P; check the relabeled solution.
    def apply_path(s, path):
        for m in path:
            s = s[gens[m]]
        return s

    def inv_path(path):
        return [(m[1:] if m.startswith("-") else "-" + m) for m in reversed(path)]

    path = [names[rng.integers(0, 18)] for _ in range(40)]
    s = apply_path(np.arange(n), path)                          # scramble with known solution
    sol = inv_path(path)
    assert np.array_equal(apply_path(s, sol), np.arange(n)), "sanity: sol solves s"

    i = 7                                                       # pick any non-identity symmetry
    P, Pinv = sym[i], sym_inv[i]
    s_rot = apply_symmetry(s, P, Pinv)                          # s' = P . s . P^-1
    sigma = move_relabel(P, Pinv, gens)
    sigma_inv = {v: k for k, v in sigma.items()}
    sol_rot = [sigma_inv[m] for m in sol]                       # solution of s'
    ok = np.array_equal(apply_path(s_rot, sol_rot), np.arange(n))
    print(f"sym-ensemble round-trip (relabeled solution solves the rotated scramble): {ok}")
    assert ok
    print("OK")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Demo / self-test for the 96 cube symmetries.")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--puzzle-info", default="puzzle_info.json")
    a = ap.parse_args()
    _demo(a.data_dir, a.puzzle_info)
