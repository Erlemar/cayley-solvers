"""Whole-cube rotation symmetries of the 4x4x4 COLOR cube (group order 24).

The three whole-cube turns are products of all FOUR co-axial layers (the 4x4x4
has 4 layers per axis, unlike the 3x3x3's 3):

    R_f = f0 . f1 . f2 . f3
    R_r = r0 . r1 . r2 . r3
    R_d = d0 . d1 . d2 . d3

BFS from the identity over {R_f, R_r, R_d} yields all 24 rotations.

COLOR RELABELLING -- the part that differs from the picture cube
---------------------------------------------------------------
On a picture cube (unique stickers) a rotation R maps the solved state to a
permuted-but-still-"solved-looking" state and distance is preserved by
conjugation alone. Here the state is a *coloring*: rotating the solved cube
physically moves colors onto different faces, so `central[R]` is NOT
`central` -- it is a recolored solved cube.

So each rotation carries a color bijection `pi_R` with

    pi_R[ central[R] ] == central          (elementwise)

and the distance-preserving symmetry map on an arbitrary state is

    sym(s, R) = pi_R[ s[R] ]

which sends solved -> solved and preserves the move set, hence preserves
distance-to-solved. All 24 rotations admit such a `pi_R` (verified in
`build_rotations`).

Usage in a sym-ensemble beam: solve `sym(s, R)` instead of `s`; a solution word
for the rotated frame must be translated back through the corresponding move
relabelling (see `build_move_relabel`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cube444.puzzle import Cube444


def _compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Permutation composition matching `new = old[perm]`: apply a, then b."""
    return a[b]


def build_rotations(puzzle: Cube444) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotations, color_maps).

    rotations:  (24, 96) int64 -- slot permutations of the whole-cube rotation group.
    color_maps: (24, 6)  int64 -- color_maps[k][c] is the new label for color c.

    Guarantees `color_maps[k][ central[rotations[k]] ] == central` for every k.
    Index 0 is always the identity.
    """
    G = {k: np.array(v, dtype=np.int64) for k, v in puzzle.generators.items()}
    central = np.array(puzzle.solved_state, dtype=np.int64)

    axes = []
    for axis in ("f", "r", "d"):
        R = np.arange(len(central), dtype=np.int64)
        for layer in range(4):
            R = _compose(R, G[f"{axis}{layer}"])
        axes.append(R)

    ident = np.arange(len(central), dtype=np.int64)
    seen = {ident.tobytes(): ident}
    order = [ident]
    frontier = [ident]
    while frontier:
        nxt = []
        for x in frontier:
            for R in axes:
                y = _compose(x, R)
                if y.tobytes() not in seen:
                    seen[y.tobytes()] = y
                    order.append(y)
                    nxt.append(y)
        frontier = nxt

    rotations = np.stack(order)
    assert rotations.shape[0] == 24, f"expected 24 rotations, got {rotations.shape[0]}"

    color_maps = np.zeros((rotations.shape[0], 6), dtype=np.int64)
    for k, R in enumerate(rotations):
        rotated = central[R]
        pi = {}
        for a, b in zip(rotated.tolist(), central.tolist()):
            if a in pi:
                assert pi[a] == b, f"rotation {k}: color {a} maps to both {pi[a]} and {b}"
            pi[a] = b
        assert len(pi) == 6 and len(set(pi.values())) == 6, f"rotation {k}: not a color bijection"
        for c, nc in pi.items():
            color_maps[k, c] = nc

    # Verify the defining property.
    for k in range(rotations.shape[0]):
        assert np.array_equal(color_maps[k][central[rotations[k]]], central), \
            f"rotation {k} does not restore the solved coloring"

    return rotations, color_maps


def build_move_relabel(puzzle: Cube444, rotations: np.ndarray) -> np.ndarray:
    """Return (24, 24) int64 table: relabel[k, g] = index of the move that acts on
    the ORIGINAL frame the way move g acts on rotated frame k.

    Defining identity (what the table is built to satisfy):

        sym(apply(s, g), R_k) == apply(sym(s, R_k), relabel[k, g])

    i.e. `relabel[k, g]` is the move that, in rotated frame k, does what move g
    does in the original frame. So a solution word found in frame k is
    translated back to the original frame by mapping each move through the
    INVERSE of row k (see `invert_move_relabel`).

    Derivation, with `_compose(a, b) = a[b]` (apply a, then b; associative):
        sym(s, R)      = pi[ s[R] ]
        LHS = pi[ s[ _compose(G_g, R) ] ]
        RHS = pi[ s[ _compose(R, G_g') ] ]
    so  _compose(G_g, R) == _compose(R, G_g'), hence

        G_g' = _compose(Rinv, _compose(G_g, R)) = Rinv[ G_g[ R ] ]
    """
    names = list(puzzle.move_names)
    G = np.array([puzzle.generators[n] for n in names], dtype=np.int64)
    n_gen = len(names)
    lookup = {G[i].tobytes(): i for i in range(n_gen)}

    relabel = np.zeros((rotations.shape[0], n_gen), dtype=np.int64)
    for k, R in enumerate(rotations):
        Rinv = np.argsort(R)
        for g in range(n_gen):
            conj = _compose(Rinv, _compose(G[g], R))
            idx = lookup.get(conj.tobytes())
            assert idx is not None, f"conjugate of {names[g]} by rotation {k} is not a generator"
            relabel[k, g] = idx
        assert len(set(relabel[k].tolist())) == n_gen, f"rotation {k}: relabel is not a bijection"
    return relabel


def invert_move_relabel(relabel: np.ndarray) -> np.ndarray:
    """Row-wise inverse of `build_move_relabel`.

    If a beam solved frame k and returned move indices g'_1..g'_n, the original
    frame is solved by `inv[k, g'_1] .. inv[k, g'_n]` (same order, no reversal).
    """
    inv = np.zeros_like(relabel)
    for k in range(relabel.shape[0]):
        inv[k, relabel[k]] = np.arange(relabel.shape[1])
    return inv


def apply_symmetry(states: np.ndarray, rotation: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    """sym(s, R) = pi_R[ s[R] ] for a batch of states (N, 96)."""
    return color_map[states[:, rotation]]


def save_tables(puzzle: Cube444, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rotations, color_maps = build_rotations(puzzle)
    relabel = build_move_relabel(puzzle, rotations)
    inv_relabel = invert_move_relabel(relabel)
    np.save(out_dir / "rotations_24.npy", rotations)
    np.save(out_dir / "color_maps_24.npy", color_maps)
    np.save(out_dir / "move_relabel_24.npy", relabel)
    np.save(out_dir / "move_relabel_inv_24.npy", inv_relabel)
    print(f"wrote rotations_24.npy {rotations.shape}, color_maps_24.npy {color_maps.shape}, "
          f"move_relabel_24.npy {relabel.shape}, move_relabel_inv_24.npy {inv_relabel.shape} "
          f"-> {out_dir}")
