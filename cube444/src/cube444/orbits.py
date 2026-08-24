"""Slot-orbit structure of the 4x4x4 colour cube, and the phase-1 reduction target.

This is the foundation for the two-phase (orbit-staged) solver. Everything here is
verified empirically by `scripts/10_verify_orbits.py` against the raw generators --
nothing below is assumed from geometry alone.

LAYOUT
------
The 96-vector is face-major: slot = face*16 + row*4 + col, faces 0..5, each face a
row-major 4x4 grid. Within a face:

    0  1  2  3        corners  0  3 12 15
    4  5  6  7        centers  5  6  9 10
    8  9 10 11        wings    1  2  4  7  8 11 13 14
   12 13 14 15

MEASURED ORBIT STRUCTURE
------------------------
Under the FULL 24-generator group the 96 slots split into 4 closed orbits of 24:
corners, centres, wing-A, wing-B. Closure is the load-bearing fact -- it means each
orbit's colour projection evolves autonomously, so a value function may be defined
on a sub-orbit projection with no loss.

Under the 12 OUTER-layer generators (layers 0 and 3 of each axis) the centres break
into 6 orbits of 4 -- one per face -- while corners, wing-A and wing-B each stay a
single 24-orbit. So outer moves can never move a centre sticker off its face, and
never separate the two wings of an edge.

THE PHASE-1 TARGET
------------------
    R(s)  :=  every face f's 4 centre slots carry colour f   (ABSOLUTE, not merely
                                                              monochrome)
          AND every (face, edge) wing pair is monochrome

R is invariant along outer-only walks (verified to 20k steps) and holds for solved,
so the reduction subgroup orbit <outer>.solved is contained in R. R is therefore the
correct phase-1 goal: reach R with all 24 generators, then finish inside R using only
the 12 outer generators, where the puzzle is a 3x3x3 in quarter-turn metric.

WHY THE CENTRE CONDITION IS ABSOLUTE
------------------------------------
Outer moves permute a face's 4 centre slots among themselves and can never move a
centre sticker off its face, so the colour multiset of each centre block is frozen
for the whole of phase 2. If phase 1 stops at a state whose centres are monochrome
but carry the *wrong* colour (which inner moves can easily produce), phase 2 is
unsolvable -- the cube would be solved only up to a whole-cube rotation, and whole-
cube rotations are not in our generator set. Requiring `colour == face` costs
nothing and removes that dead end. Merely-monochrome centres are still exposed as
`center_block_defects` for diagnostics.

R is *necessary* but not quite sufficient: a state can satisfy R yet be unsolvable by
outer moves alone (the classic 4x4x4 OLL / PLL parity). `phase2.py` detects that case
from the solver, and the two-phase driver simply moves on to the next endpoint.

CORNERS ARE FREE IN PHASE 1
---------------------------
R places no constraint on the 24 corner slots, and corners are a closed orbit, so the
distance-to-R of a state is *exactly* a function of its 72 non-corner slots. The
phase-1 value network is therefore defined on the 72-slot projection -- an exact
quotient, not an approximation. See `project_noncorner`.
"""

from __future__ import annotations

import numpy as np

STATE_SIZE = 96
FACE_SIZE = 16
N_FACES = 6
NUM_CLASSES = 6

# --- within-face slot roles -------------------------------------------------
CORNER_IN_FACE = (0, 3, 12, 15)
CENTER_IN_FACE = (5, 6, 9, 10)
# The 4 edges of a face, each holding 2 wing stickers: top, left, right, bottom.
WING_PAIRS_IN_FACE = ((1, 2), (4, 8), (7, 11), (13, 14))

# --- global slot sets -------------------------------------------------------
CORNER_SLOTS = np.array(
    [f * FACE_SIZE + i for f in range(N_FACES) for i in CORNER_IN_FACE], dtype=np.int64
)
CENTER_SLOTS = np.array(
    [f * FACE_SIZE + i for f in range(N_FACES) for i in CENTER_IN_FACE], dtype=np.int64
)
WING_SLOTS = np.array(
    [f * FACE_SIZE + i
     for f in range(N_FACES)
     for pair in WING_PAIRS_IN_FACE
     for i in pair],
    dtype=np.int64,
)
NONCORNER_SLOTS = np.array(
    sorted(set(range(STATE_SIZE)) - set(CORNER_SLOTS.tolist())), dtype=np.int64
)

# (6, 4) -- one centre block per face
CENTER_BLOCKS = np.array(
    [[f * FACE_SIZE + i for i in CENTER_IN_FACE] for f in range(N_FACES)], dtype=np.int64
)
# (24, 2) -- one wing pair per (face, edge)
WING_PAIRS = np.array(
    [[f * FACE_SIZE + a, f * FACE_SIZE + b]
     for f in range(N_FACES)
     for a, b in WING_PAIRS_IN_FACE],
    dtype=np.int64,
)

N_NONCORNER = len(NONCORNER_SLOTS)          # 72
# 24 centre slots that must each carry their own face colour, + 24 wing pairs.
MAX_DEFECTS = len(CENTER_SLOTS) + len(WING_PAIRS)  # 24 + 24 = 48

# The colour every centre slot must carry, aligned with CENTER_SLOTS. The solved
# state is face-major (16x colour 0, 16x colour 1, ...), so face f wants colour f.
CENTER_TARGET = CENTER_SLOTS // FACE_SIZE

# Outer-layer generators: layer index 0 or 3 on each axis. Phase 2's move set.
OUTER_LAYERS = ("0", "3")
INNER_LAYERS = ("1", "2")


def is_outer_move(name: str) -> bool:
    """f0/-f0/f3/... are outer; f1/f2/... are inner. Name form is [-]<axis><layer>."""
    return name.lstrip("-")[1] in OUTER_LAYERS


def outer_move_names(move_names) -> list[str]:
    return [m for m in move_names if is_outer_move(m)]


def inner_move_names(move_names) -> list[str]:
    return [m for m in move_names if not is_outer_move(m)]


def _as_batch(states: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(states)
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False


def center_defects(states: np.ndarray) -> np.ndarray:
    """Per state, the number of centre slots not carrying their own face colour.

    0..24. This is the ABSOLUTE condition phase 2 requires -- see module docstring.
    """
    arr, single = _as_batch(states)
    out = (arr[:, CENTER_SLOTS] != CENTER_TARGET[None, :]).sum(-1)
    return out[0] if single else out


def center_block_defects(states: np.ndarray) -> np.ndarray:
    """Diagnostic only: sum over faces of (distinct colours in the centre block - 1).

    The weaker 'merely monochrome' condition. Not the phase-1 goal -- a state can
    score 0 here and still be unsolvable by outer moves. See module docstring.
    """
    arr, single = _as_batch(states)
    blocks = np.sort(arr[:, CENTER_BLOCKS], axis=-1)    # (B, 6, 4)
    n_distinct = (blocks[..., 1:] != blocks[..., :-1]).sum(-1) + 1
    out = (n_distinct - 1).sum(-1)
    return out[0] if single else out


def wing_defects(states: np.ndarray) -> np.ndarray:
    """Per state, the number of (face, edge) wing pairs that are not monochrome."""
    arr, single = _as_batch(states)
    pairs = arr[:, WING_PAIRS]                          # (B, 24, 2)
    out = (pairs[..., 0] != pairs[..., 1]).sum(-1)
    return out[0] if single else out


def reduction_defects(states: np.ndarray) -> np.ndarray:
    """Total distance-to-R surrogate: centre defects + wing defects, in [0, 48]."""
    return center_defects(states) + wing_defects(states)


def is_reduced(states: np.ndarray) -> np.ndarray:
    """R(s): the phase-1 goal predicate. True iff the state is reduced."""
    return reduction_defects(states) == 0


def project_noncorner(states: np.ndarray) -> np.ndarray:
    """Drop the 24 corner slots -> the (…, 72) projection phase-1 V is defined on.

    Exact, not lossy for phase 1: corners are a closed orbit and R does not
    constrain them, so distance-to-R depends only on this projection.
    """
    return np.asarray(states)[..., NONCORNER_SLOTS]


def full_orbits(generators: dict[str, np.ndarray]) -> list[list[int]]:
    """Connected components of the slot graph induced by the given generators."""
    adj: list[list[int]] = [[] for _ in range(STATE_SIZE)]
    for perm in generators.values():
        p = np.asarray(perm)
        for i in range(STATE_SIZE):
            j = int(p[i])
            if j != i:
                adj[i].append(j)
                adj[j].append(i)
    seen = [False] * STATE_SIZE
    orbits: list[list[int]] = []
    for s in range(STATE_SIZE):
        if seen[s]:
            continue
        comp, stack, seen[s] = [], [s], True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        orbits.append(sorted(comp))
    return orbits


def verify_structure(generators: dict[str, np.ndarray], solved: np.ndarray) -> dict:
    """Assert every structural claim this module relies on. Raises on violation."""
    gens = {k: np.asarray(v) for k, v in generators.items()}
    report: dict[str, object] = {}

    # 1. four closed orbits of 24 under the full group
    orbits = full_orbits(gens)
    sizes = sorted(len(o) for o in orbits)
    assert sizes == [24, 24, 24, 24], f"expected 4 orbits of 24, got {sizes}"
    report["full_orbit_sizes"] = sizes

    # 2. corners == the slots fixed by every inner-layer move
    inner = [g for g in gens if not is_outer_move(g)]
    fixed = set(range(STATE_SIZE))
    for g in inner:
        p = gens[g]
        fixed &= {i for i in range(STATE_SIZE) if int(p[i]) == i}
    assert fixed == set(CORNER_SLOTS.tolist()), "corner slots != inner-fixed slots"
    report["n_inner_fixed"] = len(fixed)

    # 3. each declared slot set is a union of full-group orbits (closure)
    orbit_of = {}
    for idx, o in enumerate(orbits):
        for s in o:
            orbit_of[s] = idx
    for name, slots in (("corners", CORNER_SLOTS), ("centers", CENTER_SLOTS)):
        ids = {orbit_of[int(s)] for s in slots}
        assert len(ids) == 1, f"{name} spans {len(ids)} orbits, expected 1"
    wing_ids = {orbit_of[int(s)] for s in WING_SLOTS}
    assert len(wing_ids) == 2, f"wings span {len(wing_ids)} orbits, expected 2"
    report["wing_orbit_count"] = len(wing_ids)

    # 4. centres break into exactly the 6 per-face blocks under outer moves
    outer = {g: gens[g] for g in gens if is_outer_move(g)}
    out_orbits = full_orbits(outer)
    small = {tuple(o) for o in out_orbits if len(o) == 4}
    declared = {tuple(sorted(b.tolist())) for b in CENTER_BLOCKS}
    assert small == declared, "outer size-4 orbits != declared centre blocks"
    report["n_outer_size4_orbits"] = len(small)

    # 5. R holds at solved, and is invariant along an outer-only walk
    assert bool(is_reduced(solved)), "solved state is not reduced"
    rng = np.random.default_rng(0)
    outer_names = list(outer)
    s = np.asarray(solved).copy()
    for _ in range(5000):
        s = s[outer[outer_names[rng.integers(len(outer_names))]]]
        if not bool(is_reduced(s)):
            raise AssertionError("R broken by an outer-only walk")
    report["outer_walk_steps_checked"] = 5000

    # 6. the non-corner projection is autonomous: no move maps a corner slot
    #    into a non-corner slot or vice versa
    corner_set = set(CORNER_SLOTS.tolist())
    for name, p in gens.items():
        for i in range(STATE_SIZE):
            if (i in corner_set) != (int(p[i]) in corner_set):
                raise AssertionError(f"generator {name} mixes corner and non-corner slots")
    report["projection_autonomous"] = True

    report["max_defects"] = MAX_DEFECTS
    return report
