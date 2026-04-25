"""Symmetries v2: derive the icosahedral rotational symmetries of Megaminx.

V1 failed (found 0/60) because it propagated P via single-generator BFS, which only
covers one orbit of the generator action — face-centers (untouched by face turns) form
their own singleton orbits and the algorithm never reached them.

V2 fixes this by using a different invariant: a sticker permutation P is a symmetry iff
applying P preserves the SOLVED STATE and conjugates each generator into another
generator. We don't propagate via BFS through generator action — instead we treat each
candidate face permutation sigma : {0..11} -> {0..11} as the "guess" and solve for the
unique consistent sticker permutation by a constraint-propagation that walks every
sticker (not just generator-orbits).

Strategy:
  1. Use the cayleypy library's Megaminx graph if it provides explicit symmetry data.
  2. Otherwise: enumerate face permutations sigma and, for each, propagate the sticker
     mapping P[i] from a small set of "anchor" choices, checking ALL constraints
     (P preserves solved state + P conjugates each generator g to g_{sigma(g_face)}).
  3. We also try the direct approach: find all P such that P is a permutation, P
     fixes the solved state (always true since solved is the identity), and for every
     generator g, P^{-1} . g . P is also in the generator set.

The icosahedral rotation group has order 60. Output: rotations.npy (60, 120) int8.
"""

import os, sys, subprocess, time, json
from pathlib import Path
import numpy as np


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, x):
        for s in self.streams:
            try: s.write(x); s.flush()
            except Exception: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except Exception: pass


_LOG = open("/kaggle/working/run.log", "w")
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)
t_start = time.time()


# Try cayleypy first — it may have explicit symmetry data.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cayleypy"])


DATA_IN = None
for cand in [Path("/kaggle/input/cayley-megaminx-snapshot"),
             Path("/kaggle/input/datasets/artgor/cayley-megaminx-snapshot")]:
    if cand.exists():
        DATA_IN = cand; break
if DATA_IN is None:
    raise SystemExit("dataset not found")
puzzle_info = json.load(open(DATA_IN / "megaminx" / "data" / "puzzle_info.json"))
gens_dict = {n: tuple(perm) for n, perm in puzzle_info["generators"].items()}
gen_names = list(gens_dict.keys())
state_size = len(puzzle_info["central_state"])
print(f"state_size={state_size}, n_gen={len(gen_names)}", flush=True)


# Try cayleypy's puzzle definition first
try:
    from cayleypy.puzzles.puzzles import Puzzles
    if hasattr(Puzzles, "megaminx"):
        meg = Puzzles.megaminx()
        print(f"cayleypy megaminx: type={type(meg).__name__}, attrs={dir(meg)[:10]}", flush=True)
        # Look for any "symmetr" attr / method
        sym_attrs = [a for a in dir(meg) if "sym" in a.lower() or "rot" in a.lower()]
        print(f"  sym/rot attributes: {sym_attrs}", flush=True)
except Exception as e:
    print(f"cayleypy lookup failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Direct approach: find all permutations P of [0..119] such that for every
# generator g, P^{-1} . g . P is in the generator set.
#
# Key property: a sticker permutation P is determined by where it sends a SMALL
# set of "anchor" stickers (e.g., where it sends sticker 0 + sticker 1). Then
# the rest is determined by composing with the generator action AND with the
# conjugation property simultaneously.
#
# For Megaminx, each face has 11 stickers (1 center + 10 around). The 12 faces
# induce a partition of stickers into orbits-by-face. A symmetry maps each
# face-orbit to another face-orbit.
# ---------------------------------------------------------------------------

def compose(p, q):
    return tuple(p[qi] for qi in q)


def inverse(p):
    n = len(p)
    inv = [0] * n
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)


def conjugate(P, g):
    """Compute P^{-1} . g . P."""
    P_inv = inverse(P)
    return compose(P_inv, compose(g, P))


gens_tuple = [gens_dict[n] for n in gen_names]
gens_set = set(gens_tuple)


# ---------------------------------------------------------------------------
# Approach: enumerate symmetries by (1) noting which orbits exist under the
# generator action, (2) trying face-center mappings, (3) extending to all stickers.
# ---------------------------------------------------------------------------

# Find orbits under generator action: do BFS from each starting sticker and accumulate.
seen = [False] * state_size
orbits = []
for start in range(state_size):
    if seen[start]:
        continue
    orbit = set([start])
    queue = [start]
    while queue:
        new_queue = []
        for s in queue:
            for g in gens_tuple:
                t = g[s]
                if t not in orbit:
                    orbit.add(t)
                    new_queue.append(t)
        queue = new_queue
    for s in orbit:
        seen[s] = True
    orbits.append(sorted(orbit))

print(f"orbits found: {len(orbits)}", flush=True)
for i, orb in enumerate(orbits[:8]):
    print(f"  orbit {i}: {len(orb)} stickers; sample: {orb[:8]}", flush=True)


# ---------------------------------------------------------------------------
# A symmetry must MAP ORBITS TO ORBITS OF THE SAME SIZE.
# So we restrict the search: for each pair (orbit_a -> orbit_b) with |a|=|b|,
# try the bijection. The total number of such bijections is huge in general,
# but we further constrain via the conjugation property.
#
# Practical approach: enumerate using the FACE permutation sigma (which faces map
# to which) — the icosahedral rotation group acts as a subgroup of S_12 of order 60.
# For each candidate sigma, build P by composing per-face within-face rotations.
#
# To get the 60 elements of the icosahedral group:
#   The rotation group of the icosahedron (= rotation group of the dodecahedron) has
#   1 identity + 6 axes through pairs of opposite vertices (5-fold, 4 each = 24)
#   + 10 axes through pairs of opposite faces (3-fold, 2 each = 20)
#   + 15 axes through pairs of opposite edges (2-fold, 1 each = 15)
#   = 60 rotations.
#
# In our puzzle, faces are vertices of the icosahedron (dual). So:
#   - 5-fold rotations cycle 5 faces around a fixed FACE (and its opposite).
#     12 faces -> 6 axes -> 4 non-trivial each = 24
#   - 3-fold rotations cycle 3 faces around a vertex.
#     20 vertices in icosahedron -> 10 axes -> 2 non-trivial each = 20
#   - 2-fold rotations swap pairs of faces around an edge.
#     30 edges -> 15 axes -> 1 each = 15
#
# We don't have the geometry directly, but we know which faces are ADJACENT
# (share an edge) from the generator structure: each face has 5 neighbors.
# ---------------------------------------------------------------------------

# Build face adjacency from generators. Each generator (face turn) cycles the 11 stickers
# on its face, but the FACE itself doesn't move. The ADJACENCY of faces shows up because
# turning face A also moves stickers on faces adjacent to A (stickers on face A's edges).
#
# For Megaminx with 24 generators (12 faces × CW/CCW), the adjacency structure:
# stickers belonging to face A are exactly those that move under A's rotation.
# Two faces A, B are adjacent if some sticker s exists where s belongs-to A and B's
# rotation moves s — i.e., the orbits of s under {A, B} are non-trivial.
#
# Easier proxy: face A's "stickers" = positions moved by A's CW generator. We'll use
# this to identify which 12 sets of stickers correspond to which 12 faces.

# Compute moved-set per CW generator
cw_names = [n for n in gen_names if not n.startswith("-")]
print(f"CW generators: {cw_names}", flush=True)
moved_sets = {}
for n in cw_names:
    g = gens_dict[n]
    moved = frozenset(i for i in range(state_size) if g[i] != i)
    moved_sets[n] = moved
    if len(moved_sets) <= 3:
        print(f"  {n}: moves {len(moved)} stickers, sample={sorted(moved)[:8]}", flush=True)

# Each face turn moves stickers on the face itself + stickers on adjacent faces' edges.
# Two faces A, B are adjacent if their moved-sets overlap.
adjacency = {}
for a in cw_names:
    adj = []
    for b in cw_names:
        if b != a and moved_sets[a] & moved_sets[b]:
            adj.append(b)
    adjacency[a] = adj
    print(f"  {a} adjacent to: {len(adj)} faces", flush=True)


# Each face on a Megaminx has 5 adjacent faces (sharing an edge). Verify.
n_adjacent_per_face = [len(adjacency[a]) for a in cw_names]
print(f"adjacency degrees: {n_adjacent_per_face}", flush=True)


# ---------------------------------------------------------------------------
# Now enumerate face permutations sigma : {12 faces} -> {12 faces} that PRESERVE
# adjacency. These are exactly the icosahedral rotation group elements (as a
# subgroup of S_12).
# ---------------------------------------------------------------------------

# Use face indices 0..11 mapped to cw_names in order.
n_faces = len(cw_names)
adj_set = {i: set(cw_names.index(b) for b in adjacency[cw_names[i]]) for i in range(n_faces)}


# BFS over face permutations sigma that preserve adjacency, starting from identity.
# Cap by face mapping size.
from itertools import permutations as _perms

valid_sigmas = []
n_tested = 0
budget = 1_000_000  # cap brute-force; should hit 60 well before

# Only try permutations that fix face 0 (representative); compose with translations later.
# Actually: enumerate all permutations of 12 faces and filter by adjacency-preservation.
# But 12! = 479M which is too many. Better: BFS in S_12.

# Cleaner: for each face j in {0,1,...,11}, try mapping face 0 -> j and use a
# constraint-propagation to fill in the rest based on the adjacency structure.

def propagate_sigma_from_partial(sigma_partial):
    """sigma_partial: dict mapping face indices -> face indices.
    Try to extend uniquely using adjacency preservation. Return full sigma if
    consistent, else None.
    """
    sigma = dict(sigma_partial)
    changed = True
    while changed:
        changed = False
        # For each known mapping i -> sigma[i], the adjacency set must map.
        for i, si in list(sigma.items()):
            i_adj = adj_set[i]
            si_adj = adj_set[si]
            i_adj_known = {j: sigma[j] for j in i_adj if j in sigma}
            unmapped_i_adj = i_adj - sigma.keys()
            unmapped_si_adj = si_adj - set(sigma.values())
            if not unmapped_i_adj:
                # All adjacent of i are mapped — verify they go to adj of si
                if set(i_adj_known.values()) != si_adj:
                    return None
            elif len(unmapped_i_adj) == 1 and len(unmapped_si_adj) == 1:
                k = next(iter(unmapped_i_adj))
                v = next(iter(unmapped_si_adj))
                sigma[k] = v
                changed = True
    if len(sigma) != n_faces:
        return None
    if len(set(sigma.values())) != n_faces:
        return None
    # Final adjacency check
    for i in range(n_faces):
        if {sigma[j] for j in adj_set[i]} != adj_set[sigma[i]]:
            return None
    return tuple(sigma[i] for i in range(n_faces))


# Try mapping face 0 -> j for each j; for each that yields a valid sigma, also try
# all compatible orientations (rotations of face 0's adjacency).
print("\nenumerating valid face permutations...", flush=True)
for j in range(n_faces):
    # face 0 -> j is forced; need to also pin one of face 0's neighbors to determine orientation
    face0_adj = sorted(adj_set[0])  # 5 faces
    j_adj = sorted(adj_set[j])      # 5 faces
    for k in face0_adj:
        # k is adjacent to face 0; map k -> some neighbor of j
        for v in j_adj:
            partial = {0: j, k: v}
            sigma = propagate_sigma_from_partial(partial)
            if sigma and sigma not in valid_sigmas:
                valid_sigmas.append(sigma)
print(f"  found {len(valid_sigmas)} valid face permutations (target: 60)", flush=True)


# ---------------------------------------------------------------------------
# For each valid face permutation sigma, derive the unique sticker permutation P.
#
# Constraint: for each CW generator g_i (which acts on face i), P^{-1} . g_i . P =
# g_{sigma(i)} (since the symmetry maps face i's CW rotation to face sigma(i)'s CW).
# This determines how the 11 stickers around each face are permuted.
# ---------------------------------------------------------------------------

def find_sticker_perm_for_face_map(sigma):
    """Given face permutation sigma (12-tuple), find a sticker permutation P such that
    P^{-1} . cw_gens[i] . P = cw_gens[sigma[i]] for all i."""
    cw_gens = [gens_dict[cw_names[i]] for i in range(n_faces)]

    # Constraint: P[g_{sigma_i}[k]] = g_i[P[k]] for every i, k.
    # We pick a starting anchor: try each possible value for P[0] (sticker 0 -> ?).
    for start in range(state_size):
        P = [-1] * state_size
        P[0] = start
        worklist = [0]
        ok = True
        while worklist and ok:
            new_wl = []
            for k in worklist:
                pk = P[k]
                for i in range(n_faces):
                    target_k = cw_gens[sigma[i]][k]
                    target_v = cw_gens[i][pk]
                    if P[target_k] == -1:
                        P[target_k] = target_v
                        new_wl.append(target_k)
                    elif P[target_k] != target_v:
                        ok = False; break
                if not ok: break
            worklist = new_wl
        if not ok:
            continue
        # If P is incomplete, try seeding more anchors (a NEW orbit). Pick smallest unset.
        while ok and any(p == -1 for p in P):
            unset = [k for k, v in enumerate(P) if v == -1]
            seed_k = unset[0]
            # Try every value for P[seed_k]
            for seed_v in range(state_size):
                if seed_v in P[:seed_k] or seed_v in P[seed_k+1:]:
                    continue
                # No: the check above is too expensive. Just check it's not in current P.
                if seed_v in [v for v in P if v >= 0]:
                    continue
                P_try = list(P)
                P_try[seed_k] = seed_v
                wl = [seed_k]
                ok2 = True
                while wl and ok2:
                    new_wl = []
                    for k in wl:
                        pk = P_try[k]
                        for i in range(n_faces):
                            target_k = cw_gens[sigma[i]][k]
                            target_v = cw_gens[i][pk]
                            if P_try[target_k] == -1:
                                P_try[target_k] = target_v
                                new_wl.append(target_k)
                            elif P_try[target_k] != target_v:
                                ok2 = False; break
                        if not ok2: break
                    wl = new_wl
                if ok2:
                    P = P_try
                    break
            else:
                ok = False  # no consistent extension
                break
        if not ok:
            continue
        if any(p == -1 for p in P):
            continue
        if len(set(P)) != state_size:
            continue
        # Verify on all generators (CW + CCW)
        Pt = tuple(P)
        valid = True
        for n in gen_names:
            face_idx = cw_names.index(n[1:] if n.startswith("-") else n)
            target_n = ("-" if n.startswith("-") else "") + cw_names[sigma[face_idx]]
            if conjugate(Pt, gens_dict[n]) != gens_dict[target_n]:
                valid = False; break
        if valid:
            return Pt
    return None


print("\nderiving sticker permutations for each valid face perm...", flush=True)
found = []
for sigma in valid_sigmas:
    Pt = find_sticker_perm_for_face_map(sigma)
    if Pt is not None and Pt not in {tuple(p) for p in found}:
        found.append(Pt)
        print(f"  symmetry #{len(found)}/60: face_map[:6]={sigma[:6]}...", flush=True)
        if len(found) >= 60:
            break

print(f"\ntotal symmetries found: {len(found)} (target 60). dt={time.time()-t_start:.1f}s", flush=True)
arr = np.array(found, dtype=np.int8)
out_path = Path("/kaggle/working/rotations.npy")
np.save(out_path, arr)
print(f"wrote {out_path}: shape={arr.shape}", flush=True)
