"""Symmetries v3: derive the 60 icosahedral rotational symmetries of Megaminx.

V1 used single-orbit BFS over generator action — failed (face-centers in singleton
orbits, never reached). V2 enumerated face permutations and tried to extend
sticker maps but the multi-orbit seed-search was unprincipled.

V3 observation (computed locally before writing this):
  - No sticker is fixed by ALL generators (no face-center singletons).
  - The 120 stickers split into exactly 2 orbits of size 60 under generator action.

So the structure is: 2 orbits, 60 each. A symmetry P is uniquely determined by
its image of one anchor in each orbit, given a face permutation sigma that
preserves face-adjacency. The icosahedral rotation group has 60 such sigmas.

Algorithm:
  1. Compute orbits and face-adjacency.
  2. Enumerate sigmas in S_12 preserving face-adjacency (target: 60).
  3. For each sigma:
     - For each candidate anchor target in orbit 0 (60 choices), propagate via
       P[g_sigma[i][k]] = g_i[P[k]] within orbit 0.
     - For each candidate anchor target in orbit 1 (60 choices), propagate within
       orbit 1 similarly.
     - Verify the full P against ALL 24 conjugation constraints.
  4. Output 60 distinct sticker permutations to data/rotations.npy.

Expected runtime: a few minutes on a single CPU core (60 sigmas * ~3600 candidates
per sigma * O(120 * 24) propagation ~= 600M ops).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    puzzle_info = json.load(open(PROJECT / "data" / "puzzle_info.json"))
    gens_dict = {n: tuple(perm) for n, perm in puzzle_info["generators"].items()}
    gen_names = list(gens_dict.keys())
    state_size = len(puzzle_info["central_state"])
    gens = [gens_dict[n] for n in gen_names]
    cw_names = [n for n in gen_names if not n.startswith("-")]
    cw_gens = [gens_dict[n] for n in cw_names]
    n_faces = len(cw_names)
    print(f"state_size={state_size}, n_gen={len(gen_names)}, n_cw={n_faces}", flush=True)

    # --- Step 1: orbits under generator action ---
    seen = [False] * state_size
    orbits: list[list[int]] = []
    for s in range(state_size):
        if seen[s]:
            continue
        orbit = {s}
        stack = [s]
        while stack:
            u = stack.pop()
            for g in gens:
                v = g[u]
                if v not in orbit:
                    orbit.add(v)
                    stack.append(v)
        for x in orbit:
            seen[x] = True
        orbits.append(sorted(orbit))
    print(f"orbits: {[len(o) for o in orbits]}", flush=True)
    if len(orbits) != 2 or any(len(o) != 60 for o in orbits):
        print(f"WARNING: expected 2 orbits of 60, got {[len(o) for o in orbits]}",
              flush=True)
    orbit_of = [0] * state_size
    for oi, orb in enumerate(orbits):
        for s in orb:
            orbit_of[s] = oi

    # --- Step 2: face adjacency (two faces share an edge iff their CW gens overlap) ---
    moved_sets = {
        n: frozenset(i for i in range(state_size) if gens_dict[n][i] != i)
        for n in cw_names
    }
    adj_set: dict[int, set[int]] = {}
    for i, a in enumerate(cw_names):
        adj_set[i] = {j for j, b in enumerate(cw_names)
                      if a != b and moved_sets[a] & moved_sets[b]}
    deg = [len(adj_set[i]) for i in range(n_faces)]
    print(f"face adjacency degrees: {deg}", flush=True)
    if not all(d == 5 for d in deg):
        print("WARNING: expected every face to have 5 neighbors (Megaminx topology)",
              flush=True)

    # --- Step 3: enumerate sigmas in S_12 preserving face-adjacency (target: 60) ---
    # Backtracking with forward-checking: a candidate target for unmapped face F
    # must be adjacent to images of all mapped neighbors of F, and non-adjacent
    # to images of all mapped non-neighbors. Seed with two anchors (face 0 -> j
    # and one neighbor of 0 -> some neighbor of j) since the icosahedral rotation
    # group acts simply transitively on (face, oriented-neighbor) pairs.
    seen_sigmas: set[tuple[int, ...]] = set()
    face_0_nbr = sorted(adj_set[0])[0]
    n_recurse_calls = [0]

    def recurse(s: dict[int, int]) -> None:
        n_recurse_calls[0] += 1
        if len(s) == n_faces:
            for i in range(n_faces):
                if {s[j] for j in adj_set[i]} != adj_set[s[i]]:
                    return
            seen_sigmas.add(tuple(s[i] for i in range(n_faces)))
            return
        face = next(i for i in range(n_faces) if i not in s)
        used = set(s.values())
        candidates = set(range(n_faces)) - used
        for n in adj_set[face]:
            if n in s:
                candidates &= adj_set[s[n]]
        for nn in range(n_faces):
            if nn != face and nn not in adj_set[face] and nn in s:
                candidates -= adj_set[s[nn]]
        for t in candidates:
            s[face] = t
            recurse(s)
            del s[face]

    print("enumerating valid face permutations sigma...", flush=True)
    for j in range(n_faces):
        for k in adj_set[j]:
            recurse({0: j, face_0_nbr: k})
    valid_sigmas: list[tuple[int, ...]] = sorted(seen_sigmas)
    print(f"  found {len(valid_sigmas)} sigmas (target 60); recurse calls: {n_recurse_calls[0]}",
          flush=True)

    # --- Step 4: for each sigma, derive the sticker permutation P ---
    # Constraint: P[g_sigma[i][k]] = g_i[P[k]] for all i, k. (Equivalently:
    # P^{-1} . cw_gens[i] . P = cw_gens[sigma[i]].)
    #
    # Per orbit O, fixing P[anchor_O] = target uniquely determines P over all of O.
    # Try every candidate target in the same orbit as anchor (60 choices); keep
    # the one whose extension is consistent with the constraints AND with the
    # other orbit's choice on full verification.

    gens_by_name = gens_dict
    anchor_per_orbit = [orbits[0][0], orbits[1][0]]

    def propagate_in_orbit(anchor: int, target: int, sigma: tuple[int, ...],
                           sig_cw: list[tuple[int, ...]]) -> list[int] | None:
        P = [-1] * state_size
        P[anchor] = target
        worklist = [anchor]
        while worklist:
            new_wl: list[int] = []
            for k in worklist:
                pk = P[k]
                for i in range(n_faces):
                    target_k = sig_cw[i][k]
                    target_v = cw_gens[i][pk]
                    if P[target_k] == -1:
                        P[target_k] = target_v
                        new_wl.append(target_k)
                    elif P[target_k] != target_v:
                        return None
            worklist = new_wl
        return P

    def merge_partial_Ps(P0: list[int], P1: list[int]) -> list[int] | None:
        merged = [-1] * state_size
        for k in range(state_size):
            if P0[k] != -1 and P1[k] != -1 and P0[k] != P1[k]:
                return None
            merged[k] = P0[k] if P0[k] != -1 else P1[k]
        return merged

    def verify_P(P: list[int], sigma: tuple[int, ...]) -> bool:
        if any(p == -1 for p in P):
            return False
        if len(set(P)) != state_size:
            return False
        Pt = tuple(P)
        Pinv = [0] * state_size
        for i, v in enumerate(Pt):
            Pinv[v] = i
        for n in gen_names:
            base = n[1:] if n.startswith("-") else n
            face_idx = cw_names.index(base)
            target_n = ("-" if n.startswith("-") else "") + cw_names[sigma[face_idx]]
            g_n = gens_by_name[n]
            g_target = gens_by_name[target_n]
            for k in range(state_size):
                if Pinv[g_n[Pt[k]]] != g_target[k]:
                    return False
        return True

    print("\nderiving P for each sigma...", flush=True)
    found: list[tuple[int, ...]] = []
    found_set: set[tuple[int, ...]] = set()
    t_start = time.time()
    for s_idx, sigma in enumerate(valid_sigmas):
        sig_cw = [cw_gens[sigma[i]] for i in range(n_faces)]
        # Both orbits MIGHT swap (orbit_0 -> orbit_1) under some symmetries; try both.
        candidate_targets_per_orbit: list[list[list[int]]] = [[], []]
        for o_idx, orbit in enumerate(orbits):
            anchor = anchor_per_orbit[o_idx]
            targets_same = orbits[o_idx]
            targets_swap = orbits[1 - o_idx]
            for tgt in targets_same + targets_swap:
                P_partial = propagate_in_orbit(anchor, tgt, sigma, sig_cw)
                if P_partial is not None:
                    candidate_targets_per_orbit[o_idx].append(P_partial)
        # Combine: try all (orbit_0_partial, orbit_1_partial) merges.
        for P0 in candidate_targets_per_orbit[0]:
            for P1 in candidate_targets_per_orbit[1]:
                merged = merge_partial_Ps(P0, P1)
                if merged is None:
                    continue
                if not verify_P(merged, sigma):
                    continue
                Pt = tuple(merged)
                if Pt not in found_set:
                    found_set.add(Pt)
                    found.append(Pt)
        if (s_idx + 1) % 10 == 0 or s_idx == len(valid_sigmas) - 1:
            dt = time.time() - t_start
            print(f"  sigma {s_idx+1}/{len(valid_sigmas)}: total Ps found {len(found)} ({dt:.1f}s)",
                  flush=True)

    print(f"\nDONE: {len(found)} symmetries found (target 60)", flush=True)

    # Sanity check: identity should be among them.
    identity = tuple(range(state_size))
    if identity in found_set:
        print("  identity present in found set: yes")
    else:
        print("  identity present in found set: NO (bug)")

    arr = np.array(found, dtype=np.int8)
    out_path = PROJECT / "data" / "rotations.npy"
    np.save(out_path, arr)
    print(f"\nwrote {out_path}: shape={arr.shape}", flush=True)

    # Quick group-closure check: pick a few random pairs (a, b) and verify a*b is in found.
    if len(found) > 1:
        rng = np.random.default_rng(0)
        n_check = min(20, len(found))
        ok_count = 0
        for _ in range(n_check):
            a, b = rng.choice(len(found), size=2, replace=True)
            Pa = found[a]
            Pb = found[b]
            comp = tuple(Pa[Pb[i]] for i in range(state_size))
            if comp in found_set:
                ok_count += 1
        print(f"  group closure spot check: {ok_count}/{n_check} compositions in set",
              flush=True)

    # Group test: every composition a*b must be in the set.
    closure_ok = True
    arr_int = arr.astype(np.int64)
    sset = set(map(tuple, arr_int.tolist()))
    for i, Pa in enumerate(arr_int):
        for Pb in arr_int:
            if tuple(Pa[Pb].tolist()) not in sset:
                closure_ok = False
                break
        if not closure_ok:
            break
    print(f"  closure (full): {'OK' if closure_ok else 'FAILED'}")
    return 0 if (len(found) >= 60 and closure_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
