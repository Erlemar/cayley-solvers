"""Symmetries v720: extend v3 to find all 720 valid sticker permutations.

Diagnosis from instrumentation: 19_symmetry_v3.py's recurse already enumerates
**120** face-graph automorphisms (the full icosahedral group: 60 rotations
+ 60 reflections). The reason the saved rotations.npy has only 360 entries
is that v3's verify_P enforces sign-PRESERVING conjugation:

    P^-1 . cw_gen_i . P = cw_gen_sigma[i]      (sign-preserving)

This validates only the 60 rotation sigmas (× 6 internal relabelings = 360).
The 60 reflection sigmas satisfy sign-FLIPPING conjugation instead:

    P^-1 . cw_gen_i . P = ccw_gen_sigma[i]     (sign-flipping)

This script tries both sign conventions per sigma and keeps every valid P.

Expected output: 720 unique sticker permutations (60 rotation sigmas × 6
relabelings + 60 reflection sigmas × 6 relabelings).

The notebook's Cell 8 conj_idx logic already supports sign-flipping P's: the
perm_to_name lookup indexes ALL 24 generators (CW + CCW), so a P that conjugates
CW to CCW just produces a conj_idx pointing at a "-X" generator. No solve-side
code change required.

Output: data/rotations_720.npy (shape (720, 120), int8). The existing
data/rotations.npy is left untouched.
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
    ccw_gens = [tuple(int(x) for x in np.argsort(g)) for g in cw_gens]
    for i, nm in enumerate(cw_names):
        assert ccw_gens[i] == gens_dict["-" + nm], f"ccw[{nm}] mismatch"
    n_faces = len(cw_names)
    print(f"state_size={state_size}, n_cw_faces={n_faces}", flush=True)

    # --- Orbits under generator action (same as v3). ---
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
    print(f"orbits: {[len(o) for o in orbits]}")
    assert len(orbits) == 2 and all(len(o) == 60 for o in orbits)

    # --- Face adjacency (same as v3). ---
    moved_sets = {
        n: frozenset(i for i in range(state_size) if gens_dict[n][i] != i)
        for n in cw_names
    }
    adj_set: dict[int, set[int]] = {}
    for i, a in enumerate(cw_names):
        adj_set[i] = {j for j, b in enumerate(cw_names)
                      if a != b and moved_sets[a] & moved_sets[b]}
    deg = [len(adj_set[i]) for i in range(n_faces)]
    assert all(d == 5 for d in deg), f"unexpected degrees: {deg}"

    # --- Enumerate sigmas (same recursion as v3 — yields the full 120). ---
    seen_sigmas: set[tuple[int, ...]] = set()
    face_0_nbr = sorted(adj_set[0])[0]

    def recurse(s: dict[int, int]) -> None:
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

    print("enumerating face-graph automorphisms (target=120)...", flush=True)
    for j in range(n_faces):
        for k in adj_set[j]:
            recurse({0: j, face_0_nbr: k})
    valid_sigmas: list[tuple[int, ...]] = sorted(seen_sigmas)
    print(f"  found {len(valid_sigmas)} sigmas")
    assert len(valid_sigmas) == 120, f"expected 120 sigmas, got {len(valid_sigmas)}"

    anchor_per_orbit = [orbits[0][0], orbits[1][0]]

    # Propagation rule (derived in the docstring):
    #   sign = +1:  P[cw_sigma[i](k)] = cw_i(P[k])
    #   sign = -1:  P[cw_sigma[i](k)] = ccw_i(P[k])
    # Only the RHS gen changes. The LHS (which P-slot we're filling) is the
    # same orbit-traversal in either case.
    def propagate_in_orbit(anchor: int, target: int, sigma: tuple[int, ...],
                           sign: int) -> list[int] | None:
        sig_cw = [cw_gens[sigma[i]] for i in range(n_faces)]
        rhs_gens = cw_gens if sign == +1 else ccw_gens
        P = [-1] * state_size
        P[anchor] = target
        worklist = [anchor]
        while worklist:
            new_wl: list[int] = []
            for k in worklist:
                pk = P[k]
                for i in range(n_faces):
                    target_k = sig_cw[i][k]
                    target_v = rhs_gens[i][pk]
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

    def verify_P(P: list[int], sigma: tuple[int, ...], sign: int) -> bool:
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
            n_is_ccw = n.startswith("-")
            target_is_ccw = n_is_ccw if sign == +1 else (not n_is_ccw)
            target_n = ("-" if target_is_ccw else "") + cw_names[sigma[face_idx]]
            g_n = gens_dict[n]
            g_target = gens_dict[target_n]
            for k in range(state_size):
                if Pinv[g_n[Pt[k]]] != g_target[k]:
                    return False
        return True

    print("\nderiving P for each (sigma, sign)...", flush=True)
    found: list[tuple[int, ...]] = []
    found_set: set[tuple[int, ...]] = set()
    n_preserving = 0
    n_flipping = 0
    t_start = time.time()
    for s_idx, sigma in enumerate(valid_sigmas):
        for sign in (+1, -1):
            candidate_targets_per_orbit: list[list[list[int]]] = [[], []]
            for o_idx in range(2):
                anchor = anchor_per_orbit[o_idx]
                targets_same = orbits[o_idx]
                targets_swap = orbits[1 - o_idx]
                for tgt in targets_same + targets_swap:
                    P_partial = propagate_in_orbit(anchor, tgt, sigma, sign)
                    if P_partial is not None:
                        candidate_targets_per_orbit[o_idx].append(P_partial)
            sigma_sign_count = 0
            for P0 in candidate_targets_per_orbit[0]:
                for P1 in candidate_targets_per_orbit[1]:
                    merged = merge_partial_Ps(P0, P1)
                    if merged is None:
                        continue
                    if not verify_P(merged, sigma, sign):
                        continue
                    Pt = tuple(merged)
                    if Pt not in found_set:
                        found_set.add(Pt)
                        found.append(Pt)
                        sigma_sign_count += 1
            if sigma_sign_count > 0:
                if sign == +1:
                    n_preserving += 1
                else:
                    n_flipping += 1
        if (s_idx + 1) % 20 == 0 or s_idx == len(valid_sigmas) - 1:
            dt = time.time() - t_start
            print(f"  sigma {s_idx+1}/{len(valid_sigmas)}: total Ps {len(found)} "
                  f"(+1 sigmas: {n_preserving}, -1 sigmas: {n_flipping}) ({dt:.1f}s)",
                  flush=True)

    print(f"\nDONE: {len(found)} sticker perms found (target 720)")
    print(f"  sigmas valid under +1: {n_preserving} (expected 60)")
    print(f"  sigmas valid under -1: {n_flipping} (expected 60)")

    identity = tuple(range(state_size))
    print(f"  identity in set: {identity in found_set}")

    arr = np.array(found, dtype=np.int8)
    out_path = PROJECT / "data" / "rotations_720.npy"
    np.save(out_path, arr)
    print(f"\nwrote {out_path}: shape={arr.shape}")

    # Verify the existing 360-entry rotations.npy is a subset of the 720.
    old_path = PROJECT / "data" / "rotations.npy"
    if old_path.exists():
        old_arr = np.load(old_path)
        old_set = set(map(tuple, old_arr.tolist()))
        missing = old_set - found_set
        print(f"  old 360 ⊆ new 720: {'YES' if not missing else f'NO ({len(missing)} missing)'}")

    # Full group closure check.
    closure_ok = True
    arr_int = arr.astype(np.int64)
    sset = set(map(tuple, arr_int.tolist()))
    for Pa in arr_int:
        for Pb in arr_int:
            if tuple(Pa[Pb].tolist()) not in sset:
                closure_ok = False
                break
        if not closure_ok:
            break
    print(f"  closure (full): {'OK' if closure_ok else 'FAILED'}")
    return 0 if (len(found) == 720 and closure_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
