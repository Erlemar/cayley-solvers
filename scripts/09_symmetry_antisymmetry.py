"""Enumerate the IHES Picture Cube symmetry + inverse-antisymmetry group.

Cube analog of megaminx/scripts/19b_symmetry_720.py, for the 72-facelet
picture cube (8 corners x 3 + 12 edges x 2 + 6 centers x 4 = 72 facelets).

A *symmetry* is a relabeling P of the 72 facelets that conjugates the
generator set to itself:

    P^-1 . g_n . P = g_{sigma(n)}        for every generator n,

for some bijection sigma of the 18 directed generators. Conjugating any state
by such a P preserves distance-to-solved (it maps solved->solved and the
move set onto itself), so P . s . P^-1 has the same optimal solution length as
s. These are the relabelings the trainer can apply for free as augmentation
(src/cayley/symmetry.py currently uses only the 24 rotation subgroup).

An *inverse antisymmetry* additionally inverts the state first:

    s  -->  P . s^-1 . P^-1 .

This is distance-preserving because every generator's inverse is also a
generator, so inverting a length-L solution yields another length-L solution
(this is the same fact the NISS code leans on via PictureCube.invert_state).
The full distance-preserving relabeling group is the 2|S| union of the
symmetries and the antisymmetries.

Method (complete; no cube geometry hard-coded):
  1. Orbits of the 72 facelets under the generator group (corners/edges/centers).
  2. Enumerate candidate sigma by backtracking under three NECESSARY conditions:
       - cycle-type(sigma(n)) == cycle-type(n),
       - inverse-consistency: sigma(n^-1) == sigma(n)^-1,
       - commutation-preservation: [g_a,g_b]=1  iff  [g_sigma(a),g_sigma(b)]=1.
     (Over-generation is fine; verify_P below is the ground truth.)
  3. For each sigma, reconstruct every P by anchor propagation inside each
     orbit  ( P[g_sigma(n)[k]] = g_n[P[k]] ), then take the product across
     orbits. Each P determines its own sigma, so no cross-sigma duplicates.
  4. verify_P re-checks the exact conjugation identity for all 18 generators.
  5. Group-closure check; assert the 24 known rotations are a subgroup.

Outputs (data/):
  cube_symmetries.npy        (S, 72)  int16  -- the symmetry perms P
  cube_symmetries_inv.npy    (S, 72)  int16  -- their inverses P^-1
  cube_symmetry_group.npz    full 2S group: perms, perms_inv, is_antisymmetry
  cube_symmetries_meta.json  counts + provenance

Run:  .venv/Scripts/python.exe scripts/09_symmetry_antisymmetry.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# permutation helpers (arrays are puzzle-info convention: apply(s,g)[i]=s[g[i]])
# --------------------------------------------------------------------------- #
def perm_inverse(g: list[int]) -> list[int]:
    inv = [0] * len(g)
    for i, v in enumerate(g):
        inv[v] = i
    return inv


def perm_mul(a: list[int], b: list[int]) -> tuple[int, ...]:
    """(a after b)[i] = a[b[i]] -- only used for the commutation test."""
    return tuple(a[b[i]] for i in range(len(b)))


def cycle_type(g: list[int]) -> tuple[int, ...]:
    n = len(g)
    seen = [False] * n
    lens: list[int] = []
    for s in range(n):
        if seen[s]:
            continue
        ln = 0
        x = s
        while not seen[x]:
            seen[x] = True
            x = g[x]
            ln += 1
        lens.append(ln)
    return tuple(sorted(lens))


def main() -> int:
    t0 = time.time()
    info = json.load(open(PROJECT / "data" / "puzzle_info.json", encoding="utf-8"))
    gens_dict = {n: list(int(x) for x in perm) for n, perm in info["generators"].items()}
    gen_names = list(gens_dict.keys())
    gens = [gens_dict[n] for n in gen_names]
    n_gen = len(gens)
    state_size = len(info["central_state"])
    print(f"state_size={state_size}, n_generators={n_gen}", flush=True)
    print(f"generators: {gen_names}", flush=True)

    name_to_idx = {n: i for i, n in enumerate(gen_names)}

    def inverse_name(n: str) -> str:
        return n[1:] if n.startswith("-") else "-" + n

    inv_idx = [name_to_idx[inverse_name(n)] for n in gen_names]
    # sanity: every generator . its inverse == identity
    ident = list(range(state_size))
    for i in range(n_gen):
        assert perm_mul(gens[i], gens[inv_idx[i]]) == tuple(ident), f"{gen_names[i]} inverse broken"

    # --- (1) sticker orbits under the full generator group -------------------
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
    orbit_sizes = [len(o) for o in orbits]
    print(f"orbits: {len(orbits)} of sizes {orbit_sizes}", flush=True)
    assert sum(orbit_sizes) == state_size

    # --- generator invariants for sigma pruning ------------------------------
    cyc = [cycle_type(g) for g in gens]
    commute = [[perm_mul(gens[i], gens[j]) == perm_mul(gens[j], gens[i])
                for j in range(n_gen)] for i in range(n_gen)]
    domain = [[j for j in range(n_gen) if cyc[j] == cyc[i]] for i in range(n_gen)]
    print("distinct cycle types among generators: "
          f"{sorted(set(cyc))}", flush=True)

    # --- (2) backtrack candidate sigma (necessary conditions only) -----------
    sigma = [-1] * n_gen
    used = [False] * n_gen
    candidate_sigmas: list[tuple[int, ...]] = []

    def consistent(i: int, j: int) -> bool:
        # commutation must be preserved against every already-assigned generator
        for a in range(n_gen):
            sa = sigma[a]
            if sa == -1 or a == i:
                continue
            if commute[a][i] != commute[sa][j]:
                return False
        return True

    def backtrack(start: int) -> None:
        idx = start
        while idx < n_gen and sigma[idx] != -1:
            idx += 1
        if idx == n_gen:
            candidate_sigmas.append(tuple(sigma))
            return
        i = idx
        ii = inv_idx[i]
        for j in domain[i]:
            jj = inv_idx[j]
            if used[j] or used[jj]:
                continue
            if (ii == i) != (jj == j):  # self-inverse must map to self-inverse
                continue
            if not consistent(i, j):
                continue
            # tentatively assign i and its inverse partner
            sigma[i] = j
            used[j] = True
            assigned_inv = False
            if ii != i:
                if consistent(ii, jj):
                    sigma[ii] = jj
                    used[jj] = True
                    assigned_inv = True
                else:
                    sigma[i] = -1
                    used[j] = False
                    continue
            backtrack(idx + 1)
            sigma[i] = -1
            used[j] = False
            if assigned_inv:
                sigma[ii] = -1
                used[jj] = False

    backtrack(0)
    print(f"candidate sigmas (necessary conditions): {len(candidate_sigmas)} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # --- (3) reconstruct P per sigma via anchor propagation ------------------
    def propagate(anchor: int, target: int, sg: tuple[int, ...]) -> list[int] | None:
        P = [-1] * state_size
        P[anchor] = target
        wl = [anchor]
        while wl:
            nxt: list[int] = []
            for k in wl:
                pk = P[k]
                for n in range(n_gen):
                    slot = gens[sg[n]][k]
                    val = gens[n][pk]
                    if P[slot] == -1:
                        P[slot] = val
                        nxt.append(slot)
                    elif P[slot] != val:
                        return None
            wl = nxt
        return P

    def verify_P(P: list[int], sg: tuple[int, ...]) -> bool:
        if len(set(P)) != state_size or -1 in P:
            return False
        Pinv = perm_inverse(P)
        for n in range(n_gen):
            gn = gens[n]
            gt = gens[sg[n]]
            for k in range(state_size):
                if Pinv[gn[P[k]]] != gt[k]:
                    return False
        return True

    found: list[tuple[int, ...]] = []
    found_set: set[tuple[int, ...]] = set()
    sigmas_realized = 0
    for sg in candidate_sigmas:
        # valid partial P's per orbit (each fills exactly its orbit's positions)
        partials_per_orbit: list[list[tuple[list[int], frozenset]]] = []
        ok_sigma = True
        for opos in orbits:
            anchor = opos[0]
            seen_vals: set[tuple[int, ...]] = set()
            valids: list[tuple[list[int], frozenset]] = []
            for t in range(state_size):
                P = propagate(anchor, t, sg)
                if P is None:
                    continue
                vals = [P[p] for p in opos]
                if -1 in vals or len(set(vals)) != len(opos):
                    continue
                key = tuple(vals)
                if key in seen_vals:
                    continue
                seen_vals.add(key)
                valids.append((P, frozenset(vals)))
            if not valids:
                ok_sigma = False
                break
            partials_per_orbit.append(valids)
        if not ok_sigma:
            continue

        # cartesian product across orbits, requiring disjoint value sets
        acc: list[tuple[list[int], frozenset]] = [([-1] * state_size, frozenset())]
        for opos, valids in zip(orbits, partials_per_orbit):
            new_acc: list[tuple[list[int], frozenset]] = []
            for accP, accV in acc:
                for pP, pV in valids:
                    if accV & pV:
                        continue
                    merged = accP[:]
                    for p in opos:
                        merged[p] = pP[p]
                    new_acc.append((merged, accV | pV))
            acc = new_acc

        realized = False
        for P, _ in acc:
            if verify_P(P, sg):
                Pt = tuple(P)
                if Pt not in found_set:
                    found_set.add(Pt)
                    found.append(Pt)
                    realized = True
        if realized:
            sigmas_realized += 1

    n_sym = len(found)
    print(f"\nfacelet-automorphism group |S| = {n_sym}   (from {sigmas_realized} "
          f"realized sigmas)  ({time.time() - t0:.1f}s)", flush=True)

    # --- map each P to the generator-permutation sigma it induces ------------
    gen_index_of = {tuple(g): i for i, g in enumerate(gens)}

    def sigma_of(P: tuple[int, ...]) -> tuple[int, ...]:
        Pinv = perm_inverse(list(P))
        return tuple(gen_index_of[tuple(Pinv[gens[n][P[k]]] for k in range(state_size))]
                     for n in range(n_gen))

    # centralizer = P's inducing the identity sigma (commute with every move).
    # These act TRIVIALLY on any state reachable from solved (P . s . P^-1 = s
    # whenever P commutes with s in G), so they are useless as augmentation --
    # exactly the megaminx 'x6 internal relabeling' phenomenon (720 = 120 x 6).
    cent = [P for P in found if sigma_of(P) == tuple(range(n_gen))]
    print(f"  internal-relabeling factor (centralizer order): {len(cent)}", flush=True)

    # --- group-closure check on the full facelet-automorphism group ----------
    arr = np.array(found, dtype=np.int64)
    sset = set(map(tuple, arr.tolist()))
    closure_ok = True
    for Pa in arr:
        for Pb in arr:
            if tuple(Pa[Pb].tolist()) not in sset:
                closure_ok = False
                break
        if not closure_ok:
            break
    ident_in = tuple(range(state_size)) in found_set
    print(f"  identity present: {ident_in};  group closure: "
          f"{'OK' if closure_ok else 'FAILED'}", flush=True)

    # --- the 24 known rotations (geometric subgroup) -------------------------
    rot_set: set[tuple[int, ...]] = set()
    try:
        sys.path.insert(0, str(PROJECT / "src"))
        from cayley.puzzle import PictureCube
        from cayley.symmetry import compute_rotations
        puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
        rots = compute_rotations(puzzle)
        rot_set = set(map(tuple, rots.tolist()))
        print(f"  compute_rotations() 24-subgroup of S: "
              f"{'YES' if rot_set <= found_set else 'NO'}", flush=True)
    except Exception as e:  # pragma: no cover - cross-check only
        print(f"  (rotation cross-check skipped: {e})", flush=True)

    # --- effective symmetries: one representative per induced sigma ----------
    # P and Q induce the same sigma iff they differ by a centralizer element, so
    # they act IDENTICALLY on every reachable state. The distinct-on-states group
    # is S / centralizer; keep one P per sigma (preferring a geometric rotation).
    by_sigma: dict[tuple[int, ...], tuple[int, ...]] = {}
    for P in found:
        sg = sigma_of(P)
        if sg not in by_sigma or (P in rot_set and by_sigma[sg] not in rot_set):
            by_sigma[sg] = P
    eff = sorted(by_sigma.values())
    n_eff = len(eff)
    n_rot_in_eff = sum(1 for P in eff if P in rot_set)
    print(f"  EFFECTIVE distinct-on-states symmetries: {n_eff} "
          f"(= |S| / centralizer; {n_rot_in_eff} of them rotations)", flush=True)

    # --- save artifacts ------------------------------------------------------
    out = PROJECT / "data"

    def inv_stack(perms: list[tuple[int, ...]]) -> np.ndarray:
        return np.stack([np.array(perm_inverse(list(p)), dtype=np.int16) for p in perms])

    eff_arr = np.array(eff, dtype=np.int16)
    eff_inv = inv_stack(eff)
    full_arr = np.array(sorted(found), dtype=np.int16)

    # PRIMARY drop-in set: effective symmetries (flag 0) + inverse antisymmetries
    # (flag 1, applied as P . invert_state(s) . P^-1).
    np.save(out / "cube_symmetries.npy", eff_arr)
    np.save(out / "cube_symmetries_inv.npy", eff_inv)
    np.savez(out / "cube_symmetry_group.npz",
             perms=np.concatenate([eff_arr, eff_arr]),
             perms_inv=np.concatenate([eff_inv, eff_inv]),
             is_antisymmetry=np.concatenate([np.zeros(n_eff, bool), np.ones(n_eff, bool)]))
    # FULL facelet-automorphism group (incl. the trivial-on-state internal factor)
    np.save(out / "cube_symmetries_full.npy", full_arr)

    meta = {
        "state_size": state_size,
        "n_generators": n_gen,
        "facelet_automorphism_group_order": int(n_sym),
        "centralizer_order": int(len(cent)),
        "effective_symmetries": int(n_eff),
        "effective_with_antisymmetry": int(2 * n_eff),
        "full_with_antisymmetry": int(2 * n_sym),
        "group_closure_ok": bool(closure_ok),
        "identity_present": bool(ident_in),
        "rotations_subgroup_ok": (bool(rot_set <= found_set) if rot_set else None),
        "antisymmetry": "s -> P . invert_state(s) . P^-1 (state inversion + conjugation)",
        "note": ("centralizer elements commute with every move, so they act "
                 "trivially on any state reachable from solved; distinct "
                 "augmentations on real states number 'effective_symmetries'."),
        "files": {
            "cube_symmetries.npy": f"effective symmetry perms P, shape ({n_eff}, {state_size})",
            "cube_symmetries_inv.npy": f"their inverses, shape ({n_eff}, {state_size})",
            "cube_symmetry_group.npz": f"effective sym+antisym, shape ({2 * n_eff}, {state_size})",
            "cube_symmetries_full.npy": f"full facelet-automorphism group, shape ({n_sym}, {state_size})",
        },
    }
    with open(out / "cube_symmetries_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nwrote (data/): cube_symmetries.npy "
          f"{eff_arr.shape}, cube_symmetries_inv.npy {eff_inv.shape}, "
          f"cube_symmetry_group.npz (2x{n_eff}), cube_symmetries_full.npy {full_arr.shape}")
    print("\nSUMMARY")
    print(f"  facelet-relabeling automorphisms (theoretical max) : {n_sym}"
          f"   (= 48 octahedral x {len(cent)} internal)")
    print(f"  ... with inverse antisymmetry                      : {2 * n_sym}")
    print(f"  EFFECTIVE distinct symmetries on real states       : {n_eff}")
    print(f"  ... with inverse antisymmetry (the useful number)  : {2 * n_eff}")
    print(f"  ({time.time() - t0:.1f}s total)")
    return 0 if (closure_ok and ident_in and n_eff == 48) else 1


# --------------------------------------------------------------------------- #
# convenience appliers (numpy), matching src/cayley/symmetry.apply_random_rotation
# --------------------------------------------------------------------------- #
def apply_symmetry(state: np.ndarray, P: np.ndarray, P_inv: np.ndarray) -> np.ndarray:
    """new[p] = P[state[P_inv[p]]]  ==  P . state . P^-1  (conjugation)."""
    return P[state[P_inv]]


def apply_antisymmetry(state: np.ndarray, P: np.ndarray, P_inv: np.ndarray) -> np.ndarray:
    """new = P . state^-1 . P^-1  (group-invert the state, then conjugate)."""
    inv_state = np.empty_like(state)
    inv_state[state] = np.arange(state.shape[0])
    return P[inv_state[P_inv]]


if __name__ == "__main__":
    sys.exit(main())
