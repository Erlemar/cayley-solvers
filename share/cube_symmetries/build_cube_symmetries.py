"""Build the 96 distance-preserving relabelings of the IHES Picture Cube.

Self-contained: needs only `numpy` and the competition's `puzzle_info.json`
(72-facelet picture cube: 8 corners x 3 + 12 edges x 2 + 6 centers x 4 = 72,
18 directed generators f/r/d x layers 0/1/2 x {CW, CCW}).

WHAT IT FINDS
-------------
A *symmetry* is a relabeling P of the 72 facelets that maps the move set onto
itself by conjugation:

    P^-1 . g_n . P = g_{sigma(n)}        for every generator n,

for some permutation sigma of the 18 generators. Conjugating any state by such
a P preserves distance-to-solved, so it is a free training augmentation / a
valid sym-ensemble rotation for search.

An *inverse antisymmetry* additionally inverts the state first:

    s  -->  P . s^-1 . P^-1 ,

distance-preserving because every move's inverse is also a move (so inverting a
length-L solution gives another length-L solution).

THE 48 / 96 (and why not 1152)
------------------------------
Enumerating *all* facelet permutations that conjugate the move set to itself
gives a group of order 1152 = 48 x 24.  The factor 24 is the centralizer:
relabelings that commute with EVERY move.  Those act trivially on any state
reachable from solved (P . s . P^-1 = s when P commutes with s), so they are
useless duplicates.  Quotienting them out leaves:

    48  spatial symmetries          (the full octahedral group: 24 rotations + 24 mirrors)
    48  inverse antisymmetries      (same 48, applied to the inverted state)
    --
    96  distinct distance-preserving relabelings.

(The megaminx has the identical structure: 720 = 120 x 6, effective 120.)

OUTPUTS  (into --out-dir)
-------------------------
  cube_symmetries.npy        (48, 72) int16  -- spatial symmetry perms P
  cube_symmetries_inv.npy    (48, 72) int16  -- their inverses P^-1
  cube_symmetry_group.npz    the 96: perms, perms_inv, is_antisymmetry
  cube_symmetries_meta.json  counts + provenance

USAGE
-----
  python build_cube_symmetries.py --puzzle-info puzzle_info.json --out-dir .
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# permutation helpers.  Arrays use the puzzle-info convention apply(s,g)[i]=s[g[i]]
# --------------------------------------------------------------------------- #
def perm_inverse(g):
    inv = [0] * len(g)
    for i, v in enumerate(g):
        inv[v] = i
    return inv


def perm_mul(a, b):
    """(a after b)[i] = a[b[i]] -- used only for the commutation test."""
    return tuple(a[b[i]] for i in range(len(b)))


def cycle_type(g):
    n = len(g)
    seen = [False] * n
    lens = []
    for s in range(n):
        if seen[s]:
            continue
        ln, x = 0, s
        while not seen[x]:
            seen[x] = True
            x = g[x]
            ln += 1
        lens.append(ln)
    return tuple(sorted(lens))


def compute_rotations(gens, n):
    """The 24 whole-cube rotations, as a set of permutation tuples (cross-check)."""
    def apply_move(s, g):
        return tuple(s[g[i]] for i in range(n))

    def compose(a, b):
        return tuple(b[a[p]] for p in range(n))

    ident = tuple(range(n))
    bases = []
    for axis in ("f", "r", "d"):
        r = ident
        for layer in (0, 1, 2):
            r = apply_move(r, gens[f"{axis}{layer}"])
        bases.append(r)
    rots = {ident}
    frontier = [ident]
    while frontier:
        nf = []
        for r in frontier:
            for base in bases:
                nr = compose(r, base)
                if nr not in rots:
                    rots.add(nr)
                    nf.append(nr)
        frontier = nf
    return rots


# --------------------------------------------------------------------------- #
def build(puzzle_info_path: Path, out_dir: Path) -> int:
    t0 = time.time()
    info = json.load(open(puzzle_info_path, encoding="utf-8"))
    gens_dict = {nm: [int(x) for x in perm] for nm, perm in info["generators"].items()}
    names = list(gens_dict.keys())
    gens = [gens_dict[nm] for nm in names]
    n_gen = len(gens)
    n = len(info["central_state"])
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    inv_idx = [name_to_idx[nm[1:] if nm.startswith("-") else "-" + nm] for nm in names]
    print(f"state_size={n}, generators={n_gen}: {names}", flush=True)

    # (1) facelet orbits under the move group (expect corners/edges/centers)
    seen = [False] * n
    orbits = []
    for s in range(n):
        if seen[s]:
            continue
        orbit, stack = {s}, [s]
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
    orbit_of = [0] * n
    for oi, opos in enumerate(orbits):
        for p in opos:
            orbit_of[p] = oi
    print(f"orbits: {len(orbits)} of sizes {[len(o) for o in orbits]}", flush=True)

    # generator signatures for pruning sigma: a real symmetry preserves both the
    # cycle type and the SET of orbits a generator disturbs (outer turns move
    # corners+edges+centers; slice turns move only edges+centers).
    def moved_orbits(g):
        return frozenset(orbit_of[i] for i in range(n) if g[i] != i)

    sig = [(cycle_type(g), len(moved_orbits(g))) for g in gens]
    domain = [[j for j in range(n_gen) if sig[j] == sig[i]] for i in range(n_gen)]
    commute = [[perm_mul(gens[i], gens[j]) == perm_mul(gens[j], gens[i])
                for j in range(n_gen)] for i in range(n_gen)]

    # (2) backtrack candidate sigma under NECESSARY conditions (verify filters later)
    sigma = [-1] * n_gen
    used = [False] * n_gen
    candidates = []

    def consistent(i, j):
        for a in range(n_gen):
            sa = sigma[a]
            if sa != -1 and a != i and commute[a][i] != commute[sa][j]:
                return False
        return True

    def backtrack(start):
        idx = start
        while idx < n_gen and sigma[idx] != -1:
            idx += 1
        if idx == n_gen:
            candidates.append(tuple(sigma))
            return
        i, ii = idx, inv_idx[idx]
        for j in domain[i]:
            jj = inv_idx[j]
            if used[j] or used[jj] or (ii == i) != (jj == j) or not consistent(i, j):
                continue
            sigma[i] = j
            used[j] = True
            did_inv = False
            if ii != i:
                if not consistent(ii, jj):
                    sigma[i] = -1
                    used[j] = False
                    continue
                sigma[ii] = jj
                used[jj] = True
                did_inv = True
            backtrack(idx + 1)
            sigma[i] = -1
            used[j] = False
            if did_inv:
                sigma[ii] = -1
                used[jj] = False

    backtrack(0)
    print(f"candidate sigmas (necessary conditions): {len(candidates)} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # (3) reconstruct every P for each sigma by anchor propagation per orbit:
    #     condition  P^-1 g_n P = g_sigma(n)  =>  P[g_sigma(n)[k]] = g_n[P[k]].
    def propagate(anchor, target, sg):
        P = [-1] * n
        P[anchor] = target
        wl = [anchor]
        while wl:
            nxt = []
            for k in wl:
                pk = P[k]
                for m in range(n_gen):
                    slot, val = gens[sg[m]][k], gens[m][pk]
                    if P[slot] == -1:
                        P[slot] = val
                        nxt.append(slot)
                    elif P[slot] != val:
                        return None
            wl = nxt
        return P

    def verify(P, sg):
        if -1 in P or len(set(P)) != n:
            return False
        Pinv = perm_inverse(P)
        for m in range(n_gen):
            gm, gt = gens[m], gens[sg[m]]
            for k in range(n):
                if Pinv[gm[P[k]]] != gt[k]:
                    return False
        return True

    found, found_set = [], set()
    for sg in candidates:
        partials = []
        ok = True
        for opos in orbits:
            anchor = opos[0]
            seen_vals, vlist = set(), []
            for t in range(n):
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
                vlist.append((P, frozenset(vals)))
            if not vlist:
                ok = False
                break
            partials.append(vlist)
        if not ok:
            continue
        acc = [([-1] * n, frozenset())]
        for opos, vlist in zip(orbits, partials):
            new_acc = []
            for accP, accV in acc:
                for pP, pV in vlist:
                    if accV & pV:
                        continue
                    merged = accP[:]
                    for p in opos:
                        merged[p] = pP[p]
                    new_acc.append((merged, accV | pV))
            acc = new_acc
        for P, _ in acc:
            Pt = tuple(P)
            if Pt not in found_set and verify(P, sg):
                found_set.add(Pt)
                found.append(Pt)

    n_raw = len(found)
    print(f"facelet-automorphism group order: {n_raw}  ({time.time() - t0:.1f}s)",
          flush=True)

    # closure + identity sanity
    arr = np.array(found, dtype=np.int64)
    sset = set(map(tuple, arr.tolist()))
    closure_ok = all(tuple(a[b].tolist()) in sset for a in arr for b in arr)
    ident_in = tuple(range(n)) in found_set
    print(f"  group closure: {'OK' if closure_ok else 'FAILED'}; identity present: "
          f"{ident_in}", flush=True)

    # (4) collapse the trivial-on-states centralizer: keep one P per induced sigma
    gen_index_of = {tuple(g): i for i, g in enumerate(gens)}

    def sigma_of(P):
        Pinv = perm_inverse(list(P))
        return tuple(gen_index_of[tuple(Pinv[gens[m][P[k]]] for k in range(n))]
                     for m in range(n_gen))

    cent = sum(1 for P in found if sigma_of(P) == tuple(range(n_gen)))
    rot_set = compute_rotations(gens_dict, n)
    by_sigma = {}
    for P in found:
        sg = sigma_of(P)
        if sg not in by_sigma or (P in rot_set and by_sigma[sg] not in rot_set):
            by_sigma[sg] = P
    eff = sorted(by_sigma.values())
    n_eff = len(eff)
    print(f"  centralizer (trivial-on-states factor): {cent}", flush=True)
    print(f"  rotations subgroup present: {rot_set <= found_set}", flush=True)
    print(f"  EFFECTIVE distinct-on-states symmetries: {n_eff}", flush=True)

    # (5) save: 48 symmetries, their inverses, and the 96 (sym + antisym)
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_arr = np.array(eff, dtype=np.int16)
    eff_inv = np.stack([np.array(perm_inverse(list(p)), dtype=np.int16) for p in eff])
    np.save(out_dir / "cube_symmetries.npy", eff_arr)
    np.save(out_dir / "cube_symmetries_inv.npy", eff_inv)
    np.savez(out_dir / "cube_symmetry_group.npz",
             perms=np.concatenate([eff_arr, eff_arr]),
             perms_inv=np.concatenate([eff_inv, eff_inv]),
             is_antisymmetry=np.concatenate([np.zeros(n_eff, bool), np.ones(n_eff, bool)]))
    with open(out_dir / "cube_symmetries_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "state_size": n,
            "effective_symmetries": int(n_eff),
            "with_inverse_antisymmetry": int(2 * n_eff),
            "facelet_automorphism_group_order": int(n_raw),
            "centralizer_order": int(cent),
            "group_closure_ok": bool(closure_ok),
            "rotations_subgroup_ok": bool(rot_set <= found_set),
            "antisymmetry": "s -> P . invert(s) . P^-1 (group-invert the state, then conjugate)",
        }, f, indent=2)

    print(f"\nwrote -> {out_dir}\\cube_symmetries.npy {eff_arr.shape}, "
          f"cube_symmetry_group.npz ({2 * n_eff}, {n})")
    print(f"SUMMARY: {n_eff} symmetries + {n_eff} inverse-antisymmetries = "
          f"{2 * n_eff} distance-preserving relabelings ({time.time() - t0:.1f}s)")
    return 0 if (closure_ok and ident_in and n_eff == 48) else 1


def main():
    ap = argparse.ArgumentParser(description="Build the 96 IHES cube symmetries.")
    ap.add_argument("--puzzle-info", default="puzzle_info.json", type=Path)
    ap.add_argument("--out-dir", default=".", type=Path)
    args = ap.parse_args()
    raise SystemExit(build(args.puzzle_info, args.out_dir))


if __name__ == "__main__":
    main()
