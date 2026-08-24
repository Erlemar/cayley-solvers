"""Build the 24 distance-preserving relabelings of the Professor Tetraminx.

A *symmetry* is a relabeling P of the 88 facelets that maps the move set onto
itself by conjugation.  With the puzzle-info convention apply(s,g)[i] = s[g[i]]
(so states and generators are both functions index->value composed as
(a.b)[i] = a[b[i]]), the condition we solve for is

    P^-1 . g_{sigma(m)} . P  =  g_m       for every generator index m.

Consequence used everywhere downstream -- with conj(s) = P^-1 . s . P,

    apply(conj(s), m)  =  conj(apply(s, sigma(m)))

so a path (m_1..m_L) that solves conj(s) maps to (sigma(m_1)..sigma(m_L))
solving the ORIGINAL s.  That is the move-relabel table this script emits.

An *inverse antisymmetry* additionally inverts the state first: solving
s^-1 and then reversing+inverting the path solves s (valid because every
generator's inverse is also a generator).  So the usable ensemble is 24 x 2 = 48
independent search frames per puzzle.

WHY 24
------
The full facelet-automorphism group has order 15552 = 24 x 648, where 648 is the
centralizer (relabelings commuting with every move -- they act trivially on any
state reachable from solved, so they are useless duplicates).  The quotient is
the full tetrahedral group Td: 12 rotations (even axis permutations, direction
preserved) + 12 mirrors (odd axis permutations, direction flipped).
Same structure as the cube (1152 = 48 x 24) and megaminx (720 = 120 x 6).

OUTPUTS (into --out-dir)
------------------------
  tetra_symmetries.npy       (24, 88) int16  -- spatial relabelings P
  tetra_symmetries_inv.npy   (24, 88) int16  -- their inverses P^-1
  tetra_move_relabel.npy     (24, 24) int16  -- sigma per symmetry (move index map)
  tetra_symmetries_meta.json  counts + provenance + verification results

USAGE
-----
  .venv/Scripts/python.exe tetraminx/scripts/01_build_symmetries.py
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

AXES = ["D", "F", "BL", "BR"]
LAYERS = ["2", "3", "4"]


def move_name(sign: int, layer: str, axis: str) -> str:
    return ("-" if sign < 0 else "") + layer + axis


def build_sigma(name_to_idx: dict[str, int], perm: list[int], flip: bool) -> list[int]:
    """Candidate generator relabeling: permute the 4 axes, optionally flip direction.

    Layer is forced to be preserved because the three layer classes have distinct
    cycle types (they move 21 / 15 / 9 facelets), so no symmetry can mix them.
    """
    sigma = [-1] * len(name_to_idx)
    for a_i, axis in enumerate(AXES):
        for layer in LAYERS:
            for sign in (1, -1):
                src = move_name(sign, layer, axis)
                dst = move_name(sign * (-1 if flip else 1), layer, AXES[perm[a_i]])
                sigma[name_to_idx[src]] = name_to_idx[dst]
    return sigma


def find_relabeling(gens: list[list[int]], sigma: list[int], n: int) -> list[int] | None:
    """Solve for P satisfying P[g_m[i]] = g_{sigma(m)}[P[i]] by propagation + backtracking.

    Knowing P at one facelet determines P on that facelet's whole orbit, so the
    search only ever branches once per orbit (11 orbits here).
    """
    P = [-1] * n
    used = [False] * n

    def propagate(seed: int) -> tuple[bool, list[int]]:
        touched: list[int] = []
        stack = [seed]
        ok = True
        while stack:
            i = stack.pop()
            for m in range(len(gens)):
                j = gens[m][i]
                v = gens[sigma[m]][P[i]]
                if P[j] == -1:
                    if used[v]:
                        ok = False
                        stack = []
                        break
                    P[j] = v
                    used[v] = True
                    touched.append(j)
                    stack.append(j)
                elif P[j] != v:
                    ok = False
                    stack = []
                    break
        return ok, touched

    def backtrack() -> bool:
        if -1 not in P:
            return True
        i = P.index(-1)
        for v in range(n):
            if used[v]:
                continue
            P[i] = v
            used[v] = True
            ok, touched = propagate(i)
            if ok and backtrack():
                return True
            for j in touched:
                used[P[j]] = False
                P[j] = -1
            used[v] = False
            P[i] = -1
        return False

    return list(P) if backtrack() else None


def perm_inverse(p: list[int]) -> list[int]:
    inv = [0] * len(p)
    for i, v in enumerate(p):
        inv[v] = i
    return inv


def build(puzzle_info: Path, out_dir: Path) -> int:
    t0 = time.time()
    info = json.loads(puzzle_info.read_text(encoding="utf-8"))
    gens_dict = {nm: [int(x) for x in perm] for nm, perm in info["generators"].items()}
    names = list(gens_dict.keys())
    gens = [gens_dict[nm] for nm in names]
    n = len(info["central_state"])
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    print(f"state_size={n}, generators={len(names)}", flush=True)

    found: dict[tuple, list[int]] = {}
    for perm in itertools.permutations(range(4)):
        for flip in (False, True):
            sigma = build_sigma(name_to_idx, list(perm), flip)
            P = find_relabeling(gens, sigma, n)
            if P is not None:
                found[(perm, flip)] = P
    print(f"symmetries found: {len(found)} / {24 * 2} candidate sigmas "
          f"({time.time() - t0:.1f}s)", flush=True)

    # ---- strict verification of the conjugation identity -------------------
    keys = sorted(found.keys())
    perms, perms_inv, relabels = [], [], []
    for key in keys:
        perm, flip = key
        P = found[key]
        Pi = perm_inverse(P)
        sigma = build_sigma(name_to_idx, list(perm), flip)
        for m in range(len(gens)):
            conj = [Pi[gens[sigma[m]][P[i]]] for i in range(n)]
            assert conj == gens[m], f"conjugation failed for sigma={key}, move {names[m]}"
        perms.append(P)
        perms_inv.append(Pi)
        relabels.append(sigma)

    perms_a = np.array(perms, dtype=np.int16)
    perms_inv_a = np.array(perms_inv, dtype=np.int16)
    relabels_a = np.array(relabels, dtype=np.int16)
    assert len(np.unique(perms_a, axis=0)) == len(perms_a), "duplicate symmetries"
    assert (perms_a[0] == np.arange(n)).all(), "first symmetry is not the identity"

    # ---- end-to-end check: conjugate a scramble, solve-by-relabel, verify ---
    rng = np.random.default_rng(0)
    gens_np = [np.array(g) for g in gens]
    n_checked = 0
    for _ in range(20):
        word = rng.integers(0, len(gens), size=25)
        state = np.arange(n)
        for m in word:
            state = state[gens_np[m]]
        # solution of `state` is the reversed-inverted word
        inv_idx = [name_to_idx[nm[1:] if nm.startswith("-") else "-" + nm] for nm in names]
        solution = [inv_idx[m] for m in word[::-1]]
        for k in range(len(perms)):
            P, Pi, sigma = perms[k], perms_inv[k], relabels[k]
            conj = np.array([Pi[state[P[i]]] for i in range(n)])
            # a path solving `conj` maps through sigma to a path solving `state`:
            # so first find the path that solves conj by relabeling backwards.
            sigma_inv = perm_inverse(sigma)
            conj_solution = [sigma_inv[m] for m in solution]
            cur = conj.copy()
            for m in conj_solution:
                cur = cur[gens_np[m]]
            assert (cur == np.arange(n)).all(), f"sym {k}: relabeled solution failed"
            n_checked += 1
    print(f"conjugation + relabel round-trip verified on {n_checked} cases", flush=True)

    # ---- inverse antisymmetry check ---------------------------------------
    for _ in range(20):
        word = rng.integers(0, len(gens), size=25)
        state = np.arange(n)
        for m in word:
            state = state[gens_np[m]]
        inv_state = np.empty(n, dtype=state.dtype)
        inv_state[state] = np.arange(n)
        inv_idx = [name_to_idx[nm[1:] if nm.startswith("-") else "-" + nm] for nm in names]
        sol_of_state = [inv_idx[m] for m in word[::-1]]
        # solution of s^-1 = reverse + invert the solution of s
        sol_of_inv = [inv_idx[m] for m in sol_of_state[::-1]]
        cur = inv_state.copy()
        for m in sol_of_inv:
            cur = cur[gens_np[m]]
        assert (cur == np.arange(n)).all(), "inverse antisymmetry failed"
    print("inverse antisymmetry verified on 20 cases", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tetra_symmetries.npy", perms_a)
    np.save(out_dir / "tetra_symmetries_inv.npy", perms_inv_a)
    np.save(out_dir / "tetra_move_relabel.npy", relabels_a)
    meta = {
        "state_size": n,
        "n_generators": len(names),
        "move_names": names,
        "effective_symmetries": len(perms),
        "with_inverse_antisymmetry": 2 * len(perms),
        "facelet_automorphism_group_order": 15552,
        "centralizer_order": 648,
        "rotations": 12,
        "mirrors": 12,
        "sigma_axis_perms": [list(k[0]) for k in keys],
        "sigma_direction_flip": [bool(k[1]) for k in keys],
        "conjugation": "conj(s)[i] = P_inv[s[P[i]]]",
        "relabel": "path m_1..m_L solving conj(s) maps to sigma(m_1)..sigma(m_L) solving s",
        "antisymmetry": "solve s^-1, then reverse the path and invert each move",
        "verified": True,
    }
    (out_dir / "tetra_symmetries_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(perms)} symmetries to {out_dir} ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle-info", type=Path,
                    default=HERE / "puzzle_info.json")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    return build(args.puzzle_info, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
