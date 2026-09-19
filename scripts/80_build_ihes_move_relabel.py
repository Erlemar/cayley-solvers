"""Build the (48, 18) IHES move-relabel table in the tetraminx trainer's convention.

`tetraminx/scripts/51_train_sparse_q.py::Symmetries` (and `30_solve.py`) conjugate as

    conj_k(s)[i] = P_inv[k][ s[ P[k][i] ] ]          apply(s, m)[i] = s[ g_m[i] ]

and load `sigma` such that

    apply(conj_k(s), sigma_inv[k][a]) == conj_k(apply(s, a))     for every state s,

with `sigma_inv[k]` the inverse permutation of `sigma[k]`. The IHES symmetry code
(`scripts/09_symmetry_antisymmetry.py`) uses the opposite conjugation, so the table is
derived here by brute force against scrambled states instead of by formula, and then
re-verified the same way the trainer does at startup.

    python scripts/80_build_ihes_move_relabel.py \
        --sym data/cube_symmetries.npy --sym-inv data/cube_symmetries_inv.npy \
        --out data/cube_move_relabel.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def conj(P: np.ndarray, P_inv: np.ndarray, s: np.ndarray) -> np.ndarray:
    """conj(s)[i] = P_inv[s[P[i]]] for a batch of states (rows)."""
    return P_inv[s[:, P]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive the IHES move-relabel table.")
    ap.add_argument("--puzzle", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--sym", type=Path, default=PROJECT / "data" / "cube_symmetries.npy")
    ap.add_argument("--sym-inv", type=Path, default=PROJECT / "data" / "cube_symmetries_inv.npy")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "cube_move_relabel.npy")
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    info = json.loads(args.puzzle.read_text(encoding="utf-8"))
    names = list(info["generators"].keys())
    gens = np.array([info["generators"][n] for n in names], dtype=np.int64)
    A, S = gens.shape
    solved = np.array(info["central_state"], dtype=np.int64)
    assert np.array_equal(solved, np.arange(S)), "IHES solved state should be the identity"

    P = np.load(args.sym).astype(np.int64)
    P_inv = np.load(args.sym_inv).astype(np.int64)
    K = P.shape[0]
    assert P.shape == (K, S) and P_inv.shape == (K, S)
    rows = np.arange(S)
    for k in range(K):
        assert np.array_equal(P_inv[k][P[k]], rows), f"P_inv[{k}] is not the inverse of P[{k}]"
    ident = [k for k in range(K) if np.array_equal(P[k], rows)]
    print(f"symmetries: {K} x {S}, identity rows {ident}; generators: {A} ({' '.join(names)})")

    rng = np.random.default_rng(args.seed)
    states = np.tile(solved, (args.probes, 1))
    for _ in range(40):  # scramble the probes well past the d<=6 ball
        mv = rng.integers(0, A, size=args.probes)
        states = np.take_along_axis(states, gens[mv], axis=1)

    sigma_inv = np.full((K, A), -1, dtype=np.int64)
    for k in range(K):
        cs = conj(P[k], P_inv[k], states)
        for a in range(A):
            target = conj(P[k], P_inv[k], states[:, gens[a]])
            hits = [b for b in range(A) if np.array_equal(cs[:, gens[b]], target)]
            if len(hits) != 1:
                raise SystemExit(f"k={k} a={a}: expected exactly one transported move, got {hits}")
            sigma_inv[k, a] = hits[0]
        if sorted(sigma_inv[k].tolist()) != list(range(A)):
            raise SystemExit(f"k={k}: transport is not a permutation of the moves")

    sigma = np.argsort(sigma_inv, axis=1)
    # the trainer's own check, on fresh probes
    fresh = np.tile(solved, (args.probes, 1))
    for _ in range(37):
        mv = rng.integers(0, A, size=args.probes)
        fresh = np.take_along_axis(fresh, gens[mv], axis=1)
    back = np.zeros_like(sigma)
    for k in range(K):
        back[k, sigma[k]] = np.arange(A)
        for a in range(A):
            lhs = conj(P[k], P_inv[k], fresh)[:, gens[back[k, a]]]
            rhs = conj(P[k], P_inv[k], fresh[:, gens[a]])
            assert np.array_equal(lhs, rhs), f"re-verification failed at k={k} a={a}"
    id_rows = np.nonzero((sigma == np.arange(A)).all(axis=1))[0].tolist()
    assert len(id_rows) == 1, f"expected exactly one identity relabel row, got {id_rows}"

    # action orbits under the group (what bounds symmetry label coverage)
    orbits, seen = [], set()
    for a in range(A):
        if a in seen:
            continue
        orb = sorted({int(sigma_inv[k, a]) for k in range(K)})
        seen.update(orb)
        orbits.append([names[i] for i in orb])
    print(f"identity relabel row: {id_rows[0]}; action orbits: "
          + "; ".join(f"{len(o)}: {' '.join(o)}" for o in orbits))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, sigma.astype(np.int16))
    print(f"wrote {args.out} shape {sigma.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
