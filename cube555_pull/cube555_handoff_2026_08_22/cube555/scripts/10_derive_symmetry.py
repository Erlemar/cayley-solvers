"""Derive the symmetry frames of the 5x5x5 PICTURE cube -- expect 48.

    python cube555/scripts/10_derive_symmetry.py

WHAT A FRAME IS HERE. cube444 is a colouring, so a frame is a recolouring conjugation
`color_map[s[rot]]`. This puzzle is a permutation, so a frame is a plain CONJUGATION

    sym(s, M) = Minv[ s[M] ]          (= M^-1 o s o M as functions)

which sends solved (the identity) to solved automatically, and satisfies

    sym(s o g_a, M) = sym(s, M) o g_{relabel[M, a]}     with   g_{relabel[M,a]} = M^-1 g_a M

so d(sym(s,M)) == d(s) and a Q label rides from column a to column relabel[M,a].
Any M whose conjugation permutes the generator SET is admissible -- it does not have to
be geometrically realisable -- because those two properties are all the recipe uses.

NOTHING IS HARDCODED. For each candidate generator relabel sigma (axis permutation x
per-axis layer reversal x global direction flip = 6 x 8 x 2 = 96 candidates), the slot
map M is SOLVED for by constraint propagation on

    M[g_a[p]] = g_{sigma(a)}[M[p]]

seeded once per closed slot orbit, then accepted only if M is a bijection AND the full
transport identity holds on scrambled states. Both rotations and mirrors come out of the
same search: the rotations are the 24 sigmas that are geometrically proper, and the
script does not need to know which is which.

Writes cube555/data/:
    sym_slots_48.npy        (48, 150) int64  -- M
    sym_slots_inv_48.npy    (48, 150) int64  -- Minv
    sym_move_relabel_48.npy (48, 30)  int64  -- relabel[k, a],  g_{relabel} = M^-1 g_a M
    sym_move_relabel_inv_48.npy                -- the inverse relabel, for path translation
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube555.puzzle import Cube555, N_AXES, N_LAYERS, STATE_SIZE, gen_index  # noqa: E402


def candidate_sigmas(n_gen: int):
    """Axis permutation x PER-AXIS layer reversal x PER-AXIS direction flip.

    The direction flip must be per-axis, not global: a 90-degree rotation about d sends
    f -> r but r -> -f, so a global flip cannot express it. With a global flip this
    search finds only 12 of the 48 frames, which is silent -- every frame it does find
    verifies -- so the count is the only tell. (cube444's script has the narrower family;
    it happens to suffice there.)
    """
    for aperm in permutations(range(N_AXES)):
        for rev_mask in range(1 << N_AXES):
            for flip_mask in range(1 << N_AXES):
                sigma = np.empty(n_gen, dtype=np.int64)
                for a in range(N_AXES):
                    rev = (rev_mask >> a) & 1
                    flip = (flip_mask >> a) & 1
                    for lay in range(N_LAYERS):
                        for s in (0, 1):
                            sigma[gen_index(a, lay, s)] = gen_index(
                                aperm[a],
                                (N_LAYERS - 1 - lay) if rev else lay,
                                (1 - s) if flip else s,
                            )
                if len(set(sigma.tolist())) == n_gen:
                    yield sigma, (aperm, rev_mask, flip_mask)


def slot_orbits(moves: np.ndarray, n_slots: int) -> list[list[int]]:
    """Closed orbits of the slot set under the generator moves."""
    seen: set[int] = set()
    orbits: list[list[int]] = []
    for s in range(n_slots):
        if s in seen:
            continue
        comp, stack = {s}, [s]
        while stack:
            p = stack.pop()
            for g in range(moves.shape[0]):
                q = int(moves[g][p])
                if q not in comp:
                    comp.add(q)
                    stack.append(q)
        seen |= comp
        orbits.append(sorted(comp))
    return orbits


def propagate(sigma, moves, seed, target, n_slots):
    """Extend {seed: target} by M[g_a[p]] = g_sigma(a)[M[p]]. None if inconsistent."""
    trial = {seed: target}
    stack = [seed]
    n_gen = moves.shape[0]
    while stack:
        p = stack.pop()
        mp = trial[p]
        for g in range(n_gen):
            q = int(moves[g][p])
            t = int(moves[sigma[g]][mp])
            prev = trial.get(q)
            if prev is None:
                trial[q] = t
                stack.append(q)
            elif prev != t:
                return None
    return trial


def solve_all_M(sigma, moves, orbits, n_slots):
    """Every bijection M consistent with sigma. Usually 0 or 1."""
    per_orbit = []
    for orbit in orbits:
        seed = orbit[0]
        cands = []
        for target in range(n_slots):
            tr = propagate(sigma, moves, seed, target, n_slots)
            if tr is None:
                continue
            if len(tr) != len(orbit):
                continue  # image is not a same-size closed orbit
            cands.append(tr)
        if not cands:
            return []
        per_orbit.append(cands)
    out = []

    def combine(i, acc, used):
        if i == len(per_orbit):
            M = np.empty(n_slots, dtype=np.int64)
            for k, v in acc.items():
                M[k] = v
            out.append(M)
            return
        for tr in per_orbit[i]:
            vals = set(tr.values())
            if vals & used:
                continue
            combine(i + 1, {**acc, **tr}, used | vals)

    combine(0, {}, set())
    return out


def verify_frame(M, Minv, relabel, moves, n_gen, rng, n=8, depth=8) -> bool:
    """sym(s o g_a, M) == sym(s, M) o g_relabel[a], on scrambled states."""
    states = np.tile(np.arange(STATE_SIZE), (n, 1))
    for _ in range(depth):
        mv = rng.integers(0, n_gen, size=n)
        states = np.take_along_axis(states, moves[mv], axis=1)
    conj = Minv[states[:, M]]
    for a in range(n_gen):
        lhs = Minv[states[:, moves[a]][:, M]]
        rhs = conj[:, moves[relabel[a]]]
        if not np.array_equal(lhs, rhs):
            return False
    return True


def main() -> int:
    data = PROJECT / "data"
    puz = Cube555.load(data / "puzzle_info.json")
    names = list(puz.move_names)
    n_gen = len(names)
    moves = np.array([puz.generators[n] for n in names], dtype=np.int64)
    expected = [
        f"{'-' if s else ''}{ax}{lay}"
        for ax in ("f", "r", "d")
        for lay in range(N_LAYERS)
        for s in (0, 1)
    ]
    if names != expected:
        raise SystemExit(
            f"generator name order is not axis*10+layer*2+sign.\n got {names}\n"
            f" expected {expected}\nThe sigma family below assumes that order."
        )

    orbits = slot_orbits(moves, STATE_SIZE)
    print(f"slot orbits: {len(orbits)} of sizes {sorted(len(o) for o in orbits)}")

    rng = np.random.default_rng(555)
    frames, relabels, tags = [], [], []
    for sigma, tag in candidate_sigmas(n_gen):
        for M in solve_all_M(sigma, moves, orbits, STATE_SIZE):
            if len(set(M.tolist())) != STATE_SIZE:
                continue
            Minv = np.empty_like(M)
            Minv[M] = np.arange(STATE_SIZE)
            # relabel[a] = b such that g_b = M^-1 g_a M, found by construction not search
            relabel = np.empty(n_gen, dtype=np.int64)
            ok = True
            key = {tuple(moves[b]): b for b in range(n_gen)}
            for a in range(n_gen):
                cand = Minv[moves[a][M]]
                b = key.get(tuple(cand))
                if b is None:
                    ok = False
                    break
                relabel[a] = b
            if not ok or not verify_frame(M, Minv, relabel, moves, n_gen, rng):
                continue
            # DEDUPE BY ACTION, not by M. The slot centraliser here has order 24 (any C
            # commuting with every generator satisfies C^-1 s C = s for all s in G), so
            # 24 distinct slot maps share each relabel and act IDENTICALLY on every
            # reachable state. Keeping them would inflate the frame count 24x and make
            # `sym_rows` draw the same augmentation over and over.
            if any(np.array_equal(relabel, r) for r in relabels):
                continue
            frames.append(M)
            relabels.append(relabel)
            tags.append(tag)

    print(f"frames found and VERIFIED (deduped by action): {len(frames)}")
    if len(frames) != 48:
        print("  WARNING: expected 48. Proceeding with what verified.")

    # identity first, so frames(K) is prefix-nested and row 0 is always the original
    ident = [
        i for i, M in enumerate(frames) if np.array_equal(M, np.arange(STATE_SIZE))
    ]
    if len(ident) != 1:
        raise SystemExit(f"expected exactly one identity frame, got {len(ident)}")
    order = ident + [i for i in range(len(frames)) if i != ident[0]]
    M_all = np.stack([frames[i] for i in order])
    R_all = np.stack([relabels[i] for i in order])
    Minv_all = np.empty_like(M_all)
    for k in range(M_all.shape[0]):
        Minv_all[k, M_all[k]] = np.arange(STATE_SIZE)
    Rinv_all = np.empty_like(R_all)
    for k in range(R_all.shape[0]):
        Rinv_all[k, R_all[k]] = np.arange(n_gen)

    # generator orbits under the frame group -- the sparse-Q column-coverage ceiling
    n_f = R_all.shape[0]
    seen, orb = set(), []
    for a in range(n_gen):
        if a in seen:
            continue
        o = sorted({int(R_all[k, a]) for k in range(n_f)})
        seen |= set(o)
        orb.append(o)
    print(
        f"generator orbits under {n_f} frames: {len(orb)} of sizes "
        f"{sorted(len(o) for o in orb)}"
    )
    print(
        "  -> a labelled (undo, next) pair can reach at most "
        f"{max(len(o) for o in orb) * 2} of {n_gen} columns"
    )

    np.save(data / "sym_slots_48.npy", M_all)
    np.save(data / "sym_slots_inv_48.npy", Minv_all)
    np.save(data / "sym_move_relabel_48.npy", R_all)
    np.save(data / "sym_move_relabel_inv_48.npy", Rinv_all)
    (data / "symmetry_report.json").write_text(
        json.dumps(
            {
                "n_frames": int(n_f),
                "slot_orbit_sizes": sorted(len(o) for o in orbits),
                "generator_orbit_sizes": sorted(len(o) for o in orb),
                "tags": [list(map(str, tags[i])) for i in order],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {data}/sym_*_48.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
