"""Admissible lower bounds for the 4x4x4, and what they certify about our score.

Two independent bound families:

1. ORBIT-PROJECTION BOUNDS (per-state, certified).
   Each of the 4 slot orbits (corners, centres, wing-A, wing-B) is closed under the
   full group, so the colour projection onto an orbit evolves autonomously. Any
   solution of the full cube projects to a solution of the projection, hence
       d_full(s)  >=  d_proj(pi(s))   for every orbit.
   We BFS each projected graph from solved as far as memory allows. States inside
   the ball get their EXACT projected distance; states outside get depth_limit + 1.
   Taking the max over orbits is admissible (max of lower bounds, not a sum -- the
   orbits share moves, so adding would NOT be admissible).

   A 24-slot projection over 6 colours encodes exactly into one uint64 (6^24 =
   4.74e18 < 2^63), so dedup is exact -- important, because a hash collision that
   dropped a state would inflate the "outside the ball" bound and silently make it
   UNSOUND.

2. COUNTING BOUND (aggregate, statistical).
   |ball(d)| <= 1 + n_gen * sum_{i<d} b^i. With N = 1.7763e47 total states, any d
   with |ball(d)| < N cannot cover the space, so almost every state is deeper. This
   is a statement about almost-all states, not a certificate for a specific one --
   but ~1000 of the 1043 test pids are random walks past the mixing length, so it is
   sound in aggregate for the score.

Run:  .venv/Scripts/python.exe cube444/scripts/13_lower_bounds.py --max-states 12000000
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube444.orbits import full_orbits  # noqa: E402
from cube444.puzzle import Cube444  # noqa: E402

N_STATES = 1.7763e47      # measured state count for the 4x4x4 colour cube
BRANCHING = 19.18         # measured, stable over d=2..5
N_GEN = 24


def encode(proj: np.ndarray) -> np.ndarray:
    """(N, 24) uint8 over {0..5} -> (N,) uint64, exact (6^24 < 2^63)."""
    n = proj.shape[1]
    powers = (6 ** np.arange(n, dtype=object)).astype(np.uint64)
    return (proj.astype(np.uint64) * powers[None, :]).sum(axis=1, dtype=np.uint64)


def orbit_bfs(local_perms: np.ndarray, start: np.ndarray, max_states: int,
              max_depth: int, verbose: bool = True):
    """BFS a projected orbit graph from `start`. Returns (code->dist dict arrays, depth).

    local_perms: (n_gen, k) index arrays acting on the projection.
    Returns sorted codes and their exact distances, plus the depth fully expanded.
    """
    frontier = start[None, :].astype(np.uint8)
    all_codes = [encode(frontier)]
    all_dists = [np.zeros(1, dtype=np.uint8)]
    seen = encode(frontier)
    depth = 0
    while depth < max_depth:
        n = frontier.shape[0]
        if n == 0:
            break
        kids = np.empty((n * len(local_perms), frontier.shape[1]), dtype=np.uint8)
        for i, lp in enumerate(local_perms):
            kids[i * n:(i + 1) * n] = frontier[:, lp]
        codes = encode(kids)
        order = np.argsort(codes, kind="stable")
        codes, kids = codes[order], kids[order]
        keep = np.empty(len(codes), dtype=bool)
        keep[0] = True
        np.not_equal(codes[1:], codes[:-1], out=keep[1:])
        codes, kids = codes[keep], kids[keep]
        fresh = ~np.isin(codes, seen, assume_unique=True)
        codes, kids = codes[fresh], kids[fresh]
        if len(codes) == 0:
            break
        depth += 1
        all_codes.append(codes)
        all_dists.append(np.full(len(codes), depth, dtype=np.uint8))
        seen = np.union1d(seen, codes)
        frontier = kids
        if verbose:
            print(f"      depth {depth:2d}: +{len(codes):>12,}  total {len(seen):>12,}")
        if len(seen) > max_states:
            if verbose:
                print(f"      stopping: {len(seen):,} > cap {max_states:,}")
            break
    codes = np.concatenate(all_codes)
    dists = np.concatenate(all_dists)
    order = np.argsort(codes, kind="stable")
    return codes[order], dists[order], depth


def counting_bound(branching: float, target: float = N_STATES) -> int:
    """Smallest d with |ball(d)| >= target, given a branching upper bound."""
    total, d = 1.0, 0
    while total < target:
        d += 1
        total += N_GEN * (branching ** (d - 1))
    return d


# Reachable reduced states: a reduced 4x4x4 colouring is exactly a 3x3x3 state, so
# the <outer> orbit of solved has |3x3x3| = 4.3252e19 elements. Inner moves can also
# reach reduced states outside that orbit (the OLL/PLL parity classes); we allow a
# generous factor of 4 for those, which only makes the bound weaker (safer).
N_3X3X3 = 4.3252003274489856e19
PARITY_CLASSES = 4.0


def phase1_counting_bound(branching: float = BRANCHING) -> int:
    """Lower bound on distance-to-R for almost all states.

    Phase 1 has to move a uniform state into R, and |R| is a vanishing fraction of
    the space, so the same counting argument applies with N replaced by N/|R|.
    This is what makes the two-phase budget checkable BEFORE building the beam.
    """
    return counting_bound(branching, N_STATES / (N_3X3X3 * PARITY_CLASSES))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-states", type=int, default=12_000_000)
    ap.add_argument("--max-depth", type=int, default=20)
    ap.add_argument("--submissions", nargs="*", default=None)
    args = ap.parse_args()

    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    gens = {k: np.array(v, dtype=np.int64) for k, v in puz.generators.items()}
    solved = np.array(puz.solved_state, dtype=np.int64)

    print("=== counting bounds (aggregate, not per-state certificates) ===")
    print(f"  N = {N_STATES:.4g} states, {N_GEN} generators")
    print(f"  rigorous  (branching <= {N_GEN}):    d >= {counting_bound(N_GEN)}")
    print(f"  measured  (branching  = {BRANCHING}): d >= {counting_bound(BRANCHING)}")
    p1b = phase1_counting_bound()
    print(f"\n  PHASE-1 budget check (distance-to-R, |R| = {N_3X3X3:.4g} x "
          f"{PARITY_CLASSES:.0f} parity classes):")
    print(f"    distance-to-R  >= {p1b} moves for almost all states "
          f"(rigorous variant: {phase1_counting_bound(N_GEN)})")
    print(f"    so two-phase total >= {p1b} + (phase-2 cost)")

    orbits = full_orbits(gens)
    orbit_names = []
    for o in orbits:
        within = [i % 16 for i in o[:4]]
        if set(i % 16 for i in o) <= {0, 3, 12, 15}:
            orbit_names.append("corners")
        elif set(i % 16 for i in o) <= {5, 6, 9, 10}:
            orbit_names.append("centres")
        else:
            orbit_names.append(f"wings-{len([n for n in orbit_names if n.startswith('wings')]) + 1}")

    tables = {}
    for oname, o in zip(orbit_names, orbits):
        slots = np.array(o, dtype=np.int64)
        pos = {int(s): i for i, s in enumerate(slots)}
        local = np.stack([
            np.array([pos[int(gens[n][int(s)])] for s in slots], dtype=np.int64)
            for n in names
        ])
        print(f"\n=== orbit '{oname}' ({len(slots)} slots) BFS ===")
        t0 = time.time()
        codes, dists, depth = orbit_bfs(local, solved[slots].astype(np.uint8),
                                        args.max_states, args.max_depth)
        print(f"    fully expanded to depth {depth} in {time.time() - t0:.1f}s "
              f"({len(codes):,} states) -> LB cap {depth + 1}")
        tables[oname] = (slots, codes, dists, depth)

    # --- per-pid bounds -----------------------------------------------------
    init = {}
    with open(PROJECT / "data" / "test.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            init[int(row["initial_state_id"])] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int64)
    pids = sorted(init)
    states = np.stack([init[p] for p in pids])

    lb = np.zeros(len(pids), dtype=np.int64)
    per_orbit = {}
    for oname, (slots, codes, dists, depth) in tables.items():
        proj = states[:, slots].astype(np.uint8)
        c = encode(proj)
        idx = np.searchsorted(codes, c)
        idx = np.clip(idx, 0, len(codes) - 1)
        hit = codes[idx] == c
        vals = np.where(hit, dists[idx].astype(np.int64), depth + 1)
        per_orbit[oname] = vals
        lb = np.maximum(lb, vals)
        print(f"  orbit {oname:>9}: LB mean {vals.mean():6.2f}  "
              f"exact-hits {int(hit.sum())}/{len(pids)}")

    cb = counting_bound(BRANCHING)
    print(f"\n  max-over-orbits LB: mean {lb.mean():.2f}  min {lb.min()}  max {lb.max()}")

    # --- compare with the best paths we actually have ------------------------
    subdir = PROJECT / "submissions"
    files = args.submissions or [str(p) for p in sorted(subdir.glob("*.csv"))]
    best = {}
    src = {}
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows or "path" not in rows[0]:
            continue
        for row in rows:
            pid = int(row["initial_state_id"])
            ln = len(row["path"].split(".")) if row["path"].strip() else 0
            if pid not in best or ln < best[pid]:
                best[pid] = ln
                src[pid] = Path(fp).name
    if not best:
        print("\n  no submission CSVs found; skipping gap report")
        return 0

    lens = np.array([best[p] for p in pids], dtype=np.int64)
    from collections import Counter
    print(f"\n=== per-pid floor over {len(files)} CSVs (Rule 26 n-way min) ===")
    print(f"  total {int(lens.sum()):,}  mean {lens.mean():.2f}")
    for name, cnt in Counter(src.values()).most_common():
        print(f"    {cnt:5d} pids best in {name}")

    print("\n=== gap report ===")
    print(f"  our per-pid floor          mean {lens.mean():6.2f}   total {int(lens.sum()):,}")
    print(f"  Rokicki (public LB #1)     mean {46298 / 1043:6.2f}   total 46,298")
    print(f"  counting bound (measured)  mean {cb:6.2f}   total {cb * len(pids):,}")
    print(f"  orbit-projection LB        mean {lb.mean():6.2f}   (certified per-pid)")
    print(f"\n  slack vs Rokicki:        {lens.mean() - 46298 / 1043:6.2f} moves/pid")
    print(f"  slack vs counting bound: {lens.mean() - cb:6.2f} moves/pid")
    print(f"  pids where the certified LB is tight (== our length): "
          f"{int((lb == lens).sum())}")

    out = PROJECT / "data" / "lower_bounds.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "our_len", "lb_max_orbit"] + [f"lb_{k}" for k in per_orbit])
        for i, p in enumerate(pids):
            w.writerow([p, int(lens[i]), int(lb[i])] +
                       [int(per_orbit[k][i]) for k in per_orbit])
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
