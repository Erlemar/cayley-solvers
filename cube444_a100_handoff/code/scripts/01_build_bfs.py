"""Build exact BFS distance anchors for the 4x4x4 cube.

    python3 cube444/scripts/01_build_bfs.py --max-exact 5 --d6-sample 20000000 \
        --out cube444/data/bfs_anchors.pt

Measured ball sizes (branching factor 19.18, dead stable):
    d=0            1
    d=1           24
    d=2          468
    d=3        9,000
    d=4      172,914
    d=5    3,316,744      -> ball(<=5) = 3,499,151   (~10 s)
    d=6  ~63,600,000      -> sampled, not stored whole (6.1 GB if kept)

d=6 correctness: a state reachable in 6 moves that is NOT in ball(<=5) is at
distance exactly 6. So we expand frontier(5) in chunks, drop anything already in
ball(<=5), and label the rest 6 -- no full d=6 dedup needed for exactness.

Note: we dedup within each chunk only. Across chunks a d=6 state reachable by
several 6-paths can appear more than once, which mildly over-weights
high-in-degree states in the sample. That is acceptable for an anchor mixin
(it is not a probability model) and avoids a 6 GB sort.

d=7 is ~1.2e9 states -- out of reach, do not try.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube444.puzzle import Cube444

VT = np.dtype((np.void, 96))


def _as_void(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a).view(VT).ravel()


def _unique_rows(a: np.ndarray) -> np.ndarray:
    v = _as_void(a)
    order = np.argsort(v, kind="stable")
    vs = v[order]
    keep = np.empty(len(vs), dtype=bool)
    keep[0] = True
    keep[1:] = vs[1:] != vs[:-1]
    return a[order[keep]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle-info", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "bfs_anchors.pt")
    ap.add_argument("--max-exact", type=int, default=5,
                    help="build the exact ball up to this depth (5 = 3.5M states, ~10 s)")
    ap.add_argument("--d6-sample", type=int, default=20_000_000,
                    help="number of distance-6 states to sample (0 to skip)")
    ap.add_argument("--chunk", type=int, default=400_000,
                    help="frontier rows expanded per chunk in the d6 pass")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    puz = Cube444.load(args.puzzle_info)
    P = np.array([puz.generators[n] for n in puz.move_names], dtype=np.int64)
    n_gen = P.shape[0]
    central = np.array(puz.solved_state, dtype=np.uint8)

    # ---- exact BFS ----
    all_states = [central.reshape(1, -1)]
    all_dists = [np.zeros(1, dtype=np.int8)]
    seen_v = _as_void(central.reshape(1, -1)).copy()
    seen_v.sort()
    frontier = central.reshape(1, -1)
    print(f"  d=0  |sphere|={1:>12,}  |ball|={1:>12,}", flush=True)
    total = 1
    for d in range(1, args.max_exact + 1):
        t0 = time.time()
        nxt = _unique_rows(frontier[:, P.ravel()].reshape(-1, 96))
        nv = _as_void(nxt)
        pos = np.clip(np.searchsorted(seen_v, nv), 0, len(seen_v) - 1)
        fresh = nxt[seen_v[pos] != nv]
        seen_v = np.sort(np.concatenate([seen_v, _as_void(fresh)]))
        total += len(fresh)
        all_states.append(fresh)
        all_dists.append(np.full(len(fresh), d, dtype=np.int8))
        print(f"  d={d}  |sphere|={len(fresh):>12,}  |ball|={total:>12,}  "
              f"({time.time()-t0:.1f}s)", flush=True)
        frontier = fresh

    # ---- sampled d = max_exact + 1 ----
    if args.d6_sample > 0:
        d_next = args.max_exact + 1
        expected = int(len(frontier) * 19.18)
        frac = min(1.0, args.d6_sample / max(1, expected))
        print(f"  d={d_next} sampling: expected ~{expected:,} states, keeping frac={frac:.4f}",
              flush=True)
        kept, n_seen = [], 0
        t0 = time.time()
        for i in range(0, len(frontier), args.chunk):
            blk = frontier[i : i + args.chunk]
            nxt = _unique_rows(blk[:, P.ravel()].reshape(-1, 96))
            nv = _as_void(nxt)
            pos = np.clip(np.searchsorted(seen_v, nv), 0, len(seen_v) - 1)
            fresh = nxt[seen_v[pos] != nv]
            n_seen += len(fresh)
            if frac < 1.0:
                fresh = fresh[rng.random(len(fresh)) < frac]
            kept.append(fresh)
            if (i // args.chunk) % 2 == 0:
                print(f"    chunk {i//args.chunk:>3d}: seen {n_seen:>12,}  kept "
                      f"{sum(len(k) for k in kept):>12,}  ({time.time()-t0:.0f}s)", flush=True)
        d6 = np.concatenate(kept)
        print(f"  d={d_next} total fresh {n_seen:,}  sampled {len(d6):,}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        all_states.append(d6)
        all_dists.append(np.full(len(d6), d_next, dtype=np.int8))

    states = np.concatenate(all_states).astype(np.int8)
    dists = np.concatenate(all_dists)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": torch.from_numpy(states), "distances": torch.from_numpy(dists)},
               args.out)
    size_gb = states.nbytes / 1e9
    print(f"\nwrote {args.out}  ({len(states):,} states, {size_gb:.2f} GB)")
    for d in range(0, args.max_exact + 2):
        n = int((dists == d).sum())
        if n:
            print(f"  d={d}: {n:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
