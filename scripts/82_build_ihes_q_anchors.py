"""Exact all-18-column Q anchors for every IHES state at depth <= 6 (exclusion labelling).

Stage-3 (Q-Bellman) anchor pool, the 444 `build_anchors_deep.py` trick: anchors at depth D
need a table complete to D, not D+1. A child of a depth-d state sits at d-1 or d+1 (every
generator is an odd permutation, so never at d). If the table is complete to 6, a child that
MISSES it is deeper than 6 by completeness and at most d+1 <= 7 by adjacency -- exactly 7.

Output uses the `BakedAnchors` payload of tetraminx/scripts/51_train_sparse_q.py:
    states (N, 72) uint8, q_targets (N, 18) uint8, depths (N,) uint8, table_depth int

    python scripts/82_build_ihes_q_anchors.py --out data/ihes_q_anchors_d6.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build exact d<=6 IHES Q anchors.")
    ap.add_argument("--states", type=Path, default=PROJECT / "data" / "bfs_d6_train.pt")
    ap.add_argument("--table", type=Path, default=PROJECT / "data" / "bfs_table_d6_hash.npz")
    ap.add_argument("--puzzle", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "ihes_q_anchors_d6.pt")
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    t0 = time.time()

    info = json.loads(args.puzzle.read_text(encoding="utf-8"))
    names = list(info["generators"].keys())
    gens = torch.tensor([info["generators"][n] for n in names], dtype=torch.int64, device=dev)
    A, S = gens.shape

    blob = torch.load(args.states, map_location="cpu", weights_only=False)
    states = blob["states"]
    depths = blob["depths"].to(torch.int64)
    N = states.size(0)
    z = np.load(args.table)
    hashes = torch.from_numpy(z["hashes"]).to(dev)
    tdepth = torch.from_numpy(z["depths"]).to(dev).to(torch.int64)
    ztab = torch.from_numpy(z["ztab"]).to(dev)
    D = int(z["max_depth"])
    assert hashes.numel() == N, f"table has {hashes.numel()} states, state file {N}"
    assert int(depths.max()) == D
    print(f"{N:,} states d<={D}; table {hashes.numel():,}; {A} moves; device {dev}", flush=True)

    q = torch.empty((N, A), dtype=torch.uint8)
    n_excl = 0
    bad = 0
    for lo in range(0, N, args.chunk):
        hi = min(N, lo + args.chunk)
        s = states[lo:hi].to(dev)                                   # uint8 (b, S)
        d0 = depths[lo:hi].to(dev)
        b = s.size(0)
        ch = s[:, gens]                                             # uint8 (b, A, S)
        flat = ch.reshape(b * A, S)
        h = torch.zeros(b * A, dtype=torch.int64, device=dev)
        for i in range(S):
            h ^= ztab[i].index_select(0, flat[:, i].long())
        pos = torch.searchsorted(hashes, h).clamp_max(hashes.numel() - 1)
        hit = hashes.index_select(0, pos) == h
        cd = torch.where(hit, tdepth.index_select(0, pos), torch.full_like(h, D + 1)).view(b, A)
        # exclusion is only legal for depth-D parents; shallower parents must hit every child
        miss = ~hit.view(b, A)
        bad += int((miss & (d0 < D).unsqueeze(1)).sum())
        n_excl += int(miss.sum())
        # adjacency: every child is at d0 - 1 or d0 + 1
        bad += int(((cd - d0.unsqueeze(1)).abs() != 1).sum())
        q[lo:hi] = cd.to(torch.uint8).cpu()
        if (lo // args.chunk) % 20 == 0:
            print(f"  {hi:,}/{N:,}  excluded so far {n_excl:,}  bad {bad}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    if bad:
        raise SystemExit(f"{bad} consistency violations -- table and state file disagree")

    # self-checks: mean target by depth, and the d6 exclusion rate vs the level sizes
    level = torch.bincount(depths, minlength=D + 1).tolist()
    means = {}
    for d in range(D + 1):
        m = depths == d
        means[d] = float(q[m].float().mean()) if bool(m.any()) else float("nan")
    print("level sizes:", level)
    print("mean target by depth:", {d: round(v, 4) for d, v in means.items()})
    print("mean target - depth: ", {d: round(v - d, 4) for d, v in means.items()})
    m6 = depths == D
    rate = float((q[m6] == D + 1).float().mean())
    # expected: edge counting between levels (no same-level edges: all moves are odd),
    # down(d) = N(d-1) * up(d-1) / N(d), up(d) = A - down(d); excluded share at D = up(D)/A
    up = float(A)
    for d in range(1, D + 1):
        up = A - level[d - 1] * up / level[d]
    print(f"d{D} children excluded (labelled {D + 1}): {rate:.4%}  expected {up / A:.4%}  "
          f"(total {n_excl:,})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"states": states.contiguous(), "q_targets": q, "depths": depths.to(torch.uint8),
                "table_depth": D + 1,
                "meta": {"source_states": str(args.states.name), "table": str(args.table.name),
                         "rule": "exclusion: a child missing the complete d<=6 table is at 7"}},
               args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 2**30:.2f} GiB) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
