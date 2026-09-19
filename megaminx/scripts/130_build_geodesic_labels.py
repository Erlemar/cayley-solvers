"""Derive geodesic backward-move labels for every state in the BFS-d6 table.

For state s, a move a is geodesic (toward solved) iff apply(s, gens[a]) is one
BFS layer closer. We label s with the move leading to the MINIMUM-distance child
(a valid geodesic move; ties broken arbitrarily). Computed via GPU 64-bit state
hashing + searchsorted lookup of child distances -- a few seconds over 19.4M
states, vs 465M Python dict lookups.

Output: data/geodesic_labels_d6.pt with {states (N,120) int8, moves (N,) int8,
distances (N,) int8}. States at the identity (dist 0) are excluded (no move).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/130_build_geodesic_labels.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from megaminx.puzzle import Megaminx


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puzzle.move_names)
    gens = torch.tensor([list(puzzle.generators[n]) for n in names],
                        dtype=torch.long, device=device)  # (24,120)
    n_gen, s_n = gens.shape

    t0 = time.time()
    d = torch.load(PROJECT / "data" / "bfs_d6_train.pt", map_location="cpu",
                   weights_only=False)
    states_cpu = d["states"]         # (N,120) int8, stays on CPU
    dists_cpu = d["distances"]       # (N,) int8, CPU
    n = states_cpu.shape[0]
    print(f"loaded {n} states to depth {int(dists_cpu.max())} ({time.time()-t0:.0f}s)",
          flush=True)

    g = torch.Generator(device="cpu").manual_seed(12345)
    weights = torch.randint(-(2**62), 2**62, (s_n,), dtype=torch.long,
                            generator=g).to(device)
    chunk = 200_000

    def hash_states(st):  # (M,120) long on device -> (M,) int64
        return (st * weights.unsqueeze(0)).sum(-1)

    # parent hashes in chunks (states move to GPU one chunk at a time)
    parent_hash = torch.empty(n, dtype=torch.long, device=device)
    for i in range(0, n, chunk):
        sb = states_cpu[i:i+chunk].to(device).long()
        parent_hash[i:i+chunk] = hash_states(sb)
    order = torch.argsort(parent_hash)
    sorted_hash = parent_hash[order]
    dists_dev = dists_cpu.to(device).long()
    sorted_dist = dists_dev[order]
    del parent_hash, order
    NOTFOUND = 99

    def lookup_dist(h):  # (M,) hashes -> (M,) distances (NOTFOUND if absent)
        pos = torch.searchsorted(sorted_hash, h).clamp(max=sorted_hash.shape[0]-1)
        hit = sorted_hash[pos] == h
        return torch.where(hit, sorted_dist[pos], torch.full_like(h, NOTFOUND))

    labels = torch.empty(n, dtype=torch.long, device="cpu")
    min_child = torch.empty(n, dtype=torch.long, device="cpu")
    for i in range(0, n, chunk):
        sb = states_cpu[i:i+chunk].to(device).long()  # (c,120)
        c = sb.shape[0]
        children = torch.gather(
            sb.unsqueeze(1).expand(c, n_gen, s_n), 2,
            gens.unsqueeze(0).expand(c, n_gen, s_n))  # (c,24,120)
        ch = hash_states(children.reshape(c * n_gen, s_n))
        cd = lookup_dist(ch).reshape(c, n_gen)  # (c,24) child distances
        best = cd.argmin(dim=1)  # move to min-distance child = geodesic
        labels[i:i+chunk] = best.cpu()
        min_child[i:i+chunk] = cd.gather(1, best.unsqueeze(1)).squeeze(1).cpu()
    print(f"labels computed ({time.time()-t0:.0f}s)", flush=True)

    dists_l = dists_cpu.long()
    keep = dists_l > 0
    good = (min_child == (dists_l - 1))
    print(f"geodesic-consistency: {float((good[keep]).float().mean()):.4f} of"
          f" non-solved states have a child at dist-1 (expect ~1.0)", flush=True)
    keep = keep & good

    out = {
        "states": states_cpu[keep].clone(),
        "moves": labels[keep].to(torch.int8),
        "distances": dists_cpu[keep].clone(),
        "move_names": names,
    }
    path = PROJECT / "data" / "geodesic_labels_d6.pt"
    torch.save(out, path)
    print(f"saved {int(keep.sum())} labeled states -> {path} ({time.time()-t0:.0f}s)",
          flush=True)
    # per-depth counts
    for dd in range(1, int(dists.max()) + 1):
        print(f"  depth {dd}: {int((out['distances']==dd).sum())} states")


if __name__ == "__main__":
    main()
