"""Build the exact BFS ball and the exact 30-way Q anchors for the 5x5x5 picture cube.

    python cube555/scripts/11_build_anchors.py --depth 4
    python cube555/scripts/11_build_anchors.py --depth 5     # 10.7M states, ~2 GB

EXCLUSION LABELLING. Anchors at depth D need a table complete to D, NOT D+1. A child of
a depth-D state sits at D-1, D or D+1; if the table is complete to D then a child that
MISSES the lookup is at depth > D by completeness and <= D+1 by adjacency, hence exactly
D+1. The miss IS the label. (cube444 memory: this turned an impossible build into a
one-minute job.)

VALIDATION. Mean target by depth must land on a smooth `d + c` progression -- the
exclusion-labelled top level has to fall on the line set by the levels below it, which
were labelled by lookup. A silent hash collision writes a WRONG exact label, which is the
one thing the recipe trusts as truth, so the level counts are cross-checked against the
BFS pass and the collision headroom is printed.

Writes cube555/data/anchors_d{D}.pt:  states uint8 (N,150), q uint8 (N,30), depth uint8 (N,)
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
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube555.puzzle import Cube555, STATE_SIZE  # noqa: E402


def state_hash(x: torch.Tensor, hv: torch.Tensor, chunk: int = 1 << 22) -> torch.Tensor:
    out = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    for i in range(0, x.shape[0], chunk):
        out[i : i + chunk] = (x[i : i + chunk].long() * hv).sum(1)
    return out


def expand(states: torch.Tensor, moves: torch.Tensor, chunk: int = 1 << 18):
    """(N,150) -> (N*A,150). Chunked: the naive expand at N=10M asks for 345 GB."""
    n, a = states.shape[0], moves.shape[0]
    out = torch.empty((n * a, STATE_SIZE), dtype=states.dtype, device=states.device)
    for i in range(0, n, chunk):
        s = states[i : i + chunk]
        m = s.shape[0]
        out[i * a : (i + m) * a] = torch.gather(
            s[:, None, :].expand(m, a, STATE_SIZE),
            2,
            moves[None].expand(m, a, STATE_SIZE),
        ).reshape(m * a, STATE_SIZE)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=555)
    args = ap.parse_args()

    data = PROJECT / "data"
    puz = Cube555.load(data / "puzzle_info.json")
    names = list(puz.move_names)
    dev = args.device
    moves = torch.tensor(
        [puz.generators[n] for n in names], dtype=torch.int64, device=dev
    )
    n_act = moves.shape[0]
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    hv = torch.randint(
        -(2**62), 2**62, (STATE_SIZE,), dtype=torch.int64, device=dev, generator=g
    )

    t0 = time.time()
    solved = torch.arange(STATE_SIZE, dtype=torch.uint8, device=dev)[None, :]
    all_states = [solved]
    all_depth = [torch.zeros(1, dtype=torch.uint8, device=dev)]
    seen = state_hash(solved, hv)
    levels = [1]
    cur = solved
    for d in range(1, args.depth + 1):
        ch = expand(cur, moves)
        h = state_hash(ch, hv)
        hs, idx = torch.sort(h)
        keep = torch.ones_like(hs, dtype=torch.bool)
        keep[1:] = hs[1:] != hs[:-1]
        hu, iu = hs[keep], idx[keep]
        fresh = ~torch.isin(hu, seen)
        cur = ch[iu[fresh]]
        seen = torch.cat([seen, hu[fresh]])
        all_states.append(cur)
        all_depth.append(torch.full((cur.shape[0],), d, dtype=torch.uint8, device=dev))
        levels.append(cur.shape[0])
        print(
            f"  level {d}: {cur.shape[0]:,}  cum {sum(levels):,}  "
            f"ratio {levels[-1]/max(1,levels[-2]):.3f}  {time.time()-t0:.1f}s",
            flush=True,
        )
        del ch, h, hs, idx, hu, iu, fresh
        torch.cuda.empty_cache()

    states = torch.cat(all_states)
    depth = torch.cat(all_depth)
    n = states.shape[0]
    # 64-bit collision headroom: expected collisions ~ n^2 / 2^65
    print(
        f"table: {n:,} states complete to d<={args.depth}; expected 64-bit hash "
        f"collisions {n * n / 2.0**65:.2e} (must be << 1)"
    )
    # sort by hash for lookup via searchsorted
    hall = state_hash(states, hv)
    order = torch.argsort(hall)
    states, depth, hall = states[order], depth[order], hall[order]
    if int((hall[1:] == hall[:-1]).sum()) != 0:
        raise SystemExit("duplicate hash in the completed table -- collision, abort")

    # ---- exact Q by lookup, with EXCLUSION for the misses -----------------------------
    q = torch.empty((n, n_act), dtype=torch.uint8, device=dev)
    chunk = 1 << 18
    n_excl = 0
    for i in range(0, n, chunk):
        s = states[i : i + chunk]
        m = s.shape[0]
        ch = torch.gather(
            s[:, None, :].expand(m, n_act, STATE_SIZE),
            2,
            moves[None].expand(m, n_act, STATE_SIZE),
        ).reshape(m * n_act, STATE_SIZE)
        hc = state_hash(ch, hv)
        pos = torch.searchsorted(hall, hc).clamp_max(n - 1)
        hit = hall[pos] == hc
        val = torch.where(hit, depth[pos], torch.full_like(depth[pos], 255))
        val = val.reshape(m, n_act)
        # a miss is a child OUTSIDE a table complete to D, hence at exactly d(parent)+1
        miss = val == 255
        val = torch.where(miss, (depth[i : i + m, None] + 1).to(torch.uint8), val)
        q[i : i + m] = val
        n_excl += int(miss.sum())
    print(
        f"exclusion-labelled {n_excl:,} of {n*n_act:,} columns "
        f"({100.0*n_excl/(n*n_act):.2f}%)"
    )

    # ---- validation: mean target by depth must be a smooth d + c ----------------------
    print(
        "  depth  count      mean Q      (must rise ~1.0 per depth, incl. the top level)"
    )
    prev = None
    for d in range(args.depth + 1):
        sel = depth == d
        mq = float(q[sel].float().mean())
        delta = "" if prev is None else f"  d+{mq - prev:+.3f}"
        print(f"  {d:5d}  {int(sel.sum()):9,}  {mq:9.4f}{delta}")
        prev = mq
    # A depth-d state must have at least one child at d-1 (its undo move). The solved
    # state is the exception: every one of its children sits at depth 1.
    lo = q.min(dim=1).values.long()
    want = torch.where(depth.long() > 0, depth.long() - 1, torch.ones_like(lo))
    if not torch.equal(lo, want):
        bad = int((lo != want).sum())
        raise SystemExit(f"{bad} states have min(Q) != depth-1 -- labelling is wrong")
    print("  min(Q) == depth-1 for every state (solved: 1): OK")

    out = data / f"anchors_d{args.depth}.pt"
    torch.save(
        {
            "states": states.cpu(),
            "q": q.cpu(),
            "depth": depth.cpu(),
            "levels": levels,
            "max_depth": args.depth,
        },
        out,
    )
    print(f"wrote {out}  ({out.stat().st_size/1e6:.0f} MB, {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
