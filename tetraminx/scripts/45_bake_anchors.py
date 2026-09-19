"""Bake exact 24-way Q anchors from the BFS train file + endgame table.

`BFSAnchors` (51_train_sparse_q.py) does the child expansion, Zobrist hash and
`searchsorted` against the endgame table on EVERY training step. That is fine on a
GPU but it is three extra ops in the hot path, and on TPU the table has to live in
HBM alongside the model. `BakedAnchors` consumes the same labels precomputed, so
this script does that expansion once.

The output is byte-for-byte the same supervision `BFSAnchors` would have produced --
`--verify` asserts it on a random sample against the live BFSAnchors class, so the
substitution is a matched control, not a re-derivation.

    python tetraminx/scripts/45_bake_anchors.py \
        --train tetraminx/data/bfs_d6_train.pt \
        --endgame tetraminx/data/bfs_endgame.npz \
        --max-depth 5 \
        --out tetraminx/data/baked_anchors_d5.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.puzzle import Tetraminx


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute exact 24-way Q anchors.")
    ap.add_argument("--train", type=Path, default=PROJECT / "tetraminx/data/bfs_d6_train.pt")
    ap.add_argument("--endgame", type=Path, default=PROJECT / "tetraminx/data/bfs_endgame.npz")
    ap.add_argument("--puzzle", type=Path, default=PROJECT / "tetraminx/data/puzzle_info.json")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=65536)
    ap.add_argument("--verify", type=int, default=4096,
                    help="rows to cross-check against the live BFSAnchors class (0 to skip)")
    args = ap.parse_args()

    dev = args.device
    puzzle = Tetraminx.load(args.puzzle)
    names = list(puzzle.move_names)
    gen = torch.tensor([puzzle.generators[n] for n in names], dtype=torch.int64, device=dev)
    A, S = gen.shape

    blob = torch.load(args.train, map_location="cpu", weights_only=False)
    states_all, dists_all = blob["states"], blob["distances"]
    keep = dists_all <= args.max_depth
    states = states_all[keep].contiguous()
    depths = dists_all[keep].contiguous()
    n = states.size(0)
    print(f"anchors: {n:,} states at d<={args.max_depth} (of {states_all.size(0):,})", flush=True)

    z = np.load(args.endgame)
    hashes = torch.from_numpy(z["hashes"]).to(dev)
    table_depths = torch.from_numpy(z["depths"]).to(dev)
    ztab = torch.from_numpy(z["ztab"]).to(dev)
    max_depth = int(z["max_depth"])
    assert args.max_depth < max_depth, (
        f"max_depth={args.max_depth} needs a table of depth >={args.max_depth + 1}, "
        f"got d{max_depth}")
    assert bool(torch.all(hashes[1:] > hashes[:-1])), "endgame hashes must be sorted"
    print(f"endgame table: {hashes.numel():,} states at d<={max_depth}", flush=True)

    def zhash(flat: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(flat.size(0), dtype=torch.int64, device=flat.device)
        for i in range(flat.size(1)):
            out = torch.bitwise_xor(out, ztab[i].index_select(0, flat[:, i]))
        return out

    q = torch.empty((n, A), dtype=torch.int8)
    t0 = time.time()
    for lo in range(0, n, args.chunk):
        hi = min(lo + args.chunk, n)
        s = states[lo:hi].to(dev).long()
        children = torch.gather(s.unsqueeze(1).expand(-1, A, -1), 2,
                                gen.unsqueeze(0).expand(s.size(0), -1, -1))
        h = zhash(children.reshape(-1, S))
        pos = torch.searchsorted(hashes, h).clamp_max(hashes.numel() - 1)
        hit = hashes.index_select(0, pos) == h
        assert bool(hit.all()), f"anchor child missing from the endgame table at rows {lo}:{hi}"
        q[lo:hi] = table_depths.index_select(0, pos).view(s.size(0), A).to(torch.int8).cpu()
        if (lo // args.chunk) % 10 == 0:
            done = hi / n
            print(f"  {hi:,}/{n:,} ({done:5.1%})  {time.time() - t0:6.1f}s", flush=True)
    print(f"expansion done in {time.time() - t0:.1f}s", flush=True)

    # Sanity: every child of a d<=k state must be at d in {k-1, k, k+1}.
    d0 = depths.to(torch.int16).unsqueeze(1)
    qq = q.to(torch.int16)
    assert bool(((qq - d0).abs() <= 1).all()), "triangle inequality violated in baked targets"
    # Every state at d>=1 has a geodesic child at d-1. The solved state is the one
    # exception: it has no d-1, and all 24 of its children sit at d=1.
    deep = depths.to(torch.int16) >= 1
    assert bool((qq[deep].min(dim=1).values == depths.to(torch.int16)[deep] - 1).all()), \
        "every state at d>=1 must have a distance-reducing child"
    assert bool((qq[~deep] == 1).all()), "children of solved must all be at d=1"
    print("invariants ok: |Q - d| <= 1, min(Q) == d-1 for d>=1", flush=True)

    if args.verify > 0:
        # Cross-check against the class the trainer actually used on the GPU run.
        sys.path.insert(0, str(PROJECT / "tetraminx" / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "trainmod", PROJECT / "tetraminx/scripts/51_train_sparse_q.py")
        trainmod = importlib.util.module_from_spec(spec)
        sys.modules["trainmod"] = trainmod
        spec.loader.exec_module(trainmod)
        live = trainmod.BFSAnchors(args.train, args.endgame, gen, args.max_depth, dev)
        rng = np.random.default_rng(0)
        idx = torch.from_numpy(rng.choice(n, size=min(args.verify, n), replace=False))
        s_ref = live.states.index_select(0, idx).to(dev).long()
        children = torch.gather(s_ref.unsqueeze(1).expand(-1, A, -1), 2,
                                gen.unsqueeze(0).expand(s_ref.size(0), -1, -1))
        h = live._hash(children.reshape(-1, S))
        pos = torch.searchsorted(live.hashes, h).clamp_max(live.hashes.numel() - 1)
        ref = live.table_depths.index_select(0, pos).view(s_ref.size(0), A).to(torch.int8).cpu()
        got = q.index_select(0, idx)
        assert torch.equal(ref, got), "baked targets differ from BFSAnchors"
        assert torch.equal(live.states.index_select(0, idx), states.index_select(0, idx)), \
            "baked states differ from BFSAnchors"
        print(f"VERIFIED: {idx.numel():,} rows identical to BFSAnchors targets", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        states=states.numpy(),
        q_targets=q.numpy(),
        depths=depths.numpy(),
        table_depth=np.int64(max_depth),
        max_anchor_depth=np.int64(args.max_depth),
        move_names=np.array(names),
    )
    mb = args.out.stat().st_size / 2**20
    print(f"wrote {args.out} ({mb:.1f} MB compressed, {n:,} anchors x {A} actions)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
