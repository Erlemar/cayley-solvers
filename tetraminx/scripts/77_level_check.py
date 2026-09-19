"""Matched level check: score ONE fixed state population with every AVI checkpoint.

`E[target]` across rounds is NOT a usable health signal in this setup. The scramble
sample is seeded from the round number (deliberately -- a fixed seed would silently
re-draw the same puzzles every round), so consecutive shards share almost no pids and
the shard mean confounds "the model's level moved" with "these scrambles were harder".
Round 0 -> 1 moved 17.047 -> 16.608 on a 2-of-20 pid overlap, which is unreadable.

This fixes the population instead. A seeded subsample of beam states from one shard is
scored by every checkpoint, so the only thing that varies is the weights:

    implied distance   V_hat(s) = 1 + min_a Q(s, a)      -- what the beam ranks on
    value head         V(s)                              -- what qv_consistency reads
    |Q - (V-1)|        self-disagreement                 -- the qv_consistency penalty

Read it with the anchor loss beside it. The level RISING is the healthy direction here:
the incumbent under-predicts at depth (V@d80 = 18.7-23.9 against a true ~28), and the
probe measured the consequence directly -- beam-selected states score 0.102 against a
known-good state's 0.246 at the same depth, i.e. the beam selects hardest for exactly
this optimism. AVI is supposed to pull the level up. A FLAT level would mean the
bootstrap carried no new information (cube444 s8: its own flat control was the
pathological arm, and the climb it feared was the cure).

    .venv/Scripts/python.exe tetraminx/scripts/77_level_check.py \
        --shard tetraminx/runs/avi/shards/r000.pt \
        --ckpt incumbent=tetraminx/models/mx_tf_az/epoch_1500.pt \
        --ckpt r000=tetraminx/models/avi/r000.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.models import model_from_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Score one fixed state set with every checkpoint.")
    ap.add_argument("--shard", required=True, type=Path, help="source of the fixed states")
    ap.add_argument("--ckpt", action="append", required=True, help="name=path.pt")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    blob = torch.load(args.shard, map_location="cpu", weights_only=False)
    child = blob["sparse"]["child"]
    tgt = blob["sparse"]["target"]
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(child.size(0), generator=g)[:args.n]
    states = child.index_select(0, idx).to(dev).long()
    shard_t = tgt.index_select(0, idx)
    print(f"fixed population: {states.size(0):,} beam states from {args.shard.name}; "
          f"their harvested targets mean {shard_t.mean():.3f}", flush=True)

    print(f"\n{'checkpoint':>12} {'V_hat=1+minQ':>13} {'V head':>9} {'|Q-(V-1)|':>11} "
          f"{'meanQ':>8} {'sdQ':>7}")
    print("-" * 66)
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT / p
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = model_from_config(dict(ck["model_config"])).to(dev).eval()
        sd = {k.removeprefix("_orig_mod."): v for k, v in ck.get("state_dict", ck).items()}
        model.load_state_dict(sd)
        has_v = bool(getattr(model, "has_value_head", False))
        model.return_value = has_v
        qs, vs = [], []
        with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
            for i in range(0, states.size(0), args.chunk_size):
                out = model(states[i:i + args.chunk_size])
                q, v = out if has_v else (out, None)
                qs.append(q.float().cpu())
                if v is not None:
                    vs.append(v.float().cpu())
        q = torch.cat(qs)
        vhat = 1.0 + q.min(dim=1).values
        if vs:
            v = torch.cat(vs)
            disagree = (q - (v - 1.0).unsqueeze(1)).abs().mean()
            vm = f"{v.mean():9.3f}"
            dm = f"{disagree:11.3f}"
        else:
            vm, dm = f"{'--':>9}", f"{'--':>11}"
        print(f"{name:>12} {vhat.mean():>13.3f} {vm} {dm} {q.mean():>8.3f} {q.std():>7.3f}")
        model.return_value = False

    print("\nRISING V_hat is the healthy direction: the incumbent under-predicts at "
          "depth,\nand the beam selects hardest for exactly that optimism. FLAT would "
          "mean the\nbootstrap carried no new information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
