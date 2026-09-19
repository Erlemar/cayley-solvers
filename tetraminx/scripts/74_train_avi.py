"""Beam-AVI training: one round of supervised fitting on a harvested shard.

Step 3 of the AVI loop. There is NO target network here -- `73_gen_harvest.py` already
baked the frozen net's opinion into the shard's numbers, so this is ordinary supervised
training at plain throughput instead of paying 24 forwards per state for a live backup.
The shard is a static replay buffer for exactly one round; staleness is bounded by one
round by construction, because the next round regenerates from the updated weights.

FOUR TERMS.

  sparse   masked MSE on the ONE column named by the harvested action. The other 23
           outputs get no gradient from the row. Same shape the incumbent trainer
           already consumes (`51_train_sparse_q.py`), which is why this drops in.
  dense    MSE over all 24 columns of a `--full-expand` parent. Unbiased -- those
           parents were drawn without reference to Q. The probe measured the bias this
           corrects at 0.246 vs 0.102 (survivor-selected vs known-good states at the
           same depth), so it is not a formality.
  anchor   MSE over all 24 columns of an EXACT d<=5 BFS anchor. Pins the absolute
           scale. DO NOT SOFTEN THIS -- the beam takes a global top-B across
           (parent, action) pairs, so cross-parent comparability of the raw level is
           what it runs on; cube444 measured Huber-capping the level term taking the
           beam from 18/18 to 5/18 solved while every offline metric improved.
  value    MSE on the AZ value head, supervised where V is free: V(s) = 1 + min_a t[a]
           on dense rows, V = depth on anchors. Needed because `qv_consistency` reads
           |Q - (V-1)| on an ABSOLUTE scale -- AVI moves Q's level, and a value head
           left behind would make the deployed penalty charge correct children.

SYMMETRY IS ONE RANDOM FRAME PER ROW, not the 14-row expansion the incumbent uses.
Shard rows are plentiful, so augmentation should cost 1 row rather than 14; that also
keeps the step inside 16 GB (the incumbent's 512 x 14 peaks at 17.3 GB and Windows
WDDM pages rather than OOMing). Only the 24 SPATIAL frames are legal for a Q row: the
inverse antisymmetry transports V but not Q, because d((s.a)^-1) = d(a^-1 . s^-1) is a
LEFT multiplication and is not of the form d(s^-1 . a').

    .venv/Scripts/python.exe tetraminx/scripts/74_train_avi.py \
        --shard tetraminx/runs/avi/shards/r000.pt \
        --init tetraminx/models/mx_tf_az/epoch_1500.pt \
        --out tetraminx/models/avi/r000.pt --steps 4000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.models import model_from_config
from tetraminx.puzzle import Tetraminx


def _load_trainer_module():
    """Reuse Symmetries + BakedAnchors from 51_train_sparse_q.py rather than copy them."""
    path = PROJECT / "tetraminx" / "scripts" / "51_train_sparse_q.py"
    spec = importlib.util.spec_from_file_location("_tetra_train51", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Train one Beam-AVI round on a harvested shard.")
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path, help="warm-start checkpoint")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--anchors", type=Path, default=None,
                    help="default data-dir/baked_anchors_d5.npz")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1024, help="sparse rows per step")
    ap.add_argument("--dense-batch", type=int, default=128)
    ap.add_argument("--anchor-batch", type=int, default=256)
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--anchor-weight", type=float, default=2.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=3e-3)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--log-every", type=int, default=250)
    args = ap.parse_args()

    dev = args.device
    torch.manual_seed(args.seed)
    t51 = _load_trainer_module()

    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    move_names = list(puzzle.move_names)
    name_to_idx = {nm: i for i, nm in enumerate(move_names)}
    n_act = len(move_names)
    gens = torch.tensor([puzzle.generators[nm] for nm in move_names],
                        dtype=torch.int64, device=dev)
    inv_idx = torch.tensor([name_to_idx[puzzle.inverse_name(nm)] for nm in move_names],
                           dtype=torch.int64, device=dev)

    sym = t51.Symmetries(args.data_dir, dev)
    sym.verify(gens)                       # asserts the conj/relabel convention holds
    anchors = t51.BakedAnchors(args.anchors or (args.data_dir / "baked_anchors_d5.npz"),
                               dev, resident=True)

    # ---- shard ------------------------------------------------------------
    blob = torch.load(args.shard, map_location="cpu", weights_only=False)
    sp = blob["sparse"]
    child = sp["child"].to(dev)                                  # uint8 (N, 88)
    move = sp["move"].to(dev).long()                             # (N,)
    tgt = sp["target"].to(dev)                                   # (N,)
    N = child.size(0)
    dn = blob["dense"]
    d_state = dn["state"].to(dev) if dn is not None else None    # uint8 (M, 88)
    d_tgt = dn["target"].to(dev) if dn is not None else None     # (M, 24)
    M = 0 if d_state is None else d_state.size(0)
    meta = blob.get("meta", {})
    print(f"shard {args.shard.name}: sparse {N:,} rows, dense {M:,} rows, "
          f"target mean {tgt.mean():.3f} (round {meta.get('round')}, "
          f"beam {meta.get('beam'):,}, {meta.get('exact_overrides', 0):,} exact)", flush=True)

    # ---- model ------------------------------------------------------------
    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    mcfg = dict(ckpt["model_config"])
    model = model_from_config(mcfg).to(dev)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt.get("state_dict", ckpt).items()}
    model.load_state_dict(sd)
    has_v = bool(getattr(model, "has_value_head", False))
    model.return_value = has_v
    model.train()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {mcfg.get('arch')} {n_par/1e6:.2f}M params, value_head={has_v}, "
          f"warm from {args.init.name}", flush=True)
    if args.value_weight > 0 and not has_v:
        raise SystemExit("--value-weight > 0 needs an az_head checkpoint")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, fused=(dev == "cuda"))

    def sample_sparse(n: int):
        """Rows are stored as (child, move); the PARENT is one inverse gather away.

        parent = child[gen[inv[m]]]  -- exact, asserted by 73_gen_harvest.py --self-test.
        Storing only the child halves the shard.
        """
        i = torch.randint(0, N, (n,), device=dev)
        c = child.index_select(0, i).long()
        m = move.index_select(0, i)
        parent = torch.gather(c, 1, gens.index_select(0, inv_idx.index_select(0, m)))
        t = tgt.index_select(0, i)
        k = torch.randint(0, sym.n_sym, (n,), device=dev)        # one random frame
        return sym.conjugate(parent, k), sym.sigma_inv[k, m], t

    def sample_dense(n: int):
        i = torch.randint(0, M, (n,), device=dev)
        s = d_state.index_select(0, i).long()
        t = d_tgt.index_select(0, i)
        k = torch.randint(0, sym.n_sym, (n,), device=dev)
        # A whole 24-vector transports by PERMUTING the columns: the value that sat on
        # column sigma(j) for s belongs on column j for conj(s).
        return sym.conjugate(s, k), torch.gather(t, 1, sym.sigma.index_select(0, k))

    hist: list[dict] = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for gparam in opt.param_groups:
            gparam["lr"] = lr

        s_s, s_a, s_t = sample_sparse(args.batch)
        d_s, d_t = sample_dense(args.dense_batch)
        a_s, a_t, a_d = anchors.sample(args.anchor_batch, None)
        xs = torch.cat((s_s, d_s, a_s), 0)
        nb, nd, na = args.batch, args.dense_batch, args.anchor_batch

        with torch.autocast(dev, dtype=torch.bfloat16):
            out = model(xs)
            q_all, v_all = out if has_v else (out, None)
        q_all = q_all.float()
        rows = torch.arange(nb, device=dev)
        sparse = q_all[:nb][rows, s_a].sub(s_t).pow(2).mean()
        dense = q_all[nb:nb + nd].sub(d_t).pow(2).mean()
        anchor = q_all[nb + nd:].sub(a_t).pow(2).mean()
        if has_v:
            v_all = v_all.float()
            # V is supervised only where it is FREE: dense rows carry all 24 targets so
            # V(s) = 1 + min_a t[a], and anchors carry the exact depth. No extra forward.
            v_dense = v_all[nb:nb + nd].sub(1.0 + d_t.min(dim=1).values).pow(2).mean()
            v_anchor = v_all[nb + nd:].sub(a_d).pow(2).mean()
            value = v_dense + v_anchor
        else:
            value = q_all.new_zeros(())

        loss = (sparse + args.dense_weight * dense + args.anchor_weight * anchor
                + args.value_weight * value)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            rec = dict(step=step, loss=float(loss.detach()), sparse=float(sparse.detach()),
                       dense=float(dense.detach()), anchor=float(anchor.detach()),
                       value=float(value.detach()),
                       e_target=float(s_t.mean()), lr=lr, wall=time.time() - t0)
            hist.append(rec)
            print(f"  step {step:5d}/{args.steps} loss {rec['loss']:8.4f} "
                  f"sparse {rec['sparse']:7.4f} dense {rec['dense']:7.4f} "
                  f"anchor {rec['anchor']:7.4f} value {rec['value']:7.4f} "
                  f"E[tgt] {rec['e_target']:6.3f} {rec['wall']:6.0f}s", flush=True)

    model.return_value = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": mcfg,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "avi": dict(shard=str(args.shard), init=str(args.init), steps=args.steps,
                            batch=args.batch, lr=args.lr,
                            anchor_weight=args.anchor_weight,
                            dense_weight=args.dense_weight,
                            value_weight=args.value_weight,
                            shard_meta=meta, history=hist)}, args.out)
    print(f"\nwrote {args.out}  ({time.time()-t0:.0f}s, "
          f"{args.steps * args.batch / max(1, N):.2f} passes over the shard)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
