"""IHES Beam-AVI training: one round of supervised fitting on a harvested shard.

Port of `tetraminx/scripts/74_train_avi.py`. No target network here: `89_ihes_gen_harvest.py`
already baked the frozen net's backup into the shard, so this is plain supervised training.

LOSS TERMS
  sparse   masked MSE on the ONE harvested column (parent reconstructed as
           child[gen[inv[move]]]).
  dense    MSE over all 18 columns of the unbiased `--full-expand` parents.
  anchor   MSE over all 18 columns of EXACT d<=6 anchors (`data/ihes_q_anchors_d6.pt`), weight
           2.0 -- pins the absolute level. Never soften it (global top-B runs on the level).
  value    AZ value head: V = 1 + min_a t[a] on dense rows, V = depth on anchors.
  gap      OPTIONAL (`--gap-weight`, default 0 = the faithful method). The anti-flattening
           mechanism: on shallow random-walk pivots, mean((Q(s,next) - Q(s,undo) - 2)^2).
           The label DIFFERENCE is exactly 2 whenever the walk is locally geodesic, so the
           loss cannot be lowered by shrinking the child gap -- the failure that killed the
           tetraminx port (deep gap01 0.438 -> 0.270, `tetraminx/BEAM_AVI_PLAN.md` s5b). It
           constrains only the gap, never the absolute level, so it does not fight the
           bootstrap. Same term cube444 found to resist Bellman compression
           (`cube444/scripts/12_train_phase1.py`, lambda_margin).

  rehearse OPTIONAL (`--rehearse-weight`, default 0). Forgetting guard: MSE over all 18 columns
           against a FROZEN teacher (default: the --init checkpoint; the loop passes the
           original incumbent every round) on random-walk pivot states. Measured
           2026-09-17: one faithful round cut v1b's next-minus-undo gap at walk depth
           9-12 from 1.78 to 1.11 -- beam-state-only fitting erodes what the walk-trained
           model knew. The teacher keeps that region; the AVI targets act on beam states.

SYMMETRY: one random frame (of the 48 spatial frames) per row. Only spatial frames transport
a Q row; the inverse antisymmetry does not.

    python scripts/90_ihes_train_avi.py --shard runs/ihes_avi/A/shards/r000.pt \
        --init models/ihes_tf_b_e1300a_s3/step_02000.pt --out runs/ihes_avi/A/models/r000.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.puzzle import PictureCube  # noqa: E402
from tetraminx.models import model_from_config  # noqa: E402


def _t51():
    path = PROJECT / "tetraminx" / "scripts" / "51_train_sparse_q.py"
    spec = importlib.util.spec_from_file_location("_t51_sparse_q", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_model(ck: dict, ckpt_path: Path, dev: str):
    cfg = dict(ck["model_config"])
    lp = cfg.get("layout_path")
    if lp and not Path(lp).is_absolute() and not Path(lp).exists():
        rel = Path(str(lp).replace("\\", "/"))
        for cand in (PROJECT / rel, ckpt_path.parent / rel.name):
            if cand.exists():
                cfg["layout_path"] = str(cand)
                break
    model = model_from_config(cfg).to(dev)
    model.load_state_dict({k.removeprefix("_orig_mod."): v
                           for k, v in ck.get("state_dict", ck).items()})
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="Train one IHES Beam-AVI round.")
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--round", type=int, default=-1)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--anchors", type=Path, default=PROJECT / "data" / "ihes_q_anchors_d6.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dense-batch", type=int, default=128)
    ap.add_argument("--anchor-batch", type=int, default=256)
    ap.add_argument("--sparse-weight", type=float, default=1.0,
                    help="0 drops the sparse stream (its rows are beam-SELECTED moves, so their teacher Q is biased low -- the winner's curse)")
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--anchor-weight", type=float, default=2.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--gap-weight", type=float, default=0.0)
    ap.add_argument("--gap-batch", type=int, default=256)
    ap.add_argument("--gap-k-max", type=int, default=12,
                    help="walk length cap for gap pivots (pivots <= k-1); shallow so the "
                         "asserted gap of 2 is mostly true")
    ap.add_argument("--rehearse-weight", type=float, default=0.0)
    ap.add_argument("--rehearse-batch", type=int, default=512)
    ap.add_argument("--rehearse-k-max", type=int, default=22)
    ap.add_argument("--rehearse-tilt", type=float, default=0.5)
    ap.add_argument("--teacher", type=Path, default=None,
                    help="frozen rehearsal teacher (default: the --init checkpoint)")
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
    t51 = _t51()
    puzzle = PictureCube.load(args.data_dir / "puzzle_info.json")
    names = list(puzzle.move_names)
    A = len(names)
    gens = torch.tensor([puzzle.generators[n] for n in names], dtype=torch.int64, device=dev)
    inv_idx = torch.tensor([names.index(puzzle.inverse_name(n)) for n in names],
                           dtype=torch.int64, device=dev)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=dev)

    sym = t51.Symmetries(args.data_dir, dev, prefix="cube")
    sym.verify(gens)
    anchors = t51.BakedAnchors(args.anchors, dev, resident=True)
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed + 7)
    gap_sampler = (t51.SparseQSampler(gens, inv_idx, solved, 2, args.gap_k_max, 0.0, g)
                   if args.gap_weight > 0 else None)

    blob = torch.load(args.shard, map_location="cpu", weights_only=False)
    sp = blob["sparse"]
    child = sp["child"].to(dev)
    move = sp["move"].to(dev).long()
    tgt = sp["target"].to(dev).float()
    N = child.size(0)
    dn = blob["dense"]
    d_state = dn["state"].to(dev) if dn is not None else None
    d_tgt = dn["target"].to(dev).float() if dn is not None else None
    M = 0 if d_state is None else d_state.size(0)
    meta = blob.get("meta", {})
    print(f"shard {args.shard}: sparse {N:,}, dense {M:,}, target mean {tgt.mean():.3f}, "
          f"round {meta.get('round')}, beam {meta.get('beam')}, exact {meta.get('exact_overrides')}",
          flush=True)

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    model = build_model(ck, args.init, dev)
    has_v = bool(getattr(model, "has_value_head", False))
    if args.value_weight > 0 and not has_v:
        raise SystemExit("--value-weight > 0 needs an az_head checkpoint")
    model.return_value = has_v
    model.train()
    print(f"init {args.init} ({sum(p.numel() for p in model.parameters()):,} params, "
          f"value_head={has_v}); gap_weight={args.gap_weight} (k_max {args.gap_k_max})",
          flush=True)
    teacher, rh_sampler = None, None
    if args.rehearse_weight > 0:
        t_path = args.teacher or args.init
        teacher = build_model(torch.load(t_path, map_location="cpu", weights_only=False),
                              t_path, dev).eval()
        teacher.return_value = False
        for prm in teacher.parameters():
            prm.requires_grad_(False)
        g_r = torch.Generator(device=dev)
        g_r.manual_seed(args.seed + 11)
        rh_sampler = t51.SparseQSampler(gens, inv_idx, solved, 2, args.rehearse_k_max,
                                        args.rehearse_tilt, g_r)
        print(f"rehearsal: weight {args.rehearse_weight}, batch {args.rehearse_batch}, walk k "
              f"2-{args.rehearse_k_max} (tilt {args.rehearse_tilt}), teacher {t_path}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            fused=dev.startswith("cuda"))

    def sample_sparse(n: int):
        i = torch.randint(0, N, (n,), device=dev)
        c = child.index_select(0, i).long()
        m = move.index_select(0, i)
        parent = torch.gather(c, 1, gens.index_select(0, inv_idx.index_select(0, m)))
        k = torch.randint(0, sym.n_sym, (n,), device=dev)
        return sym.conjugate(parent, k), sym.sigma_inv[k, m], tgt.index_select(0, i)

    def sample_dense(n: int):
        i = torch.randint(0, M, (n,), device=dev)
        s = d_state.index_select(0, i).long()
        t = d_tgt.index_select(0, i)
        k = torch.randint(0, sym.n_sym, (n,), device=dev)
        # a full row transports by permuting its columns
        return sym.conjugate(s, k), torch.gather(t, 1, sym.sigma.index_select(0, k))

    def sample_gap(n: int):
        s, _p, prev, nxt = gap_sampler.sample(n)
        k = torch.randint(0, sym.n_sym, (n,), device=dev)
        return sym.conjugate(s, k), sym.sigma_inv[k, prev], sym.sigma_inv[k, nxt]

    nb = args.batch if args.sparse_weight > 0 else 0
    nd, na = (args.dense_batch if M else 0), args.anchor_batch
    if nb == 0 and nd == 0:
        raise SystemExit("nothing to train on: --sparse-weight 0 and no dense rows")
    ng = args.gap_batch if gap_sampler is not None else 0
    nr = args.rehearse_batch if rh_sampler is not None else 0
    hist = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(1, args.warmup))
        for pg in opt.param_groups:
            pg["lr"] = lr
        parts, extras = [], {}
        if nb:
            s_s, s_a, s_t = sample_sparse(nb)
            parts.append(s_s)
        if nd:
            d_s, d_t = sample_dense(nd)
            parts.append(d_s)
        a_s, a_t, a_d = anchors.sample(na, g)
        parts.append(a_s)
        if ng:
            g_s, g_prev, g_next = sample_gap(ng)
            parts.append(g_s)
        if nr:
            r_s = rh_sampler.sample(nr)[0]
            r_s = sym.conjugate(r_s, torch.randint(0, sym.n_sym, (nr,), device=dev))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=dev.startswith("cuda")):
                r_t = teacher(r_s).float()
            parts.append(r_s)
        xs = torch.cat(parts, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            out = model(xs)
        q_all, v_all = out if has_v else (out, None)
        q_all = q_all.float()
        sparse = q_all.new_zeros(())
        if nb:
            rows = torch.arange(nb, device=dev)
            sparse = q_all[:nb][rows, s_a].sub(s_t).pow(2).mean()
        o = nb
        dense = q_all.new_zeros(())
        if nd:
            dense = q_all[o:o + nd].sub(d_t).pow(2).mean()
            o += nd
        anchor = q_all[o:o + na].sub(a_t).pow(2).mean()
        a_off = o
        o += na
        gap = q_all.new_zeros(())
        gap_mean = q_all.new_zeros(())
        if ng:
            qg = q_all[o:o + ng]
            r = torch.arange(ng, device=dev)
            d = qg[r, g_next] - qg[r, g_prev]
            gap = d.sub(2.0).pow(2).mean()
            gap_mean = d.mean().detach()
            o += ng
        rehearse = q_all.new_zeros(())
        if nr:
            rehearse = q_all[o:o + nr].sub(r_t).pow(2).mean()
            o += nr
        value = q_all.new_zeros(())
        if has_v:
            v_all = v_all.float()
            if nd:
                value = value + v_all[nb:nb + nd].sub(1.0 + d_t.min(dim=1).values).pow(2).mean()
            value = value + v_all[a_off:a_off + na].sub(a_d).pow(2).mean()
        loss = (args.sparse_weight * sparse + args.dense_weight * dense + args.anchor_weight * anchor
                + args.value_weight * value + args.gap_weight * gap
                + args.rehearse_weight * rehearse)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step % args.log_every == 0 or step == 1:
            rec = dict(step=step, loss=float(loss.detach()), sparse=float(sparse.detach()),
                       dense=float(dense.detach()), anchor=float(anchor.detach()),
                       value=float(value.detach()), gap=float(gap.detach()),
                       rehearse=float(rehearse.detach()),
                       gap_mean=float(gap_mean),
                       e_target=float(s_t.mean() if nb else d_t.min(dim=1).values.mean() + 1.0),
                       lr=lr,
                       wall=time.time() - t0)
            hist.append(rec)
            print(f"  step {step:5d}/{args.steps} loss {rec['loss']:7.4f} sparse {rec['sparse']:7.4f} "
                  f"dense {rec['dense']:7.4f} anchor {rec['anchor']:7.4f} value {rec['value']:7.4f} "
                  f"gap {rec['gap']:6.3f} (mean {rec['gap_mean']:5.2f}) reh {rec['rehearse']:6.4f} "
                  f"E[tgt] {rec['e_target']:6.2f} "
                  f"{rec['wall']:5.0f}s", flush=True)

    model.return_value = False
    mcfg = dict(ck["model_config"])
    if not Path(str(mcfg.get("layout_path", ""))).is_absolute():
        mcfg["layout_path"] = "data/ihes_piece_layout.json"       # repo-relative, portable
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": mcfg,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": ck.get("epoch"), "bellman_step": ck.get("bellman_step"),
                "avi": dict(round=args.round, shard=str(args.shard), init=str(args.init),
                            steps=args.steps, batch=args.batch, lr=args.lr,
                            anchor_weight=args.anchor_weight, sparse_weight=args.sparse_weight,
                            dense_weight=args.dense_weight, dense_batch=nd,
                            value_weight=args.value_weight, gap_weight=args.gap_weight,
                            gap_k_max=args.gap_k_max, rehearse_weight=args.rehearse_weight,
                            rehearse_batch=nr, rehearse_k_max=args.rehearse_k_max,
                            rehearse_tilt=args.rehearse_tilt,
                            teacher=str(args.teacher or args.init) if nr else None,
                            shard_meta=meta, history=hist)},
               args.out)
    print(f"wrote {args.out} ({time.time() - t0:.0f}s, "
          f"{args.steps * nb / max(1, N):.2f} passes over the sparse rows, "
          f"{args.steps * nd / max(1, M):.2f} over the dense rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
