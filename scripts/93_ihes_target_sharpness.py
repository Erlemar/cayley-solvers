"""Is the Beam-AVI bootstrap target flatter than the model it trains? (IHES diagnostic, ~1 min)

The 2026-09-17 ablation on arm A's r000 shard showed that the UNBIASED dense stream alone
flattens Q-space almost as much as the beam-selected sparse stream (deep gap01 0.387 -> 0.253
vs 0.231). So the flattening lives in the target, not in which rows were harvested.

Suspected mechanism: `1 + min_b Q(child, b)` is biased low under noise, and more so when the
child has many near-minimal successors. An UPHILL child always has at least one (the undo
move), plus one per downhill parent move that commutes with the move just made, so its target
is pulled down harder than a downhill child's. The gap between good and bad moves shrinks.

This script scores N beam parents from a shard's dense stream under several target
constructions and prints their within-state sharpness next to the model's own Q:
  Q0     the model, 1 frame                    Q8   the model, averaged over F frames
  T1     1 + min_b Q(child) (the harvest target, 1 frame)
  T8     1 + min_b [frame-averaged Q(child)]   average THEN min: less noise, less min-bias
  T8b    frame-average of (1 + min_b Q_k(child))  min THEN average: same bias as T1
  TV1    V_head(child), 1 frame                no min operator at all
  TV8    V_head(child), frame-averaged
If T8 is much sharper than T1 while T8b is not, noise-driven min-bias is the mechanism, and
frame-averaged targets are the fix to try.

    python scripts/93_ihes_target_sharpness.py --shard runs/ihes_avi/A/shards/r000.pt \
        --ckpt models/ihes_tf_b_e1300a_s3/step_02000.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.puzzle import PictureCube  # noqa: E402


def _t90():
    path = PROJECT / "scripts" / "90_ihes_train_avi.py"
    spec = importlib.util.spec_from_file_location("_ihes_t90", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sharp(m: torch.Tensor, ref_arg: torch.Tensor | None = None) -> dict:
    s = m.sort(dim=1).values
    out = {"level": float((1.0 + s[:, 0]).mean()), "gap01": float((s[:, 1] - s[:, 0]).mean()),
           "n_red": float((m < s[:, :1] + 1.0).sum(dim=1).float().mean()),
           "wsd": float(m.std(dim=1).mean()),
           "rest_gap": float((s[:, 1:].mean(dim=1) - s[:, 0]).mean())}
    if ref_arg is not None:
        out["argmin_agree"] = float((m.argmin(dim=1) == ref_arg).float().mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Within-state sharpness of Beam-AVI targets.")
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--n-parents", type=int, default=4096)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--deep", type=float, default=14.0)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    t90 = _t90()
    t51 = t90._t51()
    puzzle = PictureCube.load(args.data_dir / "puzzle_info.json")
    gens = torch.tensor([puzzle.generators[n] for n in puzzle.move_names], dtype=torch.int64,
                        device=dev)
    A = gens.size(0)
    sym = t51.Symmetries(args.data_dir, dev, prefix="cube")
    sym.verify(gens)

    blob = torch.load(args.shard, map_location="cpu", weights_only=False)
    dn = blob["dense"]
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(dn["state"].size(0), generator=g)[:args.n_parents]
    P = dn["state"].index_select(0, idx).to(dev).long()
    stored = dn["target"].index_select(0, idx).to(dev).float()
    N = P.size(0)
    others = [k for k in range(sym.n_sym) if k != sym.identity]
    pick = torch.randperm(len(others), generator=g)[:args.frames - 1].tolist()
    frames = [sym.identity] + [others[i] for i in pick]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = t90.build_model(ck, args.ckpt, dev).eval()
    has_v = bool(getattr(model, "has_value_head", False))
    model.return_value = has_v

    @torch.no_grad()
    def qv(x: torch.Tensor, k: int):
        kv = torch.full((x.size(0),), k, dtype=torch.int64, device=dev)
        xc = sym.conjugate(x, kv)
        qs, vs = [], []
        for i in range(0, x.size(0), args.chunk_size):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                out = model(xc[i:i + args.chunk_size])
            q, v = out if has_v else (out, None)
            qs.append(q.float())
            vs.append(v.float().reshape(-1) if v is not None else None)
        q = torch.cat(qs)
        q = torch.gather(q, 1, sym.sigma_inv[k].unsqueeze(0).expand(q.size(0), -1))
        return q, (torch.cat(vs) if has_v else None)

    C = P[:, gens].reshape(N * A, -1)                  # child_a = parent[gens[a]]
    q0, _ = qv(P, sym.identity)
    q_sum = torch.zeros_like(q0)
    qc_sum = torch.zeros(N * A, A, device=dev)
    t_minavg = torch.zeros(N * A, device=dev)
    vc_sum = torch.zeros(N * A, device=dev)
    qc1 = vc1 = None
    for j, k in enumerate(frames):
        qp, _ = qv(P, k)
        q_sum += qp
        qc, vc = qv(C, k)
        if j == 0:
            qc1, vc1 = qc, vc
        qc_sum += qc
        t_minavg += 1.0 + qc.min(dim=1).values
        if vc is not None:
            vc_sum += vc
    F = len(frames)
    rows = {
        "Q0 model 1fr": q0,
        f"Q{F} model avg": q_sum / F,
        "stored target": stored,
        "T1 1+minQ 1fr": (1.0 + qc1.min(dim=1).values).view(N, A),
        f"T{F} avg-then-min": (1.0 + (qc_sum / F).min(dim=1).values).view(N, A),
        f"T{F}b min-then-avg": (t_minavg / F).view(N, A),
    }
    if has_v:
        rows["TV1 V(child) 1fr"] = vc1.view(N, A)
        rows[f"TV{F} V(child) avg"] = (vc_sum / F).view(N, A)
    for m in rows.values():
        m.clamp_(0.0, 32.0)
    ref = q0.argmin(dim=1)
    deep = (1.0 + q0.min(dim=1).values) >= args.deep
    print(f"{N:,} beam parents from {args.shard.name}; {int(deep.sum()):,} deep (model level >= "
          f"{args.deep}); frames {frames}; |stored - T1| mean "
          f"{float((stored - rows['T1 1+minQ 1fr']).abs().mean()):.4f}")
    hdr = (f"{'target':>22} | {'level':>6} {'gap01':>6} {'n_red':>5} {'wsd':>5} {'rest':>5} "
           f"{'agree':>5} | deep: {'level':>6} {'gap01':>6} {'n_red':>5} {'wsd':>5} {'rest':>5}")
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows.items():
        a, d = sharp(m, ref), sharp(m[deep])
        print(f"{name:>22} | {a['level']:6.2f} {a['gap01']:6.3f} {a['n_red']:5.2f} {a['wsd']:5.2f} "
              f"{a['rest_gap']:5.2f} {a['argmin_agree']:5.3f} | deep: {d['level']:6.2f} "
              f"{d['gap01']:6.3f} {d['n_red']:5.2f} {d['wsd']:5.2f} {d['rest_gap']:5.2f}",
              flush=True)
    print("\nrest = mean over the 17 non-best columns minus the best. A target FLATTER than Q0 "
          "(lower gap01/wsd/rest, higher n_red) pulls the model flat when fitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
