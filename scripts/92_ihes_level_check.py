"""Matched flattening check for IHES Beam-AVI checkpoints (~1-2 min, run after EVERY round).

Port of `tetraminx/scripts/77_level_check.py` plus the within/between-state decomposition
that predicted BOTH negative tetraminx gates before either ran (`tetraminx/BEAM_AVI_PLAN.md`
s5b). One FIXED population of beam states (from one shard) is scored by every checkpoint,
so only the weights vary.

Columns (all on the fixed population; "deep" = harvested target >= --deep):
  V_hat       mean of 1 + min_a Q(s,a)            the level the beam ranks on
  Vhead       mean AZ value head
  btw_sd      sd over states of V_hat              cross-parent spread; the global top-B
                                                   runs on it. FALLING = flattening
  gap01       mean of Q_(2) - Q_(1)                top-1 discrimination. FALLING = flattening
  n_red       mean #children with Q < min Q + 1    believed distance-reducing children.
                                                   RISING = flattening
  wsd         mean within-state sd of Q
  anc_top1    exact d5/d6 anchors: argmin Q is a true downhill move
  anc_mae     exact d5/d6 anchors: mean |Q - exact|

Tetraminx's failing rounds: deep gap01 0.438 -> 0.270, btw_sd 6.83 -> 6.23, n_red 1.90 -> 2.65,
while V_hat ROSE and the loss FELL. Watch gap01 / btw_sd / n_red, not the level.

    python scripts/92_ihes_level_check.py --shard runs/ihes_avi/A/shards/r000.pt \
        --ckpt v1b=models/ihes_tf_b_e1300a_s3/step_02000.pt --ckpt r000=runs/ihes_avi/A/models/r000.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))


def _t90():
    path = PROJECT / "scripts" / "90_ihes_train_avi.py"
    spec = importlib.util.spec_from_file_location("_ihes_t90", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def score(model, states, chunk, dev):
    has_v = bool(getattr(model, "has_value_head", False))
    model.return_value = has_v
    qs, vs = [], []
    for i in range(0, states.size(0), chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            out = model(states[i:i + chunk])
        q, v = out if has_v else (out, None)
        qs.append(q.float())
        if v is not None:
            vs.append(v.float())
    model.return_value = False
    return torch.cat(qs), (torch.cat(vs) if vs else None)


def stats(q: torch.Tensor) -> dict:
    qs = q.sort(dim=1).values
    vhat = 1.0 + qs[:, 0]
    return {"V_hat": float(vhat.mean()), "btw_sd": float(vhat.std()),
            "gap01": float((qs[:, 1] - qs[:, 0]).mean()),
            "n_red": float((q < qs[:, :1] + 1.0).sum(dim=1).float().mean()),
            "wsd": float(q.std(dim=1).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed-population flattening check.")
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--ckpt", action="append", required=True, help="name=path.pt")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--deep", type=float, default=14.0)
    ap.add_argument("--anchors", type=Path, default=PROJECT / "data" / "ihes_q_anchors_d6.pt")
    ap.add_argument("--n-anchor", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--json", type=Path, default=None, help="append results here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device
    t90 = _t90()

    blob = torch.load(args.shard, map_location="cpu", weights_only=False)
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(blob["sparse"]["child"].size(0), generator=gen)[:args.n]
    states = blob["sparse"]["child"].index_select(0, idx).to(dev).long()
    shard_t = blob["sparse"]["target"].index_select(0, idx).float().to(dev)
    deep = shard_t >= args.deep
    a = torch.load(args.anchors, map_location="cpu", weights_only=False)
    a_keep = torch.nonzero(a["depths"] >= 5).squeeze(1)
    a_idx = a_keep[torch.randperm(a_keep.numel(), generator=gen)[:args.n_anchor]]
    a_s = a["states"].index_select(0, a_idx).to(dev).long()
    a_q = a["q_targets"].index_select(0, a_idx).to(dev).float()
    a_d = a["depths"].index_select(0, a_idx).to(dev).float()
    print(f"fixed population: {states.size(0):,} beam states from {args.shard.name} "
          f"(harvested target mean {shard_t.mean():.3f}; deep >= {args.deep}: "
          f"{int(deep.sum()):,}); {a_s.size(0):,} exact anchors at d5-6", flush=True)

    hdr = (f"{'ckpt':>10} | {'V_hat':>6} {'Vhead':>6} {'btw_sd':>6} {'gap01':>6} {'n_red':>5} "
           f"{'wsd':>5} | deep: {'V_hat':>6} {'btw_sd':>6} {'gap01':>6} {'n_red':>5} | "
           f"{'anc_top1':>8} {'anc_mae':>7}")
    print(hdr)
    print("-" * len(hdr))
    results = []
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        p = Path(path) if Path(path).is_absolute() else PROJECT / path
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = t90.build_model(ck, p, dev).eval()
        q, v = score(model, states, args.chunk_size, dev)
        qa, _ = score(model, a_s, args.chunk_size, dev)
        allv, dp = stats(q), stats(q[deep])
        down = a_q <= (a_d - 1.0).unsqueeze(1) + 0.5        # true distance-reducing moves
        top1 = float(down.gather(1, qa.argmin(dim=1, keepdim=True)).float().mean())
        mae = float((qa - a_q).abs().mean())
        vh = float(v.mean()) if v is not None else float("nan")
        results.append({"ckpt": name, "path": str(p), "all": allv, "deep": dp,
                        "vhead": vh, "anc_top1": top1, "anc_mae": mae})
        print(f"{name:>10} | {allv['V_hat']:6.2f} {vh:6.2f} {allv['btw_sd']:6.3f} "
              f"{allv['gap01']:6.3f} {allv['n_red']:5.2f} {allv['wsd']:5.2f} | deep: "
              f"{dp['V_hat']:6.2f} {dp['btw_sd']:6.3f} {dp['gap01']:6.3f} {dp['n_red']:5.2f} | "
              f"{top1:8.4f} {mae:7.3f}", flush=True)
        del model
        torch.cuda.empty_cache() if dev.startswith("cuda") else None
    print("\nFLATTENING = gap01 and btw_sd FALLING, n_red RISING (tetraminx: deep gap01 "
          "0.438 -> 0.270 predicted a losing gate). A rising V_hat alone is not health.")
    if args.json:
        prev = json.loads(args.json.read_text(encoding="utf-8")) if args.json.exists() else []
        prev.append({"shard": str(args.shard), "n": states.size(0), "deep_thr": args.deep,
                     "results": results})
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(prev, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
