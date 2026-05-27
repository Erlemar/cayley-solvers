"""§3.5 v1: Bridge-Bellman true-distance D (the 'do it right' fix).

The v0 probe trained D to REGRESS to the window length j-i. That teaches
window-length, not true distance -- useless for finding compressible windows
(a window where true_dist << j-i), which is the whole point of bridge.

This script fixes the target. Because apply(s,a) maps the residual X=make_residual(s,t)
to apply(X,a), the Bridge-Bellman recursion D(s,t)=1+min_a D(apply(s,a),t) is exactly
distance-to-solved Bellman ON THE RESIDUAL:

    D(X) = clip( 1 + min_a D_target(apply(X, a)),  lower=0,  upper=wlen )

with the solved boundary D(identity)=0. wlen (the window length) enters as a per-state
UPPER-BOUND CLIP, not a regression target -- so Bellman can pull D BELOW wlen wherever
the residual is genuinely compressible (a shorter path exists), which is the signal the
deployed bridge needs. We reuse cayley.bellman._bellman_targets (battle-tested
solved-boundary + clip + softmin) and warm-start from the v0 D (which already climbs to
~52 at wlen-70), so Bellman REFINES a non-saturated estimate toward true distance rather
than bootstrapping from scratch (which tends to collapse).

THE GATE: after refinement, the held-out per-window-length calibration table. We want D
to STAY CLIMBING past d=40 (it escaped saturation AND got pulled toward true distance on
compressible windows) -- not collapse back to V's ~25 plateau (Bellman saturated on
residuals too -> the ceiling is intrinsic, §3.5 fails).

Run:
    python megaminx/scripts/87_train_bridge_bellman.py \
        --data megaminx/data/bridge_distance_dataset.pt \
        --warmstart megaminx/models/m_bridge_d_v0/best.pt \
        --out-dir megaminx/models/m_bridge_bellman_v0 --n-epochs 80
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.bellman import _apply_all_generators, _bellman_targets
from cayley.data import GeneratorTable
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx

GATE_WLENS = (10, 20, 30, 40, 50, 60, 70)


@torch.no_grad()
def predict(model, residuals, device, batch=16384):
    model.eval()
    n = residuals.shape[0]
    out = torch.empty(n, dtype=torch.float32, device=device)
    for i in range(0, n, batch):
        x = residuals[i : i + batch].to(device=device, dtype=torch.long)
        pred = model(x)
        out[i : i + batch] = (pred.squeeze(-1) if pred.dim() > 1 else pred).float()
    return out


def saturation_table(d_pred, v_pred, wlen, d0_pred=None):
    """Per-window-length means + d30->d70 slope. d0_pred = the v0 D curve (optional)."""
    rows = []
    for w in GATE_WLENS:
        mask = wlen == w
        n = int(mask.sum())
        if n == 0:
            rows.append((w, 0, float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        dm = float(d_pred[mask].mean())
        ds = float(d_pred[mask].std())
        vm = float(v_pred[mask].mean()) if v_pred is not None else float("nan")
        d0 = float(d0_pred[mask].mean()) if d0_pred is not None else float("nan")
        rows.append((w, n, dm, ds, vm, d0))

    def slope(col):
        a = next((r for r in rows if r[0] == 30 and r[1] > 0), None)
        b = next((r for r in rows if r[0] == 70 and r[1] > 0), None)
        return (b[col] - a[col]) if (a and b) else float("nan")

    return rows, slope(2), slope(4), slope(5)   # D-slope, V-slope, v0-D-slope


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--warmstart", type=Path, required=True,
                    help="v0 D checkpoint to refine (non-saturated start)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--target-update-every", type=int, default=5)
    ap.add_argument("--target-net-chunk", type=int, default=16384)
    ap.add_argument("--softmin-temperature", type=float, default=0.0)
    ap.add_argument("--no-clip-upper", action="store_true",
                    help="disable the wlen upper-bound clip (debug: pure Bellman, no cap)")
    ap.add_argument("--n-anchor-v0", type=int, default=64)
    ap.add_argument("--n-anchor-d1", type=int, default=4)
    ap.add_argument("--loss", choices=["huber", "mse"], default="huber")
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=87)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})", flush=True)

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    S = int(blob["meta"]["state_size"])
    train_res = blob["train_residuals"].to(device)                     # uint8 (N,120)
    train_wlen = blob["train_wlen"].to(device=device, dtype=torch.float32)
    val_res = blob["val_residuals"]
    val_wlen = blob["val_wlen"]
    val_v = blob.get("val_v_pred")
    print(f"data: train {train_res.shape[0]:,} | val {val_res.shape[0]:,}", flush=True)

    # Warm-start the model from the v0 D checkpoint (same arch).
    wck = torch.load(args.warmstart, map_location="cpu", weights_only=False)
    mcfg = wck["model_config"]
    model = ResMLPDistance(
        state_size=int(mcfg["state_size"]), num_classes=int(mcfg["num_classes"]),
        hidden_dims=tuple(mcfg["hidden_dims"]), num_res_blocks=int(mcfg["num_res_blocks"]),
        encoding=mcfg.get("encoding", "embedding"), embed_dim=int(mcfg.get("embed_dim", 16)),
        output_dim=1, inference_chunk_size=None,
    ).to(device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in wck["state_dict"].items()}
    model.load_state_dict(sd)
    print(f"warm-started D from {args.warmstart.name} ({sum(p.numel() for p in model.parameters()):,} params)",
          flush=True)

    # Keep the v0 D curve for the head-to-head (frozen copy, never updated).
    v0_model = copy.deepcopy(model).eval()
    for p in v0_model.parameters():
        p.requires_grad = False
    val_d0 = predict(v0_model, val_res, device).cpu()

    # Target net for Bellman (refreshed every N epochs).
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    generators = torch.from_numpy(GeneratorTable.from_puzzle(puzzle).perms).to(device)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)
    anchor_v0 = solved_state.unsqueeze(0)                              # (1,S) long
    anchor_d1 = _apply_all_generators(anchor_v0, generators).squeeze(0)  # (24,S) long

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    loss_fn = F.smooth_l1_loss if args.loss == "huber" else F.mse_loss
    clip_upper = not args.no_clip_upper
    N = train_res.shape[0]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"Bellman refine: clip_upper(wlen)={clip_upper} softmin_T={args.softmin_temperature} "
          f"target_update_every={args.target_update_every} anchors V0x{args.n_anchor_v0} "
          f"d1x{args.n_anchor_d1} | epochs={args.n_epochs} lr={args.lr}", flush=True)

    best_slope = -1e9
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(N, generator=gen, device=device)
        tot = 0.0
        nb = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs = train_res[idx].long()
            wl = train_wlen[idx]
            target = _bellman_targets(
                target_model, bs, wl, generators, solved_state,
                chunk_size=args.target_net_chunk, clip_upper=clip_upper, clip_lower=True,
                softmin_temperature=args.softmin_temperature,
            )
            bs_parts, tgt_parts = [bs], [target]
            if args.n_anchor_v0 > 0:
                bs_parts.append(anchor_v0.expand(args.n_anchor_v0, -1))
                tgt_parts.append(torch.zeros(args.n_anchor_v0, device=device))
            if args.n_anchor_d1 > 0:
                bs_parts.append(anchor_d1.repeat(args.n_anchor_d1, 1))
                tgt_parts.append(torch.ones(24 * args.n_anchor_d1, device=device))
            bs_all = torch.cat(bs_parts, dim=0)
            tgt_all = torch.cat(tgt_parts, dim=0)
            if autocast is not None:
                with autocast:
                    pred = model(bs_all)
                    pred = pred.squeeze(-1) if pred.dim() > 1 else pred
                    loss = loss_fn(pred.float(), tgt_all)
            else:
                pred = model(bs_all)
                pred = pred.squeeze(-1) if pred.dim() > 1 else pred
                loss = loss_fn(pred.float(), tgt_all)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if (epoch + 1) % args.target_update_every == 0:
            target_model.load_state_dict(model.state_dict())
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | {args.loss} {tot/max(nb,1):.4f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time()-t0:.1f}s", flush=True)

        if (epoch + 1) % args.val_every == 0 or epoch == args.n_epochs - 1:
            d_pred = predict(model, val_res, device).cpu()
            val_mae = float((d_pred - val_wlen.float()).abs().mean())
            rows, d_slope, v_slope, d0_slope = saturation_table(d_pred, val_v, val_wlen, val_d0)
            print(f"  [val e{epoch}] MAE-vs-wlen {val_mae:.2f} | CALIBRATION "
                  f"(true wlen -> Bellman-D vs v0-D vs V):", flush=True)
            for w, n, dm, dstd, vm, d0 in rows:
                vs = f"{vm:5.1f}" if vm == vm else "  n/a"
                d0s = f"{d0:5.1f}" if d0 == d0 else "  n/a"
                print(f"    wlen {w:3d} (n={n:5d}): Bellman-D {dm:5.1f}+/-{dstd:4.1f}   "
                      f"v0-D {d0s}   V {vs}", flush=True)
            print(f"    >>> slope d30->d70:  Bellman-D {d_slope:+5.1f}   v0-D {d0_slope:+5.1f}   "
                  f"V {v_slope:+5.1f}   (want Bellman-D climbing, not collapsed to V)", flush=True)
            ckpt = {
                "epoch": epoch, "state_dict": model.state_dict(),
                "model_config": {"model_class": "ResMLPDistance", **{
                    k: mcfg[k] for k in ("state_size", "num_classes", "hidden_dims",
                                         "num_res_blocks", "encoding", "embed_dim")},
                    "output_dim": 1},
                "val_mae_vs_wlen": val_mae, "d_slope_30_70": d_slope,
                "v_slope_30_70": v_slope, "v0_slope_30_70": d0_slope, "saturation_rows": rows,
            }
            torch.save(ckpt, args.out_dir / f"epoch_{epoch:04d}.pt")
            if d_slope > best_slope:   # keep the least-saturated (steepest deep slope)
                best_slope = d_slope
                torch.save(ckpt, args.out_dir / "best.pt")
                print(f"    new best deep-slope {d_slope:+.1f} -> best.pt", flush=True)

    print(f"done. best deep-slope d30->d70 {best_slope:+.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
