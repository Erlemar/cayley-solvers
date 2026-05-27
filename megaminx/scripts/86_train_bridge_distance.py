"""Train the §3.5 residual-distance model and run the SATURATION PROBE.

A residual is just a state, so D is a ResMLPDistance (matched to the production V
arch: hidden_dims=[2048,512], rb=2, encoding=embedding) trained on residual states
to predict the window length j-i (an upper bound on, and for near-optimal paths
~=, the true distance between the two endpoints).

THE GATE (the whole point of this run): a held-out per-window-length calibration
table comparing, at each true window length, the model's D_pred against the
production V's prediction (precomputed in the dataset). The deployed bridge fails
because V SATURATES (~25-30) on deep residuals. So:

    PASS  if D_pred keeps climbing with true distance past ~30 while V flatlines
          -> in-distribution residual training escapes the saturation wall;
             build the full §3.5 stack (Bridge-Bellman true-distance + policy head).
    FAIL  if D_pred also flatlines ~25-30 (tracks V) -> the ceiling is intrinsic
          to large-permutation-distance estimation; §3.5 inherits it. Stop.

Better fit (lower MSE) is NOT the gate -- the GT-V/GT-Q lesson (rule 23) is that
fit does not imply usefulness. The slope past d=30 is the gate.

Run (GCP, after transferring the dataset .pt):
    python megaminx/scripts/86_train_bridge_distance.py \
        --data megaminx/data/bridge_distance_dataset.pt \
        --out-dir megaminx/models/m_bridge_d_v0 --n-epochs 150
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.model import ResMLPDistance

GATE_WLENS = (10, 20, 30, 40, 50, 60, 70)


@torch.no_grad()
def predict(model, residuals, device, batch=16384):
    """D(residual) for every row. residuals: uint8/long (N,120) on any device."""
    model.eval()
    n = residuals.shape[0]
    out = torch.empty(n, dtype=torch.float32, device=device)
    for i in range(0, n, batch):
        x = residuals[i : i + batch].to(device=device, dtype=torch.long)
        pred = model(x)
        out[i : i + batch] = (pred.squeeze(-1) if pred.dim() > 1 else pred).float()
    return out


def saturation_table(d_pred, v_pred, wlen):
    """Per-window-length means + the d=30->70 slope for D and V. Returns (rows, verdict)."""
    rows = []
    for w in GATE_WLENS:
        mask = wlen == w
        n = int(mask.sum())
        if n == 0:
            rows.append((w, 0, float("nan"), float("nan"), float("nan")))
            continue
        dm = float(d_pred[mask].mean())
        ds = float(d_pred[mask].std())
        vm = float(v_pred[mask].mean()) if v_pred is not None else float("nan")
        rows.append((w, n, dm, ds, vm))

    def slope(idx):
        a = next((r for r in rows if r[0] == 30 and r[1] > 0), None)
        b = next((r for r in rows if r[0] == 70 and r[1] > 0), None)
        if a is None or b is None:
            return float("nan")
        return b[idx] - a[idx]

    return rows, slope(2), slope(4)   # D-slope (col dm), V-slope (col vm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--loss", choices=["huber", "mse"], default="huber")
    # arch (defaults match production V m_dd_v0)
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--encoding", default="embedding")
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=86)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    state_size = int(blob["meta"]["state_size"])
    train_res = blob["train_residuals"].to(device)            # uint8 (N,120)
    train_y = blob["train_wlen"].to(device=device, dtype=torch.float32)
    val_res = blob["val_residuals"]                           # keep on cpu; predict() moves chunks
    val_wlen = blob["val_wlen"]
    val_v = blob.get("val_v_pred")
    val_v_dev = val_v.to(device) if val_v is not None else None
    print(f"data: train {train_res.shape[0]:,} | val {val_res.shape[0]:,} | "
          f"corpus={[Path(c).name for c in blob['meta']['corpus']]}", flush=True)
    if val_v is not None:
        print(f"  val V baseline present (from {Path(blob['meta'].get('v_checkpoint','?')).name})", flush=True)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPDistance(
        state_size=state_size, num_classes=state_size, hidden_dims=hidden_dims,
        num_res_blocks=args.num_res_blocks, encoding=args.encoding,
        embed_dim=args.embed_dim, output_dim=1, inference_chunk_size=None,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model_config = {
        "model_class": "ResMLPDistance", "state_size": state_size, "num_classes": state_size,
        "hidden_dims": list(hidden_dims), "num_res_blocks": args.num_res_blocks,
        "encoding": args.encoding, "embed_dim": args.embed_dim, "output_dim": 1,
    }
    print(f"model: {n_params:,} params  hidden={hidden_dims} rb={args.num_res_blocks} "
          f"enc={args.encoding} (residual-distance D)", flush=True)
    print(f"loss={args.loss}  epochs={args.n_epochs} batch={args.batch_size} lr={args.lr}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    loss_fn = (F.smooth_l1_loss if args.loss == "huber" else F.mse_loss)
    N = train_res.shape[0]
    gen = torch.Generator(device=device).manual_seed(args.seed)

    best_val = float("inf")
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(N, generator=gen, device=device)
        tot = 0.0
        nb = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = train_res[idx].long()
            yb = train_y[idx]
            if autocast is not None:
                with autocast:
                    pred = model(xb)
                    pred = pred.squeeze(-1) if pred.dim() > 1 else pred
                    loss = loss_fn(pred.float(), yb)
            else:
                pred = model(xb)
                pred = pred.squeeze(-1) if pred.dim() > 1 else pred
                loss = loss_fn(pred.float(), yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | {args.loss} {tot / max(nb,1):.4f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time()-t0:.1f}s", flush=True)

        if (epoch + 1) % args.val_every == 0 or epoch == args.n_epochs - 1:
            d_pred = predict(model, val_res, device).cpu()
            val_mae = float((d_pred - val_wlen.float()).abs().mean())
            rows, d_slope, v_slope = saturation_table(d_pred, val_v, val_wlen)
            print(f"  [val e{epoch}] MAE {val_mae:.2f} | SATURATION TABLE "
                  f"(true wlen -> D_pred +/- std vs V_pred):", flush=True)
            for w, n, dm, dstd, vm in rows:
                vs = f"{vm:5.1f}" if vm == vm else "  n/a"
                print(f"    wlen {w:3d} (n={n:5d}): D {dm:5.1f} +/-{dstd:4.1f}   V {vs}", flush=True)
            print(f"    >>> slope d30->d70:  D {d_slope:+5.1f}   V {v_slope:+5.1f}   "
                  f"(PASS if D climbs and V is flat)", flush=True)
            ckpt = {
                "epoch": epoch, "state_dict": model.state_dict(), "model_config": model_config,
                "val_mae": val_mae, "d_slope_30_70": d_slope, "v_slope_30_70": v_slope,
                "saturation_rows": rows, "data": str(args.data),
            }
            torch.save(ckpt, args.out_dir / f"epoch_{epoch:04d}.pt")
            if val_mae < best_val:
                best_val = val_mae
                torch.save(ckpt, args.out_dir / "best.pt")
                print(f"    new best val MAE {val_mae:.2f} -> best.pt", flush=True)

    print(f"done. best val MAE {best_val:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
