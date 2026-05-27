"""Perceiver IO V model (doc 3.9) -- saturation-gate probe.

Two-stage recipe (CLAUDE rule 18: a new-shape model needs a from-scratch MSE
pretrain before Bellman, since Bellman from random init rarely converges):
    Stage 1: walk-depth MSE pretrain (predict random-walk depth).
    Stage 2: Bellman refine (reuse cayley.bellman._bellman_targets) + V0/d1 anchors.

THE GATE (CLAUDE rule 23): after Bellman, does the Perceiver-encoded V SATURATE
near the puzzle diameter on deep random walks -- like the working ResMLP V (V@d80
~= 29) -- or keep drifting like the rejected GraphTransformer V (V@d80 = 42)?
Saturation is the load-bearing property for beam; it is problem-intrinsic and has
defeated every alt encoder so far (GT-V, state_inv, the 3.5 Bellman collapse). The
canary reports mean V by random-walk depth; the binding check is V@d80 - V@d40.

    PASS  V@d80 - V@d40 <= ~10 AND V(V0)~=0, V(d1)~=1  -> saturates; promote to a
          full strat-51 eval (wire into cayley.search first).
    FAIL  V@d80 - V@d40 > ~10 (drifts)                 -> same wall; reject.

Run (GCP L4):
    python megaminx/scripts/88_perceiver_v_probe.py \
        --out-dir megaminx/models/m_perceiver_v0 \
        --pretrain-epochs 40 --bellman-epochs 120
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
from cayley.data import GeneratorTable, generate_walks_torch
from megaminx.perceiver_v import PerceiverV
from megaminx.puzzle import Megaminx

CANARY_DEPTHS = (10, 20, 30, 40, 60, 80)


@torch.no_grad()
def saturation_canary(model, puzzle, generators, solved, device, n_per=2048, k_max=90,
                      seed=999, halfwidth=2):
    """Mean V by random-walk depth + V0/d1 calibration. Returns (rows, v0, vd1, gap)."""
    model.eval()
    n_walks = max(1, (n_per * len(CANARY_DEPTHS) * 3) // k_max)
    states, dep = generate_walks_torch(puzzle, n_walks=n_walks, k_max=k_max, seed=seed,
                                       device=device, n_back=1)
    rows = []
    means = {}
    for c in CANARY_DEPTHS:
        idx = torch.nonzero((dep - c).abs() <= halfwidth, as_tuple=True)[0]
        if idx.numel() == 0:
            rows.append((c, 0, float("nan")))
            continue
        idx = idx[:n_per]
        v = model(states[idx]).float().mean().item()
        means[c] = v
        rows.append((c, int(idx.numel()), v))
    v0 = float(model(solved.unsqueeze(0)).float().item())
    d1 = _apply_all_generators(solved.unsqueeze(0), generators).squeeze(0)
    vd1 = float(model(d1).float().mean().item())
    gap = (means.get(80, float("nan")) - means.get(40, float("nan")))
    return rows, v0, vd1, gap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pretrain-epochs", type=int, default=40)
    ap.add_argument("--bellman-epochs", type=int, default=120)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--target-update-every", type=int, default=10)
    ap.add_argument("--target-net-chunk", type=int, default=4096,
                    help="Target-net forward chunk. Keep small: the Perceiver's attention "
                         "thrashes at large batch (557s/epoch at 16384 on 16GB). 4096 is safe.")
    ap.add_argument("--n-anchor-v0", type=int, default=32)
    ap.add_argument("--n-anchor-d1", type=int, default=4)
    # arch
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-latents", type=int, default=128)
    ap.add_argument("--n-self-layers", type=int, default=4)
    ap.add_argument("--n-self-heads", type=int, default=8)
    ap.add_argument("--n-cross-heads", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    ap.add_argument("--canary-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'cpu'})", flush=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    feats = torch.load(PROJECT / "data" / "bipartite_features.pt", map_location="cpu",
                       weights_only=False)
    generators = torch.from_numpy(GeneratorTable.from_puzzle(puzzle).perms).to(device)
    solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)

    model = PerceiverV(
        feats, d_model=args.d_model, n_latents=args.n_latents,
        n_cross_heads=args.n_cross_heads, n_self_layers=args.n_self_layers,
        n_self_heads=args.n_self_heads, ffn_dim=args.ffn_dim, output_dim=1,
        inference_chunk_size=None,
    ).to(device)
    print(f"model: {model.num_parameters():,} params  d={args.d_model} latents={args.n_latents} "
          f"self_layers={args.n_self_layers} (Perceiver V, T={model.n_tok}, no attn mask)", flush=True)

    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    def run_canary(tag):
        rows, v0, vd1, gap = saturation_canary(model, puzzle, generators, solved, device)
        body = "  ".join(f"d{c}:{v:.1f}" for c, n, v in rows if n > 0)
        verdict = "SATURATES" if (gap == gap and gap <= 10) else "DRIFTS" if gap == gap else "?"
        print(f"  [canary {tag}] V0 {v0:+.2f}  V(d1) {vd1:.2f}  | {body}  | "
              f"V@d80-V@d40 = {gap:+.1f}  -> {verdict}", flush=True)
        return gap, v0, vd1

    # ---------------- Stage 1: walk-depth MSE pretrain ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.pretrain_epochs))
    n_walks = max(1, args.samples_per_epoch // args.k_max)
    print(f"=== Stage 1: walk-depth MSE pretrain ({args.pretrain_epochs} ep) ===", flush=True)
    for epoch in range(args.pretrain_epochs):
        t0 = time.time()
        states, depths = generate_walks_torch(puzzle, n_walks=n_walks, k_max=args.k_max,
                                              seed=args.seed + epoch, device=device, n_back=1)
        depths_f = depths.float()
        N = states.shape[0]
        perm = torch.randperm(N, generator=batch_gen, device=device)
        model.train()
        tot = 0.0
        nb = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs, yb = states[idx], depths_f[idx]
            if autocast is not None:
                with autocast:
                    loss = F.mse_loss(model(bs).float(), yb)
            else:
                loss = F.mse_loss(model(bs).float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if epoch % 5 == 0 or epoch == args.pretrain_epochs - 1:
            print(f"  [pre {epoch:3d}] mse {tot/max(nb,1):.4f} | {time.time()-t0:.1f}s", flush=True)
    run_canary("post-pretrain")
    torch.save({"epoch": -1, "stage": "pretrain", "state_dict": model.state_dict(),
                "model_config": model.get_model_config()}, args.out_dir / "pretrain.pt")

    # ---------------- Stage 2: Bellman refine ----------------
    target_model = copy.deepcopy(model).eval()
    for p in target_model.parameters():
        p.requires_grad = False
    anchor_v0 = solved.unsqueeze(0)
    anchor_d1 = _apply_all_generators(anchor_v0, generators).squeeze(0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.bellman_epochs))
    print(f"=== Stage 2: Bellman refine ({args.bellman_epochs} ep, target_update={args.target_update_every}, "
          f"anchors V0x{args.n_anchor_v0} d1x{args.n_anchor_d1}) ===", flush=True)
    best_gap = float("inf")
    for epoch in range(args.bellman_epochs):
        t0 = time.time()
        states, depths = generate_walks_torch(puzzle, n_walks=n_walks, k_max=args.k_max,
                                              seed=args.seed + 1000 + epoch, device=device, n_back=1)
        depths_f = depths.float()
        N = states.shape[0]
        perm = torch.randperm(N, generator=batch_gen, device=device)
        model.train()
        tot = 0.0
        nb = 0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs, bd = states[idx], depths_f[idx]
            # bf16 target forwards (autocast): ~2x faster + half the attn memory. The
            # target net is frozen so reduced precision on the bootstrap is fine.
            if autocast is not None:
                with autocast:
                    target = _bellman_targets(target_model, bs, bd, generators, solved,
                                              chunk_size=args.target_net_chunk,
                                              clip_upper=True, clip_lower=True)
            else:
                target = _bellman_targets(target_model, bs, bd, generators, solved,
                                          chunk_size=args.target_net_chunk,
                                          clip_upper=True, clip_lower=True)
            bs_parts, tgt_parts = [bs], [target]
            if args.n_anchor_v0 > 0:
                bs_parts.append(anchor_v0.expand(args.n_anchor_v0, -1).to(bs.dtype))
                tgt_parts.append(torch.zeros(args.n_anchor_v0, device=device))
            if args.n_anchor_d1 > 0:
                bs_parts.append(anchor_d1.repeat(args.n_anchor_d1, 1).to(bs.dtype))
                tgt_parts.append(torch.ones(24 * args.n_anchor_d1, device=device))
            bs_all = torch.cat(bs_parts, dim=0)
            tgt_all = torch.cat(tgt_parts, dim=0)
            if autocast is not None:
                with autocast:
                    loss = F.mse_loss(model(bs_all).float(), tgt_all)
            else:
                loss = F.mse_loss(model(bs_all).float(), tgt_all)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        sched.step()
        if (epoch + 1) % args.target_update_every == 0:
            target_model.load_state_dict(model.state_dict())
        if epoch % 5 == 0 or epoch == args.bellman_epochs - 1:
            print(f"  [bel {epoch:3d}] mse {tot/max(nb,1):.4f} | lr {float(sched.get_last_lr()[0]):.2e} "
                  f"| {time.time()-t0:.1f}s", flush=True)
        if (epoch + 1) % args.canary_every == 0 or epoch == args.bellman_epochs - 1:
            gap, v0, vd1 = run_canary(f"bel-e{epoch}")
            ckpt = {"epoch": epoch, "stage": "bellman", "state_dict": model.state_dict(),
                    "model_config": model.get_model_config(),
                    "canary_gap_d80_d40": gap, "v0": v0, "vd1": vd1}
            torch.save(ckpt, args.out_dir / f"epoch_{epoch:04d}.pt")
            if gap == gap and abs(gap) < abs(best_gap):
                best_gap = gap
                torch.save(ckpt, args.out_dir / "best.pt")

    print(f"\n=== VERDICT === best V@d80-V@d40 gap: {best_gap:+.1f}  "
          f"({'PASS (saturates)' if best_gap <= 10 else 'FAIL (drifts, same wall as GT-V)'})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
