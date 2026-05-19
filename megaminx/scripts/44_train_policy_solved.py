"""m_pi_v0: train a policy head on solved-path tuples (Idea 4).

Loads `data/policy_train.pt` (built by 43_build_policy_dataset.py) and trains a
24-output policy head π(a|s) with weighted cross-entropy. Weight per tuple is
`1 / (1 + suffix_len)` so moves close to solved (where the solver's path is
near-optimal and most informative) are upweighted.

Body weights warmstart from m05 (V-model); the head is reinitialized at
output_dim=n_gen. Distinct from the existing 11_train_policy_head.py which
uses inverse-RW labels and unweighted CE.

The trained policy is consumed by beam search as
    score(child) = V(child) - λ · log π(a|parent)
where λ ∈ {0.05, 0.1, 0.2, 0.5} is swept on strat-5.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/44_train_policy_solved.py \\
        --warmstart megaminx/models/m05_bellman_warm/epoch_0499.pt \\
        --data megaminx/data/policy_train.pt \\
        --out-dir megaminx/models/m_pi_v0 \\
        --n-epochs 200
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
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path, required=True,
                    help="V-model checkpoint (e.g. m05) to warmstart body from")
    ap.add_argument("--data", type=Path, required=True,
                    help="policy_train.pt produced by 43_build_policy_dataset.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=410)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=25)
    ap.add_argument("--hidden-dims", type=str, default="2048,512",
                    help="match m05's body shape so warmstart loads cleanly")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--use-weights", action=argparse.BooleanOptionalAction, default=True,
                    help="weight CE by 1/(1+suffix_len) (default on; --no-use-weights for plain CE)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)

    # Load training data.
    d = torch.load(args.data, map_location="cpu", weights_only=False)
    states_cpu = d["states"]              # (N, 120) int8
    actions_cpu = d["actions"].to(torch.long)   # (N,) int64 (cross_entropy needs long)
    weights_cpu = d["weights"]            # (N,) float32
    md = d.get("metadata", {})
    print(f"data: {states_cpu.size(0):,} tuples from {md.get('submission', '?')}")

    # Filter out sentinel actions (=-1 from --include-final-solved-state).
    valid_mask = actions_cpu >= 0
    if int(valid_mask.sum()) < states_cpu.size(0):
        states_cpu = states_cpu[valid_mask]
        actions_cpu = actions_cpu[valid_mask]
        weights_cpu = weights_cpu[valid_mask]
        print(f"  filtered to {states_cpu.size(0):,} (removed {(~valid_mask).sum().item()} sentinels)")
    N = states_cpu.size(0)

    # Move to GPU once.
    states = states_cpu.to(device)
    actions = actions_cpu.to(device)
    weights = weights_cpu.to(device)
    if not args.use_weights:
        weights = torch.ones_like(weights)
        print("  (--no-use-weights) using uniform weights")

    # Build student. output_dim = n_gen (policy logits).
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPDistance(
        state_size=120, num_classes=120,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16,
        output_dim=n_gen,
    ).to(device)

    # Warmstart body from V-model. Drop head keys (different output_dim).
    teacher_ckpt = torch.load(args.warmstart, map_location=device, weights_only=False)
    teacher_sd = teacher_ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in teacher_sd):
        teacher_sd = {k.removeprefix("_orig_mod."): v for k, v in teacher_sd.items()}
    body_sd = {k: v for k, v in teacher_sd.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(body_sd, strict=False)
    n_missing_head = sum(1 for k in missing if k.startswith("head."))
    n_missing_body = sum(1 for k in missing if not k.startswith("head."))
    print(f"warmstart from {args.warmstart}: body loaded "
          f"({n_missing_head} head keys reinit, {n_missing_body} body keys missing)")

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0,
        fused=device == "cuda",
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"epochs: {args.n_epochs}  batch: {args.batch_size}  N={N:,}  "
          f"weighted_ce={args.use_weights}")

    best_acc = 0.0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        model.train()
        total_loss, total_acc, total_w_acc, n_batches = 0.0, 0.0, 0.0, 0
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs = states[idx]
            bl = actions[idx]
            bw = weights[idx]

            if autocast_ctx is not None:
                with autocast_ctx:
                    logits = model(bs)            # (bs, n_gen)
                    # Weighted CE: F.cross_entropy with reduction='none' then weighted mean.
                    losses = F.cross_entropy(logits, bl, reduction="none")  # (bs,)
                    loss = (losses * bw).sum() / bw.sum().clamp(min=1e-6)
            else:
                logits = model(bs)
                losses = F.cross_entropy(logits, bl, reduction="none")
                loss = (losses * bw).sum() / bw.sum().clamp(min=1e-6)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == bl).float()
                total_acc += float(acc.mean().item())
                total_w_acc += float(((acc * bw).sum() / bw.sum().clamp(min=1e-6)).item())
            n_batches += 1
        sched.step()

        avg_loss = total_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)
        avg_w_acc = total_w_acc / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | CE {avg_loss:.4f} | top1 {avg_acc:.3f} "
                  f"(w {avg_w_acc:.3f}) | lr {float(sched.get_last_lr()[0]):.2e} | "
                  f"{time.time() - t0:.1f}s", flush=True)

        if avg_w_acc > best_acc:
            best_acc = avg_w_acc

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.n_epochs - 1:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "ce_loss": avg_loss,
                "top1_acc": avg_acc,
                "weighted_top1_acc": avg_w_acc,
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": list(hidden_dims),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "embedding", "embed_dim": 16,
                    "output_dim": n_gen,
                },
                "warmstart_from": str(args.warmstart),
                "data_path": str(args.data),
                "head_kind": "policy",
                "use_weights": args.use_weights,
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    print(f"final: best weighted top1 acc = {best_acc:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
