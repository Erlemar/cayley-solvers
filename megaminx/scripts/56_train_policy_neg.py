"""m_pi_v3_neg: Policy with explicit negative examples.

Existing m_pi_v2 trains policy with weighted CE on (state, optimal_action) pairs from
solved paths only — purely POSITIVE signal.

This trainer adds NEGATIVE signal: at each training state, compute V on all 24
children using a frozen V model (m_curr_v3), then KL-distill against
softmax(-V_children/T). Combined with the existing CE loss, the policy learns:
  - Predict the optimal action (positive CE)
  - PUSH DOWN probability on high-V children (negative KL)

Loss = alpha * CE(state, optimal_action) + beta * KL(softmax(student), softmax(-V_children/T))

Usage:
    .venv/Scripts/python.exe megaminx/scripts/56_train_policy_neg.py \\
        --warmstart megaminx/models/m_pi_v2/epoch_0199.pt \\
        --v-model megaminx/models/m_curr_v3/epoch_0499.pt \\
        --policy-data megaminx/data/policy_train.pt \\
        --out-dir megaminx/models/m_pi_v3_neg \\
        --n-epochs 200 --alpha 1.0 --beta 0.5 --temperature 2.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def load_v_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 512])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        output_dim=mc.get("output_dim", 1),
    )
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path, required=True,
                    help="warmstart from m_pi_v2 (output_dim=24 policy model)")
    ap.add_argument("--v-model", type=Path, required=True,
                    help="frozen V model (e.g., m_curr_v3) for child V evaluation")
    ap.add_argument("--policy-data", type=Path, required=True,
                    help=".pt file from 43_build_policy_dataset.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    # Combined-loss knobs
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weight on CE loss (predict optimal action)")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="weight on KL loss (negative distillation)")
    ap.add_argument("--temperature", type=float, default=2.0,
                    help="temperature for soft target softmax(-V/T)")
    ap.add_argument("--use-suffix-weights", action="store_true",
                    help="weight CE by 1/(1+suffix_len) like m_pi_v2")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    # Load policy dataset
    data = torch.load(args.policy_data, map_location=device, weights_only=False)
    states = data["states"].to(device)        # (N, 120) int8
    actions = data["actions"].to(device).long()  # (N,) int8 -> long for CE
    weights = data.get("weights")
    if weights is not None:
        weights = weights.to(device).float()
    N = states.size(0)
    print(f"data: {N:,} (state, action) pairs", flush=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)
    state_size = len(puzzle.solved_state)
    all_moves = torch.zeros((n_gen, state_size), dtype=torch.int64, device=device)
    for i, name in enumerate(puzzle.move_names):
        all_moves[i] = torch.tensor(puzzle.generators[name], dtype=torch.int64)

    # Frozen V model for distillation target
    v_model = load_v_model(args.v_model, device)
    print(f"V model loaded: {sum(p.numel() for p in v_model.parameters()):,} params", flush=True)

    # Policy model — load from warmstart (output_dim=24)
    p_ckpt = torch.load(args.warmstart, map_location=device, weights_only=False)
    p_sd = p_ckpt.get("state_dict", p_ckpt)
    p_sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in p_sd.items()}
    pmc = p_ckpt.get("model_config", {})
    policy = ResMLPDistance(
        state_size=pmc.get("state_size", 120),
        num_classes=pmc.get("num_classes", 120),
        hidden_dims=tuple(pmc.get("hidden_dims", [2048, 512])),
        num_res_blocks=pmc.get("num_res_blocks", 2),
        encoding=pmc.get("encoding", "embedding"),
        embed_dim=pmc.get("embed_dim", 16),
        output_dim=pmc.get("output_dim", 24),
    )
    policy.load_state_dict(p_sd, strict=False)
    policy = policy.to(device).train()
    print(f"policy: {sum(p.numel() for p in policy.parameters()):,} params, "
          f"warmstart from {args.warmstart}", flush=True)

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"epochs: {args.n_epochs}  batch: {args.batch_size}  alpha={args.alpha}  "
          f"beta={args.beta}  T={args.temperature}", flush=True)

    best_acc = 0.0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        policy.train()
        total_ce = 0.0
        total_kl = 0.0
        total_acc = 0.0
        n_batches = 0
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs = states[idx]                       # (bs, 120) int8
            bl = actions[idx]                       # (bs,) long
            bw = weights[idx] if (weights is not None and args.use_suffix_weights) else None

            # Compute V on all 24 children — using frozen V model
            with torch.no_grad():
                # children: (bs, 24, 120) — apply each move to each state
                children = bs[:, all_moves]         # (bs, 24, 120) int8
                children_flat = children.reshape(-1, state_size)  # (bs*24, 120)
                if autocast_ctx is not None:
                    with autocast_ctx:
                        v_children_flat = v_model(children_flat.long()).flatten()
                else:
                    v_children_flat = v_model(children_flat.long()).flatten()
                v_children = v_children_flat.float().reshape(-1, n_gen)  # (bs, 24)
                # Soft target: low V → high probability
                soft_target = F.softmax(-v_children / args.temperature, dim=-1)

            if autocast_ctx is not None:
                with autocast_ctx:
                    logits = policy(bs)             # (bs, n_gen)
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    # CE on optimal action
                    ce_per = F.cross_entropy(logits, bl, reduction="none")
                    if bw is not None:
                        ce_loss = (ce_per * bw).sum() / bw.sum().clamp(min=1e-6)
                    else:
                        ce_loss = ce_per.mean()
                    # KL(soft_target || student_softmax)
                    kl_loss = -(soft_target * log_probs).sum(dim=-1).mean()
                    loss = args.alpha * ce_loss + args.beta * kl_loss
            else:
                logits = policy(bs)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                ce_per = F.cross_entropy(logits, bl, reduction="none")
                if bw is not None:
                    ce_loss = (ce_per * bw).sum() / bw.sum().clamp(min=1e-6)
                else:
                    ce_loss = ce_per.mean()
                kl_loss = -(soft_target * log_probs).sum(dim=-1).mean()
                loss = args.alpha * ce_loss + args.beta * kl_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_ce += float(ce_loss.item())
            total_kl += float(kl_loss.item())
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                total_acc += float((pred == bl).float().mean().item())
            n_batches += 1
        sched.step()

        avg_ce = total_ce / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)
        print(f"epoch {epoch:4d} | ce {avg_ce:.4f} | kl {avg_kl:.4f} | "
              f"acc top-1 {avg_acc:.4f} | lr {float(sched.get_last_lr()[0]):.2e} | "
              f"{time.time()-t0:.1f}s", flush=True)

        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save({
                "epoch": epoch,
                "state_dict": policy.state_dict(),
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": [2048, 512], "num_res_blocks": 2,
                    "encoding": "embedding", "embed_dim": 16, "output_dim": 24,
                },
                "alpha": args.alpha, "beta": args.beta, "temperature": args.temperature,
            }, args.out_dir / "best.pt")

        if (epoch + 1) % args.checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "state_dict": policy.state_dict(),
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": [2048, 512], "num_res_blocks": 2,
                    "encoding": "embedding", "embed_dim": 16, "output_dim": 24,
                },
            }, args.out_dir / f"epoch_{epoch:04d}.pt")

    print(f"\nfinal: best top-1 acc {best_acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
