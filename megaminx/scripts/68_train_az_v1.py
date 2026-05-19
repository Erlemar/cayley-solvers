"""m_az_v1: AlphaZero-style training on real solver paths.

Trains ResMLPGFlowNet (policy + value heads) on (state, action_taken, remaining)
tuples from a previous submission CSV. Warm-starts trunk from m_dd_v0 50ep, and
optionally inits value_head from the V model's distance head (no sign flip:
value_target = remaining_distance, same orientation as V).

Loss:
    α * CE(policy_logits, action_taken)
  + β * MSE(value, remaining_distance)

Usage:
    .venv/Scripts/python.exe megaminx/scripts/68_train_az_v1.py \\
        --dataset megaminx/data/az_dataset_78029.pt \\
        --warmstart-trunk megaminx/models/m_dd_v0/epoch_0049.pt \\
        --value-init-from-distance \\
        --output megaminx/models/m_az_v1 \\
        --epochs 200 --batch-size 4096 \\
        --lr 5e-4 --alpha 1.0 --beta 0.5
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

from cayley.gflow_model import ResMLPGFlowNet
from megaminx.puzzle import Megaminx


def warmstart_from_v_model(model: ResMLPGFlowNet, ckpt_path: Path,
                            init_value_from_distance: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

    trunk_sd = {}
    for k, v in sd.items():
        if k.startswith(("embedding.", "input_stack.", "res_blocks.")):
            trunk_sd[k] = v
    model.load_state_dict(trunk_sd, strict=False)
    print(f"  warm-start: loaded {len(trunk_sd)} trunk tensors", flush=True)

    if init_value_from_distance:
        if "head.weight" in sd and "head.bias" in sd:
            with torch.no_grad():
                # Value head trained to match remaining_distance directly (NO sign flip)
                model.value_head.weight.copy_(sd["head.weight"])
                model.value_head.bias.copy_(sd["head.bias"])
            print(f"  initialized value_head = V_distance_head (same sign for distance)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--warmstart-trunk", type=Path, default=None)
    ap.add_argument("--value-init-from-distance", action="store_true")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="weight on policy CE loss")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="weight on value MSE loss")
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=68)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    n_gen = len(puzzle.move_names)

    print(f"loading dataset: {args.dataset}", flush=True)
    data = torch.load(args.dataset, map_location=args.device, weights_only=False)
    states_t = data["states"].to(args.device).long()
    actions_t = data["actions"].to(args.device).long()
    values_t = data["values"].to(args.device).float()
    N = states_t.size(0)
    print(f"  {N:,} tuples, value range [{values_t.min():.0f}, {values_t.max():.0f}]",
          flush=True)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPGFlowNet(
        state_size=state_size, num_classes=state_size,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16, n_actions=n_gen,
    ).to(args.device)
    print(f"model params: {model.num_parameters():,}", flush=True)

    if args.warmstart_trunk:
        warmstart_from_v_model(model, args.warmstart_trunk,
                                init_value_from_distance=args.value_init_from_distance)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=args.device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else None

    batch_gen = torch.Generator(device=args.device).manual_seed(args.seed)
    print(f"epochs: {args.epochs}  batch: {args.batch_size}  alpha={args.alpha}  beta={args.beta}",
          flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        perm = torch.randperm(N, generator=batch_gen, device=args.device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs = states_t[idx]
            ba = actions_t[idx]
            bv = values_t[idx]

            if autocast_ctx is not None:
                with autocast_ctx:
                    logits, value = model(bs)
                    p_loss = F.cross_entropy(logits, ba)
                    v_loss = F.mse_loss(value.float(), bv)
                    loss = args.alpha * p_loss + args.beta * v_loss
            else:
                logits, value = model(bs)
                p_loss = F.cross_entropy(logits, ba)
                v_loss = F.mse_loss(value.float(), bv)
                loss = args.alpha * p_loss + args.beta * v_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_p_loss += float(p_loss.item())
            total_v_loss += float(v_loss.item())
            with torch.no_grad():
                pred_a = logits.argmax(dim=1)
                total_acc += float((pred_a == ba).float().mean().item())
            n_batches += 1
        sched.step()

        avg_p = total_p_loss / max(1, n_batches)
        avg_v = total_v_loss / max(1, n_batches)
        avg_acc = total_acc / max(1, n_batches)
        wall = time.time() - t0
        print(f"epoch {epoch:4d} | p_loss {avg_p:.4f} | v_loss {avg_v:.4f} | "
              f"top-1 acc {avg_acc:.4f} | lr {sched.get_last_lr()[0]:.2e} | {wall:.1f}s",
              flush=True)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            sd = model.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": sd,
                "model_config": {
                    "state_size": state_size, "num_classes": state_size,
                    "hidden_dims": list(hidden_dims),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "embedding", "embed_dim": 16, "n_actions": n_gen,
                },
                "p_loss": avg_p, "v_loss": avg_v, "top1_acc": avg_acc,
            }, args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
