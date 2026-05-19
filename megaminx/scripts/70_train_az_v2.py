"""m_az_v2: AZ training with off-path sibling augmentation.

Trains ResMLPGFlowNet (policy + value heads):
  - Policy CE on (states_p, actions_p): path states with known optimal actions.
  - Value MSE on (states_v, values_v): path states + 23 siblings each, labeled by
    teacher V (broad coverage to fix beam distribution shift).

Each batch samples both: a policy mini-batch (B_p path tuples) and a value mini-batch
(B_v from the full 1.87M set). Loss = α·CE(policy_logits, actions) + β·MSE(value).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/70_train_az_v2.py \\
        --dataset megaminx/data/az_dataset_v2.pt \\
        --warmstart-trunk megaminx/models/m_dd_v0/epoch_0049.pt \\
        --value-init-from-distance \\
        --output megaminx/models/m_az_v2 \\
        --epochs 100 --batch-size-policy 1024 --batch-size-value 4096
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
    trunk_sd = {k: v for k, v in sd.items()
                if k.startswith(("embedding.", "input_stack.", "res_blocks."))}
    model.load_state_dict(trunk_sd, strict=False)
    print(f"  warm-start: loaded {len(trunk_sd)} trunk tensors", flush=True)
    if init_value_from_distance and "head.weight" in sd:
        with torch.no_grad():
            model.value_head.weight.copy_(sd["head.weight"])
            model.value_head.bias.copy_(sd["head.bias"])
        print(f"  initialized value_head = V_distance_head", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--warmstart-trunk", type=Path, default=None)
    ap.add_argument("--value-init-from-distance", action="store_true")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size-policy", type=int, default=1024,
                    help="path-state batch size for policy CE + value MSE")
    ap.add_argument("--batch-size-value", type=int, default=4096,
                    help="value-only batch size (path + siblings)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="policy CE weight")
    ap.add_argument("--beta", type=float, default=0.5, help="value MSE weight")
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=70)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    n_gen = len(puzzle.move_names)

    print(f"loading dataset: {args.dataset}", flush=True)
    data = torch.load(args.dataset, map_location=args.device, weights_only=False)
    # Policy training: states_p, actions_p, values_p
    states_p = data["states_p"].to(args.device).long()
    actions_p = data["actions_p"].to(args.device).long()
    values_p = data["values_p"].to(args.device).float()
    # Value training: states_v, values_v (path + siblings)
    states_v = data["states_v"].to(args.device).long()
    values_v = data["values_v"].to(args.device).float()
    Np = states_p.size(0)
    Nv = states_v.size(0)
    print(f"  policy: {Np:,} tuples (paths only)", flush=True)
    print(f"  value:  {Nv:,} tuples (paths + siblings)", flush=True)

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

    gen_p = torch.Generator(device=args.device).manual_seed(args.seed)
    gen_v = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    n_batches_per_epoch = max(1, Np // args.batch_size_policy)

    print(f"epochs: {args.epochs}  Bp: {args.batch_size_policy}  Bv: {args.batch_size_value}  "
          f"alpha={args.alpha}  beta={args.beta}", flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        total_p_loss = 0.0
        total_v_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        perm_p = torch.randperm(Np, generator=gen_p, device=args.device)
        for b in range(n_batches_per_epoch):
            idx_p = perm_p[b * args.batch_size_policy : (b + 1) * args.batch_size_policy]
            bs_p = states_p[idx_p]
            ba_p = actions_p[idx_p]
            bv_p = values_p[idx_p]

            # Sample value batch independently (random sample with replacement is fine for big set)
            idx_v = torch.randint(0, Nv, (args.batch_size_value,),
                                  generator=gen_v, device=args.device)
            bs_v = states_v[idx_v]
            bv_v = values_v[idx_v]

            if autocast_ctx is not None:
                with autocast_ctx:
                    # Policy + value on policy batch
                    logits_p, value_p = model(bs_p)
                    p_loss = F.cross_entropy(logits_p, ba_p)
                    v_loss_p = F.mse_loss(value_p.float(), bv_p)
                    # Value only on value batch
                    _, value_v = model(bs_v)
                    v_loss_v = F.mse_loss(value_v.float(), bv_v)
                    # Total value loss is weighted by Bv (since the value batch
                    # dominates the value training)
                    v_loss = (v_loss_p * args.batch_size_policy + v_loss_v * args.batch_size_value) \
                              / (args.batch_size_policy + args.batch_size_value)
                    loss = args.alpha * p_loss + args.beta * v_loss
            else:
                logits_p, value_p = model(bs_p)
                p_loss = F.cross_entropy(logits_p, ba_p)
                v_loss_p = F.mse_loss(value_p.float(), bv_p)
                _, value_v = model(bs_v)
                v_loss_v = F.mse_loss(value_v.float(), bv_v)
                v_loss = (v_loss_p * args.batch_size_policy + v_loss_v * args.batch_size_value) \
                          / (args.batch_size_policy + args.batch_size_value)
                loss = args.alpha * p_loss + args.beta * v_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_p_loss += float(p_loss.item())
            total_v_loss += float(v_loss.item())
            with torch.no_grad():
                total_acc += float((logits_p.argmax(dim=1) == ba_p).float().mean().item())
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
