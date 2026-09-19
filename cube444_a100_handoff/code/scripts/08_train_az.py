"""Stage 4 -- AZ dual-head training for the 4x4x4 cube.

    python3 cube444/scripts/08_train_az.py \
        --warmstart-trunk cube444/models/c_bells/epoch_0299.pt \
        --policy-dataset cube444/data/az_dataset_54754.pt \
        --bfs-anchors cube444/data/bfs_anchors.pt \
        --output cube444/models/c_az \
        --hidden-dims 2048,512 --num-res-blocks 2 \
        --epochs 100 --checkpoint-every 5

Port of megaminx/scripts/71_train_az_v3.py, trimmed to the recipe that actually
worked (AZ v4). The experimental knobs in the megaminx version -- orbit loss,
preserve loss, path-value loss, path-rank loss -- all default to zero and are
dropped here. The orbit loss in particular is NOT portable: it conjugates states
by a rotation permutation, which is the picture-cube form and ignores the color
relabel a color cube needs.

Value head: exactly the Stage-3 Bellman recipe (random walks + BFS anchors +
V0/d=1 anchors + a target net refreshed every N epochs).
Policy head: cross-entropy on solver-path actions.
Shared trunk, warm-started from the Stage-3 V.

WHAT THIS IS FOR
----------------
The policy head is NOT expected to be usable at inference. Rule 15 (a qshort
distilled from one V regresses another V), allneighbor_qhead_rejected, and the
V-only-wins-on-TPU finding all point the same way. Stage 4's value is multi-task
regularization of the trunk to get a BETTER V HEAD.

AND IT ONLY WORKS EARLY. On megaminx, m_az_v3 at 100 epochs was WORSE than the
pure V it started from (bench 943 vs 871); only the early-stopped m_az_v4 ep24
beat it (strat-5 51/51, mean 87.5 vs 89.4). Over-training makes the policy
memorize and wrecks V calibration. Hence --checkpoint-every 5, and run
04_eval_v.py on several early checkpoints rather than trusting the final one.

Rule 13: pass --rw-batch-size / --policy-batch-size EXPLICITLY. AZ v4's first
launch took script defaults (4096/512 instead of 8192/1024), quadrupling the
gradient steps per epoch and destroying V calibration.
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

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.gflow_model import ResMLPGFlowNet
from cube444.puzzle import Cube444


def warmstart_trunk(model: ResMLPGFlowNet, ckpt_path: Path) -> None:
    """Copy trunk (+ value head if shape-compatible) from a Stage-3 V checkpoint.

    Uses strict=False, so a SHAPE MISMATCH IS SILENT. Rule 18: a trunk-shape
    change turns a "warm start" into a from-scratch run with only the embedding
    carried over. The counts printed below are the check -- read them.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    trunk_sd = {k: v for k, v in sd.items()
                if k.startswith(("embedding.", "input_stack.", "res_blocks."))}
    msd = model.state_dict()
    compatible = {k: v for k, v in trunk_sd.items()
                  if k in msd and tuple(msd[k].shape) == tuple(v.shape)}
    skipped = len(trunk_sd) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    print(f"  warm-start: loaded {len(compatible)}/{len(trunk_sd)} trunk tensors "
          f"({skipped} shape-mismatched SKIPPED)", flush=True)
    if skipped:
        print("  *** WARNING: shape mismatch -- this is NOT a warm start (Rule 18) ***",
              flush=True)
    for src, dst in (("head", model.value_head), ("value_head", model.value_head)):
        if f"{src}.weight" in sd and tuple(dst.weight.shape) == tuple(sd[f"{src}.weight"].shape):
            with torch.no_grad():
                dst.weight.copy_(sd[f"{src}.weight"])
                dst.bias.copy_(sd[f"{src}.bias"])
            print(f"  initialized value_head from '{src}'", flush=True)
            break


def value_only_forward(model, states: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    outs = []
    for i in range(0, states.size(0), chunk):
        h = model.trunk(states[i : i + chunk])
        outs.append(model.value_head(h).squeeze(-1))
    return torch.cat(outs, dim=0)


def bellman_targets(target_model, states, walk_depths, generators, solved_state,
                    chunk: int = 8192) -> torch.Tensor:
    """target = clip(1 + min_a V_target(child_a), 0, walk_depth)."""
    B, S = states.shape
    n_gen = generators.shape[0]
    children = states.unsqueeze(1).expand(B, n_gen, S).clone()
    children = torch.gather(children, 2, generators.unsqueeze(0).expand(B, n_gen, S))
    flat = children.reshape(B * n_gen, S)
    is_solved = (flat == solved_state).all(dim=1)
    target_model.eval()
    cv = value_only_forward(target_model, flat, chunk).float()
    cv = torch.where(is_solved, torch.zeros_like(cv), cv).view(B, n_gen)
    target = 1.0 + cv.min(dim=1).values
    return torch.clamp(torch.minimum(target, walk_depths), min=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--warmstart-trunk", required=True, type=Path)
    ap.add_argument("--policy-dataset", required=True, type=Path)
    ap.add_argument("--bfs-anchors", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--rw-batch-size", type=int, default=8192)
    ap.add_argument("--policy-batch-size", type=int, default=1024)
    ap.add_argument("--anchor-v0", type=int, default=32)
    ap.add_argument("--anchor-d1", type=int, default=4)
    ap.add_argument("--bfs-fraction", type=float, default=0.10)
    ap.add_argument("--k-max", type=int, default=60)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=1.0, help="policy CE weight")
    ap.add_argument("--beta", type=float, default=1.0, help="value MSE weight")
    ap.add_argument("--target-update-every-epochs", type=int, default=10)
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=444)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puz.solved_state)
    n_gen = len(puz.move_names)
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))

    # num_classes=6, NOT state_size. The megaminx original hardcodes
    # num_classes=state_size, which would silently build a 96x96 one-hot here.
    model = ResMLPGFlowNet(
        state_size=state_size, num_classes=6,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="onehot", n_actions=n_gen,
    ).to(args.device)
    print(f"model params: {model.num_parameters():,}  "
          f"(state_size={state_size} num_classes=6 n_actions={n_gen})", flush=True)
    warmstart_trunk(model, args.warmstart_trunk)

    target_model = copy.deepcopy(model).to(args.device).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    gens = GeneratorTable.from_puzzle(puz)
    generators = torch.from_numpy(gens.perms).to(args.device)
    solved = torch.tensor(puz.solved_state, dtype=torch.int64, device=args.device)
    anchor_v0 = solved.unsqueeze(0)
    anchor_d1 = torch.gather(anchor_v0.expand(n_gen, state_size), 1, generators)

    pdata = torch.load(args.policy_dataset, map_location=args.device, weights_only=False)
    states_p = pdata["states"].to(args.device).long()
    actions_p = pdata["actions"].to(args.device).long()
    Np = states_p.size(0)
    print(f"policy dataset: {Np:,} (state, action) pairs", flush=True)

    bfs_states = bfs_dists = None
    bfs_per_batch = 0
    if args.bfs_anchors:
        b = torch.load(args.bfs_anchors, map_location="cpu", weights_only=False)
        bfs_states, bfs_dists = b["states"], b["distances"]
        bfs_per_batch = max(1, int(round(args.rw_batch_size * args.bfs_fraction)))
        print(f"BFS anchors: {bfs_states.size(0):,} states, "
              f"mixin {bfs_per_batch}/{args.rw_batch_size}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=args.device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    use_amp = str(args.device).startswith("cuda")
    rng_p = torch.Generator(device=args.device).manual_seed(args.seed)
    rng_bfs = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"epochs: {args.epochs}  rw_batch: {args.rw_batch_size}  "
          f"policy_batch: {args.policy_batch_size}  alpha={args.alpha}  beta={args.beta}  "
          f"k_max={args.k_max}  target_update_every={args.target_update_every_epochs}",
          flush=True)

    rw_per_batch = args.rw_batch_size - bfs_per_batch - args.anchor_v0 - n_gen * args.anchor_d1
    if rw_per_batch <= 0:
        raise ValueError(f"rw_batch_size too small: {rw_per_batch} rows left for random walks")

    for epoch in range(args.epochs):
        t0 = time.time()
        if epoch > 0 and epoch % args.target_update_every_epochs == 0:
            target_model.load_state_dict(model.state_dict())
            target_model.eval()

        n_walks = max(1, args.samples_per_epoch // args.k_max)
        rw_states, rw_depths = generate_walks_torch(
            puz, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch * 1000, device=args.device, n_back=args.n_back)
        rw_depths_f = rw_depths.to(torch.float32)
        N_rw = rw_states.size(0)

        ep_bfs_s = ep_bfs_d = None
        if bfs_states is not None:
            need = bfs_per_batch * (N_rw // rw_per_batch + 1)
            need = min(need, bfs_states.size(0))
            perm = torch.randperm(bfs_states.size(0), generator=rng_bfs)[:need]
            ep_bfs_s = bfs_states[perm].to(args.device).long()
            ep_bfs_d = bfs_dists[perm].to(args.device).float()

        n_batches = max(1, N_rw // rw_per_batch)
        cur = 0
        tp = tv = ta = 0.0
        for b in range(n_batches):
            lo, hi = b * rw_per_batch, min((b + 1) * rw_per_batch, N_rw)
            bs_rw, bd_rw = rw_states[lo:hi], rw_depths_f[lo:hi]
            with torch.no_grad():
                tgt_rw = bellman_targets(target_model, bs_rw, bd_rw, generators, solved)

            parts_s, parts_t = [bs_rw], [tgt_rw]
            if args.anchor_v0 > 0:
                parts_s.append(anchor_v0.expand(args.anchor_v0, -1))
                parts_t.append(torch.zeros(args.anchor_v0, device=args.device))
            if args.anchor_d1 > 0:
                parts_s.append(anchor_d1.repeat(args.anchor_d1, 1))
                parts_t.append(torch.ones(n_gen * args.anchor_d1, device=args.device))
            if ep_bfs_s is not None and cur + bfs_per_batch <= ep_bfs_s.size(0):
                parts_s.append(ep_bfs_s[cur : cur + bfs_per_batch])
                parts_t.append(ep_bfs_d[cur : cur + bfs_per_batch])
                cur += bfs_per_batch
            bs_v = torch.cat(parts_s, 0)
            tgt_v = torch.cat(parts_t, 0)

            idx = torch.randint(0, Np, (args.policy_batch_size,),
                                generator=rng_p, device=args.device)
            bs_p, ba_p = states_p[idx], actions_p[idx]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred_v = model.value_head(model.trunk(bs_v)).squeeze(-1)
                v_loss = F.mse_loss(pred_v.float(), tgt_v)
                logits = model.policy_head(model.trunk(bs_p))
                p_loss = F.cross_entropy(logits, ba_p)
                loss = args.alpha * p_loss + args.beta * v_loss

            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            with torch.no_grad():
                tp += p_loss.item()
                tv += v_loss.item()
                ta += (logits.argmax(-1) == ba_p).float().mean().item()

        sched.step()
        n = max(1, n_batches)
        print(f"epoch {epoch:4d} | p_loss {tp/n:.4f} | v_loss {tv/n:.4f} | "
              f"top-1 acc {ta/n:.4f} | lr {sched.get_last_lr()[0]:.2e} | "
              f"{time.time()-t0:.1f}s", flush=True)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "model_config": {
                    "model_class": "ResMLPGFlowNet",
                    "state_size": state_size, "num_classes": 6,
                    "hidden_dims": list(hidden_dims),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "onehot", "n_actions": n_gen,
                },
            }, args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
