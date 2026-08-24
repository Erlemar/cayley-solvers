"""Train a geodesic-imitation policy head on the AZ v5 trunk.

Warm-starts the trunk from an AZ v5 checkpoint (the production V trunk by default),
attaches a fresh output_dim=24 policy head, and CE-trains on exact BFS-d6 geodesic
move labels (data/geodesic_labels_d6.pt from script 130). This is the
warm-start half of the GFN recipe (BFS-geodesic imitation) transplanted to our
AZ model -- HEAD-ONLY when --freeze-trunk, else a light warm-started fine-tune.

Saves a load_model_checkpoint-compatible checkpoint (model_config with
output_dim=24) so 09b_eval_q_recall_by_depth.py can score it as a --student.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/131_train_geodesic_pi.py \
        --init megaminx/models/m_az_v5_73614_orbit_v_only.pt \
        --freeze-trunk --iters 30000 --save megaminx/models/m_geo_pi_frozen.pt
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=str(PROJECT / "models" / "m_az_v5_73614_orbit_v_only.pt"),
                   help="AZ checkpoint to warm-start the trunk from")
    p.add_argument("--labels", default=str(PROJECT / "data" / "geodesic_labels_d6.pt"))
    p.add_argument("--freeze-trunk", action="store_true")
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--trunk-lr", type=float, default=1e-4, help="lr for trunk if not frozen")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    mcfg = dict(ck["model_config"])
    print(f"trunk config: {mcfg}", flush=True)
    mcfg_pi = {**mcfg, "output_dim": 24}
    model = ResMLPDistance(
        state_size=mcfg["state_size"], num_classes=mcfg["num_classes"],
        hidden_dims=tuple(mcfg["hidden_dims"]), num_res_blocks=mcfg["num_res_blocks"],
        encoding=mcfg["encoding"], embed_dim=mcfg["embed_dim"], output_dim=24,
    ).to(device)
    # load trunk only (drop the scalar V head; policy head stays random-init)
    trunk_sd = {k: v for k, v in ck["state_dict"].items()
                if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(trunk_sd, strict=False)
    print(f"warm-started {len(trunk_sd)} trunk tensors; fresh policy head "
          f"(missing head: {[m for m in missing]})", flush=True)

    if args.freeze_trunk:
        for name, prm in model.named_parameters():
            if not name.startswith("head"):
                prm.requires_grad_(False)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        print("trunk FROZEN; training head only", flush=True)
    else:
        opt = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if n.startswith("head")], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("head")], "lr": args.trunk_lr},
        ])
        print(f"trunk fine-tuned at lr {args.trunk_lr}", flush=True)

    d = torch.load(args.labels, map_location="cpu", weights_only=False)
    states = d["states"]        # (N,120) int8 CPU
    moves = d["moves"].long()   # (N,) CPU
    dists = d["distances"]      # (N,) int8 CPU
    n = states.shape[0]
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(args.val_frac * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"dataset {n} states: {len(train_idx)} train / {len(val_idx)} val", flush=True)

    # keep val on GPU for fast per-depth eval
    val_states = states[val_idx].to(device).long()
    val_moves = moves[val_idx].to(device)
    val_dists = dists[val_idx].to(device)

    @torch.inference_mode()
    def eval_by_depth():
        model.eval()
        lines = []
        for dd in range(1, int(val_dists.max()) + 1):
            m = val_dists == dd
            if not bool(m.any()):
                continue
            sub = val_states[m]
            preds = []
            for i in range(0, sub.shape[0], 32768):
                preds.append(model(sub[i:i+32768]).argmax(-1))
            pred = torch.cat(preds)
            acc = float((pred == val_moves[m]).float().mean())
            lines.append(f"d{dd}:{acc:.3f}")
        model.train()
        return " ".join(lines)

    model.train()
    t0 = time.time()
    tn = len(train_idx)
    for it in range(args.iters):
        bi = train_idx[torch.randint(0, tn, (args.batch,), generator=g)]
        sb = states[bi].to(device).long()
        mb = moves[bi].to(device)
        logits = model(sb)
        loss = F.cross_entropy(logits, mb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (it + 1) % args.log_every == 0:
            acc = float((logits.argmax(-1) == mb).float().mean())
            print(f"iter {it+1} ce={float(loss):.4f} top1={acc:.3f} | "
                  f"val {eval_by_depth()} ({time.time()-t0:.0f}s)", flush=True)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_config": mcfg_pi,
                "iter": args.iters, "trained_on": "geodesic_labels_d6"}, args.save)
    print(f"saved -> {args.save}", flush=True)
    print(f"FINAL val by depth: {eval_by_depth()}", flush=True)


if __name__ == "__main__":
    main()
