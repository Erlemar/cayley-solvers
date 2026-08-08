"""Cell 5: a TRUE trajectory-balance GFlowNet for the Professor Tetraminx.

This is the only cell of the architecture matrix that changes the OBJECTIVE rather
than the architecture. The other cells all optimise sparse-Q; here the model is a
`GFNPolicyNet` (backward-policy logits + forward-policy logits + stop) trained by
prefix trajectory balance, per arXiv:2603.01786 as ported in
`cayley/gfn_pathfinding.py`.

Why the `gflownet` architecture cell was DROPPED and this one kept: a
`ResMLPGFlowNet` trained with sparse-Q is bit-identical to the plain ResMLP cell
(same trunk, `policy_head` is the same Linear as `head`; measured identical loss to
4 decimals). The net shape contributes nothing -- what makes a GFlowNet a GFlowNet
is the TB objective, so that is what this trains.

RISK, on the record before spending the GPU. `gfn_pathfinding_ported_gated`:
  * on-policy prefix-TB MODE-COLLAPSES without eps_explore (both policies), so
    `--eps-explore` defaults non-zero here;
  * no CONSTANT flow-reg lambda spans both the flow climb and the equilibrium --
    the recipe is two-stage: tiny lambda until TB balances, then RESUME with an
    equilibrium-targeted lambda. A single big lambda from scratch froze training
    via a reg~1e27 outlier-flow blowup;
  * learnable logZ at 10x lr is the better recipe at large lnZ (verdict reversed
    from the earlier fixed-true-logZ runs);
  * full megaminx needed TPU budget and only reached diameter ~41 at laptop scale.
Tetraminx is diameter ~28-30 over 1.5e32 states, so a from-scratch TB model that
solves nothing is a plausible outcome. That IS the experiment's answer if so; it
is not a bug to debug indefinitely.

    python3 tetraminx/scripts/53_train_gfn_tb.py --output tetraminx/models/mx_gfn \
        --epochs 1500 --stage1-epochs 300
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.gfn_pathfinding import (
    GFNPolicyNet, build_env, count_params, regularized_tb_loss,
    sample_forward_trajectories,
)
from tetraminx.models import GFlowNetTBQ
from tetraminx.puzzle import Tetraminx


def build_tetraminx_env(data_dir: Path, nmax: int, device: str):
    puzzle = Tetraminx.load(data_dir / "puzzle_info.json")
    names = list(puzzle.move_names)
    gens = [list(puzzle.generators[n]) for n in names]
    inv_idx = [names.index(puzzle.inverse_name(n)) for n in names]
    a = len(names)
    # Prefix-TB over fixed-length nmax trajectories: Z is the number of
    # non-backtracking words of that length, so ln Z ~ ln A + (nmax-1) ln (A-1).
    true_log_z = math.log(a) + (nmax - 1) * math.log(a - 1)
    env = build_env(gens, inv_idx, list(puzzle.solved_state),
                    num_classes=len(puzzle.solved_state), true_log_z=true_log_z)
    return puzzle, env.to(device), true_log_z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--steps-per-epoch", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--nmax", type=int, default=32, help="prefix-TB trajectory length")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--num-res-blocks", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--logz-lr-mult", type=float, default=10.0,
                    help="learnable logZ at 10x lr -- the reversed-verdict recipe")
    ap.add_argument("--eps-explore", type=float, default=0.05,
                    help="on-policy prefix-TB mode-collapses at 0")
    ap.add_argument("--stage1-epochs", type=int, default=300,
                    help="epochs at --reg-stage1 before switching to --reg-stage2")
    ap.add_argument("--reg-stage1", type=float, default=0.0,
                    help="tiny/zero lambda while TB balances")
    ap.add_argument("--reg-stage2", type=float, default=1e-8,
                    help="equilibrium-targeted lambda; a big lambda from scratch froze training")
    ap.add_argument("--reg-mode", default="logflow", choices=("flow", "logflow"))
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    dev = args.device
    puzzle, env, true_log_z = build_tetraminx_env(args.data_dir, args.nmax, dev)
    A, S = env.n_actions, env.state_size

    model = GFNPolicyNet(state_size=S, num_classes=len(puzzle.solved_state), n_actions=A,
                         hidden=args.hidden, num_res_blocks=args.num_res_blocks,
                         encoding="embedding", embed_dim=16).to(dev)
    log_z = torch.nn.Parameter(torch.tensor(float(true_log_z), device=dev))
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": args.lr},
        {"params": [log_z], "lr": args.lr * args.logz_lr_mult},
    ])

    start_epoch = 1
    if args.resume is not None and args.resume.exists():
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        with torch.no_grad():
            log_z.copy_(torch.tensor(float(ck["log_z"]), device=dev))
        opt.load_state_dict(ck["optimizer"])
        start_epoch = int(ck["epoch"]) + 1
        print(f"  resumed from {args.resume} at epoch {start_epoch}", flush=True)

    print("=" * 72, flush=True)
    print(f"GFN trajectory balance | device {dev} | actions {A} | state {S}", flush=True)
    print(f"  params {count_params(model):,} | nmax {args.nmax} | init lnZ {true_log_z:.2f}", flush=True)
    print(f"  eps_explore {args.eps_explore} | reg {args.reg_stage1} -> {args.reg_stage2} "
          f"at epoch {args.stage1_epochs} (mode {args.reg_mode})", flush=True)

    log_path = args.output / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("epoch,loss,tb,reg,residual_rms,log_flow_d1,log_flow_last,log_z,secs\n",
                            encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        reg = args.reg_stage1 if epoch <= args.stage1_epochs else args.reg_stage2
        t0 = time.time()
        acc = {k: 0.0 for k in ("loss", "tb", "reg", "residual_rms", "log_flow_d1", "log_flow_last")}
        model.train()
        for _ in range(args.steps_per_epoch):
            states, actions = sample_forward_trajectories(
                model, env, args.batch_size, args.nmax, eps_explore=args.eps_explore)
            loss, met = regularized_tb_loss(model, env, states, actions, reg,
                                            reg_mode=args.reg_mode, log_z=log_z)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc["loss"] += float(loss.detach())
            for k in ("tb", "reg", "residual_rms", "log_flow_d1", "log_flow_last"):
                acc[k] += met[k]
        for k in acc:
            acc[k] /= args.steps_per_epoch
        secs = time.time() - t0
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{epoch},{acc['loss']:.5f},{acc['tb']:.5f},{acc['reg']:.3e},"
                     f"{acc['residual_rms']:.5f},{acc['log_flow_d1']:.3f},"
                     f"{acc['log_flow_last']:.3f},{float(log_z):.3f},{secs:.1f}\n")
        print(f"epoch {epoch:5d} | loss {acc['loss']:10.4f} tb {acc['tb']:10.4f} "
              f"reg {acc['reg']:.2e} | rms {acc['residual_rms']:7.3f} "
              f"logF(d1) {acc['log_flow_d1']:7.2f} logF(last) {acc['log_flow_last']:7.2f} "
              f"lnZ {float(log_z):7.2f} | {secs:.1f}s", flush=True)

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            # Save BOTH the raw policy and the beam-ready Q adapter, so the eval and
            # solve paths need no special casing for this cell.
            adapter = GFlowNetTBQ(model, env.s0_preimages.cpu(), n_actions=A)
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                "log_z": float(log_z), "metrics": acc,
                "model_config": {"arch": "gfn_tb", "n_actions": A, "az_head": False,
                                 "state_size": S, "num_classes": len(puzzle.solved_state),
                                 "hidden": args.hidden, "num_res_blocks": args.num_res_blocks,
                                 "nmax": args.nmax},
                "adapter_state_dict": adapter.state_dict(),
            }, args.output / f"epoch_{epoch:04d}.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
