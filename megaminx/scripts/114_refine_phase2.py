"""Calibration refine for the Phase-2 V: add V0/depth-1 anchors so V(solved)~0.

The pretrain-only m_phase2_v0 never saw the solved state (generate_walks emits
depths 1..k_max), so V(solved)=4.47 while V(1-move)=1.0 -- solved is NOT the
V-minimum, which breaks the wide TPU SPMD beam's endgame (it descends away from
solved and ranks it out). Fix: warm-start from m_phase2_v0 and MSE-refine on
12-gen walks PLUS anchors {solved->0, each 1-move->1}. Mid-range (walks) is already
good, so this just calibrates the low end. Lowest-risk vs a full Bellman refine.

    .venv/Scripts/python.exe megaminx/scripts/114_refine_phase2.py \
        --in-checkpoint megaminx/models/m_phase2_v0/epoch_1999.pt \
        --output megaminx/models/m_phase2_v1 --epochs 120
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import generate_walks_torch
from megaminx.puzzle import Megaminx
from megaminx.two_phase import build_phase2_puzzle, load_two_phase_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--samples-per-epoch", type=int, default=600_000)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--k-max", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5.0e-4)
    ap.add_argument("--n-v0", type=int, default=6144, help="solved-state anchors per epoch")
    ap.add_argument("--n-d1", type=int, default=512, help="copies of each 1-move anchor per epoch")
    ap.add_argument("--checkpoint-every", type=int, default=40)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    full = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    puzzle = build_phase2_puzzle(full)
    dev = args.device

    model = load_two_phase_model(args.in_checkpoint, dev)  # plain ResMLP, _orig_mod stripped
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, fused=(dev == "cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # Anchors: solved (label 0) + each of the 12 restricted 1-move states (label 1).
    solved = torch.tensor([list(puzzle.solved_state)], dtype=torch.int64, device=dev)  # (1,S)
    d1 = torch.stack([
        torch.tensor(list(puzzle.apply_move(puzzle.solved_state, nm)), dtype=torch.int64, device=dev)
        for nm in puzzle.move_names
    ])  # (12, S)
    a_states = torch.cat([solved.repeat(args.n_v0, 1), d1.repeat_interleave(args.n_d1, dim=0)], dim=0)
    a_depths = torch.cat([
        torch.zeros(args.n_v0, device=dev),
        torch.ones(d1.shape[0] * args.n_d1, device=dev),
    ])
    n_anchor = a_states.shape[0]

    def v_at(states_list):
        model.eval()
        with torch.no_grad():
            t = torch.tensor(states_list, dtype=torch.int64, device=dev)
            out = model(t).float().cpu().tolist()
        model.train()
        return out

    print(f"refine phase2: warm {args.in_checkpoint.name}, {args.epochs} ep, "
          f"anchors/epoch V0={args.n_v0} d1=12x{args.n_d1}={n_anchor - args.n_v0}", flush=True)
    sv = list(puzzle.solved_state)
    print(f"  pre-refine V(solved)={v_at([sv])[0]:.3f}", flush=True)

    n_walks = max(1, args.samples_per_epoch // args.k_max)
    gen = torch.Generator(device=dev).manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        w_states, w_depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max, seed=epoch, device=dev, n_back=1)
        states = torch.cat([w_states, a_states], dim=0)
        depths = torch.cat([w_depths.float(), a_depths], dim=0)
        idx = torch.randperm(states.shape[0], generator=gen, device=dev)
        total = 0.0
        nb = 0
        for i in range(0, states.shape[0], args.batch_size):
            b = idx[i:i + args.batch_size]
            pred = model(states[b])
            loss = ((pred - depths[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        sched.step()
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} | loss {total / nb:.4f} | V(solved)={v_at([sv])[0]:.3f} | "
                  f"{time.time() - t0:.1f}s", flush=True)
        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "model_config": model.get_model_config()},
                       args.output / f"epoch_{epoch:04d}.pt")

    v1 = v_at([list(puzzle.apply_move(puzzle.solved_state, nm)) for nm in puzzle.move_names])
    print(f"DONE. V(solved)={v_at([sv])[0]:.3f}  V(1-move) min={min(v1):.3f} max={max(v1):.3f}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
