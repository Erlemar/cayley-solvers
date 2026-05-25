"""Distill GT V into a ResMLP V model. Phase 4 option (A).

Given a frozen GT teacher (V-only, output_dim=1), train a ResMLPDistance
student to match the teacher's V values on a mix of states:
  - random walks
  - frontier states (beam-frontier replay)
  - solver-trace states (real-solve trajectories)

Loss: MSE(V_student, V_teacher) + optional small auxiliary on BFS-d6 anchors
to keep the student well-calibrated near the solved state.

Why this exists: GT v0 V calibration matches baselines AND has good high-depth
discrimination but fails to solve at beam=65536. The hypothesis being tested
is that the V LANDSCAPE the GT learned is good but the GT INFERENCE PATH is
the bug (eager bf16 SDPA + Python chunking + KhoruzhiiSolver interaction).
If a distilled ResMLP solves where the GT didn't, this hypothesis is true.

Usage:
    python3 megaminx/scripts/78_distill_v_from_gt.py \\
        --teacher megaminx/models/m_gt_v0_bellman_solvertrace/epoch_0019.pt \\
        --out-dir megaminx/models/m_gt_v0_distill_resmlp \\
        --n-epochs 100
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

from cayley.data import generate_walks_torch
from cayley.model import ResMLPDistance
from cayley.search import load_model_checkpoint
from megaminx.puzzle import Megaminx


@torch.no_grad()
def teacher_v(teacher: torch.nn.Module, states: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    teacher.eval()
    outs = []
    for i in range(0, states.size(0), chunk_size):
        outs.append(teacher(states[i:i + chunk_size]).flatten().to(torch.float32))
    return torch.cat(outs, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True,
                    help="GT V-model checkpoint (output_dim=1)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=100)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=780)
    # Student arch (matches m_dd_v0 baseline shape for fast inference)
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    # Optional mixin paths (relative to PROJECT, e.g. "data/frontier_states.pt")
    ap.add_argument("--frontier-path", type=str, default="data/frontier_states.pt")
    ap.add_argument("--frontier-fraction", type=float, default=0.25)
    ap.add_argument("--solver-trace-path", type=str, default="data/solver_trace_train.pt")
    ap.add_argument("--solver-trace-fraction", type=float, default=0.20)
    ap.add_argument("--bfs-d6-path", type=str, default="data/bfs_d6_train.pt")
    ap.add_argument("--bfs-d6-fraction", type=float, default=0.10)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")

    # Teacher (frozen, bf16 for speed)
    teacher = load_model_checkpoint(args.teacher, device=device, dtype=torch.bfloat16)
    for p in teacher.parameters():
        p.requires_grad = False
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"teacher: {n_teacher:,} params from {args.teacher.name}", flush=True)

    # Student (matches m_dd_v0 shape)
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    student = ResMLPDistance(
        state_size=120, num_classes=120,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16, output_dim=1,
    ).to(device)
    n_student = sum(p.numel() for p in student.parameters())
    print(f"student: {n_student:,} params, hidden={hidden_dims}", flush=True)

    # Auxiliary data
    frontier_states = None
    if args.frontier_fraction > 0:
        fp = PROJECT / args.frontier_path
        d = torch.load(fp, map_location="cpu", weights_only=False)
        frontier_states = d["states"] if isinstance(d, dict) else d
        print(f"  frontier: {frontier_states.shape[0]:,} states", flush=True)

    st_states, st_dists = None, None
    if args.solver_trace_fraction > 0:
        sp = PROJECT / args.solver_trace_path
        d = torch.load(sp, map_location="cpu", weights_only=False)
        st_states, st_dists = d["states"], d["distances"]
        print(f"  solver-trace: {st_states.shape[0]:,} (state, dist) pairs", flush=True)

    bfs_states, bfs_dists = None, None
    if args.bfs_d6_fraction > 0:
        bp = PROJECT / args.bfs_d6_path
        d = torch.load(bp, map_location="cpu", weights_only=False)
        bfs_states, bfs_dists = d["states"], d["distances"]
        print(f"  bfs-d6: {bfs_states.shape[0]:,} (state, dist) pairs", flush=True)

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else None)
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)
    cpu_gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  "
          f"batch: {args.batch_size}  lr: {args.lr}", flush=True)
    for epoch in range(args.n_epochs):
        t0 = time.time()

        # 1. Generate random walks (largest portion of training data)
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        rw_states, _ = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch, device=device, n_back=args.n_back,
        )
        N_rw = rw_states.size(0)

        # 2. Mix in frontier / solver-trace / bfs-d6 each batch.
        frontier_per_batch = int(args.batch_size * args.frontier_fraction) if frontier_states is not None else 0
        st_per_batch = int(args.batch_size * args.solver_trace_fraction) if st_states is not None else 0
        bfs_per_batch = int(args.batch_size * args.bfs_d6_fraction) if bfs_states is not None else 0
        rw_per_batch = args.batch_size - frontier_per_batch - st_per_batch - bfs_per_batch
        if rw_per_batch <= 0:
            raise ValueError("Mixin fractions sum >= 1; reduce them.")
        n_batches = N_rw // rw_per_batch

        student.train()
        total = 0.0
        for b in range(n_batches):
            rw_b = rw_states[b * rw_per_batch:(b + 1) * rw_per_batch]
            parts = [rw_b]
            if frontier_per_batch > 0:
                idx = torch.randint(0, frontier_states.size(0), (frontier_per_batch,),
                                    generator=cpu_gen)
                parts.append(frontier_states[idx].to(device).long())
            if st_per_batch > 0:
                idx = torch.randint(0, st_states.size(0), (st_per_batch,),
                                    generator=cpu_gen)
                parts.append(st_states[idx].to(device).long())
            if bfs_per_batch > 0:
                idx = torch.randint(0, bfs_states.size(0), (bfs_per_batch,),
                                    generator=cpu_gen)
                parts.append(bfs_states[idx].to(device).long())
            batch_states = torch.cat(parts, dim=0)

            with torch.no_grad():
                target_v = teacher_v(teacher, batch_states, chunk_size=4096)

            if autocast_ctx is not None:
                with autocast_ctx:
                    pred_v = student(batch_states)
                    loss = F.mse_loss(pred_v.float(), target_v)
            else:
                pred_v = student(batch_states)
                loss = F.mse_loss(pred_v.float(), target_v)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total += float(loss.item())
        sched.step()

        avg = total / max(1, n_batches)
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | MSE {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s",
                  flush=True)

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.n_epochs - 1:
            torch.save({
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "loss": avg,
                "model_config": student.get_model_config() if hasattr(student, "get_model_config") else {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": list(hidden_dims),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "embedding", "embed_dim": 16, "output_dim": 1,
                },
                "teacher_path": str(args.teacher),
            }, args.out_dir / f"epoch_{epoch:04d}.pt")
            print(f"  saved epoch_{epoch:04d}.pt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
