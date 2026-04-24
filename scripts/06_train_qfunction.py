"""Train a Q-function neighbor-scorer via distillation from an existing V-model.

For each batch of states, the V-teacher scores all 18 neighbors. The Q-student is
trained to predict those scores in a single forward pass (output_dim=18), which
eliminates the per-neighbor inference loop inside beam search.

Reference: Vlad Kuznetsov report (CayleyPy chat 2026-04-04): on megaminx, beam 2^21
with Q-head runs in the wall-time of beam 2^18 with V-head, yielding avg path 83 vs 91.

    python scripts/06_train_qfunction.py \\
        --teacher models/e6/epoch_0499.pt \\
        --output models/qe6 \\
        --config configs/q_distill.yaml
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.model import ResMLPDistance
from cayley.puzzle import PictureCube
from cayley.search import load_model_checkpoint


@dataclass
class QDistillConfig:
    n_epochs: int = 500
    samples_per_epoch: int = 500_000   # halved vs V-training — 18× more forward targets per sample
    batch_size: int = 8000
    k_max: int = 26
    lr: float = 1e-3
    weight_decay: float = 0.0
    checkpoint_every_epochs: int = 50
    n_back: int = 1
    amp: bool = True
    fused_optimizer: bool = True
    seed: int = 0


def _compute_neighbors(states: torch.Tensor, generators: torch.Tensor) -> torch.Tensor:
    """(B, S), (n_gen, S) → (B, n_gen, S)."""
    B, S = states.shape
    n_gen = generators.shape[0]
    return torch.gather(
        states.unsqueeze(1).expand(B, n_gen, S),
        2,
        generators.unsqueeze(0).expand(B, n_gen, S),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path, help="V-model checkpoint")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--hidden-dims", type=int, nargs="+", default=[1024, 256])
    ap.add_argument("--num-res-blocks", type=int, default=1)
    ap.add_argument("--embed-dim", type=int, default=16)
    ap.add_argument("--encoding", default="embedding")
    ap.add_argument("--n-epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k-max", type=int, default=26)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    gens = GeneratorTable.from_puzzle(puzzle)
    n_gen = gens.perms.shape[0]
    generators = torch.from_numpy(gens.perms).to(args.device)

    # Teacher: frozen V-model.
    teacher = load_model_checkpoint(args.teacher, device=args.device, dtype=torch.float32).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"teacher: {teacher.num_parameters():,} params (encoding={teacher.encoding})")

    # Student: new Q-model, same trunk, output_dim=n_gen (=18).
    student = ResMLPDistance(
        state_size=72, num_classes=72,
        hidden_dims=tuple(args.hidden_dims),
        num_res_blocks=args.num_res_blocks,
        encoding=args.encoding, embed_dim=args.embed_dim,
        output_dim=n_gen,
    ).to(args.device)
    print(f"student: {student.num_parameters():,} params, output_dim={n_gen}")

    opt_kwargs = dict(lr=args.lr)
    if args.device == "cuda":
        opt_kwargs["fused"] = True
    opt = torch.optim.Adam(student.parameters(), **opt_kwargs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.n_epochs)

    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else None

    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.n_epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        states, _ = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch, device=args.device, n_back=1,
        )

        # Precompute teacher Q-targets in chunks to stay in memory.
        chunk = args.batch_size
        total_loss = 0.0; n_batches = 0
        perm = torch.randperm(states.shape[0], device=args.device)
        for i in range(0, states.shape[0], chunk):
            batch = states[perm[i : i + chunk]]
            B = batch.shape[0]
            # (B, 18, 72) children
            children = _compute_neighbors(batch, generators).reshape(B * n_gen, -1)
            with torch.no_grad():
                v_children = teacher(children).view(B, n_gen).float()
            if autocast_ctx is not None:
                with autocast_ctx:
                    pred = student(batch).float()
                    loss = ((pred - v_children) ** 2).mean()
            else:
                pred = student(batch).float()
                loss = ((pred - v_children) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        sched.step()
        avg = total_loss / max(n_batches, 1)
        print(f"ep {epoch:4d} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s")

        if (epoch + 1) % 50 == 0 or epoch == args.n_epochs - 1:
            base = getattr(student, "_orig_mod", student)
            model_cfg = dict(
                state_size=base.state_size, num_classes=base.num_classes,
                hidden_dims=args.hidden_dims, num_res_blocks=args.num_res_blocks,
                encoding=args.encoding, embed_dim=args.embed_dim,
                output_dim=n_gen,
            )
            torch.save({
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "loss": avg,
                "model_config": model_cfg,
                "teacher_path": str(args.teacher),
            }, args.output / f"epoch_{epoch:04d}.pt")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
