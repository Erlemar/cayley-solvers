"""m23: train a Q-shortlister from m05 teacher.

The Q-head is a 24-output ResMLPDistance trained to mimic teacher V(apply(s, a))
for each action a. It is intended to be used as a SHORTLISTER in beam search:
the student picks top-α·B candidates per parent in one forward, then the teacher
m05 reranks just those α·B candidates. Teacher cost drops from 24·B → α·B per
step. At α=4 that's a 6× model speedup.

Why this differs from m06 (failed Q-distill):
- m06 was trained from m07 (sharper teacher = sharper labels). We use m05.
- m06 used MSE only. We add a softmax-KL ranking term so the student's RANKING
  is preserved even if absolute values drift.
- Bigger student head: hidden_dims=(2048, 1024), 3 res blocks (~10M params).

After training, run `09_eval_recall.py` (built separately) BEFORE deploying to
beam search. Acceptance gate: recall(student top-αB ⊇ teacher top-B) ≥ 0.99
at α=4.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/09_train_q_shortlister.py \
        --teacher megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --out-dir megaminx/models/m23_q_shortlister
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

from cayley.data import GeneratorTable, generate_walks_torch
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


@torch.no_grad()
def teacher_q_targets(
    teacher: ResMLPDistance,
    states: torch.Tensor,
    generators: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute target Q-values: for each state s and action a, target_Q[s, a] = V_teacher(apply(s, a)).

    Returns (B, n_gen) float32.
    """
    B, S = states.shape
    n_gen = generators.shape[0]
    # Children: (B, n_gen, S)
    children = torch.gather(
        states.unsqueeze(1).expand(B, n_gen, S),
        2,
        generators.unsqueeze(0).expand(B, n_gen, S),
    )
    children_flat = children.reshape(B * n_gen, S)
    out = torch.empty(B * n_gen, dtype=torch.float32, device=states.device)
    teacher.eval()
    for i in range(0, B * n_gen, chunk_size):
        out[i : i + chunk_size] = teacher(children_flat[i : i + chunk_size]).flatten().to(torch.float32)
    return out.view(B, n_gen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True,
                    help="m05 (or other V-model) checkpoint to distill from")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=500)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000,
                    help="random walks; smaller than V-training because each sample now "
                         "needs 24 teacher forwards.")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=230)
    # Architecture
    ap.add_argument("--hidden-dims", type=str, default="2048,1024",
                    help="comma-separated hidden dims for student MLP")
    ap.add_argument("--num-res-blocks", type=int, default=3)
    # Loss weights
    ap.add_argument("--mse-weight", type=float, default=0.5)
    ap.add_argument("--kl-weight", type=float, default=0.5)
    ap.add_argument("--kl-temperature", type=float, default=1.0,
                    help="softmax temperature for KL ranking term. Lower = peakier.")
    ap.add_argument("--checkpoint-every-epochs", type=int, default=50)
    ap.add_argument("--n-back", type=int, default=1)
    # Rotation augmentation (sym-aware m23_v2). Per batch, with prob `--rotation-aug-prob`,
    # rotate the state via R*s*R_inv before computing teacher Q-values. Targets are
    # recomputed on the rotated state, so the student learns the teacher's view of the
    # rotated puzzle. Pairs with `--sym-ensemble` at inference: each rotated input the
    # solver feeds will have a sharper shortlist than vanilla m23 (which never saw
    # rotated states during training).
    ap.add_argument("--rotations-path", type=Path, default=None,
                    help="path to (N, state_size) int8 rotations array (sym v3).")
    ap.add_argument("--rotation-aug-prob", type=float, default=0.0,
                    help="probability per row of applying a random rotation.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # Load teacher (V-model, output_dim=1).
    teacher_ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher_sd = teacher_ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in teacher_sd):
        teacher_sd = {k.removeprefix("_orig_mod."): v for k, v in teacher_sd.items()}
    teacher_cfg = teacher_ckpt.get("model_config", {})
    teacher = ResMLPDistance(
        state_size=teacher_cfg.get("state_size", 120),
        num_classes=teacher_cfg.get("num_classes", 120),
        hidden_dims=tuple(teacher_cfg.get("hidden_dims", (2048, 512))),
        num_res_blocks=teacher_cfg.get("num_res_blocks", 2),
        encoding=teacher_cfg.get("encoding", "embedding"),
        embed_dim=teacher_cfg.get("embed_dim", 16),
    ).to(device).eval()
    teacher.load_state_dict(teacher_sd)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"teacher: {teacher_params:,} params from {args.teacher}")

    # Build student: same arch but with output_dim=n_gen (24 for Megaminx).
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    student = ResMLPDistance(
        state_size=120, num_classes=120,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16,
        output_dim=n_gen,  # KEY: 24 outputs for the 24 actions
    ).to(device)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"student: {student_params:,} params, hidden={hidden_dims}, "
          f"num_res_blocks={args.num_res_blocks}, output_dim={n_gen}")

    # Optimizer
    optim = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.0,
        fused=device == "cuda",
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)

    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(device)  # (n_gen, state_size)

    # Symmetry rotations (optional sym-aware m23_v2 mode).
    rotations_dev = None
    rotations_inv_dev = None
    rot_gen = torch.Generator(device=device).manual_seed(args.seed + 54321)
    if args.rotations_path is not None and args.rotation_aug_prob > 0:
        import numpy as np
        rot_arr = np.load(args.rotations_path)
        rotations_dev = torch.from_numpy(rot_arr).to(device)
        rotations_inv_dev = torch.from_numpy(
            np.argsort(rot_arr, axis=1).astype(np.int64)
        ).to(device)
        print(f"rotation augmentation: {rot_arr.shape[0]} rotations, "
              f"prob={args.rotation_aug_prob:.2f}")
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"loss = {args.mse_weight}*MSE + {args.kl_weight}*KL(softmax_T={args.kl_temperature})")
    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  batch: {args.batch_size}")

    for epoch in range(args.n_epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        states, _depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch, device=device, n_back=args.n_back,
        )
        N = states.shape[0]

        student.train()
        total_mse, total_kl, n_batches = 0.0, 0.0, 0
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            batch_states = states[idx]
            # Optional rotation augmentation: replace state with R*s*R_inv before
            # computing teacher Q. Targets are recomputed on the rotated state, so
            # the student learns the teacher's view of the rotated puzzle.
            if rotations_dev is not None:
                B_rw, S_ = batch_states.shape
                rot_idx = torch.randint(0, rotations_dev.size(0), (B_rw,),
                                        generator=rot_gen, device=device)
                R = rotations_dev[rot_idx].to(torch.long)
                R_inv = rotations_inv_dev[rot_idx]
                bs_long = batch_states.to(torch.long)
                step1 = torch.gather(bs_long, 1, R_inv)
                rotated = torch.gather(R, 1, step1)
                aug_mask = (torch.rand(B_rw, generator=rot_gen, device=device)
                            < args.rotation_aug_prob).unsqueeze(1)
                batch_states = torch.where(
                    aug_mask, rotated.to(batch_states.dtype), batch_states
                )
            # Teacher: target Q[s, a] = V(apply(s, a))
            with torch.no_grad():
                target_q = teacher_q_targets(
                    teacher, batch_states, generators, chunk_size=8192
                )  # (bs, n_gen)

            if autocast_ctx is not None:
                with autocast_ctx:
                    pred_q = student(batch_states)  # (bs, n_gen)
                    mse = F.mse_loss(pred_q, target_q)
                    # KL on softmax(-Q/T): lower Q = better, so use -Q for distribution
                    log_p = F.log_softmax(-pred_q.float() / args.kl_temperature, dim=-1)
                    log_q = F.log_softmax(-target_q / args.kl_temperature, dim=-1)
                    # KL(target || pred): target_distrib * (log_target - log_pred)
                    kl = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
                    loss = args.mse_weight * mse + args.kl_weight * kl
            else:
                pred_q = student(batch_states)
                mse = F.mse_loss(pred_q, target_q)
                log_p = F.log_softmax(-pred_q.float() / args.kl_temperature, dim=-1)
                log_q = F.log_softmax(-target_q / args.kl_temperature, dim=-1)
                kl = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
                loss = args.mse_weight * mse + args.kl_weight * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_mse += float(mse.item())
            total_kl += float(kl.item())
            n_batches += 1
        sched.step()

        avg_mse = total_mse / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | MSE {avg_mse:.4f} | KL {avg_kl:.4f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time() - t0:.1f}s",
                  flush=True)

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.n_epochs - 1:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "mse_loss": avg_mse, "kl_loss": avg_kl,
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": list(hidden_dims),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "embedding", "embed_dim": 16,
                    "output_dim": n_gen,
                },
                "teacher_path": str(args.teacher),
                "kl_temperature": args.kl_temperature,
                "mse_weight": args.mse_weight,
                "kl_weight": args.kl_weight,
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
