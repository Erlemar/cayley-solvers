"""GraphTransformer Q-shortlister training (Phase 5 of the GT plan).

Distills a teacher V-model into a 24-output GraphTransformerV (Q-head).
Target: Q(s, a) = V_teacher(apply(s, a)) for all 24 actions.

Loss:
    L = mse_weight * MSE(Q_pred, Q_target)
      + kl_weight  * KL( softmax(-Q_target / T) || softmax(-Q_pred / T) )

Rotation augmentation (optional): per row, with prob `--rotation-aug-prob`,
replace s with R*s*R_inv before computing teacher targets. V is rotation-
invariant, so the teacher's Q-values transfer trivially without action-relabel.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/75_train_gt_q.py \\
        --teacher megaminx/models/m_az_v4_v_only_e99.pt \\
        --out-dir megaminx/models/m_gt_q_v0 \\
        --n-epochs 200

Acceptance (Phase 5 gate):
    recall@alpha=2 ≥ 99 % vs teacher's argmin-2 children (run
    `09_eval_q_recall.py` on a sample) before deployment in 03_solve.py
    with --qshort-student.
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
from cayley.search import load_model_checkpoint
from megaminx.graph_transformer import GraphTransformerV
from megaminx.puzzle import Megaminx


@torch.no_grad()
def teacher_q_targets(
    teacher,
    states: torch.Tensor,
    generators: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """target_Q[s, a] = V_teacher(apply(s, a)).  Returns (B, n_gen) float32."""
    B, S = states.shape
    n_gen = generators.shape[0]
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
                    help="Path to a V-model checkpoint (any class supported by "
                         "load_model_checkpoint; output_dim must be 1).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=200)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=750)
    # Student architecture
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    # Loss weights
    ap.add_argument("--mse-weight", type=float, default=0.5)
    ap.add_argument("--kl-weight", type=float, default=0.5)
    ap.add_argument("--kl-temperature", type=float, default=1.0,
                    help="softmax temperature for KL ranking term.")
    ap.add_argument("--checkpoint-every-epochs", type=int, default=10)
    ap.add_argument("--n-back", type=int, default=1)
    # Rotation augmentation
    ap.add_argument("--rotations-path", type=Path, default=None,
                    help="path to (N, state_size) int8 rotations array (e.g. "
                         "data/rotations.npy). Required if --rotation-aug-prob > 0.")
    ap.add_argument("--rotation-aug-prob", type=float, default=0.0,
                    help="probability per row of applying a random rotation before "
                         "computing teacher targets.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    # ---- Teacher ----
    teacher = load_model_checkpoint(args.teacher, device=device, dtype=torch.float32)
    for p in teacher.parameters():
        p.requires_grad = False
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"teacher: {n_teacher:,} params from {args.teacher}", flush=True)
    if getattr(teacher, "output_dim", 1) != 1:
        raise ValueError("--teacher must be a V-model (output_dim=1)")

    # ---- Student ----
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)
    graph_features = torch.load(PROJECT / "data" / "graph_features.pt",
                                map_location="cpu", weights_only=False)
    student = GraphTransformerV(
        graph_features,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        output_dim=n_gen,  # Q head: 24 outputs
    ).to(device)
    n_student = student.num_parameters()
    print(f"student: {n_student:,} params  d_model={args.d_model} layers={args.n_layers} "
          f"heads={args.n_heads} ffn={args.ffn_dim}", flush=True)

    # ---- Generators ----
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(device)  # (n_gen, state_size) long

    # ---- Optional rotation augmentation ----
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
        print(f"rotation aug: {rot_arr.shape[0]} rotations, p={args.rotation_aug_prob:.2f}",
              flush=True)

    optim = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.0,
        fused=device == "cuda",
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else None)
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    print(f"loss = {args.mse_weight}*MSE + {args.kl_weight}*KL(T={args.kl_temperature})", flush=True)
    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  "
          f"batch: {args.batch_size}", flush=True)

    for epoch in range(args.n_epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        states, _depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch, device=device, n_back=args.n_back,
        )
        N = states.shape[0]
        student.train()
        total_mse = 0.0
        total_kl = 0.0
        n_batches = 0
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            batch_states = states[idx]
            # Optional rotation augmentation: replace state with R*s*R_inv.
            if rotations_dev is not None:
                B_rw, S_ = batch_states.shape
                rot_idx = torch.randint(0, rotations_dev.size(0), (B_rw,),
                                        generator=rot_gen, device=device)
                R = rotations_dev[rot_idx].to(torch.long)
                R_inv = rotations_inv_dev[rot_idx]
                bs_long = batch_states.to(torch.long)
                # rotated[i] = R[bs[R_inv[i]]]
                step1 = torch.gather(bs_long, 1, R_inv)
                rotated = torch.gather(R, 1, step1)
                aug_mask = (torch.rand(B_rw, generator=rot_gen, device=device)
                            < args.rotation_aug_prob).unsqueeze(1)
                batch_states = torch.where(
                    aug_mask, rotated.to(batch_states.dtype), batch_states
                )

            with torch.no_grad():
                target_q = teacher_q_targets(
                    teacher, batch_states, generators, chunk_size=8192
                )  # (bs, n_gen) float32

            if autocast_ctx is not None:
                with autocast_ctx:
                    pred_q = student(batch_states)
                    mse = F.mse_loss(pred_q.float(), target_q)
                    log_p = F.log_softmax(-pred_q.float() / args.kl_temperature, dim=-1)
                    log_q = F.log_softmax(-target_q / args.kl_temperature, dim=-1)
                    kl = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
                    loss = args.mse_weight * mse + args.kl_weight * kl
            else:
                pred_q = student(batch_states)
                mse = F.mse_loss(pred_q.float(), target_q)
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
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | MSE {avg_mse:.4f} | KL {avg_kl:.4f} | "
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time() - t0:.1f}s",
                  flush=True)

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.n_epochs - 1:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            sd = student.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": sd,
                "mse_loss": avg_mse,
                "kl_loss": avg_kl,
                "model_config": student.get_model_config(),
                "teacher_path": str(args.teacher),
                "kl_temperature": args.kl_temperature,
                "mse_weight": args.mse_weight,
                "kl_weight": args.kl_weight,
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
