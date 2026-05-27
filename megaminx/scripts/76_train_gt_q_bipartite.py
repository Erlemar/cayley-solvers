"""Train the Bipartite Slot-Sticker Graph Transformer as a Q-shortlister (doc 3.2).

Distills a teacher V-model into the bipartite GT's 24 action nodes:
    Q_target(s, a) = V_teacher(apply(s, a))
    loss = mse_weight * MSE(Q, Q_target)
         + kl_weight  * KL( softmax(-Q_target/T) || softmax(-Q/T) )

This is the action-centric upgrade of 75_train_gt_q.py: instead of reading all 24
Q-values from one pooled vector, each Q(a) comes from a dedicated action node that
attended to exactly the slots action a permutes. Same loss and teacher as 75 so the
two are directly comparable on recall.

Crucially, this trainer EARLY-STOPS ON DEPTH-STRATIFIED RECALL, not on MSE/KL loss.
The GraphTransformer-as-V failure (gt_v_no_saturation memory / CLAUDE rule 23) was a
value-landscape problem invisible to training loss; the analogous Q failure is recall
that sags at deep parents. So every --val-every-epochs we measure recall@alpha per
depth bucket and keep the checkpoint with the best WORST-bucket recall.

Rotation augmentation is V-style (in-frame target recomputation, no action relabel):
the teacher's Q-values for a rotated state are computed on that rotated state, so the
student simply learns the teacher's view of the rotated puzzle (matches m23_v2/v3).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/76_train_gt_q_bipartite.py \
        --teacher megaminx/models/m_az_v4_v_only_e99.pt \
        --out-dir megaminx/models/m_gt_q_bip_v0 \
        --n-epochs 150 \
        --rotations-path megaminx/data/rotations.npy --rotation-aug-prob 0.25

Acceptance (Stage 1 gate, then Stage 2 before deploy):
    recall@alpha=2 >= 0.99 vs teacher top-B, FLAT across depth buckets, and beating
    the flat GT-Q (m_gt_q_v0) / ResMLP-Q (m23_v3_az_v4_sym) at equal or lower alpha.
    Then distill into a ResMLP-Q and run strat-51 production-recipe (CLAUDE rule 21).
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
from megaminx.graph_transformer_bipartite import BipartiteGraphTransformerQ
from megaminx.puzzle import Megaminx


@torch.no_grad()
def teacher_q_targets(teacher, states, generators, chunk_size=8192):
    """target_Q[s, a] = V_teacher(apply(s, a)).  Returns (B, n_gen) float32."""
    B, S = states.shape
    n_gen = generators.shape[0]
    children = torch.gather(
        states.unsqueeze(1).expand(B, n_gen, S), 2,
        generators.unsqueeze(0).expand(B, n_gen, S),
    ).reshape(B * n_gen, S)
    out = torch.empty(B * n_gen, dtype=torch.float32, device=states.device)
    teacher.eval()
    for i in range(0, B * n_gen, chunk_size):
        out[i : i + chunk_size] = teacher(children[i : i + chunk_size]).flatten().to(torch.float32)
    return out.view(B, n_gen)


@torch.inference_mode()
def recall_by_depth(teacher, student, puzzle, generators, n_gen, depths, halfwidth,
                    parents_per_bucket, alpha, k_max, seed, device, chunk=8192):
    """Returns (per_bucket: list[(depth, n, recall)], min_recall, pooled_recall) at one alpha."""
    if device == "cuda":
        torch.cuda.empty_cache()  # release training's cached blocks before eval
    n_walks = max(1, (parents_per_bucket * len(depths) * 4) // k_max)
    pool, dep = generate_walks_torch(puzzle, n_walks=n_walks, k_max=k_max,
                                     seed=seed, device=device, n_back=1)
    per_bucket = []
    pooled = []
    student.eval()
    for c in depths:
        idx = torch.nonzero((dep - c).abs() <= halfwidth, as_tuple=True)[0]
        if idx.numel() == 0:
            per_bucket.append((c, 0, float("nan")))
            continue
        idx = idx[torch.randperm(idx.numel(), device=device)[:parents_per_bucket]]
        parents = pool[idx]
        pooled.append(parents)
        per_bucket.append((c, parents.shape[0],
                           _recall_pool(teacher, student, parents, generators, n_gen, alpha, chunk, device)))
    recs = [r for _, n, r in per_bucket if n > 0]
    min_rec = min(recs) if recs else float("nan")
    pooled_rec = float("nan")
    if pooled:
        allp = torch.cat(pooled, dim=0)
        pooled_rec = _recall_pool(teacher, student, allp, generators, n_gen, alpha, chunk, device)
    return per_bucket, min_rec, pooled_rec


@torch.inference_mode()
def _recall_pool(teacher, student, parents, generators, n_gen, alpha, chunk, device):
    P, S = parents.shape
    B = P  # top-B == #parents (realistic beam ratio: B out of P*24)
    children = torch.gather(
        parents.unsqueeze(1).expand(P, n_gen, S), 2,
        generators.unsqueeze(0).expand(P, n_gen, S),
    ).reshape(P * n_gen, S)
    n_total = children.shape[0]
    tv = torch.empty(n_total, dtype=torch.float32, device=device)
    for i in range(0, n_total, chunk):
        tv[i : i + chunk] = teacher(children[i : i + chunk]).flatten().to(torch.float32)
    sq = torch.empty((P, n_gen), dtype=torch.float32, device=device)
    # Student in bf16 autocast: eval has no checkpointing, so the (B,H,T,T)
    # attn matrix is materialized in full; fp32 would double it and OOM.
    ac = (torch.autocast("cuda", dtype=torch.bfloat16)
          if device == "cuda" else torch.autocast("cpu", enabled=False))
    for i in range(0, P, chunk):
        with ac:
            out = student(parents[i : i + chunk])
        sq[i : i + chunk] = out.to(torch.float32)
    sq = sq.reshape(-1)
    b = min(B, n_total)
    _, t_top = torch.topk(tv, b, largest=False, sorted=False)
    k = min(int(round(alpha * b)), n_total)
    _, s_top = torch.topk(sq, k, largest=False, sorted=False)
    inter = len(set(t_top.cpu().tolist()) & set(s_top.cpu().tolist()))
    return inter / max(b, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=150)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=760)
    # Student architecture
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    # Loss
    ap.add_argument("--mse-weight", type=float, default=0.5)
    ap.add_argument("--kl-weight", type=float, default=0.5)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--n-back", type=int, default=1)
    # Rotation aug
    ap.add_argument("--rotations-path", type=Path, default=None)
    ap.add_argument("--rotation-aug-prob", type=float, default=0.0)
    # Validation / early stop on recall
    ap.add_argument("--val-every-epochs", type=int, default=10)
    ap.add_argument("--val-alpha", type=float, default=2.0)
    ap.add_argument("--val-depths", type=str, default="20,40,60,80")
    ap.add_argument("--val-halfwidth", type=int, default=5)
    ap.add_argument("--val-parents-per-bucket", type=int, default=1024)
    ap.add_argument("--val-seed", type=int, default=99)
    ap.add_argument("--early-stop-patience", type=int, default=4,
                    help="stop if worst-bucket recall@val-alpha doesn't improve for "
                         "this many consecutive validations (0 disables)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    print(f"device: {device}", flush=True)
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)

    teacher = load_model_checkpoint(args.teacher, device=device, dtype=torch.float32)
    for p in teacher.parameters():
        p.requires_grad = False
    if getattr(teacher, "output_dim", 1) != 1:
        raise ValueError("--teacher must be a V-model (output_dim=1)")
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()):,} params from {args.teacher}", flush=True)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)
    feats = torch.load(PROJECT / "data" / "bipartite_features.pt", map_location="cpu", weights_only=False)
    student = BipartiteGraphTransformerQ(
        feats, d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        ffn_dim=args.ffn_dim, dropout=args.dropout, output_dim=n_gen,
        inference_chunk_size=1024,  # eval/val FP32 attn is (B,H,T,T); keep B small
    ).to(device)
    print(f"student: {student.num_parameters():,} params  d_model={args.d_model} "
          f"layers={args.n_layers} heads={args.n_heads} ffn={args.ffn_dim} (bipartite, T={student.n_tokens})",
          flush=True)

    generators = torch.from_numpy(GeneratorTable.from_puzzle(puzzle).perms).to(device)

    rotations_dev = rotations_inv_dev = None
    rot_gen = torch.Generator(device=device).manual_seed(args.seed + 54321)
    if args.rotations_path is not None and args.rotation_aug_prob > 0:
        import numpy as np
        rot_arr = np.load(args.rotations_path)
        rotations_dev = torch.from_numpy(rot_arr).to(device)
        rotations_inv_dev = torch.from_numpy(np.argsort(rot_arr, axis=1).astype(np.int64)).to(device)
        print(f"rotation aug: {rot_arr.shape[0]} rotations, p={args.rotation_aug_prob:.2f}", flush=True)

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else None
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)
    val_depths = [int(x) for x in args.val_depths.split(",")]

    print(f"loss = {args.mse_weight}*MSE + {args.kl_weight}*KL(T={args.kl_temperature}); "
          f"epochs={args.n_epochs} samples/epoch={args.samples_per_epoch:,} batch={args.batch_size}", flush=True)
    print(f"validate recall@alpha={args.val_alpha} at depths {val_depths} every "
          f"{args.val_every_epochs} epochs; early-stop patience {args.early_stop_patience}", flush=True)

    best_min_recall = -1.0
    best_epoch = -1
    stale = 0
    for epoch in range(args.n_epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        states, _ = generate_walks_torch(puzzle, n_walks=n_walks, k_max=args.k_max,
                                         seed=args.seed + epoch, device=device, n_back=args.n_back)
        N = states.shape[0]
        student.train()
        total_mse = total_kl = 0.0
        n_batches = 0
        perm = torch.randperm(N, generator=batch_gen, device=device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs_states = states[idx]
            if rotations_dev is not None:
                Bn = bs_states.shape[0]
                ri = torch.randint(0, rotations_dev.size(0), (Bn,), generator=rot_gen, device=device)
                R = rotations_dev[ri].to(torch.long)
                R_inv = rotations_inv_dev[ri]
                step1 = torch.gather(bs_states.to(torch.long), 1, R_inv)
                rotated = torch.gather(R, 1, step1)
                m = (torch.rand(Bn, generator=rot_gen, device=device) < args.rotation_aug_prob).unsqueeze(1)
                bs_states = torch.where(m, rotated.to(bs_states.dtype), bs_states)

            with torch.no_grad():
                target_q = teacher_q_targets(teacher, bs_states, generators)
            if autocast_ctx is not None:
                with autocast_ctx:
                    pred_q = student(bs_states)
                    mse = F.mse_loss(pred_q.float(), target_q)
                    log_p = F.log_softmax(-pred_q.float() / args.kl_temperature, dim=-1)
                    log_q = F.log_softmax(-target_q / args.kl_temperature, dim=-1)
                    kl = F.kl_div(log_p, log_q, reduction="batchmean", log_target=True)
                    loss = args.mse_weight * mse + args.kl_weight * kl
            else:
                pred_q = student(bs_states)
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
                  f"lr {float(sched.get_last_lr()[0]):.2e} | {time.time()-t0:.1f}s", flush=True)

        is_val = (epoch + 1) % args.val_every_epochs == 0 or epoch == args.n_epochs - 1
        if is_val:
            per_bucket, min_rec, pooled_rec = recall_by_depth(
                teacher, student, puzzle, generators, n_gen, val_depths, args.val_halfwidth,
                args.val_parents_per_bucket, args.val_alpha, args.k_max, args.val_seed, device)
            bucket_str = " ".join(f"d{c}:{r:.3f}" for c, n, r in per_bucket)
            print(f"  [val e{epoch}] recall@a={args.val_alpha}: {bucket_str} | "
                  f"min {min_rec:.3f} pooled {pooled_rec:.3f}", flush=True)
            ckpt = {
                "epoch": epoch, "state_dict": student.state_dict(),
                "mse_loss": avg_mse, "kl_loss": avg_kl,
                "model_config": student.get_model_config(),
                "teacher_path": str(args.teacher),
                "val_alpha": args.val_alpha, "val_min_recall": min_rec, "val_pooled_recall": pooled_rec,
            }
            torch.save(ckpt, args.out_dir / f"epoch_{epoch:04d}.pt")
            if min_rec > best_min_recall:
                best_min_recall = min_rec
                best_epoch = epoch
                stale = 0
                torch.save(ckpt, args.out_dir / "best.pt")
                print(f"    new best worst-bucket recall {min_rec:.3f} -> saved best.pt", flush=True)
            else:
                stale += 1
                if args.early_stop_patience and stale >= args.early_stop_patience:
                    print(f"  early stop: worst-bucket recall stale for {stale} validations "
                          f"(best {best_min_recall:.3f} @ e{best_epoch})", flush=True)
                    break

    print(f"done. best worst-bucket recall@a={args.val_alpha}: {best_min_recall:.3f} @ epoch {best_epoch}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
