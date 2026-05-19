"""m24: Macro-Q shortlister.

Idea 1 from megaminx/novel_ideas.md.

Same shape as m23 (Q-shortlister) but with the action set extended to include
curated macros from `data/curated_macros_phase2.pkl`. Output dim = n_gen +
n_macros (24 + 45 = 69 by default).

Target Q[s, a] = V_m05(apply(s, perm[a])). For primitive actions, perm[a] is
the generator's permutation; for macros, perm[a] is the macro's net 120-perm.
The beam applies a cost-aware adjustment at inference time (`score =
Q + (cost(a) - 1)`); training the Q-head on raw V values keeps the model
reusable under different cost rules.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/41_macro_qhead.py \
        --teacher megaminx/models/m05_bellman_warm/epoch_0499.pt \
        --macros megaminx/data/curated_macros_phase2.pkl \
        --out-dir megaminx/models/m24_macro_qshort \
        --n-epochs 500
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import generate_walks_torch
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def load_macros(path: Path) -> list[dict]:
    """Load curated_macros_phase2.pkl. Returns list of macro dicts with keys
    name, word_idxs, word_names, perm, hamming, source_len, best_len, ..."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    macros = d["macros"] if isinstance(d, dict) and "macros" in d else d
    assert isinstance(macros, list), f"unexpected macros format: {type(macros)}"
    return macros


@torch.no_grad()
def teacher_q_targets_expanded(
    teacher: ResMLPDistance,
    states: torch.Tensor,
    all_perms: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute target Q over an EXPANDED action set (primitives + macros).

    states: (B, S) int.
    all_perms: (n_actions, S) int64 — for action a, apply(s, a)[i] = s[all_perms[a, i]].
    Returns (B, n_actions) float32.
    """
    B, S = states.shape
    n_actions = all_perms.shape[0]
    children = torch.gather(
        states.unsqueeze(1).expand(B, n_actions, S),
        2,
        all_perms.unsqueeze(0).expand(B, n_actions, S),
    )
    children_flat = children.reshape(B * n_actions, S)
    out = torch.empty(B * n_actions, dtype=torch.float32, device=states.device)
    teacher.eval()
    for i in range(0, B * n_actions, chunk_size):
        out[i : i + chunk_size] = teacher(children_flat[i : i + chunk_size]).flatten().to(torch.float32)
    return out.view(B, n_actions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=Path, required=True,
                    help="m05 (or other V-model) checkpoint to distill from")
    ap.add_argument("--macros", type=Path,
                    default=PROJECT / "data" / "curated_macros_phase2.pkl",
                    help="curated macro library pickle (Phase 2 default)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=500)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=240)
    ap.add_argument("--hidden-dims", type=str, default="2048,1024")
    ap.add_argument("--num-res-blocks", type=int, default=3)
    ap.add_argument("--mse-weight", type=float, default=0.5)
    ap.add_argument("--kl-weight", type=float, default=0.5)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=50)
    ap.add_argument("--n-back", type=int, default=1)
    # Sym-aware augmentation (optional; matches m23_v2 recipe).
    ap.add_argument("--rotations-path", type=Path, default=None,
                    help="optional rotations.npy for R*s*R_inv augmentation")
    ap.add_argument("--rotation-aug-prob", type=float, default=0.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_gen = len(puzzle.move_names)

    # Load curated macros and build expanded action set.
    macros = load_macros(args.macros)
    n_macros = len(macros)
    n_actions = n_gen + n_macros
    print(f"loaded {n_macros} macros from {args.macros}")
    macro_costs = [int(len(m["word_names"])) if isinstance(m["word_names"], list)
                   else int(m["source_len"]) for m in macros]
    print(f"macro costs: min={min(macro_costs)} max={max(macro_costs)} "
          f"median={int(np.median(macro_costs))}")

    # Build (n_actions, S) perm tensor: gens first, then macros.
    gen_perms = np.stack([puzzle.generators[n] for n in puzzle.move_names], axis=0).astype(np.int64)
    macro_perms = np.stack([m["perm"] for m in macros], axis=0).astype(np.int64)
    all_perms_np = np.concatenate([gen_perms, macro_perms], axis=0)  # (n_actions, S)
    assert all_perms_np.shape == (n_actions, 120), all_perms_np.shape
    all_perms = torch.from_numpy(all_perms_np).to(device)

    # Teacher.
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
    print(f"teacher: {sum(p.numel() for p in teacher.parameters()):,} params")

    # Student.
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    student = ResMLPDistance(
        state_size=120, num_classes=120,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16,
        output_dim=n_actions,  # KEY: n_gen + n_macros
    ).to(device)
    print(f"student: {sum(p.numel() for p in student.parameters()):,} params, "
          f"hidden={hidden_dims}, output_dim={n_actions}")

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else None)
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    rotations_dev = rotations_inv_dev = None
    rot_gen = torch.Generator(device=device).manual_seed(args.seed + 54321)
    if args.rotations_path is not None and args.rotation_aug_prob > 0:
        rot_arr = np.load(args.rotations_path)
        rotations_dev = torch.from_numpy(rot_arr).to(device)
        rotations_inv_dev = torch.from_numpy(
            np.argsort(rot_arr, axis=1).astype(np.int64)
        ).to(device)
        print(f"rotation aug: {rot_arr.shape[0]} rots, prob={args.rotation_aug_prob:.2f}")

    print(f"loss = {args.mse_weight}*MSE + {args.kl_weight}*KL(softmax_T={args.kl_temperature})")
    print(f"epochs: {args.n_epochs}  samples/epoch: {args.samples_per_epoch:,}  "
          f"batch: {args.batch_size}")

    for epoch in range(args.n_epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        states, _ = generate_walks_torch(
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

            with torch.no_grad():
                target_q = teacher_q_targets_expanded(
                    teacher, batch_states, all_perms, chunk_size=8192
                )  # (bs, n_actions)

            if autocast_ctx is not None:
                with autocast_ctx:
                    pred_q = student(batch_states)
                    mse = F.mse_loss(pred_q, target_q)
                    log_p = F.log_softmax(-pred_q.float() / args.kl_temperature, dim=-1)
                    log_q = F.log_softmax(-target_q / args.kl_temperature, dim=-1)
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
                    "output_dim": n_actions,
                },
                "teacher_path": str(args.teacher),
                "macros_path": str(args.macros),
                "n_gen": n_gen,
                "n_macros": n_macros,
                "kl_temperature": args.kl_temperature,
                "mse_weight": args.mse_weight,
                "kl_weight": args.kl_weight,
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
