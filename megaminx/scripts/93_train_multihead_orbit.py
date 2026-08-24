"""Train a 2A/3B multi-head orbit-distilled V model.

The trunk is warm-started from an AZ dual-head checkpoint. Each value head learns
to imitate the same teacher under rotations from a different rotation subset.
Inference can then use head-voted beam selection: each head nominates candidates,
approximating a small symmetry ensemble inside one forward pass.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import generate_walks_torch
from cayley.gflow_model import ResMLPGFlowNet
from cayley.model import ResBlock
from megaminx.puzzle import Megaminx


class ResMLPMultiHeadOrbit(nn.Module):
    def __init__(
        self,
        state_size: int = 120,
        num_classes: int = 120,
        hidden_dims: tuple[int, ...] = (2048, 512),
        num_res_blocks: int = 2,
        encoding: str = "embedding",
        embed_dim: int = 16,
        n_heads: int = 4,
    ):
        super().__init__()
        if encoding != "embedding":
            raise NotImplementedError("only embedding encoding is supported")
        self.state_size = state_size
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.num_res_blocks = num_res_blocks
        self.encoding = encoding
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.output_dim = n_heads
        in_dim = state_size * embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(num_res_blocks)])
        self.value_heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(n_heads)])
        self.uncertainty_head = nn.Linear(prev, 1)

    def trunk(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.input_stack[0].weight.dtype
        h = self.embedding(x.long()).to(target_dtype).flatten(start_dim=-2)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.value_heads], dim=1)

    def forward_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        values = torch.cat([head(h) for head in self.value_heads], dim=1)
        uncertainty = F.softplus(self.uncertainty_head(h).squeeze(-1))
        return values, uncertainty

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_az_teacher(path: Path, device: str, dtype: torch.dtype) -> ResMLPGFlowNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()}
    mc = ck["model_config"]
    model = ResMLPGFlowNet(
        state_size=mc["state_size"],
        num_classes=mc["num_classes"],
        hidden_dims=tuple(mc["hidden_dims"]),
        num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        n_actions=mc.get("n_actions", 24),
    )
    model.load_state_dict(sd)
    return model.to(device=device, dtype=dtype).eval()


def init_from_teacher(student: ResMLPMultiHeadOrbit, teacher: ResMLPGFlowNet) -> None:
    with torch.no_grad():
        student.embedding.load_state_dict(teacher.embedding.state_dict())
        student.input_stack.load_state_dict(teacher.input_stack.state_dict())
        student.res_blocks.load_state_dict(teacher.res_blocks.state_dict())
        for head in student.value_heads:
            head.weight.copy_(teacher.value_head.weight)
            head.bias.copy_(teacher.value_head.bias)
        student.uncertainty_head.weight.zero_()
        student.uncertainty_head.bias.zero_()


def load_rotation_tables(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    rotations_np = np.load(path).astype(np.int64)
    rotations = torch.from_numpy(rotations_np).to(device=device, dtype=torch.long)
    inverse = torch.empty_like(rotations)
    idx = torch.arange(rotations.size(1), device=device, dtype=torch.long).expand_as(rotations)
    inverse.scatter_(1, rotations, idx)
    return rotations, inverse


def apply_state_rotations(states: torch.Tensor, rotations: torch.Tensor,
                          inverse_rotations: torch.Tensor,
                          rotation_idx: torch.Tensor) -> torch.Tensor:
    r = rotations[rotation_idx]
    r_inv = inverse_rotations[rotation_idx]
    pulled = torch.gather(states.long(), 1, r_inv)
    return torch.gather(r, 1, pulled)


@torch.no_grad()
def teacher_value(model: ResMLPGFlowNet, states: torch.Tensor, chunk_size: int) -> torch.Tensor:
    outs = []
    for i in range(0, states.size(0), chunk_size):
        chunk = states[i : i + chunk_size]
        h = model.trunk(chunk)
        outs.append(model.value_head(h).squeeze(-1).float())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def build_targets(
    teacher: ResMLPGFlowNet,
    states: torch.Tensor,
    rotations: torch.Tensor,
    inverse_rotations: torch.Tensor,
    head_rotation_ids: list[torch.Tensor],
    rng: torch.Generator,
    chunk_size: int,
    autocast_ctx,
) -> torch.Tensor:
    targets = []
    for ids in head_rotation_ids:
        pick = torch.randint(0, ids.numel(), (states.size(0),), generator=rng, device=states.device)
        rot_idx = ids[pick]
        rotated = apply_state_rotations(states, rotations, inverse_rotations, rot_idx)
        if autocast_ctx is None:
            targets.append(teacher_value(teacher, rotated, chunk_size))
        else:
            with autocast_ctx:
                targets.append(teacher_value(teacher, rotated, chunk_size))
    return torch.stack(targets, dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--policy-dataset", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--rotations", type=Path, default=PROJECT / "data" / "rotations.npy")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=500_000)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--path-fraction", type=float, default=0.50)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lr-trunk", type=float, default=0.0)
    ap.add_argument("--lambda-uncertainty", type=float, default=0.05)
    ap.add_argument("--rotation-mode", choices=["coset-random", "fixed"], default="coset-random")
    ap.add_argument("--fixed-rotation-indices", default="",
                    help="Comma-separated rotation row indices for fixed mode. Empty means identity + RNG picks.")
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--teacher-chunk", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=930)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    teacher = load_az_teacher(args.teacher, args.device, dtype)
    for p in teacher.parameters():
        p.requires_grad = False
    mc = torch.load(args.teacher, map_location="cpu", weights_only=False)["model_config"]
    student = ResMLPMultiHeadOrbit(
        state_size=mc["state_size"],
        num_classes=mc["num_classes"],
        hidden_dims=tuple(mc["hidden_dims"]),
        num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        n_heads=args.n_heads,
    ).to(args.device)
    init_from_teacher(student, teacher)
    print(f"student params: {student.num_parameters():,} heads={args.n_heads}", flush=True)
    print(f"teacher: {args.teacher}", flush=True)

    if args.lr_trunk <= 0:
        for name, p in student.named_parameters():
            if not (name.startswith("value_heads.") or name.startswith("uncertainty_head.")):
                p.requires_grad = False
        params = [p for p in student.parameters() if p.requires_grad]
    else:
        head_params = []
        trunk_params = []
        for name, p in student.named_parameters():
            (head_params if name.startswith(("value_heads.", "uncertainty_head."))
             else trunk_params).append(p)
        params = [
            {"params": trunk_params, "lr": args.lr_trunk},
            {"params": head_params, "lr": args.lr_head},
        ]
    optim = torch.optim.AdamW(params, lr=args.lr_head, weight_decay=0.0,
                              fused=args.device.startswith("cuda"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    policy_data = torch.load(args.policy_dataset, map_location=args.device, weights_only=False)
    path_states = policy_data["states"].to(args.device).long()
    rotations, inverse_rotations = load_rotation_tables(args.rotations, args.device)
    all_ids = torch.arange(rotations.size(0), device=args.device, dtype=torch.long)
    if args.rotation_mode == "coset-random":
        head_rotation_ids = [all_ids[h::args.n_heads].contiguous() for h in range(args.n_heads)]
    else:
        if args.fixed_rotation_indices:
            fixed = [int(x) for x in args.fixed_rotation_indices.split(",") if x.strip()]
            if len(fixed) != args.n_heads:
                raise ValueError("--fixed-rotation-indices must contain exactly --n-heads entries")
        else:
            identity = np.arange(rotations.size(1), dtype=np.int64)
            rot_np = rotations.detach().cpu().numpy()
            identity_hits = np.flatnonzero((rot_np == identity[None, :]).all(axis=1))
            if identity_hits.size == 0:
                raise ValueError("rotations table has no identity row")
            identity_idx = int(identity_hits[0])
            non_identity = [i for i in range(rotations.size(0)) if i != identity_idx]
            rng_np = np.random.default_rng(args.seed)
            fixed = [identity_idx] + list(
                rng_np.choice(non_identity, size=args.n_heads - 1, replace=False).astype(int)
            )
        head_rotation_ids = [
            torch.tensor([i], device=args.device, dtype=torch.long) for i in fixed
        ]
        print(f"fixed rotation indices: {fixed}", flush=True)
    print("head rotation counts: " + ", ".join(str(x.numel()) for x in head_rotation_ids),
          flush=True)

    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device.startswith("cuda") else None
    n_path = int(round(args.batch_size * args.path_fraction))
    n_rw = args.batch_size - n_path
    if n_rw <= 0 or n_path <= 0:
        raise ValueError("--path-fraction must leave both random-walk and path samples")
    print(f"epochs={args.epochs} batch={args.batch_size} rw={n_rw} path={n_path} "
          f"lr_head={args.lr_head} lr_trunk={args.lr_trunk}", flush=True)

    for epoch in range(args.epochs):
        t0 = time.time()
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        rw_states, _ = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch * 1000, device=args.device, n_back=args.n_back,
        )
        n_batches = max(1, rw_states.size(0) // n_rw)
        total_v = 0.0
        total_u = 0.0
        total_spread = 0.0
        total_target_spread = 0.0
        student.train()
        for b in range(n_batches):
            bs_rw = rw_states[b * n_rw : (b + 1) * n_rw]
            idx_p = torch.randint(0, path_states.size(0), (n_path,),
                                  generator=rng, device=args.device)
            states = torch.cat([bs_rw, path_states[idx_p]], dim=0)
            targets = build_targets(
                teacher, states, rotations, inverse_rotations, head_rotation_ids,
                rng=rng, chunk_size=args.teacher_chunk, autocast_ctx=autocast_ctx,
            )
            spread_target = torch.sqrt(torch.var(targets.float(), dim=1, unbiased=False) + 1e-6)
            if autocast_ctx is None:
                pred, uncert = student.forward_with_uncertainty(states)
                v_loss = F.mse_loss(pred.float(), targets)
                u_loss = F.mse_loss(uncert.float(), spread_target)
                head_spread = torch.sqrt(torch.var(pred.float(), dim=1, unbiased=False) + 1e-6).mean()
            else:
                with autocast_ctx:
                    pred, uncert = student.forward_with_uncertainty(states)
                    v_loss = F.mse_loss(pred.float(), targets)
                    u_loss = F.mse_loss(uncert.float(), spread_target)
                    head_spread = torch.sqrt(torch.var(pred.float(), dim=1, unbiased=False) + 1e-6).mean()
            loss = v_loss + args.lambda_uncertainty * u_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_v += float(v_loss.item())
            total_u += float(u_loss.item())
            total_spread += float(head_spread.item())
            total_target_spread += float(spread_target.mean().item())
        sched.step()
        avg_v = total_v / n_batches
        avg_u = total_u / n_batches
        avg_spread = total_spread / n_batches
        avg_target_spread = total_target_spread / n_batches
        print(f"epoch {epoch:4d} | v_loss {avg_v:.4f} | u_loss {avg_u:.4f} | "
              f"head_spread {avg_spread:.4f} | target_spread {avg_target_spread:.4f} | "
              f"lr {sched.get_last_lr()[-1]:.2e} | "
              f"{time.time() - t0:.1f}s", flush=True)
        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save({
                "epoch": epoch,
                "state_dict": student.state_dict(),
                "model_config": {
                    "model_class": "ResMLPMultiHeadOrbit",
                    "state_size": mc["state_size"],
                    "num_classes": mc["num_classes"],
                    "hidden_dims": list(mc["hidden_dims"]),
                    "num_res_blocks": mc["num_res_blocks"],
                    "encoding": mc.get("encoding", "embedding"),
                    "embed_dim": mc.get("embed_dim", 16),
                    "n_heads": args.n_heads,
                },
                "teacher": str(args.teacher),
                "v_loss": avg_v,
                "u_loss": avg_u,
                "head_spread": avg_spread,
                "target_spread": avg_target_spread,
                "head_rotation_counts": [int(x.numel()) for x in head_rotation_ids],
            }, args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
