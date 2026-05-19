"""m_v_pi_v0: Multi-head V + pi model (Idea: Option C from multitask_training_ideas.md).

Single shared body + two heads:
  - V head: scalar, trained with Bellman target on random-walk states (with optional
    warmstart from m05 / m_fr_v0)
  - pi head: n_actions logits, trained with weighted cross-entropy on solved-path
    (state, action, suffix_len) tuples from data/policy_train.pt

Joint training: per batch, half the samples are Bellman states (V loss), half are
policy states (pi loss). Body is updated by both losses; heads only by their
respective signals.

Inference plug-in: V head matches ResMLPDistance signature, so 03_solve.py auto-loads
this as a V model. Separate pi-extraction wrapper allows beam to use V on children
+ pi on parents in the same forward when desired.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/49_train_multihead_v_pi.py \\
        --warmstart megaminx/models/m_fr_v0/epoch_0499.pt \\
        --policy-data megaminx/data/policy_train.pt \\
        --out-dir megaminx/models/m_v_pi_v0 \\
        --n-epochs 500
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.bellman import _apply_all_generators, _bellman_targets
from cayley.data import GeneratorTable, generate_walks_torch
from cayley.model import ResBlock, ResMLPDistance
from megaminx.puzzle import Megaminx


class ResMLPVPi(nn.Module):
    """ResMLP body + V head (1) + pi head (n_actions). Shared body weights."""

    def __init__(
        self,
        state_size: int = 120,
        num_classes: int = 120,
        hidden_dims: tuple[int, ...] = (2048, 512),
        num_res_blocks: int = 2,
        encoding: str = "embedding",
        embed_dim: int = 16,
        n_actions: int = 24,
    ):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.encoding = encoding
        self.embed_dim = embed_dim
        self.output_dim = 1  # for 03_solve.py auto-detect: V-head model
        self.n_actions = n_actions
        self.inference_chunk_size = None

        if encoding != "embedding":
            raise NotImplementedError("only embedding encoding for now")
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
        # V head (matches ResMLPDistance.head naming for warmstart compatibility)
        self.head = nn.Linear(prev, 1)
        # pi head (separate from V head)
        self.pi_head = nn.Linear(prev, n_actions)

    def forward_body(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.input_stack[0].weight.dtype
        h = self.embedding(x.long()).to(target_dtype).flatten(start_dim=-2)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """V output. Compatible with ResMLPDistance interface for 03_solve.py."""
        h = self.forward_body(x)
        return self.head(h).squeeze(-1)

    def forward_pi(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_body(x)
        return self.pi_head(h)

    def forward_both(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.forward_body(x)
        return self.head(h).squeeze(-1), self.pi_head(h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmstart", type=Path, default=None,
                    help="V-model checkpoint to warmstart body+V_head from (e.g. m_fr_v0). "
                         "If absent, train from scratch with walk-depth target on V head.")
    ap.add_argument("--policy-data", type=Path, required=True,
                    help="policy_train.pt with (states, actions, weights)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-epochs", type=int, default=500)
    ap.add_argument("--samples-per-epoch", type=int, default=300_000,
                    help="random-walk states per epoch (V branch)")
    ap.add_argument("--batch-size", type=int, default=8192,
                    help="combined batch size: half V states, half pi states")
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha-pi", type=float, default=1.0,
                    help="weight on pi loss vs V loss in the combined objective")
    ap.add_argument("--seed", type=int, default=900)
    ap.add_argument("--checkpoint-every-epochs", type=int, default=50)
    ap.add_argument("--n-back", type=int, default=1)
    ap.add_argument("--use-bellman", action=argparse.BooleanOptionalAction, default=True,
                    help="if true, V branch uses Bellman target (requires warmstart). "
                         "if false, V branch uses walk-depth target (no warmstart needed).")
    ap.add_argument("--target-update-every-epochs", type=int, default=10)
    ap.add_argument("--clip-upper", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--clip-lower", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    n_actions = len(puzzle.move_names)

    model = ResMLPVPi(
        state_size=120, num_classes=120,
        hidden_dims=(2048, 512), num_res_blocks=2,
        encoding="embedding", embed_dim=16,
        n_actions=n_actions,
    ).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} params")

    # Warmstart body + V head from a V-model checkpoint (m05 / m_fr_v0).
    # Skip pi_head (always reinit).
    if args.warmstart is not None:
        ckpt = torch.load(args.warmstart, map_location=device, weights_only=False)
        sd = ckpt["state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        # Strip pi_head entries (none expected) + filter shape mismatches.
        body_v_sd = {k: v for k, v in sd.items() if not k.startswith("pi_head.")}
        missing, unexpected = model.load_state_dict(body_v_sd, strict=False)
        n_missing_body = sum(1 for k in missing if not k.startswith("pi_head"))
        n_missing_pi = sum(1 for k in missing if k.startswith("pi_head"))
        print(f"warmstart from {args.warmstart}: body+V loaded "
              f"({n_missing_body} body keys missing, {n_missing_pi} pi_head reinit)")

    # Bellman target net (frozen snapshot of model). Used for V branch when use_bellman.
    target_model = None
    if args.use_bellman:
        target_model = copy.deepcopy(model).eval()
        for p in target_model.parameters():
            p.requires_grad = False
        print(f"Bellman target net initialized")

    # Load policy data.
    pd = torch.load(args.policy_data, map_location="cpu", weights_only=False)
    p_states_cpu = pd["states"]              # (N, 120) int8
    p_actions_cpu = pd["actions"].to(torch.long)  # (N,) int64
    p_weights_cpu = pd["weights"]            # (N,) float32
    valid = p_actions_cpu >= 0
    p_states_cpu = p_states_cpu[valid]
    p_actions_cpu = p_actions_cpu[valid]
    p_weights_cpu = p_weights_cpu[valid]
    p_states = p_states_cpu.to(device)
    p_actions = p_actions_cpu.to(device)
    p_weights = p_weights_cpu.to(device)
    print(f"policy data: {p_states.size(0):,} (state, action) pairs")

    # Generators tensor for Bellman target.
    gens_table = GeneratorTable.from_puzzle(puzzle)
    generators = torch.from_numpy(gens_table.perms).to(device)
    solved_state = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.n_epochs)
    autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda" else None)
    batch_gen = torch.Generator(device=device).manual_seed(args.seed)

    half_batch = args.batch_size // 2
    print(f"epochs: {args.n_epochs}  batch: {args.batch_size} ({half_batch} V + {half_batch} pi)  "
          f"alpha_pi: {args.alpha_pi}  use_bellman: {args.use_bellman}")

    for epoch in range(args.n_epochs):
        t0 = time.time()
        # V branch: generate random walks for this epoch.
        n_walks = max(1, args.samples_per_epoch // args.k_max)
        v_states, v_depths = generate_walks_torch(
            puzzle, n_walks=n_walks, k_max=args.k_max,
            seed=args.seed + epoch, device=device, n_back=args.n_back,
        )
        v_depths_f = v_depths.to(torch.float32)
        N_v = v_states.shape[0]
        N_p = p_states.shape[0]

        model.train()
        total_v_loss, total_pi_loss, total_loss, n_batches = 0.0, 0.0, 0.0, 0
        v_perm = torch.randperm(N_v, generator=batch_gen, device=device)
        p_perm = torch.randperm(N_p, generator=batch_gen, device=device)
        # iterate while both branches have data
        n_steps = min(N_v // half_batch, N_p // half_batch)
        for step in range(n_steps):
            v_idx = v_perm[step * half_batch : (step + 1) * half_batch]
            p_idx = p_perm[step * half_batch : (step + 1) * half_batch]
            bs_v = v_states[v_idx]
            bd_v = v_depths_f[v_idx]
            bs_p = p_states[p_idx]
            ba_p = p_actions[p_idx]
            bw_p = p_weights[p_idx]

            # Compute Bellman target on V states (using target net).
            if args.use_bellman and target_model is not None:
                target_v = _bellman_targets(
                    target_model, bs_v, bd_v, generators, solved_state,
                    chunk_size=4096,
                    clip_upper=args.clip_upper, clip_lower=args.clip_lower,
                    softmin_temperature=0.0,
                )
            else:
                target_v = bd_v  # walk-depth target

            # Forward both branches through one body call.
            all_states = torch.cat([bs_v, bs_p], dim=0)
            if autocast_ctx is not None:
                with autocast_ctx:
                    h = model.forward_body(all_states)
                    h_v, h_p = h[:bs_v.size(0)], h[bs_v.size(0):]
                    v_pred = model.head(h_v).squeeze(-1)
                    pi_logits = model.pi_head(h_p)
                    v_loss = F.mse_loss(v_pred, target_v)
                    pi_losses_per = F.cross_entropy(pi_logits, ba_p, reduction="none")
                    pi_loss = (pi_losses_per * bw_p).sum() / bw_p.sum().clamp(min=1e-6)
                    loss = v_loss + args.alpha_pi * pi_loss
            else:
                h = model.forward_body(all_states)
                h_v, h_p = h[:bs_v.size(0)], h[bs_v.size(0):]
                v_pred = model.head(h_v).squeeze(-1)
                pi_logits = model.pi_head(h_p)
                v_loss = F.mse_loss(v_pred, target_v)
                pi_losses_per = F.cross_entropy(pi_logits, ba_p, reduction="none")
                pi_loss = (pi_losses_per * bw_p).sum() / bw_p.sum().clamp(min=1e-6)
                loss = v_loss + args.alpha_pi * pi_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_v_loss += float(v_loss.item())
            total_pi_loss += float(pi_loss.item())
            total_loss += float(loss.item())
            n_batches += 1

        sched.step()

        avg_v = total_v_loss / max(n_batches, 1)
        avg_pi = total_pi_loss / max(n_batches, 1)
        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 5 == 0 or epoch == args.n_epochs - 1:
            print(f"epoch {epoch:4d} | V {avg_v:.4f} | pi {avg_pi:.4f} | "
                  f"total {avg_loss:.4f} | lr {float(sched.get_last_lr()[0]):.2e} | "
                  f"{time.time()-t0:.1f}s", flush=True)

        # Bellman target refresh.
        if args.use_bellman and target_model is not None:
            if (epoch + 1) % args.target_update_every_epochs == 0:
                target_model.load_state_dict(model.state_dict())

        if (epoch + 1) % args.checkpoint_every_epochs == 0 or epoch == args.n_epochs - 1:
            ckpt_path = args.out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "v_loss": avg_v,
                "pi_loss": avg_pi,
                "model_config": {
                    "state_size": 120, "num_classes": 120,
                    "hidden_dims": [2048, 512], "num_res_blocks": 2,
                    "encoding": "embedding", "embed_dim": 16,
                    "n_actions": n_actions, "output_dim": 1,
                    "model_class": "ResMLPVPi",
                },
                "warmstart": str(args.warmstart) if args.warmstart else None,
                "policy_data": str(args.policy_data),
                "use_bellman": args.use_bellman,
                "alpha_pi": args.alpha_pi,
            }, ckpt_path)
            print(f"  saved {ckpt_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
