"""Trajectory Balance training (Malkin et al. 2022 / Pan et al. 2026).

Trains a GFlowNet model with shared trunk + policy head + value head + logZ scalar.
Trajectories sampled by random walks FROM solved (solved → s_k via actions),
then REVERSED to give trajectories TO solved (s_k → ... → V0 via inverse actions).

TB loss per τ: L = (logZ + Σ log P_F - Σ log P_B - log R)²
P_B uniform 1/n_gen, R = 1 at solved → log R = 0.
+ shortest-path regularizer λ·logZ to push policy onto shortest paths.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/62_train_tb.py \\
        --output megaminx/models/m_tb_v0_smoke \\
        --epochs 5 --batch-size 256 --traj-len 30
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable
from cayley.gflow_model import ResMLPGFlowNet, trajectory_balance_loss
from megaminx.puzzle import Megaminx


def sample_trajectories(
    puzzle, n_trajs: int, traj_len: int, device: str, seed: int,
):
    """Sample n_trajs trajectories from solved → s_k.

    Returns:
        states: (n_trajs, traj_len+1, state_size) int8 — s_0=V0, s_1, ..., s_{traj_len}
        actions: (n_trajs, traj_len) int — action taken at each step (a_t for t in [0, traj_len))
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    gens = GeneratorTable.from_puzzle(puzzle)
    perms = torch.from_numpy(gens.perms).to(device).long()  # (n_gen, S)
    n_gen = perms.shape[0]
    state_size = perms.shape[1]

    solved = torch.tensor(puzzle.solved_state, dtype=torch.long, device=device)
    states = torch.zeros((n_trajs, traj_len + 1, state_size), dtype=torch.long, device=device)
    actions = torch.zeros((n_trajs, traj_len), dtype=torch.long, device=device)
    states[:, 0] = solved

    for t in range(traj_len):
        a = torch.randint(0, n_gen, (n_trajs,), generator=g, device=device)
        actions[:, t] = a
        gen_rows = perms[a]  # (n_trajs, S)
        states[:, t + 1] = torch.gather(states[:, t], 1, gen_rows)

    return states, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-trajs-per-epoch", type=int, default=4096,
                    help="trajectories per epoch (sample budget)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--traj-len", type=int, default=30,
                    help="trajectory length k (uniform; in production we'd vary)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-logz", type=float, default=1e-2,
                    help="separate (typically larger) LR for logZ scalar")
    ap.add_argument("--lambda-logz", type=float, default=1e-2,
                    help="shortest-path regularizer: + lambda * logZ in loss")
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=62)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    n_gen = len(puzzle.move_names)

    # Inverse-action lookup
    gens = GeneratorTable.from_puzzle(puzzle)
    inv_idx = torch.from_numpy(gens.inverse_idx).to(args.device).long()  # (n_gen,)

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPGFlowNet(
        state_size=state_size, num_classes=state_size,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16, n_actions=n_gen,
    ).to(args.device).train()
    print(f"model params: {model.num_parameters():,}")

    # Two parameter groups: logZ gets a larger LR (typical TB practice)
    main_params = [p for n, p in model.named_parameters() if n != "log_Z"]
    optim = torch.optim.AdamW([
        {"params": main_params, "lr": args.lr},
        {"params": [model.log_Z], "lr": args.lr_logz},
    ], weight_decay=0.0, fused=args.device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else None

    # log P_B for uniform backward = -log(n_gen)
    log_pb_uniform = -math.log(n_gen)
    print(f"log P_B (uniform) = {log_pb_uniform:.3f}", flush=True)

    n_batches_per_epoch = max(1, args.n_trajs_per_epoch // args.batch_size)

    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss = 0.0
        total_log_z = 0.0
        n_batches = 0

        # Sample fresh trajectories each epoch
        traj_states, traj_actions = sample_trajectories(
            puzzle, args.n_trajs_per_epoch, args.traj_len, args.device,
            seed=args.seed * 1000 + epoch,
        )
        # traj_states: (N, T+1, S); traj_actions: (N, T)

        # Shuffle and batch
        perm = torch.randperm(args.n_trajs_per_epoch, device=args.device)
        for b in range(n_batches_per_epoch):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            bs_states = traj_states[idx]      # (B, T+1, S)
            bs_actions = traj_actions[idx]    # (B, T)
            B, T = bs_actions.shape

            # Reverse: trajectory from s_T → V0 with INVERSE actions
            # Forward actions a_0..a_{T-1} taken from V0 (=states[0]) to s_T (=states[T]).
            # Reversed traj: visit s_T, s_{T-1}, ..., s_0=V0.
            # Action at step t (in reversed order) is inv(a_{T-1-t}).
            # We want log P_F at each (state, action) where state is the t-th in reversed
            # order (s_{T-t}) and action is inv(a_{T-1-t}).

            # Reversed states (excluding final V0 — actions don't apply at terminal):
            # rev_states[b, t] = bs_states[b, T - t] for t in [0, T)
            t_idx = torch.arange(T, device=args.device)
            rev_states = bs_states[:, T - t_idx]  # (B, T, S)
            rev_actions = inv_idx[bs_actions[:, T - 1 - t_idx]]  # (B, T) — inverses, time-reversed

            # Forward through model on flattened (B*T, S)
            flat_states = rev_states.reshape(B * T, state_size)
            flat_actions = rev_actions.reshape(B * T)

            if autocast_ctx is not None:
                with autocast_ctx:
                    logits, _ = model(flat_states)  # (B*T, n_gen), (B*T,)
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    log_pf = log_probs[torch.arange(B * T, device=args.device),
                                        flat_actions]  # (B*T,)
            else:
                logits, _ = model(flat_states)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                log_pf = log_probs[torch.arange(B * T, device=args.device), flat_actions]

            log_pf_per_step = log_pf.reshape(B, T)
            traj_lengths = torch.full((B,), T, dtype=torch.long, device=args.device)
            log_R = torch.zeros(B, device=args.device)  # R=1 at solved

            tb_loss = trajectory_balance_loss(
                model.log_Z.squeeze(),
                log_pf_per_step, traj_lengths, log_pb_uniform, log_R,
            )
            loss = tb_loss + args.lambda_logz * model.log_Z.squeeze()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(tb_loss.item())
            total_log_z += float(model.log_Z.squeeze().item())
            n_batches += 1

        sched.step()
        avg_tb = total_loss / max(1, n_batches)
        avg_logz = total_log_z / max(1, n_batches)
        wall = time.time() - t0
        print(f"epoch {epoch:4d} | tb_loss {avg_tb:.4f} | logZ {avg_logz:.4f} | "
              f"lr {sched.get_last_lr()[0]:.2e} | {wall:.1f}s",
              flush=True)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            sd = model.state_dict()
            torch.save({
                "epoch": epoch,
                "state_dict": sd,
                "model_config": {
                    "state_size": state_size, "num_classes": state_size,
                    "hidden_dims": list(hidden_dims),
                    "num_res_blocks": args.num_res_blocks,
                    "encoding": "embedding", "embed_dim": 16,
                    "n_actions": n_gen,
                },
                "log_Z": float(model.log_Z.squeeze().item()),
                "tb_loss": avg_tb,
            }, args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
