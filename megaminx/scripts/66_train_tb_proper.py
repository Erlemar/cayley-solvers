"""Proper TB training: warm-start trunk from m_dd_v0 V model, variable-length walks.

Key differences vs `62_train_tb.py`:
  - `--warmstart-trunk`: load trunk weights (embedding + input_stack + res_blocks)
    from a ResMLPDistance V model (e.g., m_dd_v0). Heads initialized fresh.
    The trunk has already learned useful state representations; value/policy heads
    can build on top.
  - `--min-walk-len, --max-walk-len`: sample trajectory length uniformly from this
    range per trajectory (instead of fixed length). Forces the model to handle
    diverse depths instead of overfitting to one.
  - `--value-init-from-distance`: initialize value_head from the V model's distance
    head with sign flip (since at TB optimum, log F ~ -distance + const).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/66_train_tb_proper.py \\
        --warmstart-trunk megaminx/models/m_dd_v0/epoch_0049.pt \\
        --value-init-from-distance \\
        --output megaminx/models/m_tb_v1_proper \\
        --epochs 500 \\
        --n-trajs-per-epoch 32768 --batch-size 512 \\
        --min-walk-len 5 --max-walk-len 100 \\
        --lr 3e-4 --lr-logz 1e-2 --lambda-logz 1e-2
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


def warmstart_from_v_model(model: ResMLPGFlowNet, ckpt_path: Path,
                            init_value_from_distance: bool = False):
    """Load trunk weights from a ResMLPDistance checkpoint into ResMLPGFlowNet.

    Trunk = embedding + input_stack + res_blocks. These map state-by-state from
    permutations to feature vectors and have the same shape in both models.
    Heads are NOT loaded by default: the V model's head is single-output distance,
    while TB has policy_head (24) + value_head (1) + log_Z scalar.

    If init_value_from_distance: copy the V model's distance head into value_head
    with sign flip. At TB optimum log F(s) ~ const - dist(s, V0), so we want
    value_head ~ -head (sign flip) to start near the right calibration.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

    # Filter to trunk keys: embedding.*, input_stack.*, res_blocks.*
    trunk_sd = {}
    for k, v in sd.items():
        if k.startswith(("embedding.", "input_stack.", "res_blocks.")):
            trunk_sd[k] = v

    missing, unexpected = model.load_state_dict(trunk_sd, strict=False)
    n_loaded = len(trunk_sd)
    print(f"  warm-start: loaded {n_loaded} trunk tensors from {ckpt_path.name}",
          flush=True)

    if init_value_from_distance:
        if "head.weight" in sd and "head.bias" in sd:
            # value_head: Linear(prev, 1). head: Linear(prev, 1). Sign flip.
            with torch.no_grad():
                model.value_head.weight.copy_(-sd["head.weight"])
                model.value_head.bias.copy_(-sd["head.bias"])
            print(f"  initialized value_head = -V_distance_head (sign flip for log F)",
                  flush=True)
        else:
            print(f"  WARNING: --value-init-from-distance requested but head.* not in checkpoint",
                  flush=True)


def sample_variable_trajectories(
    puzzle, n_trajs: int, min_walk_len: int, max_walk_len: int,
    device: str, seed: int,
):
    """Sample n_trajs trajectories with random length in [min, max].

    Returns:
        states_list: list of (T+1, state_size) tensors (variable T per traj)
        actions_list: list of (T,) action tensors
        lengths: (n_trajs,) actual lengths
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)

    gens = GeneratorTable.from_puzzle(puzzle)
    perms = torch.from_numpy(gens.perms).to(device).long()
    n_gen = perms.shape[0]
    state_size = perms.shape[1]
    solved = torch.tensor(puzzle.solved_state, dtype=torch.long, device=device)

    # Sample lengths upfront on CPU
    lengths_cpu = torch.randint(min_walk_len, max_walk_len + 1, (n_trajs,), generator=g_cpu)
    lengths = lengths_cpu.to(device)

    # For efficiency, allocate max-size buffers and use mask later
    max_len = int(max_walk_len)
    states = torch.zeros((n_trajs, max_len + 1, state_size), dtype=torch.long, device=device)
    actions = torch.zeros((n_trajs, max_len), dtype=torch.long, device=device)
    states[:, 0] = solved

    for t in range(max_len):
        a = torch.randint(0, n_gen, (n_trajs,), generator=g, device=device)
        actions[:, t] = a
        gen_rows = perms[a]
        # Only update trajectories that are still "active" (t < length)
        active = (t < lengths)
        new_states = torch.gather(states[:, t], 1, gen_rows)
        states[:, t + 1] = torch.where(active.unsqueeze(1), new_states, states[:, t])

    return states, actions, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--warmstart-trunk", type=Path, default=None,
                    help="Path to ResMLPDistance checkpoint to warm-start trunk")
    ap.add_argument("--value-init-from-distance", action="store_true",
                    help="Init value_head = -V_distance_head (sign flip for log F calibration)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--n-trajs-per-epoch", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--min-walk-len", type=int, default=5)
    ap.add_argument("--max-walk-len", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-logz", type=float, default=1e-2)
    ap.add_argument("--lambda-logz", type=float, default=1e-2)
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=66)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    n_gen = len(puzzle.move_names)

    gens = GeneratorTable.from_puzzle(puzzle)
    inv_idx = torch.from_numpy(gens.inverse_idx).to(args.device).long()

    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(","))
    model = ResMLPGFlowNet(
        state_size=state_size, num_classes=state_size,
        hidden_dims=hidden_dims, num_res_blocks=args.num_res_blocks,
        encoding="embedding", embed_dim=16, n_actions=n_gen,
    ).to(args.device)
    print(f"model params: {model.num_parameters():,}", flush=True)

    if args.warmstart_trunk:
        warmstart_from_v_model(model, args.warmstart_trunk,
                                init_value_from_distance=args.value_init_from_distance)
    model.train()

    main_params = [p for n, p in model.named_parameters() if n != "log_Z"]
    optim = torch.optim.AdamW([
        {"params": main_params, "lr": args.lr},
        {"params": [model.log_Z], "lr": args.lr_logz},
    ], weight_decay=0.0, fused=args.device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else None

    log_pb_uniform = -math.log(n_gen)
    print(f"log P_B (uniform) = {log_pb_uniform:.3f}", flush=True)
    print(f"trajectory length range: [{args.min_walk_len}, {args.max_walk_len}]", flush=True)

    n_batches_per_epoch = max(1, args.n_trajs_per_epoch // args.batch_size)

    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss = 0.0
        total_log_z = 0.0
        n_batches = 0

        traj_states, traj_actions, traj_lengths = sample_variable_trajectories(
            puzzle, args.n_trajs_per_epoch, args.min_walk_len, args.max_walk_len,
            args.device, seed=args.seed * 1000 + epoch,
        )
        # traj_states: (N, max_len+1, S); traj_actions: (N, max_len); traj_lengths: (N,)

        perm = torch.randperm(args.n_trajs_per_epoch, device=args.device)
        for b in range(n_batches_per_epoch):
            idx = perm[b * args.batch_size : (b + 1) * args.batch_size]
            bs_states = traj_states[idx]      # (B, max_len+1, S)
            bs_actions = traj_actions[idx]     # (B, max_len)
            bs_lengths = traj_lengths[idx]     # (B,)
            B, T = bs_actions.shape

            # Reverse: tau visits states in order s_n, s_{n-1}, ..., V0
            # For each traj b, length n_b: for t in [0, n_b), rev_state = states[n_b - t]
            # action at step t (in reversed traj) is inv(forward_a_{n_b-1-t})
            # We zero-pad steps t >= n_b (mask later in TB loss).

            t_idx = torch.arange(T, device=args.device).unsqueeze(0)  # (1, T)
            # rev_state_pos[b, t] = bs_lengths[b] - t (clipped at 0 for t >= length)
            rev_pos = (bs_lengths.unsqueeze(1) - t_idx).clamp(min=0)  # (B, T)
            # rev_states[b, t] = bs_states[b, rev_pos[b, t]]
            batch_indices = torch.arange(B, device=args.device).unsqueeze(1).expand(B, T)
            rev_states = bs_states[batch_indices, rev_pos]  # (B, T, S)

            # rev_action_pos[b, t] = bs_lengths[b] - 1 - t (the forward action index)
            rev_action_pos = (bs_lengths.unsqueeze(1) - 1 - t_idx).clamp(min=0)
            rev_actions_forward = bs_actions[batch_indices, rev_action_pos]  # (B, T)
            rev_actions = inv_idx[rev_actions_forward]  # inverse of forward actions

            # Forward pass through model. Note: we evaluate ALL steps (including padded
            # ones), but TB loss masks them by trajectory length.
            flat_states = rev_states.reshape(B * T, state_size)
            flat_actions = rev_actions.reshape(B * T)

            if autocast_ctx is not None:
                with autocast_ctx:
                    logits, _ = model(flat_states)
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    log_pf = log_probs[torch.arange(B * T, device=args.device),
                                        flat_actions]
            else:
                logits, _ = model(flat_states)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                log_pf = log_probs[torch.arange(B * T, device=args.device), flat_actions]

            log_pf_per_step = log_pf.reshape(B, T)
            log_R = torch.zeros(B, device=args.device)

            tb_loss = trajectory_balance_loss(
                model.log_Z.squeeze(),
                log_pf_per_step, bs_lengths, log_pb_uniform, log_R,
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
              f"lr {sched.get_last_lr()[0]:.2e} | {wall:.1f}s", flush=True)

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
                "args": vars(args),
            }, args.output / f"epoch_{epoch:04d}.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
