"""AlphaZero-style training for megaminx.

Self-play with PUCT search → (state, policy_target, value_target) tuples.
Joint training of policy + value heads on this data.

This is a smoke implementation:
- Single iteration (one round of self-play + train).
- Reuses ResMLPGFlowNet (policy + value heads in one model).
- PUCT done with sequential descent (slow but correct, per 51_puct_search.py findings).
- Self-play trajectories: from scrambled state, take π_PUCT-greedy moves until V0 or budget.
- Value targets: realized 1/(1+remaining_path_len) (or 0 if didn't solve).
- Policy targets: PUCT visit-count distribution at each node.

For full production, would need iterative self-play loop:
  for iter t:
    M_t -> generate self-play data via PUCT(M_t)
    train M_{t+1} on data
This script does a single iteration (smoke).

Usage:
    .venv/Scripts/python.exe megaminx/scripts/63_train_alphazero.py \\
        --v-checkpoint megaminx/models/m_curr_v3/epoch_0499.pt \\
        --output megaminx/models/m_az_v0_smoke \\
        --n-self-play 100 --puct-budget 256 \\
        --epochs 5
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.data import GeneratorTable
from cayley.gflow_model import ResMLPGFlowNet
from cayley.model import ResMLPDistance
from megaminx.puzzle import Megaminx


def load_v_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=mc.get("state_size", 120),
        num_classes=mc.get("num_classes", 120),
        hidden_dims=tuple(mc.get("hidden_dims", [2048, 512])),
        num_res_blocks=mc.get("num_res_blocks", 2),
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
    )
    model.load_state_dict(sd, strict=False)
    return model.to(device).eval()


def simple_puct_episode(
    initial_state, puzzle, perms, inv_idx, v_model, max_nodes, max_steps,
    cpuct=1.4, device="cuda",
):
    """Run a self-play episode from `initial_state` using PUCT.

    Each step:
      1. Run PUCT from current state with `max_nodes` total expansions
      2. Get visit-count distribution at root → policy target
      3. Sample action from this distribution (greedy after warmup)
      4. Apply, advance to next state

    Stops when state == V0 or max_steps reached.

    Returns list of (state, policy_target_24, taken_action) per step,
    plus boolean reached_V0 and final path length.
    """
    state_size = perms.shape[1]
    n_gen = perms.shape[0]
    solved = torch.tensor(puzzle.solved_state, dtype=torch.long, device=device)

    cur_state = torch.tensor(list(initial_state), dtype=torch.long, device=device)
    trajectory = []  # list of (state_tensor, pi_24, action_taken)

    for step in range(max_steps):
        # Check if solved
        if torch.equal(cur_state, solved):
            return trajectory, True, step

        # PUCT root expansion: simple version — just look 1 step ahead, score by V
        # (For real PUCT we'd do tree search; smoke uses 1-ply lookahead by V.)
        # Generate all 24 children
        children = cur_state.unsqueeze(0).expand(n_gen, state_size).clone()
        children = torch.gather(children, 1, perms)  # (n_gen, state_size)

        # Evaluate V on children
        with torch.no_grad():
            v_children = v_model(children).flatten().float()

        # Convert to "preferences": low V → high prior. Use softmax(-V/T)
        T = 1.0
        log_pi = torch.log_softmax(-v_children / T, dim=0)
        pi = log_pi.exp()  # (n_gen,) — visit-distribution proxy

        # Sample action
        action = int(torch.multinomial(pi, num_samples=1).item())
        # Or greedy:
        # action = int(pi.argmax().item())

        trajectory.append((cur_state.clone(), pi.cpu().numpy().copy(), action))
        cur_state = children[action]

    return trajectory, False, max_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-checkpoint", required=True, type=Path,
                    help="V model used for value priors during self-play")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--n-self-play", type=int, default=100)
    ap.add_argument("--puct-budget", type=int, default=256, help="(unused in smoke)")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1.0, help="policy CE weight")
    ap.add_argument("--beta", type=float, default=0.5, help="value MSE weight")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=63)
    ap.add_argument("--synthetic-walks", action="store_true",
                    help="Use synthetic random-walk-reversed trajectories instead of real self-play.")
    ap.add_argument("--max-walk-len", type=int, default=20)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    n_gen = len(puzzle.move_names)

    gens = GeneratorTable.from_puzzle(puzzle)
    perms = torch.from_numpy(gens.perms).to(args.device).long()
    inv_idx = torch.from_numpy(gens.inverse_idx).to(args.device).long()

    # Self-play: use V_neural for value priors. NOTE: For first AZ smoke, the simple
    # 1-ply greedy V can't solve hard scrambles in `max_steps`. So we ALSO support a
    # synthetic-trajectory mode (`--synthetic-walks`) which generates random walks
    # from V0 of length [3, max_walk_len], reverses them to get valid solve paths,
    # and uses these as "ground-truth" self-play data. Validates the joint trainer
    # without requiring a strong PUCT search.
    print(f"loading V from {args.v_checkpoint}", flush=True)
    v_model = load_v_model(args.v_checkpoint, args.device)

    # Training student: dual-head GFlowNet model (we use it for policy + value heads;
    # discard logZ for AlphaZero context — we just train the heads)
    student = ResMLPGFlowNet(
        state_size=state_size, num_classes=state_size,
        hidden_dims=(2048, 512), num_res_blocks=2,
        encoding="embedding", embed_dim=16, n_actions=n_gen,
    ).to(args.device)
    print(f"student params: {student.num_parameters():,}", flush=True)

    all_states = []
    all_policies = []
    all_values = []
    n_solved = 0
    total_path_len = 0

    if args.synthetic_walks:
        print(f"\n=== Synthetic walks: {args.n_self_play} trajectories, "
              f"length up to {args.max_walk_len} ===", flush=True)
        rng_torch = torch.Generator(device=args.device)
        rng_torch.manual_seed(args.seed)
        solved = torch.tensor(puzzle.solved_state, dtype=torch.long, device=args.device)
        t0 = time.time()
        for i in range(args.n_self_play):
            walk_len = int(torch.randint(3, args.max_walk_len + 1, (1,),
                                          generator=rng_torch, device=args.device).item())
            cur = solved.clone()
            actions_taken = []
            for _ in range(walk_len):
                a = int(torch.randint(0, n_gen, (1,), generator=rng_torch,
                                      device=args.device).item())
                actions_taken.append(a)
                cur = cur[perms[a]]
            # Reversed solve trajectory: (s_walk, inv(a_{w-1}), s_{w-1}, ..., V0)
            inv_actions = [int(inv_idx[a].item()) for a in reversed(actions_taken)]
            # Walk states for the reversed trajectory:
            # state_t in reversed traj = state at step (walk_len - t) of forward walk
            states_traj = [solved.clone()]
            cur2 = solved.clone()
            for a in actions_taken:
                cur2 = cur2[perms[a]]
                states_traj.append(cur2.clone())
            # rev_states[t] for t in [0, walk_len): states_traj[walk_len - t]
            for t in range(walk_len):
                rev_state = states_traj[walk_len - t]
                action_taken = inv_actions[t]  # the inverse of forward action a_{walk_len-1-t}
                # Policy target: one-hot on the chosen action (since this is "ground truth")
                pi_one_hot = np.zeros(n_gen, dtype=np.float32)
                pi_one_hot[action_taken] = 1.0
                v_target = float(walk_len - t)
                all_states.append(rev_state.cpu().numpy())
                all_policies.append(pi_one_hot)
                all_values.append(v_target)
            n_solved += 1
            total_path_len += walk_len
        wall_sp = time.time() - t0
        print(f"synthetic done: {n_solved} trajectories, total {total_path_len} steps, "
              f"{wall_sp:.1f}s", flush=True)
    else:
        # Generate self-play trajectories
        print(f"\n=== Self-play: {args.n_self_play} episodes ===", flush=True)
        rng = np.random.default_rng(args.seed)
        test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))
        pids = rng.choice(len(test_rows), args.n_self_play, replace=False)

        t0 = time.time()
        for i, pid in enumerate(pids):
            s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
            traj, reached, path_len = simple_puct_episode(
                s0, puzzle, perms, inv_idx, v_model,
                max_nodes=args.puct_budget, max_steps=args.max_steps,
                device=args.device,
            )
            if reached:
                n_solved += 1
                total_path_len += path_len
                for j, (st, pi, a) in enumerate(traj):
                    v_target = float(path_len - j)
                    all_states.append(st.cpu().numpy())
                    all_policies.append(pi)
                    all_values.append(v_target)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{args.n_self_play} | solved {n_solved} | "
                      f"avg path {total_path_len/max(1,n_solved):.1f}", flush=True)
        wall_sp = time.time() - t0
        print(f"self-play done: {n_solved}/{args.n_self_play} solved, "
              f"avg path {total_path_len/max(1,n_solved):.1f}, {wall_sp:.1f}s", flush=True)

    if not all_states:
        print("ERROR: no successful self-play trajectories - try larger max_steps")
        return 1

    # Convert to tensors
    states_t = torch.tensor(np.stack(all_states), dtype=torch.long, device=args.device)
    policies_t = torch.tensor(np.stack(all_policies), dtype=torch.float32, device=args.device)
    values_t = torch.tensor(all_values, dtype=torch.float32, device=args.device)
    N = states_t.size(0)
    print(f"\n=== Training student on {N:,} (state, π, v) tuples ===", flush=True)

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0,
                              fused=args.device == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else None

    student.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        total_p_loss = 0.0
        total_v_loss = 0.0
        n_batches = 0
        perm = torch.randperm(N, device=args.device)
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            bs = states_t[idx]
            bp = policies_t[idx]
            bv = values_t[idx]

            if autocast_ctx is not None:
                with autocast_ctx:
                    logits, value = student(bs)
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    p_loss = -(bp * log_probs).sum(dim=-1).mean()
                    v_loss = F.mse_loss(value.float(), bv)
                    loss = args.alpha * p_loss + args.beta * v_loss
            else:
                logits, value = student(bs)
                log_probs = F.log_softmax(logits.float(), dim=-1)
                p_loss = -(bp * log_probs).sum(dim=-1).mean()
                v_loss = F.mse_loss(value.float(), bv)
                loss = args.alpha * p_loss + args.beta * v_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_p_loss += float(p_loss.item())
            total_v_loss += float(v_loss.item())
            n_batches += 1
        sched.step()
        avg_p = total_p_loss / max(1, n_batches)
        avg_v = total_v_loss / max(1, n_batches)
        print(f"epoch {epoch:4d} | p_loss {avg_p:.4f} | v_loss {avg_v:.4f} | "
              f"lr {sched.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s", flush=True)

    sd = student.state_dict()
    torch.save({
        "state_dict": sd,
        "model_config": {
            "state_size": state_size, "num_classes": state_size,
            "hidden_dims": [2048, 512], "num_res_blocks": 2,
            "encoding": "embedding", "embed_dim": 16, "n_actions": n_gen,
        },
        "n_self_play": args.n_self_play,
        "n_solved": n_solved,
        "n_train_samples": N,
    }, args.output / "final.pt")
    print(f"saved {args.output / 'final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
