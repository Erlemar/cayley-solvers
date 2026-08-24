"""Train a GFN-pathfinding policy (Morozov et al. 2026, arXiv:2603.01786 port).

Tasks:
  rubik2   -- 2x2x2 Rubik's cube, paper generators verbatim. Gate-0 validation:
              with paper hyperparams the beam-256 eval on their shipped test set
              must trend to ~10.64 (BFS-verified optimum; paper Table 1).
  megaminx -- 120-sticker megaminx from megaminx/data/puzzle_info.json, 24 gens,
              exact logZ from the Schreier-Sims group order (verified 2026-07-17).

Example (Gate 0):
  .venv/Scripts/python.exe megaminx/scripts/120_train_gfn_pathfinding.py \
      --task rubik2 --hidden 1024 --res-blocks 3 --encoding onehot \
      --batch-size 128 --nmax 12 --reg-coef 0.01 --reg-mode flow \
      --iters 500000 --eval-every 25000 \
      --test-npy data/gfn_ref/rubik2_test.npy \
      --save-dir models/gfn_rubik2_v0

Example (Gate 1 smoke):
  .venv/Scripts/python.exe megaminx/scripts/120_train_gfn_pathfinding.py \
      --task megaminx --hidden 1024 --res-blocks 3 \
      --batch-size 512 --nmax 40 --reg-coef 2.4e-55 --reg-mode flow \
      --iters 100000 --eval-every 10000 \
      --save-dir megaminx/models/m_gfn_v0

All output is ASCII-only (cp932 console, project Rule 24).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "megaminx" / "src"))

from cayley.gfn_pathfinding import (  # noqa: E402
    GFNPolicyNet,
    bfs_exact_states,
    build_env,
    count_params,
    policy_beam_solve,
    random_walk_states,
    regularized_tb_loss,
    rollout_solve,
    sample_forward_trajectories,
)

# 2x2x2 generators copied verbatim from GreatDrake/gfn-pathfinding train.py (MIT).
# Inverse pairing: gen k and gen (k+6) mod 12 are mutual inverses (asserted at build).
RUBIK2_GENS = [
    [0, 1, 19, 17, 6, 4, 7, 5, 2, 9, 3, 11, 12, 13, 14, 15, 16, 20, 18, 21, 10, 8, 22, 23],
    [18, 16, 2, 3, 4, 5, 6, 7, 8, 0, 10, 1, 13, 15, 12, 14, 22, 17, 23, 19, 20, 21, 11, 9],
    [0, 5, 2, 7, 4, 21, 6, 23, 10, 8, 11, 9, 3, 13, 1, 15, 16, 17, 18, 19, 20, 14, 22, 12],
    [4, 1, 6, 3, 20, 5, 22, 7, 8, 9, 10, 11, 12, 2, 14, 0, 17, 19, 16, 18, 15, 21, 13, 23],
    [0, 1, 2, 3, 4, 5, 18, 19, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 22, 20, 23, 21],
    [1, 3, 0, 2, 16, 17, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 20, 21, 22, 23],
    [0, 1, 8, 10, 5, 7, 4, 6, 21, 9, 20, 11, 12, 13, 14, 15, 16, 3, 18, 2, 17, 19, 22, 23],
    [9, 11, 2, 3, 4, 5, 6, 7, 8, 23, 10, 22, 14, 12, 15, 13, 1, 17, 0, 19, 20, 21, 16, 18],
    [0, 14, 2, 12, 4, 1, 6, 3, 9, 11, 8, 10, 23, 13, 21, 15, 16, 17, 18, 19, 20, 5, 22, 7],
    [15, 1, 13, 3, 0, 5, 2, 7, 8, 9, 10, 11, 12, 22, 14, 20, 18, 16, 19, 17, 4, 21, 6, 23],
    [0, 1, 2, 3, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 18, 19, 16, 17, 6, 7, 21, 23, 20, 22],
    [2, 0, 3, 1, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 4, 5, 18, 19, 20, 21, 22, 23],
]

# Reachable 2x2 sticker states: 3,674,160 * 24 (global orientations reachable via
# opposite-face turns). Matches the reference implementation's true_log_z.
RUBIK2_LOG_Z = math.log(3674160.0 * 24)

# Megaminx sticker group order, Schreier-Sims on the 24 generator permutations
# (sympy, verified 2026-07-17). ln = 156.582460.
MEGAMINX_ORDER = 100669616553523347122516032313645505168688116411019768627200000000000

# Two-phase Phase-2 subgroup H = <U, F, L, BL, BR, R> (12 TOP generators),
# Schreier-Sims verified 2026-07-18. ln = 97.3990, diameter ~ 41-45.
MEGAMINX_H_ORDER = 1994489344120498841166261897249423360000000

# IHES Picture Cube: 18-gen, 72-perm, order 2.1259e24 (Schreier-Sims 2026-07-21).
# ln = 56.0162. Cube-scale (between rubik3's 45.2 and megaminx's 156.6).
IHES_ORDER = 2125922464947725402112000


def build_task_env(task: str):
    if task == "rubik2":
        solved = [i // 4 for i in range(24)]
        inv_idx = [(k + 6) % 12 for k in range(12)]
        return build_env(RUBIK2_GENS, inv_idx, solved, num_classes=6, true_log_z=RUBIK2_LOG_Z)
    if task == "megaminx":
        from megaminx.puzzle import Megaminx

        puzzle = Megaminx.load(ROOT / "megaminx" / "data" / "puzzle_info.json")
        names = list(puzzle.move_names)
        gens = [list(puzzle.generators[n]) for n in names]
        inv_idx = [names.index(puzzle.inverse_name(n)) for n in names]
        return build_env(
            gens,
            inv_idx,
            list(puzzle.solved_state),
            num_classes=120,
            true_log_z=math.log(MEGAMINX_ORDER),
        )
    if task == "megaminx_h":
        # Two-phase Phase-2 subgroup: the GFN finisher operates inside H using
        # only the 12 TOP-face generators (see megaminx/two_phase.py).
        from megaminx.puzzle import Megaminx
        from megaminx.two_phase import phase2_move_names

        puzzle = Megaminx.load(ROOT / "megaminx" / "data" / "puzzle_info.json")
        names = list(phase2_move_names(puzzle))
        gens = [list(puzzle.generators[n]) for n in names]
        inv_idx = [names.index(puzzle.inverse_name(n)) for n in names]
        return build_env(gens, inv_idx, list(puzzle.solved_state),
                         num_classes=120, true_log_z=math.log(MEGAMINX_H_ORDER))
    if task == "megaminx2f":
        # 2-adjacent-face subgroup: real megaminx moves + 120-dim states at a group
        # size where the paper's recipe demonstrably converges (rubik2-scale gate).
        from megaminx.puzzle import Megaminx

        puzzle = Megaminx.load(ROOT / "megaminx" / "data" / "puzzle_info.json")
        names = ["U", "-U", "F", "-F"]
        gens = [list(puzzle.generators[n]) for n in names]
        inv_idx = [names.index(puzzle.inverse_name(n)) for n in names]
        from sympy.combinatorics import Permutation, PermutationGroup

        order = int(PermutationGroup([Permutation(g) for g in gens]).order())
        log_z = math.log(order)
        print("megaminx2f subgroup order = %.6e  lnZ = %.4f" % (float(order), log_z))
        return build_env(gens, inv_idx, list(puzzle.solved_state),
                         num_classes=120, true_log_z=log_z)
    if task == "ihes":
        # IHES Picture Cube: generic permutation puzzle from the cayley project's
        # data/puzzle_info.json (18 gens, 72-perm, solved = identity).
        info = json.load(open(ROOT / "data" / "puzzle_info.json", encoding="utf-8"))
        gens_d = info["generators"]
        names = list(gens_d.keys())
        gens = [list(gens_d[n]) for n in names]
        solved = info.get("central_state") or list(range(len(gens[0])))
        inv_name = lambda n: n[1:] if n.startswith("-") else "-" + n
        inv_idx = [names.index(inv_name(n)) for n in names]
        return build_env(gens, inv_idx, solved, num_classes=len(solved),
                         true_log_z=math.log(IHES_ORDER))
    raise ValueError(f"unknown task {task!r}")


def fmt(x: float) -> str:
    return f"{x:.4g}"


def eval_rubik2(model, env, test_states, beam_w: int, max_len: int, beam_n: int) -> str:
    model.eval()
    done_g, len_g = rollout_solve(model, env, test_states, max_steps=max_len, greedy=True)
    done_s, len_s = rollout_solve(model, env, test_states, max_steps=max_len, greedy=False)
    parts = [
        f"greedy rate={float(done_g.float().mean()):.3f}"
        f" len={float(len_g[done_g].float().mean()) if bool(done_g.any()) else -1:.2f}",
        f"sample rate={float(done_s.float().mean()):.3f}"
        f" len={float(len_s[done_s].float().mean()) if bool(done_s.any()) else -1:.2f}",
    ]
    n = min(beam_n, test_states.shape[0])
    solved_n, tot = 0, 0
    for i in range(n):
        ok, ln, _ = policy_beam_solve(model, env, test_states[i], width=beam_w, max_steps=max_len)
        if ok:
            solved_n += 1
            tot += ln
    parts.append(
        f"beam{beam_w}(n={n}) rate={solved_n / n:.3f}"
        f" len={tot / solved_n if solved_n else -1:.2f}"
    )
    model.train()
    return "  ".join(parts)


def eval_megaminx(model, env, walk_sets, exact_sets, beam_w: int, beam_n: int,
                  beam_depths: tuple[int, ...] = (20, 40, 60)) -> str:
    model.eval()
    lines = []
    for d, states in sorted(exact_sets.items()):
        done, ln = rollout_solve(model, env, states, max_steps=4 * d + 8, greedy=True)
        opt = float(((ln == d) & done).float().mean())
        line = f"exact d={d}: greedy rate={float(done.float().mean()):.2f} optimal={opt:.2f}"
        if beam_w > 1 and d >= 3:
            n = min(beam_n, states.shape[0])
            sn, on = 0, 0
            for i in range(n):
                ok, ln_b, _ = policy_beam_solve(model, env, states[i], width=beam_w,
                                                max_steps=4 * d + 8)
                sn += int(ok)
                on += int(ok and ln_b == d)
            line += f"  beam{beam_w}(n={n}) rate={sn / n:.2f} optimal={on / n:.2f}"
        lines.append(line)
    for d, states in sorted(walk_sets.items()):
        done, ln = rollout_solve(model, env, states, max_steps=max(2 * d + 20, 60), greedy=True)
        mean_len = float(ln[done].float().mean()) if bool(done.any()) else -1.0
        line = f"walk d={d}: greedy rate={float(done.float().mean()):.2f} len={mean_len:.1f}"
        if beam_w > 1 and d in beam_depths:
            n = min(beam_n, states.shape[0])
            sn, tot = 0, 0
            for i in range(n):
                ok, ln_b, _ = policy_beam_solve(
                    model, env, states[i], width=beam_w, max_steps=max(2 * d + 20, 60)
                )
                if ok:
                    sn += 1
                    tot += ln_b
            line += f"  beam{beam_w}(n={n}) rate={sn / n:.2f} len={tot / sn if sn else -1:.1f}"
        lines.append(line)
    model.train()
    return "\n    ".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["rubik2", "megaminx", "megaminx2f", "megaminx_h", "ihes"],
                   required=True)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--res-blocks", type=int, default=3, help="two-linear residual blocks")
    p.add_argument("--encoding", choices=["embedding", "onehot"], default="embedding")
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--nmax", type=int, default=12, help="training trajectory length")
    p.add_argument("--iters", type=int, default=500000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=100.0)
    p.add_argument("--reg-coef", type=float, default=0.01)
    p.add_argument("--reg-mode", choices=["flow", "logflow"], default="flow")
    p.add_argument("--eps-explore", type=float, default=0.0)
    p.add_argument("--eps-end", type=float, default=None,
                   help="anneal eps_explore -> eps_end linearly over eps-anneal-frac")
    p.add_argument("--eps-anneal-frac", type=float, default=0.5,
                   help="fraction of iters over which eps decays to eps-end")
    p.add_argument("--warmstart-bfs-depth", type=int, default=0,
                   help="BFS depth for backward-head geodesic warm-start (0=off)")
    p.add_argument("--warmstart-iters", type=int, default=20000)
    p.add_argument("--warmstart-cap", type=int, default=200000,
                   help="max states per BFS depth in the warm-start dataset")
    p.add_argument("--warmstart-batch", type=int, default=4096)
    p.add_argument("--pb-anchor", type=float, default=0.0,
                   help="KL(warmstart P_B || current P_B) penalty coef; protects the"
                        " geodesic backward policy while P_F/flows calibrate")
    p.add_argument("--pb-anchor-frac", type=float, default=0.4,
                   help="fraction of iters over which the anchor decays to 0")
    p.add_argument("--learn-logz", action="store_true",
                   help="learn logZ as a scalar (init at true value, 10x lr) instead"
                        " of fixing it; relaxes the unsatisfiable-residual squeeze")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=25000)
    p.add_argument("--ckpt-every", type=int, default=25000)
    p.add_argument("--eval-beam", type=int, default=256)
    p.add_argument("--eval-beam-n", type=int, default=100, help="states per beam eval")
    p.add_argument("--eval-max-len", type=int, default=20)
    p.add_argument("--eval-walk-n", type=int, default=50)
    p.add_argument("--test-npy", type=str, default="", help="rubik2 test set path")
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--load", type=str, default="", help="checkpoint to resume from")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = build_task_env(args.task).to(device)
    model = GFNPolicyNet(
        state_size=env.state_size,
        num_classes=env.num_classes,
        n_actions=env.n_actions,
        hidden=args.hidden,
        num_res_blocks=args.res_blocks,
        encoding=args.encoding,
        embed_dim=args.embed_dim,
    ).to(device)

    start_iter = 0
    resume_ckpt = None
    if args.load:
        resume_ckpt = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        start_iter = int(resume_ckpt.get("iter", 0))
        print(f"loaded checkpoint {args.load} at iter {start_iter}")

    log_z_param = None
    param_groups = [{"params": model.parameters()}]
    if args.learn_logz:
        init_lz = float(env.true_log_z)
        if resume_ckpt is not None and "log_z_param" in resume_ckpt:
            init_lz = float(resume_ckpt["log_z_param"])
        log_z_param = torch.nn.Parameter(torch.tensor(init_lz, device=device))
        param_groups.append({"params": [log_z_param], "lr": args.lr * 10})
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    if resume_ckpt is not None and "opt" in resume_ckpt and not args.eval_only:
        try:
            opt.load_state_dict(resume_ckpt["opt"])
            print("restored optimizer state")
        except ValueError:
            print("optimizer state incompatible (param groups changed); fresh optimizer")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Rule 13: echo the resolved config before anything else.
    cfg = {**vars(args), "device": device, "params": count_params(model),
           "true_log_z": env.true_log_z, "n_actions": env.n_actions,
           "state_size": env.state_size}
    print("resolved config:", json.dumps(cfg, sort_keys=True))
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    # Eval assets
    if args.task == "rubik2":
        if not args.test_npy:
            raise SystemExit("rubik2 requires --test-npy")
        test_states = torch.tensor(
            np.load(args.test_npy).astype(np.int64), device=device
        )
        print(f"rubik2 test set: {tuple(test_states.shape)}")
    else:
        t0 = time.time()
        if args.task == "megaminx2f":
            bfs_depth, walk_depths, beam_depths = 6, [4, 8, 12, 16, 20, 30], (12, 20, 30)
        elif args.task == "megaminx_h":
            bfs_depth, walk_depths, beam_depths = 4, [6, 10, 15, 20, 30, 45], (15, 30, 45)
        elif args.task == "ihes":
            bfs_depth, walk_depths, beam_depths = 5, [8, 14, 20, 25, 28, 32], (20, 25, 28)
        else:
            bfs_depth, walk_depths, beam_depths = 4, [6, 10, 15, 20, 30, 40, 60], (20, 40, 60)
        exact_sets = bfs_exact_states(env, max_depth=bfs_depth,
                                      sample_per_depth=args.eval_walk_n, seed=args.seed)
        walk_sets = random_walk_states(env, depths=walk_depths,
                                       n_per_depth=args.eval_walk_n, seed=args.seed)
        sizes = {d: int(v.shape[0]) for d, v in exact_sets.items()}
        print(f"eval assets: exact {sizes} + walks x{args.eval_walk_n}"
              f" ({time.time() - t0:.1f}s)")

    def run_eval() -> None:
        t0 = time.time()
        if args.task == "rubik2":
            msg = eval_rubik2(model, env, test_states, args.eval_beam,
                              args.eval_max_len, args.eval_beam_n)
        else:
            msg = eval_megaminx(model, env, walk_sets, exact_sets,
                                beam_w=8, beam_n=10, beam_depths=beam_depths)
        print(f"[eval @ {it}] ({time.time() - t0:.0f}s)\n    {msg}", flush=True)

    if args.eval_only:
        it = start_iter
        run_eval()
        return

    # ---- Stage 0: geodesic warm-start of the backward (solver) head ----
    if args.warmstart_bfs_depth > 0 and start_iter == 0:
        from cayley.gfn_pathfinding import bfs_geodesic_dataset
        t0 = time.time()
        ws_states, ws_moves = bfs_geodesic_dataset(
            env, args.warmstart_bfs_depth, cap_per_depth=args.warmstart_cap, seed=args.seed)
        print(f"warmstart dataset: {ws_states.shape[0]} states to depth "
              f"{args.warmstart_bfs_depth} ({time.time()-t0:.0f}s)", flush=True)
        model.train()
        ws_opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        n = ws_states.shape[0]
        for wi in range(args.warmstart_iters):
            idx = torch.randint(0, n, (args.warmstart_batch,), device=device)
            bwd_logits, _ = model(ws_states[idx])
            ce = torch.nn.functional.cross_entropy(bwd_logits, ws_moves[idx])
            ws_opt.zero_grad(set_to_none=True)
            ce.backward()
            ws_opt.step()
            if (wi + 1) % max(1, args.warmstart_iters // 5) == 0:
                with torch.no_grad():
                    acc = (bwd_logits.argmax(-1) == ws_moves[idx]).float().mean()
                print(f"  warmstart iter {wi+1} ce={float(ce):.4f} top1={float(acc):.3f}",
                      flush=True)
        torch.save({"model": model.state_dict(), "iter": 0, "config": cfg},
                   save_dir / "ckpt_warmstart.pt")

    # frozen reference of the warm-started backward policy, for the KL anchor
    ref_model = None
    if args.pb_anchor > 0.0 and args.warmstart_bfs_depth > 0:
        import copy
        ref_model = copy.deepcopy(model).eval()
        for p_ in ref_model.parameters():
            p_.requires_grad_(False)
        print(f"pb-anchor active: coef {args.pb_anchor}, decay over "
              f"{args.pb_anchor_frac} of run", flush=True)

    model.train()
    t_start = time.time()
    t_last = t_start
    for it in range(start_iter, args.iters):
        # eps annealing: eps_explore -> eps_end linearly over eps-anneal-frac
        if args.eps_end is not None:
            frac = min(1.0, it / max(1.0, args.eps_anneal_frac * args.iters))
            cur_eps = args.eps_explore + frac * (args.eps_end - args.eps_explore)
        else:
            cur_eps = args.eps_explore
        states, actions = sample_forward_trajectories(
            model, env, args.batch_size, args.nmax, eps_explore=cur_eps
        )
        loss, metrics = regularized_tb_loss(
            model, env, states, actions, reg_coef=args.reg_coef, reg_mode=args.reg_mode,
            log_z=log_z_param,
        )
        anchor_val = 0.0
        if ref_model is not None:
            a_frac = min(1.0, it / max(1.0, args.pb_anchor_frac * args.iters))
            a_coef = args.pb_anchor * (1.0 - a_frac)
            if a_coef > 0.0:
                flat = states.reshape(-1, env.state_size)
                cur_bwd, _ = model(flat)
                with torch.no_grad():
                    ref_bwd, _ = ref_model(flat)
                ref_logp = torch.log_softmax(ref_bwd, dim=-1)
                cur_logp = torch.log_softmax(cur_bwd, dim=-1)
                # KL(ref || cur) = sum ref_p * (ref_logp - cur_logp)
                kl = (ref_logp.exp() * (ref_logp - cur_logp)).sum(-1).mean()
                anchor = a_coef * kl
                loss = loss + anchor
                anchor_val = float(anchor.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        if (it + 1) % args.log_every == 0:
            now = time.time()
            ips = args.log_every / (now - t_last)
            t_last = now
            lz_str = f"  logZ={float(log_z_param):.2f}" if log_z_param is not None else ""
            eps_str = f"  eps={cur_eps:.3f}" if args.eps_end is not None else ""
            anc_str = f"  anchor={anchor_val:.3g}" if ref_model is not None else ""
            print(
                f"iter {it + 1}  tb={fmt(metrics['tb'])}  reg={fmt(metrics['reg'])}"
                f"  rms={fmt(metrics['residual_rms'])}"
                f"  logF_d1={fmt(metrics['log_flow_d1'])}"
                f"  logF_last={fmt(metrics['log_flow_last'])}"
                f"{lz_str}{eps_str}{anc_str}  it_per_s={ips:.2f}",
                flush=True,
            )
        if (it + 1) % args.eval_every == 0:
            run_eval()
        if (it + 1) % args.ckpt_every == 0 or (it + 1) == args.iters:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "iter": it + 1, "config": cfg}
            if log_z_param is not None:
                ck["log_z_param"] = float(log_z_param)
            torch.save(ck, save_dir / "ckpt_latest.pt")
            # immutable snapshot too: the learn-logz dynamics have a transient
            # broad-coverage peak worth recovering post-hoc (2f lesson)
            torch.save(ck, save_dir / f"ckpt_{it + 1:06d}.pt")

    it = args.iters
    run_eval()
    torch.save({"model": model.state_dict(), "iter": args.iters, "config": cfg},
               save_dir / "ckpt_final.pt")
    print(f"done in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
