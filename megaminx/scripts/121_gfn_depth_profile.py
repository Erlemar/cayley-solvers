"""Depth profile for a trained GFN-pathfinding policy (Gate 2 of the plan in
megaminx/gfn_pathfinding_analysis_2026-07-17.md).

The saturation question transplanted to policy space: does P_B stay DECISIVE as
scramble depth grows, or does it go uniform past d~30 (the V-saturation analog)?

Per walk depth, reports:
  - P_B sharpness at the scrambled states: mean entropy (nats; uniform over 24
    actions = 3.178), mean top-1 probability, mean top1-top2 log-prob gap
  - greedy rollout solve rate + mean length
  - policy-sample solve rate
  - policy-beam solve rate + mean length at each requested width

Usage:
  .venv/Scripts/python.exe megaminx/scripts/121_gfn_depth_profile.py \
      --ckpt megaminx/models/m_gfn_v0/ckpt_latest.pt \
      --depths 5,10,15,20,25,30,40,50,60,80 --n-per-depth 50 \
      --beams 8,64,512 --beam-n 20

ASCII-only output (project Rule 24).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "megaminx" / "src"))

from cayley.gfn_pathfinding import (  # noqa: E402
    GFNPolicyNet,
    policy_beam_solve,
    random_walk_states,
    rollout_solve,
)


def load_model_env(ckpt_path: str, device: str):
    sys.path.insert(0, str(ROOT / "megaminx" / "scripts"))
    import importlib

    trainer = importlib.import_module("120_train_gfn_pathfinding")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    env = trainer.build_task_env(cfg["task"]).to(device)
    model = GFNPolicyNet(
        state_size=env.state_size,
        num_classes=env.num_classes,
        n_actions=env.n_actions,
        hidden=cfg["hidden"],
        num_res_blocks=cfg["res_blocks"],
        encoding=cfg["encoding"],
        embed_dim=cfg["embed_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path}: task={cfg['task']} iter={ckpt.get('iter')}"
          f" hidden={cfg['hidden']} blocks={cfg['res_blocks']}")
    return model, env, cfg


@torch.no_grad()
def pb_sharpness(model, states) -> tuple[float, float, float, float, float]:
    bwd_logits, fwd_logits = model.forward_chunked(states)
    logp = torch.log_softmax(bwd_logits.float(), dim=-1)
    p = logp.exp()
    entropy = float((-(p * logp).sum(-1)).mean())
    top2 = logp.topk(2, dim=-1).values
    gap = float((top2[:, 0] - top2[:, 1]).mean())
    top1p = float(p.max(dim=-1).values.mean())
    # forward-policy diagnostics: entropy over non-stop actions + mean log-flow
    logf_all = torch.log_softmax(fwd_logits.float(), dim=-1)
    logf_act = torch.log_softmax(fwd_logits[:, :-1].float(), dim=-1)
    pf = logf_act.exp()
    pf_entropy = float((-(pf * logf_act).sum(-1)).mean())
    log_flow = float((-logf_all[:, -1]).mean())
    return entropy, top1p, gap, pf_entropy, log_flow


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--depths", default="5,10,15,20,25,30,40,50,60,80")
    p.add_argument("--n-per-depth", type=int, default=50)
    p.add_argument("--beams", default="8,64", help="comma-separated beam widths")
    p.add_argument("--beam-n", type=int, default=20, help="states per beam width per depth")
    p.add_argument("--max-steps-slack", type=int, default=40,
                   help="rollout cap = 2*depth + slack")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, env, cfg = load_model_env(args.ckpt, device)

    depths = [int(x) for x in args.depths.split(",")]
    beams = [int(x) for x in args.beams.split(",") if x]
    walks = random_walk_states(env, depths=depths, n_per_depth=args.n_per_depth,
                               seed=args.seed)
    uniform_h = math.log(env.n_actions)
    print(f"uniform entropy = {uniform_h:.3f} nats over {env.n_actions} actions")

    header = "depth |  H(P_B) top1p  gap | H(P_F) logF | greedy rate/len | sample rate"
    for w in beams:
        header += f" | beam{w} rate/len"
    print(header)
    for d in depths:
        states = walks[d]
        max_steps = 2 * d + args.max_steps_slack
        ent, top1p, gap, pf_ent, log_flow = pb_sharpness(model, states)
        g_done, g_len = rollout_solve(model, env, states, max_steps=max_steps, greedy=True)
        s_done, _ = rollout_solve(model, env, states, max_steps=max_steps, greedy=False)
        g_ml = float(g_len[g_done].float().mean()) if bool(g_done.any()) else -1.0
        row = (f"{d:5d} | {ent:6.3f} {top1p:.3f} {gap:5.2f}"
               f" | {pf_ent:6.3f} {log_flow:6.1f}"
               f" | {float(g_done.float().mean()):.2f} / {g_ml:6.1f}"
               f" | {float(s_done.float().mean()):.2f}      ")
        for w in beams:
            n = min(args.beam_n, states.shape[0])
            sn, tot = 0, 0
            t0 = time.time()
            for i in range(n):
                ok, ln, _ = policy_beam_solve(model, env, states[i], width=w,
                                              max_steps=max_steps)
                if ok:
                    sn += 1
                    tot += ln
            row += f" | {sn / n:.2f} / {tot / sn if sn else -1:6.1f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
