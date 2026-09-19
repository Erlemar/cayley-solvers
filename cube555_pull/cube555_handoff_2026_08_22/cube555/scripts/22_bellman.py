"""Q-Bellman refinement, WARM-STARTED. A merge member, not a replacement for Phase 1.

    python cube555/scripts/22_bellman.py --init cube555/models/q555_a/epoch_011718.pt \
        --output cube555/models/q555_a_bell --steps 6000

WHY WARM-STARTED MATTERS, and why the same idea is listed as dead elsewhere. On 444,
Q-Bellman from a NEAR-SCRATCH init failed: the bootstrap `1 + min_a' Q_target(child)`
inherits the target net's flatness at depth, so there is nothing to propagate outward
from the shallow ball. From a TRAINED init the target net already discriminates at depth
and the recursion has signal to refine -- and that arm scored 2931 on 54 held-out pids,
the only arm in the whole project to beat the floor on a pid. Same code, opposite verdict,
decided entirely by the init.

It refines the EXISTING objective rather than adding a label source, which is the pattern
that keeps winning here; adding label sources (path labels, sorted profiles, deeper
anchors) is the pattern that keeps losing.

THE COLLAPSE DIAGNOSTIC IS `E[target]`, NOT THE LOSS. Q == 0 is also a fixed point of the
recursion, so a falling loss is consistent with total collapse. On 444 a healthy run held
E[target] at 12.8 -> 14.5 over 6000 steps. Watch that line; if it trends toward 0, stop.
Exact anchors in every batch are what hold the absolute level (rule 9).
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube555.models import build_model, load_model  # noqa: E402
from cube555.puzzle import Cube555  # noqa: E402
from cube555.qtrain import ExactAnchors, PivotBuffer, SparseQSampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, type=Path, help="TRAINED checkpoint")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--n-anchor", type=int, default=256)
    ap.add_argument("--anchor-weight", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--target-refresh", type=int, default=500)
    ap.add_argument("--k-max", type=int, default=80)
    ap.add_argument("--anchor-depth", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-every", type=int, default=0,
                    help="checkpoint every N steps so intermediate points can be gated")
    ap.add_argument("--seed", type=int, default=555)
    args = ap.parse_args()

    dev = args.device
    args.output.mkdir(parents=True, exist_ok=True)
    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    names = list(puz.move_names)
    perms = torch.tensor(
        [puz.generators[n] for n in names], dtype=torch.int64, device=dev
    )
    inv_idx = torch.tensor(
        [names.index(puz.inverse_name(n)) for n in names], dtype=torch.int64, device=dev
    )
    solved = torch.tensor(puz.solved_state, dtype=torch.uint8, device=dev)
    n_act = len(names)

    model = load_model(args.init, device=dev, dtype=torch.float32)
    model.train()
    model.return_value = False
    target = copy.deepcopy(model).eval()
    for p in target.parameters():
        p.requires_grad_(False)
    print(f"warm start: {args.init}  {model.num_parameters()/1e6:.1f}M params")

    anchors = ExactAnchors(PROJECT / "data" / f"anchors_d{args.anchor_depth}.pt", dev)
    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    sg = torch.Generator(device=dev)
    sg.manual_seed(args.seed + 1)
    pivots = PivotBuffer(
        SparseQSampler(perms, inv_idx, solved, 2, args.k_max, 0.5, sg), 1 << 15
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.0, fused=True
    )

    solved_h = solved.long()
    t0 = time.time()
    for step in range(args.steps):
        if step % args.target_refresh == 0:
            target.load_state_dict(model.state_dict())
        s, _, _, _ = pivots.take(args.batch)
        b = s.shape[0]
        # every child of every rw state: (b, 30, 150)
        ch = torch.gather(
            s[:, None, :].expand(b, n_act, 150).long(),
            2,
            perms[None].expand(b, n_act, 150),
        )
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            qc = target(ch.reshape(b * n_act, 150)).float().reshape(b, n_act, n_act)
        # 1 + min_a' Q_target(child, a'), clamped at 0 for children that ARE solved
        is_solved = (ch == solved_h).all(dim=2)
        tgt = 1.0 + qc.min(dim=2).values
        tgt = torch.where(is_solved, torch.zeros_like(tgt), tgt)

        a_s, a_q, _ = anchors.sample(args.n_anchor, g)
        x = torch.cat([s, a_s], dim=0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        out = out.float()
        bell = out[:b].sub(tgt).pow(2).mean()
        anc = out[b:].sub(a_q).pow(2).mean()
        loss = bell + args.anchor_weight * anc

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if args.save_every and step and step % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": model.get_model_config(),
                    "epoch": -2,
                    "init": str(args.init),
                    "bellman_step": step,
                },
                args.output / f"bellman_{step:06d}.pt",
            )
        if step % 250 == 0 or step == args.steps - 1:
            print(
                f"step {step:6d} | loss {float(loss):7.4f} | bellman {float(bell):7.4f} "
                f"| anchor {float(anc):7.4f} | E[target] {float(tgt.mean()):7.3f} "
                f"<- watch this, not the loss | {time.time()-t0:.0f}s",
                flush=True,
            )
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.get_model_config(),
            "epoch": -2,
            "init": str(args.init),
        },
        args.output / "bellman.pt",
    )
    print(f"wrote {args.output/'bellman.pt'} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
