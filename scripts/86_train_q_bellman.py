"""Warm-started Q-Bellman refinement for the IHES PieceTransformer (the 444 "s3" stage).

    python scripts/86_train_q_bellman.py --init models/ihes_tf_a/epoch_1000.pt \
        --output models/ihes_tf_a_s3 --steps 6000 --save-every 500

Recipe (cube444 `cube444_s3_bellman_recipe_2026-08-09.md`, cube555 `22_bellman.py`):

    target      Q(s,a) = 0 if apply(s,a) is solved, else 1 + min_a' Q_target(apply(s,a))
    target net  hard copy of the online net, refreshed every --target-refresh steps
    states      --batch per step: (1 - path_frac) random-walk states + path_frac path-bank
                states (random frame of 48 x random inverse); their sparse labels are unused
    anchors     +--n-anchor exact d<=6 rows (all 18 columns, scripts/82), weight 2.0
    value head  V(s) target = 1 + min_a target(s,a) on Bellman rows, exact depth on anchors
    loss        dense MSE on all 18 columns (no Huber, no reweighting), fp32; bf16 forward
    optimizer   AdamW(betas 0.9/0.95, wd 3e-3), lr 2e-5, linear warmup then flat, clip 1.0

REQUIRED: --init must be a TRAINED stage-1 checkpoint. From near-scratch the recursion
inherits the target net's flatness and nothing propagates (measured on 444 and 555).

WATCH `E[target]`, NOT THE LOSS. Q == 0 is also a fixed point of the recursion; the exact
anchors are what pin the level. A healthy run holds E[target] near the mean depth of the
sampled states (444: 12.8 -> 14.3, drifting slightly up). If it slides toward 0, stop.
cube555: 2k Bellman steps beat 20k, so save often and beam-gate every checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from cayley.puzzle import PictureCube  # noqa: E402
from tetraminx.models import model_from_config  # noqa: E402


def _t51():
    """Reuse the stage-1 trainer's classes (Symmetries, SparseQSampler, BakedAnchors)."""
    path = PROJECT / "tetraminx" / "scripts" / "51_train_sparse_q.py"
    spec = importlib.util.spec_from_file_location("t51_sparse_q", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class EndpointWalks:
    """States at the END of non-backtracking walks, k ~ U[k_min, k_max] (test-like states)."""

    def __init__(self, gens, inv_idx, solved, k_min, k_max, g):
        self.gens, self.inv, self.solved = gens, inv_idx, solved
        self.k_min, self.k_max, self.g = k_min, k_max, g
        self.A = gens.size(0)

    def sample(self, n):
        dev = self.gens.device
        lengths = torch.randint(self.k_min, self.k_max + 1, (n,), generator=self.g, device=dev)
        s = self.solved.unsqueeze(0).expand(n, -1).clone()
        last = torch.full((n,), -1, dtype=torch.int64, device=dev)
        for step in range(self.k_max):
            if step == 0:
                mv = torch.randint(self.A, (n,), generator=self.g, device=dev)
            else:
                c = torch.randint(self.A - 1, (n,), generator=self.g, device=dev)
                forb = self.inv[last.clamp_min(0)]
                mv = c + c.ge(forb).long()
            nxt = torch.gather(s, 1, self.gens[mv])
            act = lengths > step
            s = torch.where(act.unsqueeze(1), nxt, s)
            last = torch.where(act, mv, last)
        return s, lengths


class PathBank:
    """Solution-path states under a random frame (48 conjugations) and random inverse."""

    def __init__(self, path: Path, sym, device, g):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.states = blob["states"].to(device)          # uint8 (N, 72)
        self.togo = blob["togo"].to(device)
        self.sym, self.g = sym, g
        print(f"  path bank: {self.states.size(0):,} states ({blob['meta'].get('floor')}), "
              f"held out {len(blob['meta'].get('held_out', []))} pids", flush=True)

    def sample(self, n):
        dev = self.states.device
        idx = torch.randint(self.states.size(0), (n,), generator=self.g, device=dev)
        s = self.states.index_select(0, idx).long()
        inv = torch.rand((n,), generator=self.g, device=dev) < 0.5
        s = torch.where(inv.unsqueeze(1), torch.argsort(s, dim=1), s)   # group inverse
        k = torch.randint(self.sym.n_sym, (n,), generator=self.g, device=dev)
        return self.sym.conjugate(s, k), self.togo.index_select(0, idx)


def main() -> int:
    ap = argparse.ArgumentParser(description="Warm-started Q-Bellman for the IHES transformer.")
    ap.add_argument("--init", required=True, type=Path, help="TRAINED stage-1 checkpoint")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=768)
    ap.add_argument("--path-frac", type=float, default=0.5)
    ap.add_argument("--rw-mode", choices=("pivot", "end"), default="pivot",
                    help="pivot = stage-1 sampler pivots (444 s3); end = walk endpoints (s3-rw30)")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=23)
    ap.add_argument("--pivot-tilt", type=float, default=0.0)
    ap.add_argument("--n-anchor", type=int, default=256)
    ap.add_argument("--anchor-weight", type=float, default=2.0)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--anchors", type=Path, default=PROJECT / "data" / "ihes_q_anchors_d6.pt")
    ap.add_argument("--path-bank", type=Path, default=PROJECT / "data" / "ihes_path_bank.pt")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=3e-3)
    ap.add_argument("--target-refresh", type=int, default=500)
    ap.add_argument("--child-chunk", type=int, default=16384)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", type=Path, default=None, help="a step_XXXXX.pt from this script")
    args = ap.parse_args()

    dev = args.device
    args.output.mkdir(parents=True, exist_ok=True)
    t51 = _t51()
    puz = PictureCube.load(args.data_dir / "puzzle_info.json")
    puz.verify_inverse_pairs()
    names = list(puz.move_names)
    A = len(names)
    gens = torch.tensor([puz.generators[n] for n in names], dtype=torch.int64, device=dev)
    inv_idx = torch.tensor([names.index(puz.inverse_name(n)) for n in names],
                           dtype=torch.int64, device=dev)
    solved = torch.tensor(puz.solved_state, dtype=torch.int64, device=dev)
    S = solved.numel()

    ck = torch.load(args.init if args.resume is None else args.resume,
                    map_location=dev, weights_only=False)
    mcfg = dict(ck["model_config"])
    lp = mcfg.get("layout_path")
    if lp and not Path(lp).is_absolute() and not Path(lp).exists():
        mcfg["layout_path"] = str(PROJECT / lp)
    model = model_from_config(mcfg).to(dev)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()})
    if not getattr(model, "has_value_head", False) and args.value_weight > 0:
        print("  [note] checkpoint has no value head -> value term disabled", flush=True)
        args.value_weight = 0.0
    model.train()
    model.return_value = args.value_weight > 0
    target = copy.deepcopy(model).eval()
    target.return_value = False
    for p in target.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())

    print("=" * 72, flush=True)
    print(f"Q-Bellman | init {args.init} (epoch {ck.get('epoch')}) | {n_params:,} params | {dev}",
          flush=True)
    print(f"config: {json.dumps({k: str(v) for k, v in vars(args).items()}, sort_keys=True)}",
          flush=True)

    g = torch.Generator(device=dev)
    g.manual_seed(args.seed)
    sym = t51.Symmetries(args.data_dir, dev, prefix="cube")
    sym.verify(gens)
    print(f"  symmetry transport verified over {sym.n_sym} frames x {sym.n_actions} actions",
          flush=True)
    anchors = t51.BakedAnchors(args.anchors, dev, resident=True)
    n_path = int(round(args.batch * args.path_frac))
    n_rw = args.batch - n_path
    paths = PathBank(args.path_bank, sym, dev, g) if n_path > 0 else None
    if args.rw_mode == "pivot":
        rw = t51.SparseQSampler(gens, inv_idx, solved, args.k_min, args.k_max, args.pivot_tilt, g)
    else:
        rw = EndpointWalks(gens, inv_idx, solved, max(1, args.k_min), args.k_max, g)
    print(f"  batch: {n_rw} random-walk ({args.rw_mode}, k {args.k_min}-{args.k_max}) + "
          f"{n_path} path + {args.n_anchor} anchor rows", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    start = 0
    if args.resume is not None:
        opt.load_state_dict(ck["optimizer"])
        start = int(ck["bellman_step"]) + 1
        print(f"  resumed at step {start}", flush=True)

    def save(step: int, name: str):
        payload = {
            "epoch": ck.get("epoch") if args.resume is None else ck.get("epoch"),
            "bellman_step": step,
            "init": str(args.init),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": opt.state_dict(),
            "model_config": dict(ck["model_config"]),
            "bellman_config": {k: str(v) for k, v in vars(args).items()},
        }
        torch.save(payload, args.output / name)

    t0 = time.time()
    acc = torch.zeros(6, device=dev)
    n_acc = 0
    for step in range(start, args.steps):
        if (step - start) % args.target_refresh == 0 or step % args.target_refresh == 0:
            target.load_state_dict(model.state_dict())
        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for pg in opt.param_groups:
            pg["lr"] = lr

        parts = []
        if n_rw:
            if args.rw_mode == "pivot":
                s_rw = rw.sample(n_rw)[0]
            else:
                s_rw = rw.sample(n_rw)[0]
            parts.append(s_rw)
        if n_path:
            parts.append(paths.sample(n_path)[0])
        s = torch.cat(parts, 0)                                    # (b, S) int64
        b = s.size(0)
        ch = s[:, gens].reshape(b * A, S)                          # (b*A, S)
        with torch.no_grad():
            qc = torch.empty((b * A, A), device=dev)
            for lo in range(0, b * A, args.child_chunk):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                    qc[lo:lo + args.child_chunk] = target(ch[lo:lo + args.child_chunk]).float()
            is_solved = (ch == solved).all(dim=1)
            tgt = (1.0 + qc.min(dim=1).values).clamp_min(0.0)
            tgt = torch.where(is_solved, torch.zeros_like(tgt), tgt).view(b, A)
            v_tgt = 1.0 + tgt.min(dim=1).values

        a_s, a_q, a_d = anchors.sample(args.n_anchor, g)
        x = torch.cat([s, a_s], 0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
            out = model(x)
        if model.return_value:
            q_all, v_all = out
            v_all = v_all.float()
            value = v_all[:b].sub(v_tgt).pow(2).mean() + v_all[b:].sub(a_d).pow(2).mean()
        else:
            q_all, value = out, torch.zeros((), device=dev)
        q_all = q_all.float()
        bell = q_all[:b].sub(tgt).pow(2).mean()
        anc = q_all[b:].sub(a_q).pow(2).mean()
        loss = bell + args.anchor_weight * anc + args.value_weight * value

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        acc += torch.stack([loss.detach(), bell.detach(), anc.detach(), value.detach(),
                            tgt[:n_rw].mean() if n_rw else tgt.new_zeros(()),
                            tgt[n_rw:].mean() if n_path else tgt.new_zeros(())])
        n_acc += 1
        if step % args.log_every == 0 or step == args.steps - 1:
            m = (acc / n_acc).tolist()
            print(f"step {step:6d} | loss {m[0]:7.4f} bellman {m[1]:7.4f} anchor {m[2]:7.4f} "
                  f"value {m[3]:7.4f} | E[target] rw {m[4]:6.2f} path {m[5]:6.2f} "
                  f"<- watch, not the loss | lr {lr:.1e} | {time.time() - t0:.0f}s", flush=True)
            acc.zero_()
            n_acc = 0
        if args.save_every and step > 0 and step % args.save_every == 0:
            save(step, f"step_{step:05d}.pt")
    save(args.steps - 1, f"step_{args.steps:05d}.pt")
    print(f"done: {args.steps - start} steps in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
