"""Fine-tune a Tetraminx Q checkpoint on frontier-regret supervision.

The primary loss is listwise: match a soft distribution over actions induced by
bounded-teacher cost.  A small absolute-Q loss preserves useful calibration, and
random-walk distillation against the untouched source checkpoint prevents the
frontier pilot from erasing broad state-space competence.
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))

from tetraminx.models import model_from_config
from tetraminx.puzzle import Tetraminx


def load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config", {}))
    if "arch" not in cfg:
        raise ValueError(f"{path} is not a tetraminx.models architecture checkpoint")
    model = model_from_config(cfg).to(device)
    state = {k.removeprefix("_orig_mod."): v
             for k, v in ckpt.get("state_dict", ckpt).items()}
    model.load_state_dict(state)
    model.return_value = False
    return model, ckpt, cfg


def forward_qv(model, states: torch.Tensor, want_value: bool):
    base = getattr(model, "_orig_mod", model)
    old = bool(getattr(base, "return_value", False))
    base.return_value = want_value
    try:
        out = model(states)
    finally:
        base.return_value = old
    if isinstance(out, tuple):
        return out
    return out, None


class RandomWalkSampler:
    def __init__(self, puzzle: Tetraminx, device: str, seed: int):
        self.device = device
        self.gen = torch.tensor(
            [puzzle.generators[name] for name in puzzle.move_names],
            dtype=torch.int64, device=device)
        self.solved = torch.tensor(puzzle.solved_state, dtype=torch.int64, device=device)
        names = list(puzzle.move_names)
        name_to_idx = {name: i for i, name in enumerate(names)}
        self.inverse = torch.tensor([
            name_to_idx[name[1:] if name.startswith("-") else "-" + name]
            for name in names], dtype=torch.int64, device=device)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)

    def sample(self, batch: int, k_min: int, k_max: int) -> torch.Tensor:
        depths = torch.randint(
            k_min, k_max + 1, (batch,), generator=self.rng, device=self.device)
        states = self.solved.unsqueeze(0).expand(batch, -1).clone()
        previous = torch.full((batch,), -1, dtype=torch.int64, device=self.device)
        for step in range(k_max):
            live = depths > step
            if not bool(live.any()):
                break
            move = torch.randint(
                self.gen.size(0), (batch,), generator=self.rng, device=self.device)
            bad = live & (previous >= 0) & (move == self.inverse[previous.clamp_min(0)])
            move[bad] = (move[bad] + 1) % self.gen.size(0)
            idx = torch.nonzero(live, as_tuple=True)[0]
            states[idx] = torch.gather(states[idx], 1, self.gen[move[idx]])
            previous[live] = move[live]
        return states


def transform_batch(states, costs, sym, sym_inv, relabel, rng):
    """Apply one random spatial conjugation per row and transport action costs."""
    batch = states.size(0)
    k = torch.randint(sym.size(0), (batch,), generator=rng, device=states.device)
    gathered = torch.gather(states, 1, sym.index_select(0, k))
    out_states = torch.gather(sym_inv.index_select(0, k), 1, gathered)
    # Original action a becomes sigma^-1(a), hence c_new[b] = c_old[sigma(b)].
    out_costs = torch.gather(costs, 1, relabel.index_select(0, k))
    return out_states, out_costs, k


def policy_metrics(pred: torch.Tensor, costs: torch.Tensor,
                   teacher_best: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    known = torch.isfinite(costs)
    if teacher_best is None:
        masked_cost = costs.masked_fill(~known, float("inf"))
        best_cost = masked_cost.min(dim=1).values
        teacher_best = known & (costs <= best_cost.unsqueeze(1) + 1.0e-5)
        pred_order = torch.argsort(pred.masked_fill(~known, float("inf")), dim=1)
    else:
        teacher_best = teacher_best.bool()
        best_cost = costs.masked_fill(~teacher_best, float("inf")).min(dim=1).values
        pred_order = torch.argsort(pred, dim=1)
    chosen = pred_order[:, 0]
    row = torch.arange(pred.size(0), device=pred.device)
    # Verified full-beam datasets only know a set of successful root actions;
    # their "regret" is therefore the miss rate. Bounded-lookahead datasets have
    # dense relative costs and retain the ordinary cost-regret metric.
    if bool(torch.isfinite(costs[row, chosen]).all()):
        regret = costs[row, chosen] - best_cost
    else:
        regret = (~teacher_best[row, chosen]).float()
    return {
        "top1": teacher_best[row, chosen].float().mean(),
        "top2": teacher_best.gather(1, pred_order[:, :2]).any(dim=1).float().mean(),
        "top4": teacher_best.gather(1, pred_order[:, :4]).any(dim=1).float().mean(),
        "regret": regret.mean(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--weight-decay", type=float, default=1.0e-4)
    ap.add_argument("--teacher-temperature", type=float, default=0.5)
    ap.add_argument("--policy-temperature", type=float, default=1.0)
    ap.add_argument("--absolute-weight", type=float, default=0.05)
    ap.add_argument("--preserve-weight", type=float, default=1.0)
    ap.add_argument("--preserve-value-weight", type=float, default=0.25)
    ap.add_argument("--preserve-batch", type=int, default=256)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=40)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--symmetry", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--train-head-only", action="store_true",
                    help="freeze the representation and fit only the 24-way policy/Q head")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
    student, source_ckpt, model_cfg = load_model(args.checkpoint, args.device)
    reference, _, _ = load_model(args.checkpoint, args.device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    has_value = bool(getattr(student, "has_value_head", False))
    if args.train_head_only:
        if not hasattr(student, "head"):
            raise ValueError("--train-head-only requires a model with a .head module")
        for parameter in student.parameters():
            parameter.requires_grad_(False)
        for parameter in student.head.parameters():
            parameter.requires_grad_(True)
        print(f"head-only policy fit: {sum(p.numel() for p in student.head.parameters()):,} "
              "trainable parameters", flush=True)

    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    states = blob["states"].contiguous()
    costs = blob["teacher_costs"].float().contiguous()
    teacher_best_mask = blob.get("teacher_best_mask")
    verified_targets = int(blob.get("meta", {}).get("verified_teacher_beam", 0)) > 0
    if teacher_best_mask is not None:
        teacher_best_mask = teacher_best_mask.bool().contiguous()
    pids = blob["pids"].long()
    unique_pids = torch.unique(pids).tolist()
    if len(unique_pids) < 2:
        # Smoke datasets still need executable validation; split rows rather than
        # pretending a one-pid split measures generalisation.
        perm = torch.randperm(states.size(0), generator=torch.Generator().manual_seed(args.seed))
        n_val = max(1, int(round(states.size(0) * args.val_fraction)))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
    else:
        rng_np = np.random.default_rng(args.seed)
        shuffled = np.asarray(unique_pids, dtype=np.int64)
        rng_np.shuffle(shuffled)
        n_val_pid = max(1, int(round(len(shuffled) * args.val_fraction)))
        val_pids = torch.tensor(shuffled[:n_val_pid], dtype=pids.dtype)
        val_mask = torch.isin(pids, val_pids)
        val_idx = torch.nonzero(val_mask, as_tuple=True)[0]
        train_idx = torch.nonzero(~val_mask, as_tuple=True)[0]
    if train_idx.numel() == 0 or val_idx.numel() == 0:
        raise ValueError("frontier dataset is too small for a non-empty train/validation split")

    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    walks = RandomWalkSampler(puzzle, args.device, args.seed + 1000)
    sym = torch.from_numpy(np.load(args.data_dir / "tetra_symmetries.npy").astype(np.int64)).to(args.device)
    sym_inv = torch.from_numpy(np.load(args.data_dir / "tetra_symmetries_inv.npy").astype(np.int64)).to(args.device)
    relabel = torch.from_numpy(np.load(args.data_dir / "tetra_move_relabel.npy").astype(np.int64)).to(args.device)
    aug_rng = torch.Generator(device=args.device)
    aug_rng.manual_seed(args.seed + 2000)
    sample_rng = torch.Generator()
    sample_rng.manual_seed(args.seed + 3000)

    trainable = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        fused=args.device.startswith("cuda"))
    scaler_enabled = bool(args.amp and args.device.startswith("cuda"))
    steps = args.steps_per_epoch or max(1, math.ceil(train_idx.numel() / args.batch_size))
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "train_log.csv"
    log_path.write_text(
        "epoch,loss,policy,absolute,preserve,val_top1,val_top2,val_top4,val_regret,secs\n",
        encoding="utf-8")

    def frontier_loss(batch_idx: torch.Tensor, train: bool):
        x = states.index_select(0, batch_idx).to(args.device).long()
        target_cost = costs.index_select(0, batch_idx).to(args.device)
        target_best = (teacher_best_mask.index_select(0, batch_idx).to(args.device)
                       if teacher_best_mask is not None else None)
        if train and args.symmetry:
            x, target_cost, k = transform_batch(
                x, target_cost, sym, sym_inv, relabel, aug_rng)
            if target_best is not None:
                target_best = torch.gather(target_best, 1, relabel.index_select(0, k))
        q, _ = forward_qv(student, x, False)
        if verified_targets:
            if target_best is None or not bool(target_best.any(dim=1).all()):
                raise RuntimeError("verified rows need at least one successful root action")
            known = target_best
            teacher_prob = target_best.float() / target_best.sum(dim=1, keepdim=True)
            student_logits = -q.float() / args.policy_temperature
        else:
            known = torch.isfinite(target_cost)
            if not bool((known.sum(dim=1) >= 2).all()):
                raise RuntimeError("every frontier row needs at least two teacher-scored actions")
            teacher_logits = -(target_cost - target_cost.masked_fill(~known, float("inf")).min(
                dim=1, keepdim=True).values) / args.teacher_temperature
            teacher_logits = teacher_logits.masked_fill(~known, float("-inf"))
            teacher_prob = torch.softmax(teacher_logits, dim=1)
            student_logits = (-q.float() / args.policy_temperature).masked_fill(
                ~known, float("-inf"))
        student_log_prob = torch.log_softmax(student_logits, dim=1)
        student_log_prob = torch.where(known, student_log_prob, torch.zeros_like(student_log_prob))
        policy = -(teacher_prob * student_log_prob).sum(dim=1).mean()
        absolute = F.smooth_l1_loss(q.float()[known], (target_cost - 1.0)[known])
        return policy + args.absolute_weight * absolute, policy, absolute, q.float(), target_cost

    @torch.no_grad()
    def evaluate():
        student.eval()
        all_pred, all_cost, all_best = [], [], []
        for i in range(0, val_idx.numel(), args.batch_size):
            idx = val_idx[i:i + args.batch_size]
            x = states.index_select(0, idx).to(args.device).long()
            target = costs.index_select(0, idx).to(args.device)
            q, _ = forward_qv(student, x, False)
            all_pred.append(q.float())
            all_cost.append(target)
            if teacher_best_mask is not None:
                all_best.append(teacher_best_mask.index_select(0, idx).to(args.device))
        return policy_metrics(
            torch.cat(all_pred), torch.cat(all_cost),
            torch.cat(all_best) if verified_targets else None)

    initial = evaluate()
    print(f"rows={states.size(0)} train={train_idx.numel()} val={val_idx.numel()} "
          f"pids={len(unique_pids)} | baseline val top1={float(initial['top1']):.3f} "
          f"top4={float(initial['top4']):.3f} regret={float(initial['regret']):.3f}", flush=True)
    best_regret = float(initial["regret"])

    for epoch in range(1, args.epochs + 1):
        student.train()
        t0 = time.time()
        sums = dict(loss=0.0, policy=0.0, absolute=0.0, preserve=0.0)
        for _ in range(steps):
            choice = torch.randint(
                train_idx.numel(), (args.batch_size,), generator=sample_rng)
            idx = train_idx.index_select(0, choice)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=scaler_enabled):
                loss, policy, absolute, _, _ = frontier_loss(idx, True)
                preserve = torch.zeros((), device=args.device)
                if args.preserve_weight > 0.0 and args.preserve_batch > 0:
                    rw = walks.sample(args.preserve_batch, args.k_min, args.k_max)
                    with torch.no_grad():
                        ref_q, ref_v = forward_qv(reference, rw, has_value)
                    stu_q, stu_v = forward_qv(student, rw, has_value)
                    preserve = F.smooth_l1_loss(stu_q.float(), ref_q.float())
                    if stu_v is not None and ref_v is not None:
                        preserve = preserve + args.preserve_value_weight * F.smooth_l1_loss(
                            stu_v.float(), ref_v.float())
                    loss = loss + args.preserve_weight * preserve
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            opt.step()
            sums["loss"] += float(loss.detach())
            sums["policy"] += float(policy.detach())
            sums["absolute"] += float(absolute.detach())
            sums["preserve"] += float(preserve.detach())
        for key in sums:
            sums[key] /= steps

        should_eval = epoch == 1 or epoch % args.checkpoint_every == 0 or epoch == args.epochs
        if should_eval:
            metrics = evaluate()
            secs = time.time() - t0
            with log_path.open("a", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow([
                    epoch, sums["loss"], sums["policy"], sums["absolute"], sums["preserve"],
                    float(metrics["top1"]), float(metrics["top2"]), float(metrics["top4"]),
                    float(metrics["regret"]), secs,
                ])
            print(f"epoch {epoch:4d} loss={sums['loss']:.4f} policy={sums['policy']:.4f} "
                  f"abs={sums['absolute']:.4f} keep={sums['preserve']:.4f} | "
                  f"val top1={float(metrics['top1']):.3f} top4={float(metrics['top4']):.3f} "
                  f"regret={float(metrics['regret']):.3f} | {secs:.1f}s", flush=True)
            payload = {
                "epoch": epoch,
                "state_dict": {k.removeprefix("_orig_mod."): v
                               for k, v in student.state_dict().items()},
                "optimizer": opt.state_dict(),
                "model_config": model_cfg,
                "train_config": vars(args),
                "frontier_meta": blob.get("meta", {}),
                "metrics": {k: float(v) for k, v in metrics.items()},
                "source_checkpoint": str(args.checkpoint),
            }
            torch.save(payload, args.output / f"epoch_{epoch:04d}.pt")
            if float(metrics["regret"]) <= best_regret:
                best_regret = float(metrics["regret"])
                torch.save(payload, args.output / "best.pt")
    print(f"done | best held-out regret={best_regret:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
