"""Evaluate the 2B sym-variance uncertainty head.

The uncertainty target is the teacher value spread across sampled rotations of a
state.  This script measures whether the learned uncertainty head predicts that
held-out spread, and whether raw multi-head disagreement is a better signal.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(ROOT / "src"))

from megaminx.puzzle import Megaminx


def _load_script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum()).item()
    return float((x * y).sum().item() / denom) if denom > 0 else float("nan")


def _rankdata(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--teacher", type=Path, default=None)
    ap.add_argument("--policy-dataset", type=Path, default=PROJECT / "data" / "az_dataset_73614.pt")
    ap.add_argument("--rotations", type=Path, default=PROJECT / "data" / "rotations.npy")
    ap.add_argument("--n-states", type=int, default=4096)
    ap.add_argument("--rot-samples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=950)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    bench = _load_script_module(PROJECT / "scripts" / "94_bench_multihead_voted.py", "bench_mh")
    train = _load_script_module(PROJECT / "scripts" / "93_train_multihead_orbit.py", "train_mh")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    teacher_path = args.teacher or Path(ckpt["teacher"])
    if not teacher_path.exists():
        teacher_path = ROOT / teacher_path
    if not teacher_path.exists():
        raise FileNotFoundError(f"teacher not found: {ckpt['teacher']}")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = bench.load_multihead(args.checkpoint, device=args.device, dtype=dtype)
    teacher = train.load_az_teacher(teacher_path, args.device, dtype)
    rotations, inverse_rotations = train.load_rotation_tables(args.rotations, args.device)

    data = torch.load(args.policy_dataset, map_location="cpu", weights_only=False)
    states_all = data["states"].long()
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(states_all.size(0), generator=gen)[: args.n_states]
    states = states_all[idx].to(args.device)

    rng = torch.Generator(device=args.device).manual_seed(args.seed + 1)
    n = states.size(0)
    rot_ids = torch.randint(0, rotations.size(0), (n, args.rot_samples),
                            generator=rng, device=args.device)
    target_vals = []
    for j in range(args.rot_samples):
        rotated = train.apply_state_rotations(states, rotations, inverse_rotations, rot_ids[:, j])
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            target_vals.append(train.teacher_value(teacher, rotated, args.batch_size))
    target = torch.stack(target_vals, dim=1)
    target_spread = torch.sqrt(torch.var(target.float(), dim=1, unbiased=False) + 1e-6)

    values, uncertainty = bench.multihead_predict(
        model, states, args.batch_size, with_uncertainty=True,
    )
    if uncertainty is None:
        raise RuntimeError("checkpoint/model did not return uncertainty")
    values_f = values.float()
    head_spread = torch.sqrt(torch.var(values_f, dim=1, unbiased=False) + 1e-6)

    u = uncertainty.float()
    q = max(1, n // 4)
    u_order = torch.argsort(u)
    target_low_u = target_spread[u_order[:q]].mean().item()
    target_high_u = target_spread[u_order[-q:]].mean().item()
    hs_order = torch.argsort(head_spread)
    target_low_hs = target_spread[hs_order[:q]].mean().item()
    target_high_hs = target_spread[hs_order[-q:]].mean().item()

    print(f"checkpoint: {args.checkpoint}")
    print(f"teacher:    {teacher_path}")
    print(f"states={n} rot_samples={args.rot_samples}")
    print(f"target_spread mean={target_spread.mean().item():.4f} std={target_spread.std().item():.4f}")
    print(f"uncertainty   mean={u.mean().item():.4f} std={u.std().item():.4f}")
    print(f"head_spread   mean={head_spread.mean().item():.4f} std={head_spread.std().item():.4f}")
    print(f"pearson(u,target)        {_pearson(u, target_spread):+.4f}")
    print(f"spearman(u,target)       {_spearman(u, target_spread):+.4f}")
    print(f"pearson(head,target)     {_pearson(head_spread, target_spread):+.4f}")
    print(f"spearman(head,target)    {_spearman(head_spread, target_spread):+.4f}")
    print(f"target high-u / low-u    {target_high_u:.4f} / {target_low_u:.4f} "
          f"(lift {target_high_u / max(target_low_u, 1e-9):.2f}x)")
    print(f"target high-head / low-head {target_high_hs:.4f} / {target_low_hs:.4f} "
          f"(lift {target_high_hs / max(target_low_hs, 1e-9):.2f}x)")
    if not math.isfinite(_pearson(u, target_spread)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
