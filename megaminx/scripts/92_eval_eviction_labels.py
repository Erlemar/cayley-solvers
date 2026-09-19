"""Evaluate a V checkpoint on cross-width eviction labels.

For each label, lower V should prefer the verified teacher child over the
near-cutoff boundary states from the narrow beam. This reports pairwise ranking
accuracy and margin-satisfaction rates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.search import load_model_checkpoint


@torch.no_grad()
def _predict(model: torch.nn.Module, states: torch.Tensor, chunk: int) -> torch.Tensor:
    vals: list[torch.Tensor] = []
    for i in range(0, states.size(0), chunk):
        vals.append(model(states[i : i + chunk]).float().flatten().cpu())
    return torch.cat(vals, dim=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--eviction-path", required=True, type=Path)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=8192)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    model.eval()

    data = torch.load(args.eviction_path, map_location="cpu", weights_only=False)
    goods = data["goods"].to(args.device)
    boundaries = data["boundaries"]
    mask = data["boundary_mask"].bool()
    n_labels, n_boundary, state_size = boundaries.shape

    good_v = _predict(model, goods, args.chunk_size).view(n_labels, 1)
    flat_bounds = boundaries.reshape(-1, state_size).to(args.device)
    bound_v = _predict(model, flat_bounds, args.chunk_size).view(n_labels, n_boundary)

    valid = mask
    valid_counts = valid.sum(dim=1).clamp(min=1)
    pairwise_ok = ((good_v < bound_v) & valid).sum(dim=1).float() / valid_counts.float()
    margin_ok = ((good_v + args.margin <= bound_v) & valid).sum(dim=1).float() / valid_counts.float()
    masked_bound_v = bound_v.masked_fill(~valid, float("inf"))
    best_boundary = masked_bound_v.min(dim=1).values
    cutoff_boundary = bound_v.masked_fill(~valid, float("-inf")).max(dim=1).values
    all_ok = good_v.flatten() < best_boundary
    all_margin_ok = good_v.flatten() + args.margin <= best_boundary
    cutoff_ok = good_v.flatten() < cutoff_boundary
    cutoff_margin_ok = good_v.flatten() + args.margin <= cutoff_boundary
    rank = ((bound_v < good_v) & valid).sum(dim=1) + 1

    print(f"checkpoint: {args.checkpoint}")
    print(f"labels: {n_labels:,}; boundary per label: {int(valid_counts.min())}-{int(valid_counts.max())}")
    print(f"mean V(good): {float(good_v.mean()):.4f}")
    print(f"mean min V(boundary): {float(best_boundary.mean()):.4f}")
    print(f"mean cutoff V(boundary): {float(cutoff_boundary.mean()):.4f}")
    print(f"pairwise good<boundary: {float(pairwise_ok.mean()):.4f}")
    print(f"pairwise margin ok: {float(margin_ok.mean()):.4f}")
    print(f"cutoff-survival good<max-boundary: {float(cutoff_ok.float().mean()):.4f}")
    print(f"cutoff-survival margin ok: {float(cutoff_margin_ok.float().mean()):.4f}")
    print(f"all-boundary good best: {float(all_ok.float().mean()):.4f}")
    print(f"all-boundary margin ok: {float(all_margin_ok.float().mean()):.4f}")
    print(f"mean rank of good among boundaries: {float(rank.float().mean()):.2f}")
    print(f"p90 rank: {int(torch.quantile(rank.float(), 0.90).item())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
