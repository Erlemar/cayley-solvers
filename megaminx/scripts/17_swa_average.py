"""Stochastic Weight Averaging: average a list of checkpoints into a single model.

Izmailov et al. 2018 (https://arxiv.org/abs/1803.05407): averaging weights from
multiple SGD iterates near a flat minimum yields a model with better generalization
(typically ~0.5-2% accuracy improvement, very low risk).

Megaminx use: average late-cycle Bellman ckpts (e.g. m05 epochs 400, 425, 450, 475,
499) into a single 'm05_swa.pt'. Then strat-5 vs m05 baseline.

Free win: no GPU training, just a few minutes of CPU + a strat-5.

Usage:
    python megaminx/scripts/17_swa_average.py \\
        --inputs models/m05_bellman_warm/epoch_0399.pt \\
                 models/m05_bellman_warm/epoch_0449.pt \\
                 models/m05_bellman_warm/epoch_0499.pt \\
        --output models/m05_swa_400_499.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, type=Path,
                    help="checkpoints to average (>=2)")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if len(args.inputs) < 2:
        ap.error("--inputs requires at least 2 checkpoints")

    print(f"averaging {len(args.inputs)} checkpoints:")
    for p in args.inputs:
        print(f"  {p}")
        if not p.exists():
            ap.error(f"missing: {p}")

    avg_sd: dict[str, torch.Tensor] = {}
    template = None  # for non-state-dict metadata
    for i, p in enumerate(args.inputs):
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        if i == 0:
            avg_sd = {k: v.clone().to(torch.float64) for k, v in sd.items()}
            template = ckpt
        else:
            for k, v in sd.items():
                if k not in avg_sd:
                    raise ValueError(f"checkpoint {p} has key {k} missing from first ckpt")
                avg_sd[k] += v.to(torch.float64)
    n = len(args.inputs)
    for k in avg_sd:
        avg_sd[k] /= n
        # Cast back to original dtype
        orig_dtype = template["state_dict"][k].dtype
        avg_sd[k] = avg_sd[k].to(orig_dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_ckpt = {
        "state_dict": avg_sd,
        "model_config": template.get("model_config", {}),
        "swa_inputs": [str(p) for p in args.inputs],
        "epoch": -1,  # marker for SWA
    }
    torch.save(out_ckpt, args.output)
    print(f"\nwrote {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
