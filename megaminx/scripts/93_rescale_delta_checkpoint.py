"""Create DeltaVModel checkpoints with different residual scales.

The trained delta weights can be useful as a small beam-ranking correction even
when scale=1.0 distorts the global V landscape. This utility rewrites only the
serialized delta_scale metadata; weights are copied unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--scales", required=True,
                    help="Comma-separated residual scales, e.g. 0.05,0.1,0.2")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config")
    if not isinstance(cfg, dict) or cfg.get("model_class") != "DeltaVModel":
        raise ValueError("--checkpoint must be a DeltaVModel checkpoint")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    for scale in [float(x) for x in args.scales.split(",") if x.strip()]:
        out = dict(ckpt)
        out_cfg = dict(cfg)
        out_cfg["delta_scale"] = scale
        out["model_config"] = out_cfg
        if isinstance(out.get("train_args"), dict):
            train_args = dict(out["train_args"])
            train_args["delta_scale_eval_override"] = scale
            out["train_args"] = train_args
        tag = f"{scale:g}".replace(".", "p")
        out_path = args.output_dir / f"{stem}_scale{tag}.pt"
        torch.save(out, out_path)
        print(f"{scale:g} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
