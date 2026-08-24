"""Create scaled aliases for a DeltaV checkpoint.

The trained delta checkpoint stores the composition scale in
``model_config["delta_scale"]``.  This helper writes lightweight full-checkpoint
copies under ``<prefix>_s<tag>/epoch_0023.pt`` so existing solve/sweep scripts can
evaluate several global delta strengths without retraining.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch


DEFAULT_SCALES = {
    "0": 0.0,
    "005": 0.05,
    "01": 0.10,
    "025": 0.25,
    "1": 1.0,
}


def parse_scales(raw: str) -> dict[str, float]:
    if not raw:
        return DEFAULT_SCALES
    out: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            tag, value = item.split("=", 1)
            out[tag.strip()] = float(value)
        else:
            out[item] = float(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--model-prefix", required=True,
                    help="Alias dirs are <models-dir>/<model-prefix>_s<tag>.")
    ap.add_argument("--models-dir", type=Path, default=Path("megaminx/models"))
    ap.add_argument("--epoch-name", default="epoch_0023.pt")
    ap.add_argument("--scales", default="",
                    help="Comma-separated tag=value entries. Default: 0,005,01,025,1.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config")
    if not isinstance(cfg, dict) or cfg.get("model_class") != "DeltaVModel":
        raise SystemExit("--checkpoint must be a DeltaVModel checkpoint")

    for tag, scale in parse_scales(args.scales).items():
        out_dir = args.models_dir / f"{args.model_prefix}_s{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        alias = copy.deepcopy(ckpt)
        alias["model_config"]["delta_scale"] = float(scale)
        out_path = out_dir / args.epoch_name
        torch.save(alias, out_path)
        print(f"wrote {out_path} delta_scale={scale:g}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
