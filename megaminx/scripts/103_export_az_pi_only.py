"""Export an AZ dual-head checkpoint's policy trunk as a ResMLPDistance checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--az-checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ck = torch.load(args.az_checkpoint, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()}
    keep = {}
    for k, v in sd.items():
        if k.startswith(("embedding.", "input_stack.", "res_blocks.")):
            keep[k] = v
    keep["head.weight"] = sd["policy_head.weight"]
    keep["head.bias"] = sd["policy_head.bias"]

    mc = dict(ck["model_config"])
    mc["model_class"] = "ResMLPDistance"
    mc["output_dim"] = int(mc.get("n_actions", keep["head.bias"].numel()))
    mc.pop("n_actions", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": ck.get("epoch", -1),
            "state_dict": keep,
            "model_config": mc,
            "source_az_checkpoint": str(args.az_checkpoint),
            "source_p_loss": ck.get("p_loss"),
            "source_v_loss": ck.get("v_loss"),
        },
        args.out,
    )
    print(f"saved {args.out} with {len(keep)} tensors", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
