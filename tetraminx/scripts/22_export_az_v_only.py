"""Export the value head of an AZ dual-head checkpoint as a plain
ResMLPDistance checkpoint (value_head -> head), so every existing consumer
(KhoruzhiiSolver, verify tooling, the TPU JAX loader) works unchanged.

    .venv/Scripts/python.exe tetraminx/scripts/22_export_az_v_only.py \
        --in tetraminx/models/taz_v1/epoch_0099.pt \
        --out tetraminx/models/taz_v1/taz_v1_v_only_ep99.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    assert "value_head.weight" in sd, "not an AZ dual-head checkpoint"

    out_sd = {}
    for k, v in sd.items():
        if k.startswith(("embedding.", "input_stack.", "res_blocks.")):
            out_sd[k] = v
        elif k == "value_head.weight":
            out_sd["head.weight"] = v
        elif k == "value_head.bias":
            out_sd["head.bias"] = v
        # policy_head / log_Z dropped

    mc = dict(ckpt.get("model_config", {}))
    mc.pop("n_actions", None)
    torch.save({
        "epoch": ckpt.get("epoch"),
        "state_dict": out_sd,
        "model_config": mc,
        "source": str(args.inp),
    }, args.out)

    # Round-trip check: load into ResMLPDistance and run a forward.
    from cayley.model import ResMLPDistance
    model = ResMLPDistance(
        state_size=mc["state_size"], num_classes=mc["num_classes"],
        hidden_dims=list(mc["hidden_dims"]), num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "embedding"), embed_dim=mc.get("embed_dim", 16),
    )
    model.load_state_dict(out_sd)
    model.eval()
    with torch.no_grad():
        v0 = float(model(torch.arange(mc["state_size"]).unsqueeze(0)).squeeze())
    print(f"wrote {args.out}  V(solved)={v0:.3f}  ({len(out_sd)} tensors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
