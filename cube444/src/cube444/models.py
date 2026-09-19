"""Checkpoint loading for the 4x4x4 pipeline.

`cayley.search.load_model_checkpoint` does not know about `ResMLPGFlowNet` (the
AZ dual-head model): its dispatch falls through to `_build_resmlp_from_config`,
whose `load_state_dict` then fails on the extra `policy_head.*` / `value_head.*`
/ `log_Z` keys and the missing `head.*`.

`load_v_model` here dispatches on `model_config["model_class"]` and wraps a
dual-head checkpoint in `GFlowValueAdapter` so it presents the plain
`model(states) -> (B,)` distance interface that `KhoruzhiiSolver` and the eval
gates expect. Stage-2/Stage-3 (`ResMLPDistance`) checkpoints pass straight
through to the shared loader.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from cayley.gflow_model import ResMLPGFlowNet
from cayley.search import load_model_checkpoint


class GFlowValueAdapter(nn.Module):
    """Expose only the value head of a ResMLPGFlowNet as a distance model.

    sign=+1 for the AZ regime (the value head predicts distance directly).
    sign=-1 would be the Trajectory-Balance regime (value is log F(s), so
    distance ~ -log F); we do not train TB models for this puzzle.
    """

    def __init__(self, gflow: ResMLPGFlowNet, sign: float = 1.0,
                 inference_chunk_size: int = 8192):
        super().__init__()
        self.gflow = gflow
        self.sign = sign
        self.state_size = gflow.state_size
        self.num_classes = gflow.num_classes
        self.inference_chunk_size = inference_chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= self.inference_chunk_size:
            return self.sign * self.gflow(x)[1]
        outs = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            outs.append(self.sign * self.gflow(x[i : i + self.inference_chunk_size])[1])
        return torch.cat(outs, dim=0)


def load_v_model(path: str | Path, device: str = "cuda",
                 dtype: torch.dtype = torch.float32) -> nn.Module:
    """Load a Stage-2/3 V checkpoint or a Stage-4 AZ checkpoint as a distance model."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mc = ckpt.get("model_config") or {}
    if mc.get("model_class") != "ResMLPGFlowNet":
        return load_model_checkpoint(path, device=device, dtype=dtype)

    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["state_dict"].items()}
    gflow = ResMLPGFlowNet(
        state_size=mc["state_size"],
        num_classes=mc["num_classes"],
        hidden_dims=tuple(mc["hidden_dims"]),
        num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "onehot"),
        embed_dim=mc.get("embed_dim", 16),
        n_actions=mc.get("n_actions", 24),
    )
    missing, unexpected = gflow.load_state_dict(sd, strict=False)
    if missing:
        print(f"  load_v_model: missing keys {list(missing)[:5]}")
    if unexpected:
        print(f"  load_v_model: unexpected keys {list(unexpected)[:5]}")
    return GFlowValueAdapter(gflow).to(device=device, dtype=dtype).eval()
