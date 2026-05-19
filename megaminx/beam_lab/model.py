"""ResMLP distance predictor — embedding-encoded variant matching m07/m05.

Hyperparams used for the 95,682 submission:
    state_size=120, num_classes=120, hidden_dims=(2048, 512),
    num_res_blocks=2, embed_dim=16  →  ~6.0M parameters

Self-contained copy for beam_lab. The full project has additional encodings
(onehot, piece) but only `embedding` is needed here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.relu(self.ln1(self.lin1(x)))
        h = self.ln2(self.lin2(h))
        return F.relu(x + h)


class ResMLPDistance(nn.Module):
    def __init__(
        self,
        state_size: int = 120,
        num_classes: int = 120,
        hidden_dims: tuple[int, ...] = (2048, 512),
        num_res_blocks: int = 2,
        embed_dim: int = 16,
        inference_chunk_size: int | None = 8192,
        output_dim: int = 1,
    ):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.inference_chunk_size = inference_chunk_size
        self.output_dim = output_dim

        in_dim = state_size * embed_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)

        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(num_res_blocks)])
        self.head = nn.Linear(prev, output_dim)

    def encode(self, x):
        target_dtype = self.input_stack[0].weight.dtype
        return self.embedding(x.long()).to(target_dtype).flatten(start_dim=-2)

    def _forward_single(self, x):
        h = self.encode(x)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        return out.squeeze(-1) if self.output_dim == 1 else out

    def forward(self, x):
        if self.training or self.inference_chunk_size is None or x.shape[0] <= self.inference_chunk_size:
            return self._forward_single(x)
        outs = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            outs.append(self._forward_single(x[i : i + self.inference_chunk_size]))
        return torch.cat(outs, dim=0)


def load_checkpoint(ckpt_path: str, device: str = "cuda") -> ResMLPDistance:
    """Load a saved m07/m05/m17 checkpoint into a ResMLPDistance."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

    cfg = ckpt.get("model_config", {})
    model = ResMLPDistance(
        state_size=cfg.get("state_size", 120),
        num_classes=cfg.get("num_classes", 120),
        hidden_dims=tuple(cfg.get("hidden_dims", (2048, 512))),
        num_res_blocks=cfg.get("num_res_blocks", 2),
        embed_dim=cfg.get("embed_dim", 16),
        output_dim=cfg.get("output_dim", 1),
    )
    model.load_state_dict(sd)
    return model.to(device).eval()
