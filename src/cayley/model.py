"""ResMLP distance predictor for the Picture Cube.

Input: a length-72 int permutation vector.
Output: a scalar predicting the "diffusion distance" (random-walk depth) to solved.

Two encoding choices:
 - `encoding="onehot"` (paper recipe): Linear(state_size*num_classes -> hidden[0]) ...
   At num_classes=72, this is a 5184-dim input. With beams >= 2^15 the one-hot tensor
   dominates VRAM (12+ GB at beam 2^15 on 16GB cards).
 - `encoding="embedding"` (our Phase-3 optimization): nn.Embedding(num_classes, embed_dim)
   per position, flattened to state_size*embed_dim. At embed_dim=16 this is 1152 dims —
   4.5x smaller than one-hot. Base-code analysis recommends this swap.

The base-code analysis also flagged that CayleyPy's library MLP has NO residual
connections; we keep the residual structure here because the paper (and DeepCubeA)
show it matters.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.ln1(self.lin1(x)))
        h = self.ln2(self.lin2(h))
        return F.relu(x + h)


class ResMLPDistance(nn.Module):
    """Distance predictor.

    `inference_chunk_size`: during eval, call the model on at most this many states at once
    and concatenate the outputs. Keeps VRAM bounded when beam search hands us 100k+ states.
    Set to None to disable chunking.
    """

    def __init__(
        self,
        state_size: int = 72,
        num_classes: int = 72,
        hidden_dims: tuple[int, ...] = (700, 643),
        num_res_blocks: int = 4,
        inference_chunk_size: int | None = 2048,
        encoding: str = "onehot",
        embed_dim: int = 16,
        output_dim: int = 1,
    ):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.inference_chunk_size = inference_chunk_size
        self.encoding = encoding
        self.embed_dim = embed_dim
        # output_dim > 1 turns this into a Q-head: one output per action.
        # For our 18-generator puzzle, set output_dim=18 to train a neighbor-score model.
        # Distillation target: Q(s, a) ≈ V_teacher(apply(s, a)).
        self.output_dim = output_dim

        if encoding == "onehot":
            in_dim = state_size * num_classes
            self.embedding = None
        elif encoding == "embedding":
            in_dim = state_size * embed_dim
            self.embedding = nn.Embedding(num_classes, embed_dim)
        elif encoding == "piece":
            # F3: piece decomposition. 52 features in 6 groups, one embedding table each.
            # Total flattened input: 52 * embed_dim.
            from cayley.piece_features import (
                CENTER_ORI_CARD, CORNER_ORI_CARD, EDGE_ORI_CARD,
                N_CENTERS, N_CORNERS, N_EDGES, N_FEATURES,
            )
            in_dim = N_FEATURES * embed_dim
            self.embedding = None  # unused for piece encoding
            self.corner_id_emb = nn.Embedding(N_CORNERS, embed_dim)
            self.corner_ori_emb = nn.Embedding(CORNER_ORI_CARD, embed_dim)
            self.edge_id_emb = nn.Embedding(N_EDGES, embed_dim)
            self.edge_ori_emb = nn.Embedding(EDGE_ORI_CARD, embed_dim)
            self.center_id_emb = nn.Embedding(N_CENTERS, embed_dim)
            self.center_ori_emb = nn.Embedding(CENTER_ORI_CARD, embed_dim)
            self._piece_extractor = None  # lazy init on first use (needs device)
            self._piece_boundaries = (N_CORNERS, 2 * N_CORNERS,
                                      2 * N_CORNERS + N_EDGES,
                                      2 * N_CORNERS + 2 * N_EDGES,
                                      2 * N_CORNERS + 2 * N_EDGES + N_CENTERS,
                                      N_FEATURES)
        else:
            raise ValueError(f"encoding must be 'onehot', 'embedding', or 'piece', got {encoding!r}")

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(num_res_blocks)])
        self.head = nn.Linear(prev, output_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a (B, state_size) int tensor into (B, in_dim) float.

        Uses the first linear layer's weight dtype so the model works under .to(bfloat16)
        or .to(float16) casts without a manual dtype argument.
        """
        target_dtype = self.input_stack[0].weight.dtype
        if self.encoding == "onehot":
            one_hot = F.one_hot(x.long(), num_classes=self.num_classes)
            return one_hot.to(target_dtype).flatten(start_dim=-2)
        if self.encoding == "piece":
            # Lazy-init the extractor on the state tensor's device.
            if self._piece_extractor is None:
                from cayley.piece_features import TorchFeatureExtractor
                self._piece_extractor = TorchFeatureExtractor(device=str(x.device))
            feats = self._piece_extractor(x.long())  # (B, 52) int64
            b0, b1, b2, b3, b4, b5 = self._piece_boundaries
            parts = [
                self.corner_id_emb(feats[:, 0:b0]),
                self.corner_ori_emb(feats[:, b0:b1]),
                self.edge_id_emb(feats[:, b1:b2]),
                self.edge_ori_emb(feats[:, b2:b3]),
                self.center_id_emb(feats[:, b3:b4]),
                self.center_ori_emb(feats[:, b4:b5]),
            ]
            return torch.cat(parts, dim=1).to(target_dtype).flatten(start_dim=-2)
        return self.embedding(x.long()).to(target_dtype).flatten(start_dim=-2)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        out = self.head(h)
        # Preserve existing (B,) output when output_dim==1 for drop-in compatibility.
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out  # (B, output_dim) for Q-heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.inference_chunk_size is None or x.shape[0] <= self.inference_chunk_size:
            return self._forward_single(x)
        # Eval path with large batch: process in chunks to cap memory.
        outs: list[torch.Tensor] = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            outs.append(self._forward_single(x[i : i + self.inference_chunk_size]))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_config(self) -> dict:
        """Serialization dict used by the polymorphic checkpoint loader.

        Returns a dict that can be passed back to ResMLPDistance(**cfg) to
        reconstruct an equivalent model. Used by `cayley.training.train` and
        `cayley.bellman.train_bellman` for checkpoint serialization.
        """
        return {
            "model_class": "ResMLPDistance",
            "state_size": self.state_size,
            "num_classes": self.num_classes,
            "hidden_dims": [
                layer.out_features for layer in self.input_stack
                if isinstance(layer, torch.nn.Linear)
            ],
            "num_res_blocks": len(self.res_blocks),
            "encoding": self.encoding,
            "embed_dim": self.embed_dim,
            "output_dim": self.output_dim,
            "inference_chunk_size": self.inference_chunk_size,
        }
