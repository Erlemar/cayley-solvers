"""ResMLP Q-head for the 5x5x5 picture cube.

`forward(states int (B,150)) -> (B,30)`, lower is better, comparable ACROSS parents, so
ONE forward on a parent scores all 30 children. That is the whole inference economics of
this project: a scalar-V beam step costs B x 30 network evaluations, this costs B.

ENCODING. `num_classes` is 150 here (picture cube), so one-hot is a 22,500-dim input and
spends 23M parameters in the first Linear alone. Measured on an A100 at matched parameter
count, `embedding` with embed_dim 24 (3,600-dim input) is ~2x faster at inference and
cheaper to train:

    onehot   d1024x10   44.1M   27.4 ms/step   0.69 Mstate/s
    embed24  d1024x10   24.8M   20.3 ms/step   1.34 Mstate/s

One-hot is strictly more expressive -- it is a full-rank (slot x sticker) lookup table
where the embedding is a rank-`embed_dim` factorisation of the same thing -- so it stays
selectable, but cost says embedding.

VALUE HEAD. A second Linear on the same trunk (no extra forward). Trained on the walk
index for rw rows and the exact depth for anchors. It is a DIAGNOSTIC, not a scorer:
qv-consistency is measured dead on two model families and three lambdas, because V learns
the walk index rather than the distance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from cayley.model import ResBlock  # noqa: E402


class ResMLPQ(nn.Module):
    def __init__(
        self,
        state_size: int = 150,
        num_classes: int = 150,
        output_dim: int = 30,
        encoding: str = "embedding",
        embed_dim: int = 24,
        d_model: int = 1024,
        num_res_blocks: int = 10,
        az_head: bool = True,
        inference_chunk_size: int | None = 1 << 18,
    ):
        super().__init__()
        if encoding not in ("embedding", "onehot"):
            raise ValueError(f"encoding must be embedding or onehot, got {encoding!r}")
        self.state_size = state_size
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.encoding = encoding
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.num_res_blocks = num_res_blocks
        self.az_head = bool(az_head)
        self.has_value_head = bool(az_head)
        self.inference_chunk_size = inference_chunk_size
        # When True, forward returns (q, v). The beam's _q_predict/_qv_predict both accept
        # a tuple, so this can stay on; training sets it explicitly for clarity.
        self.return_value = False

        if encoding == "embedding":
            self.embedding = nn.Embedding(num_classes, embed_dim)
            in_dim = state_size * embed_dim
        else:
            self.embedding = None
            in_dim = state_size * num_classes
        self.input_stack = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(d_model) for _ in range(num_res_blocks)]
        )
        self.q_head = nn.Linear(d_model, output_dim)
        self.v_head = nn.Linear(d_model, 1) if az_head else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.input_stack[0].weight.dtype
        xl = x.long()
        if self.encoding == "onehot":
            return F.one_hot(xl, self.num_classes).to(dt).flatten(start_dim=-2)
        return self.embedding(xl).to(dt).flatten(start_dim=-2)

    def _forward_single(self, x: torch.Tensor):
        h = self.input_stack(self.encode(x))
        for b in self.res_blocks:
            h = b(h)
        q = self.q_head(h)
        if self.return_value:
            v = (
                self.v_head(h).squeeze(-1)
                if self.v_head is not None
                else q.new_zeros(q.shape[0])
            )
            return q, v
        return q

    def enable_compiled_inference(self) -> None:
        """torch.compile the trunk and PAD every call to exactly inference_chunk_size.

        Measured 1.56x on beam throughput (0.900 -> 1.402 Mstate/s, matched A/B in one
        invocation) -- worth ~0.65 doublings of beam width, which is the strongest lever
        we have. The standing warning is "do not compile beam inference without padding
        to a fixed batch" (5.8x slowdown from reshape recompiles); padding is what makes
        it safe, so the padding is not optional here and is done inside `forward` rather
        than left to the caller.
        """
        if self.inference_chunk_size is None:
            raise ValueError("compiled inference needs a fixed inference_chunk_size")
        self._compiled = torch.compile(
            self._forward_single, mode="max-autotune-no-cudagraphs", dynamic=False
        )

    def forward(self, x: torch.Tensor):
        ics = self.inference_chunk_size
        fwd = getattr(self, "_compiled", None) or self._forward_single
        if (
            self.training
            or ics is None
            or (x.shape[0] <= ics and fwd is self._forward_single)
        ):
            return self._forward_single(x)
        n = x.shape[0]
        qs, vs = [], []
        for i in range(0, n, ics):
            chunk = x[i : i + ics]
            pad = ics - chunk.shape[0]
            if pad and fwd is not self._forward_single:
                # zeros are valid sticker ids, so the padding cannot fault; the rows are
                # sliced off below. This is what keeps every compiled call the same shape.
                chunk = torch.cat([chunk, chunk.new_zeros((pad, chunk.shape[1]))])
            out = fwd(chunk)
            if self.return_value:
                qs.append(out[0][: ics - pad] if pad else out[0])
                vs.append(out[1][: ics - pad] if pad else out[1])
            else:
                qs.append(out[: ics - pad] if pad else out)
        return (torch.cat(qs), torch.cat(vs)) if self.return_value else torch.cat(qs)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_config(self) -> dict:
        return {
            "model_class": "ResMLPQ",
            "state_size": self.state_size,
            "num_classes": self.num_classes,
            "output_dim": self.output_dim,
            "encoding": self.encoding,
            "embed_dim": self.embed_dim,
            "d_model": self.d_model,
            "num_res_blocks": self.num_res_blocks,
            "az_head": self.az_head,
            "inference_chunk_size": self.inference_chunk_size,
        }


def build_model(mc: dict) -> ResMLPQ:
    return ResMLPQ(
        state_size=int(mc.get("state_size", 150)),
        # NOT a default we can be lazy about: cube444's rule is num_classes=6, and here
        # the correct value is 150. Wrong either way is silent.
        num_classes=int(mc["num_classes"]),
        output_dim=int(mc.get("output_dim", 30)),
        encoding=str(mc.get("encoding", "embedding")),
        embed_dim=int(mc.get("embed_dim", 24)),
        d_model=int(mc.get("d_model", 1024)),
        num_res_blocks=int(mc.get("num_res_blocks", 10)),
        az_head=bool(mc.get("az_head", True)),
        inference_chunk_size=mc.get("inference_chunk_size", 1 << 18),
    )


def load_model(path, device="cuda", dtype=torch.float32) -> ResMLPQ:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    mc = dict(ck["model_config"])
    mc.pop("model_class", None)
    model = ResMLPQ(**mc)
    sd = ck["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device=device, dtype=dtype).eval()
