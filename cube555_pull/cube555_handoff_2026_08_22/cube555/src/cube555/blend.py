"""Average the Q predictions of several checkpoints inside ONE beam step.

Motivated by a measurement, not by hope: at remaining 61-80 the model's prediction varies
by SD ~12 across states nominally at the same depth, while the action-level spread within
a parent is ~0.6. The beam's top-B is therefore selected mostly by cross-state error. If
that error is even partly independent between members, averaging shrinks it as 1/sqrt(n)
while leaving the shared signal intact -- which is exactly the term that needs shrinking.

Members must be Q heads of the same width trained to the same objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BlendedQ(nn.Module):
    def __init__(self, members: list[nn.Module], weights: list[float] | None = None):
        super().__init__()
        if not members:
            raise ValueError("BlendedQ needs at least one member")
        dims = {int(getattr(m, "output_dim", -1)) for m in members}
        if len(dims) != 1:
            raise ValueError(f"members disagree on output_dim: {dims}")
        self.members = nn.ModuleList(members)
        self.output_dim = dims.pop()
        w = torch.tensor(weights if weights else [1.0] * len(members), dtype=torch.float32)
        if w.numel() != len(members):
            raise ValueError("one weight per member, including the first")
        self.register_buffer("weights", w / w.sum())
        self.has_value_head = bool(getattr(members[0], "has_value_head", False))
        self.return_value = False
        self.inference_chunk_size = getattr(members[0], "inference_chunk_size", None)

    def forward(self, x: torch.Tensor):
        out = None
        for m, wt in zip(self.members, self.weights):
            m.return_value = False
            q = m(x).float() * wt
            out = q if out is None else out + q
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
