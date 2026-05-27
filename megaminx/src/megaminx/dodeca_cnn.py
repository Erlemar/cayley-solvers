"""Dodecahedral CNN / geometric convolution value model (architecture doc section 3.8).

The doc itself calls this "essentially a local GNN": each of the 120 slots aggregates
features from its puzzle-geometric neighbours and an MLP mixes them. We use the four
neighbour relations already encoded in the slot block of bipartite_features.pt:

    same-face       (rel 5)   -- the doc's "same-face cyclic neighbors"
    generator-edge  (rel 2)   -- adjacent on a generator 5-cycle (the "generator-image"
                                  / adjacent-face boundary neighbours)
    same-piece      (rel 3)   -- corner/edge mates
    stride-2        (rel 4)   -- 2-hop on a 5-cycle (extra medium-range face structure)

Layer (the doc's spec, residual + norm added):
    h_i' = h_i + MLP( concat[ h_i, mean_sameface(h), mean_genedge(h),
                              mean_samepiece(h), mean_stride2(h) ] )

Each mean_* is a fixed row-normalised adjacency matmul `A_rel @ h` (A_rel is static,
shared across the batch -- 120x120, tiny). NO attention anywhere, so this is far cheaper
than the Perceiver / bipartite GT (no (B,H,T,T) tensors, no OOM, fast epochs).

State enters only through per-slot dynamic features (the slot's current sticker, where its
home sticker went, a solved flag); static identity features (face/piece/slot) are shared.

External contract mirrors ResMLPDistance: forward(states[B,120]) -> Tensor[B] (scalar V).
Tested the same way as the Perceiver: scripts/89_dodeca_v_probe.py runs the two-stage
pretrain->Bellman recipe and the V@d80-V@d40 saturation gate (rule 23).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

REL_GEN_EDGE, REL_SAME_PIECE, REL_STRIDE2, REL_SAME_FACE = 2, 3, 4, 5


class DodecaCNNV(nn.Module):
    MODEL_CLASS = "DodecaCNNV"

    def __init__(
        self,
        feats: dict,
        d_model: int = 256,
        n_layers: int = 4,
        mlp_hidden: Optional[int] = None,
        dropout: float = 0.0,
        output_dim: int = 1,
        inference_chunk_size: Optional[int] = 8192,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = 2 * d_model
        self.n_slots = int(feats["n_slots"])
        self.n_stickers = int(feats["n_stickers"])
        n_pieces = int(feats["n_pieces"])
        n_faces = int(feats["n_faces"])
        n_piece_slots = int(feats["n_piece_slots"])

        # external-contract attributes
        self.state_size = self.n_slots
        self.num_classes = self.n_slots
        self.encoding = "dodeca_cnn"
        self.output_dim = int(output_dim)
        self.embed_dim = d_model
        self.d_model = d_model
        self.n_layers = n_layers
        self.mlp_hidden = mlp_hidden
        self.dropout = dropout
        self.inference_chunk_size = inference_chunk_size

        nS = self.n_slots
        rel = feats["relation_full"][:nS, :nS].long()        # (120,120) slot relation block

        def norm_adj(code: int) -> torch.Tensor:
            a = (rel == code).float()
            a = a / a.sum(dim=1, keepdim=True).clamp(min=1.0)  # row-normalised mean aggregator
            return a

        self.register_buffer("A_face", norm_adj(REL_SAME_FACE), persistent=False)
        self.register_buffer("A_gen", norm_adj(REL_GEN_EDGE), persistent=False)
        self.register_buffer("A_piece", norm_adj(REL_SAME_PIECE), persistent=False)
        self.register_buffer("A_stride2", norm_adj(REL_STRIDE2), persistent=False)
        self.register_buffer("slot_piece_id", feats["piece_id_full"][:nS].long(), persistent=False)
        self.register_buffer("slot_piece_type", feats["piece_type_full"][:nS].long(), persistent=False)
        self.register_buffer("slot_piece_slot", feats["piece_slot_full"][:nS].long(), persistent=False)
        self.register_buffer("slot_face", feats["face_idx_full"][:nS].long(), persistent=False)
        self.register_buffer("arange_slots", torch.arange(nS), persistent=False)

        # static per-slot identity
        self.piece_id_emb = nn.Embedding(n_pieces + 1, d_model)
        self.piece_type_emb = nn.Embedding(3, d_model)
        self.piece_slot_emb = nn.Embedding(n_piece_slots, d_model)
        self.face_emb = nn.Embedding(n_faces + 1, d_model)
        self.slot_id_emb = nn.Embedding(nS, d_model)
        # dynamic (state-dependent)
        self.content_emb = nn.Embedding(self.n_stickers, d_model)   # sticker in slot i
        self.inv_emb = nn.Embedding(nS, d_model)                    # slot where home-sticker i went
        self.solved_emb = nn.Embedding(2, d_model)

        act = nn.GELU()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(5 * d_model, mlp_hidden), act,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mlp_hidden, d_model),
            ))
            self.norms.append(nn.LayerNorm(d_model))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), act,
                                  nn.Linear(d_model // 2, output_dim))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_config(self) -> dict:
        return {
            "model_class": self.MODEL_CLASS, "state_size": self.state_size,
            "num_classes": self.num_classes, "d_model": self.d_model,
            "n_layers": self.n_layers, "mlp_hidden": self.mlp_hidden,
            "dropout": self.dropout, "output_dim": self.output_dim,
            "encoding": self.encoding, "embed_dim": self.d_model,
            "inference_chunk_size": self.inference_chunk_size,
        }

    def _embed(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        state = state.long()
        inv = torch.empty_like(state)
        inv.scatter_(1, state, self.arange_slots.unsqueeze(0).expand(B, -1))
        solved = (state == self.arange_slots.unsqueeze(0)).long()
        static = (self.slot_id_emb(self.arange_slots)
                  + self.piece_id_emb(self.slot_piece_id)
                  + self.piece_type_emb(self.slot_piece_type)
                  + self.piece_slot_emb(self.slot_piece_slot)
                  + self.face_emb(self.slot_face).sum(dim=1))      # (120, d)
        h = static.unsqueeze(0).expand(B, -1, -1).clone()
        h = h + self.content_emb(state) + self.inv_emb(inv) + self.solved_emb(solved)
        return h

    def _forward_single(self, state: torch.Tensor) -> torch.Tensor:
        h = self._embed(state)                                     # (B, 120, d)
        for layer, norm in zip(self.layers, self.norms):
            hn = norm(h)
            m_face = torch.einsum("ij,bjd->bid", self.A_face, hn)
            m_gen = torch.einsum("ij,bjd->bid", self.A_gen, hn)
            m_piece = torch.einsum("ij,bjd->bid", self.A_piece, hn)
            m_str = torch.einsum("ij,bjd->bid", self.A_stride2, hn)
            agg = torch.cat([hn, m_face, m_gen, m_piece, m_str], dim=-1)  # (B,120,5d)
            h = h + layer(agg)
        pooled = self.final_norm(h.mean(dim=1))                    # (B, d)
        out = self.head(pooled)
        return out.squeeze(-1) if self.output_dim == 1 else out

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if (self.training or self.inference_chunk_size is None
                or state.shape[0] <= self.inference_chunk_size):
            return self._forward_single(state)
        outs = [self._forward_single(state[i : i + self.inference_chunk_size])
                for i in range(0, state.shape[0], self.inference_chunk_size)]
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


def build_dodeca_from_config(model_cfg: dict, feats: dict) -> DodecaCNNV:
    return DodecaCNNV(
        feats,
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        mlp_hidden=int(model_cfg["mlp_hidden"]) if model_cfg.get("mlp_hidden") is not None else None,
        dropout=float(model_cfg.get("dropout", 0.0)),
        output_dim=int(model_cfg.get("output_dim", 1)),
        inference_chunk_size=model_cfg.get("inference_chunk_size", 8192),
    )
