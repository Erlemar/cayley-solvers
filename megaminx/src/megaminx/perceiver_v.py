"""Perceiver IO value model (architecture doc section 3.9).

A small set of learned LATENTS cross-attend into the puzzle's token set
(120 slot tokens + 120 sticker tokens), then self-attend among themselves; the
pooled latents predict the scalar value V. The doc frames this as "a good
compromise if the graph transformer is too heavy".

Key contrast with the rejected bipartite GT (section 3.2): there is NO attention
mask anywhere. Latents attend densely to all tokens (cross), then densely to each
other (self). So the attention score tensors are tiny -- cross is (B, H, n_latents,
T=240), self is (B, H, n_latents, n_latents) -- and there is no math-kernel
(B,H,T,T) blow-up (CLAUDE rules 22/27 do not bite). It is also compile-safe.

State enters ONLY through per-token dynamic features (a slot's current sticker, a
sticker's current slot, a solved flag), exactly as in the bipartite builder; all
static identity features (type / piece / face / piece-slot) and the token set are
shared across the batch. Reuses megaminx/data/bipartite_features.pt (built by
scripts/74_build_bipartite_features.py); we use only the first 2*n_slots tokens
(slot + sticker), dropping the 24 action tokens (not needed for a V head).

External contract mirrors ResMLPDistance / GraphTransformerV:
    forward(states: LongTensor[B, 120]) -> Tensor[B]      (scalar V)

THE point of building this is the saturation gate (CLAUDE rule 23): does a
Perceiver-encoded V SATURATE near the diameter on deep random walks (like the
working ResMLP V, ~29) or keep drifting (like the rejected GraphTransformer V,
V@d80=42)? Probe + gate live in scripts/88_perceiver_v_probe.py.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _SelfBlock(nn.Module):
    """Pre-norm transformer encoder block over the latents (dense, no mask)."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class PerceiverV(nn.Module):
    MODEL_CLASS = "PerceiverV"

    def __init__(
        self,
        feats: dict,
        d_model: int = 256,
        n_latents: int = 128,
        n_cross_heads: int = 8,
        n_self_layers: int = 4,
        n_self_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        output_dim: int = 1,
        inference_chunk_size: Optional[int] = 8192,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * d_model
        self.n_slots = int(feats["n_slots"])
        self.n_stickers = int(feats["n_stickers"])
        self.n_tok = 2 * self.n_slots          # slot + sticker tokens (drop actions)
        n_pieces = int(feats["n_pieces"])
        n_faces = int(feats["n_faces"])
        n_piece_slots = int(feats["n_piece_slots"])

        # External-contract attributes.
        self.state_size = self.n_slots
        self.num_classes = self.n_slots
        self.encoding = "perceiver"
        self.output_dim = int(output_dim)
        self.embed_dim = d_model
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_cross_heads = n_cross_heads
        self.n_self_layers = n_self_layers
        self.n_self_heads = n_self_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.inference_chunk_size = inference_chunk_size

        # ---- static per-token features (slot+sticker tokens only) ----
        sl = slice(0, self.n_tok)
        self.register_buffer("token_type", feats["token_type"][sl].long(), persistent=False)
        self.register_buffer("piece_id", feats["piece_id_full"][sl].long(), persistent=False)
        self.register_buffer("piece_type", feats["piece_type_full"][sl].long(), persistent=False)
        self.register_buffer("piece_slot", feats["piece_slot_full"][sl].long(), persistent=False)
        self.register_buffer("face_idx", feats["face_idx_full"][sl].long(), persistent=False)
        self.register_buffer("arange_slots", torch.arange(self.n_slots), persistent=False)

        self.type_emb = nn.Embedding(3, d_model)
        self.piece_id_emb = nn.Embedding(n_pieces + 1, d_model)
        self.piece_type_emb = nn.Embedding(3, d_model)
        self.piece_slot_emb = nn.Embedding(n_piece_slots, d_model)
        self.face_emb = nn.Embedding(n_faces + 1, d_model)
        # dynamic (state-dependent)
        self.content_emb = nn.Embedding(self.n_stickers + 1, d_model)  # slot's sticker (+NA)
        self.loc_emb = nn.Embedding(self.n_slots + 1, d_model)         # sticker's slot (+NA)
        self.solved_emb = nn.Embedding(3, d_model)                     # not/solved/na

        # ---- Perceiver core ----
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)
        self.cross_norm_l = nn.LayerNorm(d_model)
        self.cross_norm_x = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_cross_heads, dropout=dropout,
                                                batch_first=True)
        self.self_blocks = nn.ModuleList([
            _SelfBlock(d_model, n_self_heads, ffn_dim, dropout) for _ in range(n_self_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, output_dim)
        )

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_config(self) -> dict:
        return {
            "model_class": self.MODEL_CLASS,
            "state_size": self.state_size, "num_classes": self.num_classes,
            "d_model": self.d_model, "n_latents": self.n_latents,
            "n_cross_heads": self.n_cross_heads, "n_self_layers": self.n_self_layers,
            "n_self_heads": self.n_self_heads, "ffn_dim": self.ffn_dim,
            "dropout": self.dropout, "output_dim": self.output_dim,
            "encoding": self.encoding, "embed_dim": self.d_model,
            "inference_chunk_size": self.inference_chunk_size,
        }

    def _static_tokens(self) -> torch.Tensor:
        """(n_tok, d_model) state-independent token identity embedding."""
        return (
            self.type_emb(self.token_type)
            + self.piece_id_emb(self.piece_id)
            + self.piece_type_emb(self.piece_type)
            + self.piece_slot_emb(self.piece_slot)
            + self.face_emb(self.face_idx).sum(dim=1)
        )

    def _embed(self, state: torch.Tensor) -> torch.Tensor:
        """state (B, n_slots) -> (B, n_tok, d_model) token embeddings."""
        B = state.shape[0]
        state = state.long()
        inv = torch.empty_like(state)
        inv.scatter_(1, state, self.arange_slots.unsqueeze(0).expand(B, -1))
        nS, nT = self.n_slots, self.n_tok
        na_content = self.content_emb.num_embeddings - 1
        na_loc = self.loc_emb.num_embeddings - 1
        content = torch.full((B, nT), na_content, dtype=torch.long, device=state.device)
        content[:, :nS] = state                          # slot i holds sticker state[i]
        loc = torch.full((B, nT), na_loc, dtype=torch.long, device=state.device)
        loc[:, nS:2 * nS] = inv                          # sticker j sits in slot inv[j]
        solved_slot = (state == self.arange_slots.unsqueeze(0)).long()
        solved = torch.full((B, nT), 2, dtype=torch.long, device=state.device)
        solved[:, :nS] = solved_slot
        solved[:, nS:2 * nS] = solved_slot
        h = self._static_tokens().unsqueeze(0).expand(B, -1, -1).clone()
        h = h + self.content_emb(content) + self.loc_emb(loc) + self.solved_emb(solved)
        return h

    def _forward_single(self, state: torch.Tensor) -> torch.Tensor:
        x = self._embed(state)                                   # (B, T, D)
        B = x.shape[0]
        lat = self.latents.expand(B, -1, -1)                     # (B, L, D)
        a, _ = self.cross_attn(self.cross_norm_l(lat), self.cross_norm_x(x),
                               self.cross_norm_x(x), need_weights=False)
        lat = lat + a
        for blk in self.self_blocks:
            lat = blk(lat)
        pooled = self.final_norm(lat.mean(dim=1))                # (B, D)
        out = self.head(pooled)                                  # (B, output_dim)
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


def build_perceiver_from_config(model_cfg: dict, feats: dict) -> PerceiverV:
    return PerceiverV(
        feats,
        d_model=int(model_cfg.get("d_model", 256)),
        n_latents=int(model_cfg.get("n_latents", 128)),
        n_cross_heads=int(model_cfg.get("n_cross_heads", 8)),
        n_self_layers=int(model_cfg.get("n_self_layers", 4)),
        n_self_heads=int(model_cfg.get("n_self_heads", 8)),
        ffn_dim=int(model_cfg["ffn_dim"]) if model_cfg.get("ffn_dim") is not None else None,
        dropout=float(model_cfg.get("dropout", 0.0)),
        output_dim=int(model_cfg.get("output_dim", 1)),
        inference_chunk_size=model_cfg.get("inference_chunk_size", 8192),
    )
