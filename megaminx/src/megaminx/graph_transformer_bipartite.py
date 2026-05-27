"""Bipartite Slot-Sticker Graph Transformer (architecture doc section 3.2).

Three node types over T = 264 tokens:
    slot nodes    [0, 120)   -- know which sticker they currently hold (state[i])
    sticker nodes [120, 240) -- know which slot they currently occupy (inv_state[j])
    action nodes  [240, 264) -- read Q(a) directly off their own token

The puzzle state is injected ONLY through per-node dynamic features (a slot's
current sticker, a sticker's current slot, a solved flag). Every attention
structure (relation matrix, distance buckets, the sparse local mask) is static
and shared across the batch -- so there is no per-state (B, T, T) mask blow-up.

Each layer is GraphGPS-style: a sparse LOCAL masked attention (the MPNN view,
restricted to structural edges) run in parallel with a GLOBAL attention carrying
learned per-relation + per-distance additive bias, their outputs concatenated and
mixed, then a residual FFN.

The key property vs. the flat GraphTransformerV: Q(a) is read from a dedicated
action node that has attended (via static AFFECTS edges) to exactly the ~25 slots
action a permutes, and through those slots' dynamic features to the stickers
currently sitting there. The model scores a move from what the move actually
touches, not from a single pooled vector.

External contract matches ResMLPDistance / GraphTransformerV so it plugs into
beam search, 09_eval_q_recall, and the QShortlisterSolver unchanged:

    forward(states: LongTensor[B, 120]) -> Tensor[B, n_actions]      (Q-head)

Static tables are built by megaminx/scripts/74_build_bipartite_features.py
(saved as megaminx/data/bipartite_features.pt).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class BipartiteGPSLayer(nn.Module):
    """GraphGPS block: parallel local-masked + global-biased attention, then FFN.

    q/k/v projections are shared between the two attention views (same content,
    different receptive field); the two attention outputs are concatenated and
    mixed by a single output projection.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0,
                 activation: str = "gelu"):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(2 * d_model, d_model)   # concat(local, global)
        self.norm2 = nn.LayerNorm(d_model)
        act = nn.GELU() if activation == "gelu" else nn.ReLU(inplace=True)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), act, nn.Linear(ffn_dim, d_model))
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, global_bias: torch.Tensor,
                local_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D); global_bias: (H, T, T) additive; local_mask: (T, T) bool
        B, T, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        dp = self.dropout_p if self.training else 0.0

        gb = global_bias.to(dtype=q.dtype).unsqueeze(0)          # (1, H, T, T)
        out_g = F.scaled_dot_product_attention(q, k, v, attn_mask=gb, dropout_p=dp)

        lm = local_mask.unsqueeze(0).unsqueeze(0)                # (1, 1, T, T) bool
        out_l = F.scaled_dot_product_attention(q, k, v, attn_mask=lm, dropout_p=dp)

        out = torch.cat([
            out_l.transpose(1, 2).reshape(B, T, D),
            out_g.transpose(1, 2).reshape(B, T, D),
        ], dim=-1)                                               # (B, T, 2D)
        x = x + self.o_proj(out)
        x = x + self.ffn(self.norm2(x))
        return x


class BipartiteGraphTransformerQ(nn.Module):
    MODEL_CLASS = "BipartiteGraphTransformerQ"

    def __init__(
        self,
        feats: dict,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        output_dim: Optional[int] = None,     # defaults to n_actions (Q-head)
        use_value_head: bool = False,
        activation: str = "gelu",
        inference_chunk_size: Optional[int] = 4096,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * d_model

        self.n_slots = int(feats["n_slots"])
        self.n_stickers = int(feats["n_stickers"])
        self.n_actions = int(feats["n_actions"])
        self.n_tokens = int(feats["n_tokens"])
        self.act0 = 2 * self.n_slots
        n_rel = int(feats["n_relations_full"])
        n_dist = int(feats["n_dist_buckets_full"])
        n_faces = int(feats["n_faces"])
        n_pieces = int(feats["n_pieces"])
        n_piece_slots = int(feats["n_piece_slots"])

        # External-contract attributes (mirror ResMLPDistance / GraphTransformerV).
        self.state_size = self.n_slots
        self.num_classes = self.n_slots
        self.encoding = "bipartite_graph_transformer"
        self.output_dim = int(output_dim) if output_dim is not None else self.n_actions
        self.embed_dim = d_model
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.activation = activation
        self.use_value_head = use_value_head
        self.inference_chunk_size = inference_chunk_size
        self.grad_checkpoint = grad_checkpoint

        # ---- static buffers ----
        self.register_buffer("token_type", feats["token_type"].long(), persistent=False)
        self.register_buffer("piece_id_full", feats["piece_id_full"].long(), persistent=False)
        self.register_buffer("piece_type_full", feats["piece_type_full"].long(), persistent=False)
        self.register_buffer("piece_slot_full", feats["piece_slot_full"].long(), persistent=False)
        self.register_buffer("face_idx_full", feats["face_idx_full"].long(), persistent=False)
        self.register_buffer("action_id_full", feats["action_id_full"].long(), persistent=False)
        self.register_buffer("action_dir_full", feats["action_dir_full"].long(), persistent=False)
        self.register_buffer("action_inv_full", feats["action_inv_full"].long(), persistent=False)
        self.register_buffer("relation_full", feats["relation_full"].long(), persistent=False)
        self.register_buffer("dist_bucket_full", feats["dist_bucket_full"].long(), persistent=False)
        self.register_buffer("local_mask", feats["local_mask"].bool(), persistent=False)
        self.register_buffer("arange_slots", torch.arange(self.n_slots), persistent=False)

        # ---- static identity embeddings ----
        self.type_emb = nn.Embedding(3, d_model)
        self.piece_id_emb = nn.Embedding(n_pieces + 1, d_model)     # +1 sentinel (action)
        self.piece_type_emb = nn.Embedding(3, d_model)              # corner/edge/action
        self.piece_slot_emb = nn.Embedding(n_piece_slots, d_model)
        self.face_emb = nn.Embedding(n_faces + 1, d_model)         # +1 for "none" (idx 0)
        self.action_id_emb = nn.Embedding(self.n_actions + 1, d_model)
        self.action_dir_emb = nn.Embedding(3, d_model)             # cw/ccw/na
        self.action_inv_emb = nn.Embedding(self.n_actions + 1, d_model)

        # ---- dynamic (state-dependent) embeddings ----
        self.content_emb = nn.Embedding(self.n_stickers + 1, d_model)  # slot's current sticker (+NA)
        self.loc_emb = nn.Embedding(self.n_slots + 1, d_model)         # sticker's current slot (+NA)
        self.solved_emb = nn.Embedding(3, d_model)                     # not/solved/na

        # ---- per-head attention bias tables ----
        self.relation_bias = nn.Embedding(n_rel, n_heads)
        self.dist_bias = nn.Embedding(n_dist, n_heads)
        nn.init.zeros_(self.relation_bias.weight)
        nn.init.zeros_(self.dist_bias.weight)

        self.layers = nn.ModuleList([
            BipartiteGPSLayer(d_model, n_heads, ffn_dim, dropout=dropout, activation=activation)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.q_head = nn.Linear(d_model, 1)
        if use_value_head:
            self.value_head = nn.Linear(d_model, 1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_config(self) -> dict:
        return {
            "model_class": self.MODEL_CLASS,
            "state_size": self.state_size,
            "num_classes": self.num_classes,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "output_dim": self.output_dim,
            "use_value_head": self.use_value_head,
            "activation": self.activation,
            "encoding": self.encoding,
            "embed_dim": self.d_model,
            "inference_chunk_size": self.inference_chunk_size,
            "grad_checkpoint": self.grad_checkpoint,
        }

    def _static_token_emb(self) -> torch.Tensor:
        """(T, d_model) state-independent identity embedding."""
        return (
            self.type_emb(self.token_type)
            + self.piece_id_emb(self.piece_id_full)
            + self.piece_type_emb(self.piece_type_full)
            + self.piece_slot_emb(self.piece_slot_full)
            + self.face_emb(self.face_idx_full).sum(dim=1)
            + self.action_id_emb(self.action_id_full)
            + self.action_dir_emb(self.action_dir_full)
            + self.action_inv_emb(self.action_inv_full)
        )

    def _global_bias(self) -> torch.Tensor:
        """(H, T, T) additive attention bias = relation + distance, per head."""
        rb = self.relation_bias(self.relation_full)      # (T, T, H)
        db = self.dist_bias(self.dist_bucket_full)        # (T, T, H)
        return (rb + db).permute(2, 0, 1)                 # (H, T, T)

    def _embed(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, n_slots) int -> (B, T, d_model) token embeddings."""
        B = state.shape[0]
        state = state.long()
        # inv_state[b, sticker] = slot currently holding that sticker.
        inv = torch.empty_like(state)
        inv.scatter_(1, state, self.arange_slots.unsqueeze(0).expand(B, -1))

        nS, nT = self.n_slots, self.n_tokens
        content_idx = self.content_emb.num_embeddings - 1  # NA sentinel
        loc_idx_na = self.loc_emb.num_embeddings - 1

        content = torch.full((B, nT), content_idx, dtype=torch.long, device=state.device)
        content[:, :nS] = state                            # slot i holds sticker state[i]
        loc = torch.full((B, nT), loc_idx_na, dtype=torch.long, device=state.device)
        loc[:, nS:2 * nS] = inv                            # sticker j sits in slot inv[j]

        solved_slot = (state == self.arange_slots.unsqueeze(0)).long()  # (B, nS)
        solved = torch.full((B, nT), 2, dtype=torch.long, device=state.device)  # 2 = NA (actions)
        solved[:, :nS] = solved_slot
        solved[:, nS:2 * nS] = solved_slot                 # sticker j solved iff state[j]==j

        h = self._static_token_emb().unsqueeze(0).expand(B, -1, -1).clone()
        h = h + self.content_emb(content) + self.loc_emb(loc) + self.solved_emb(solved)
        return h

    def _forward_single(self, state: torch.Tensor) -> torch.Tensor:
        h = self._embed(state)                             # (B, T, D)
        gbias = self._global_bias()                        # (H, T, T)
        # Explicit-attn_mask SDPA uses the math kernel, which saves the full
        # (B, H, T, T) score matrix for backward. With 2 SDPA x n_layers that
        # OOMs at large batch, so checkpoint each layer in training (dropout=0,
        # so recomputation is deterministic and exact).
        use_ckpt = self.training and self.grad_checkpoint and torch.is_grad_enabled()
        for layer in self.layers:
            if use_ckpt:
                h = checkpoint(layer, h, gbias, self.local_mask, use_reentrant=False)
            else:
                h = layer(h, gbias, self.local_mask)
        act = self.final_norm(h[:, self.act0:self.act0 + self.n_actions])  # (B, n_actions, D)
        q = self.q_head(act).squeeze(-1)                   # (B, n_actions)
        return q

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if (
            self.training
            or self.inference_chunk_size is None
            or state.shape[0] <= self.inference_chunk_size
        ):
            return self._forward_single(state)
        outs = []
        for i in range(0, state.shape[0], self.inference_chunk_size):
            outs.append(self._forward_single(state[i : i + self.inference_chunk_size]))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        """Optional scalar V from pooled slot+sticker tokens (requires use_value_head)."""
        if not self.use_value_head:
            raise RuntimeError("model built without a value head")
        h = self._embed(state)
        gbias = self._global_bias()
        for layer in self.layers:
            h = layer(h, gbias, self.local_mask)
        pooled = self.final_norm(h[:, : 2 * self.n_slots].mean(dim=1))
        return self.value_head(pooled).squeeze(-1)

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


def build_bipartite_from_config(model_cfg: dict, feats: dict) -> BipartiteGraphTransformerQ:
    return BipartiteGraphTransformerQ(
        feats,
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        ffn_dim=int(model_cfg["ffn_dim"]) if model_cfg.get("ffn_dim") is not None else None,
        dropout=float(model_cfg.get("dropout", 0.0)),
        output_dim=model_cfg.get("output_dim"),
        use_value_head=bool(model_cfg.get("use_value_head", False)),
        activation=str(model_cfg.get("activation", "gelu")),
        inference_chunk_size=model_cfg.get("inference_chunk_size", 4096),
        grad_checkpoint=bool(model_cfg.get("grad_checkpoint", True)),
    )
