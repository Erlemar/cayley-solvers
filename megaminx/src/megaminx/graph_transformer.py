"""Graph-biased transformer over megaminx sticker tokens.

A puzzle-native architecture with stronger inductive bias than the flat ResMLP.
120 sticker positions are tokens. Self-attention is augmented with two static
biases derived from the puzzle's group structure:

  - relation_id  in {0..5}: same_position, generator_edge, same_piece,
                            same_cycle_stride2, same_face, unrelated.
  - dist_bucket  in {0..8}: shortest-path distance on the generator graph,
                            capped at 8.

Bias tables are learned per attention head:

  attn_score[h, i, j] += relation_bias[h, relation_id[i, j]]
                      +  dist_bias[h, dist_bucket[i, j]]

The model exposes the same call signature as ResMLPDistance so it plugs into
beam search and Bellman target code with no other changes:

    forward(states: LongTensor[B, 120]) -> Tensor[B]                 (V model)
    forward(states: LongTensor[B, 120]) -> Tensor[B, n_actions]      (Q head)

`GraphTransformerVPi` adds a separate policy head sharing the trunk.

The static graph features dict is built by
`megaminx/scripts/72_build_graph_features.py` (saved as
`megaminx/data/graph_features.pt`).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_attn_bias(
    relation_bias: nn.Embedding,
    dist_bias: nn.Embedding,
    relation_id: torch.Tensor,   # (S, S) long
    dist_bucket: torch.Tensor,   # (S, S) long
    cls_pad: int = 1,
) -> torch.Tensor:
    """Returns the attention bias matrix (n_heads, S+cls_pad, S+cls_pad).

    The bias entries for the CLS row/col are zero (effectively "no bias" so the
    CLS token attends uniformly relative to the learned scale).
    """
    rb = relation_bias(relation_id)           # (S, S, H)
    db = dist_bias(dist_bucket)               # (S, S, H)
    inner = (rb + db).permute(2, 0, 1)        # (H, S, S)
    if cls_pad == 0:
        return inner
    H, S, _ = inner.shape
    full = inner.new_zeros((H, S + cls_pad, S + cls_pad))
    full[:, cls_pad:, cls_pad:] = inner
    return full


class GraphTransformerLayer(nn.Module):
    """Single pre-norm transformer block with additive attention bias support.

    Uses F.scaled_dot_product_attention with an explicit attn_mask broadcasting
    over the batch dimension.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0,
                 activation: str = "gelu"):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"unknown activation {activation!r}")
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            act,
            nn.Linear(ffn_dim, d_model),
        )
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        # x:        (B, T, D)
        # attn_bias:(H, T, T)
        B, T, D = x.shape
        h = self.attn_norm(x)
        q = self.q_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # broadcast bias over batch: (1, H, T, T)
        mask = attn_bias.to(dtype=q.dtype).unsqueeze(0)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.o_proj(attn)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GraphTransformerV(nn.Module):
    """V-head graph transformer over 120 sticker tokens with CLS readout.

    Token features:
      - sticker_id (state[i])
      - position_id (i)
      - piece_type  (corner/edge)
      - piece_id    (0..49)

    Attention bias:
      - relation_bias[h, relation_id[i,j]]
      - dist_bias[h,     dist_bucket[i,j]]

    Readout:
      - learned CLS token prepended; final head reads CLS row.
    """

    # Model identity for the polymorphic loader.
    MODEL_CLASS = "GraphTransformerV"

    def __init__(
        self,
        graph_features: dict,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        output_dim: int = 1,
        use_relation_bias: bool = True,
        use_dist_bias: bool = True,
        activation: str = "gelu",
        inference_chunk_size: Optional[int] = 4096,
    ):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * d_model

        state_size = int(graph_features["state_size"])
        n_classes = int(graph_features["n_classes"])
        n_pieces = int(graph_features["n_pieces"])
        n_relations = int(graph_features["n_relations"])
        n_dist_buckets = int(graph_features["n_dist_buckets"])
        n_generators = int(graph_features["n_generators"])

        # Compatibility attributes used by callers (cayley.search,
        # cayley.training, beam search), mirroring ResMLPDistance's interface.
        self.state_size = state_size
        self.num_classes = n_classes
        self.embed_dim = d_model
        self.encoding = "graph_transformer"
        self.output_dim = output_dim
        self.n_pieces = n_pieces
        self.n_relations = n_relations
        self.n_dist_buckets = n_dist_buckets
        self.n_generators = n_generators
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.use_relation_bias = use_relation_bias
        self.use_dist_bias = use_dist_bias
        self.activation = activation
        self.inference_chunk_size = inference_chunk_size

        # ---- buffers (static, derived from puzzle) ----
        self.register_buffer("relation_id", graph_features["relation_id"].long(), persistent=False)
        self.register_buffer("dist_bucket", graph_features["dist_bucket"].long(), persistent=False)
        self.register_buffer("piece_type", graph_features["piece_type"].long(), persistent=False)
        self.register_buffer("piece_id_table", graph_features["piece_id"].long(), persistent=False)
        self.register_buffer("positions_arange", torch.arange(state_size).unsqueeze(0), persistent=False)

        # ---- embeddings ----
        self.sticker_emb = nn.Embedding(n_classes, d_model)
        self.position_emb = nn.Embedding(state_size, d_model)
        self.piece_type_emb = nn.Embedding(2, d_model)
        self.piece_id_emb = nn.Embedding(n_pieces, d_model)

        # ---- CLS token (learned readout) ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # ---- per-head bias tables ----
        if use_relation_bias:
            self.relation_bias = nn.Embedding(n_relations, n_heads)
            nn.init.zeros_(self.relation_bias.weight)
        else:
            self.register_parameter("relation_bias", None)
        if use_dist_bias:
            self.dist_bias = nn.Embedding(n_dist_buckets, n_heads)
            nn.init.zeros_(self.dist_bias.weight)
        else:
            self.register_parameter("dist_bias", None)

        # ---- transformer stack ----
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_model, n_heads, ffn_dim, dropout=dropout, activation=activation)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

        # Default init for nn.Linear / nn.Embedding is OK for the depth/scale we use.

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_model_config(self) -> dict:
        """Serialization dict used by the polymorphic checkpoint loader."""
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
            "use_relation_bias": self.use_relation_bias,
            "use_dist_bias": self.use_dist_bias,
            "activation": self.activation,
            "encoding": self.encoding,
            "embed_dim": self.d_model,
            "inference_chunk_size": self.inference_chunk_size,
        }

    def _attn_bias(self) -> torch.Tensor:
        # Compute over the static (S,S) inner block + zero-padded CLS row/col.
        # The bias build is ~0.5 ms (4 emb lookups + add + permute + pad on a
        # 120x120x4 tensor). Profiled vs a getattr-based eval-mode cache —
        # cache was slower because the conditional + getattr cost outweighs
        # the savings.
        H = self.n_heads
        S = self.state_size
        if self.relation_bias is None and self.dist_bias is None:
            return self.cls_token.new_zeros((H, S + 1, S + 1))
        rb = (
            self.relation_bias(self.relation_id)
            if self.relation_bias is not None
            else self.cls_token.new_zeros((S, S, H))
        )
        db = (
            self.dist_bias(self.dist_bucket)
            if self.dist_bias is not None
            else self.cls_token.new_zeros((S, S, H))
        )
        inner = (rb + db).permute(2, 0, 1)   # (H, S, S)
        full = inner.new_zeros((H, S + 1, S + 1))
        full[:, 1:, 1:] = inner
        return full

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S) int. Returns (B, S+1, D) with CLS prepended."""
        B = x.shape[0]
        positions = self.positions_arange.expand(B, -1)            # (B, S)
        h = (
            self.sticker_emb(x.long())
            + self.position_emb(positions)
            + self.piece_type_emb(self.piece_type.unsqueeze(0).expand(B, -1))
            + self.piece_id_emb(self.piece_id_table.unsqueeze(0).expand(B, -1))
        )
        cls = self.cls_token.expand(B, 1, -1)
        return torch.cat([cls, h], dim=1)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encode_tokens(x)                                  # (B, S+1, D)
        bias = self._attn_bias()                                    # (H, S+1, S+1)
        for layer in self.layers:
            h = layer(h, bias)
        h_cls = self.final_norm(h[:, 0])                            # (B, D)
        out = self.head(h_cls)                                      # (B, output_dim)
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.training
            or self.inference_chunk_size is None
            or x.shape[0] <= self.inference_chunk_size
        ):
            return self._forward_single(x)
        outs: list[torch.Tensor] = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            outs.append(self._forward_single(x[i : i + self.inference_chunk_size]))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class GraphTransformerVPi(GraphTransformerV):
    """Shared-trunk variant with a separate policy head returning per-action logits.

    forward returns either (policy_logits, value) (training) or value (eval),
    matching the existing ResMLPGFlowNet convention. For beam-search loading,
    the loader can call `.value_head_forward(x)` or set output_dim=1 mode.
    """

    MODEL_CLASS = "GraphTransformerVPi"

    def __init__(self, graph_features: dict, n_actions: int = 24, **kw):
        super().__init__(graph_features, output_dim=1, **kw)
        self.n_actions = n_actions
        self.policy_head = nn.Linear(self.d_model, n_actions)
        # value_head is the inherited self.head (output_dim=1)

    def value_head_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: V only (for beam-search inference)."""
        return GraphTransformerV.forward(self, x)

    def _forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encode_tokens(x)
        bias = self._attn_bias()
        for layer in self.layers:
            h = layer(h, bias)
        return self.final_norm(h[:, 0])                             # (B, D)

    def forward(self, x: torch.Tensor):
        # Train mode returns (policy_logits, value); eval returns value only,
        # matching the V-only contract expected by beam search via load_model_checkpoint.
        if self.training:
            trunk = self._forward_trunk(x)
            pi = self.policy_head(trunk)
            v = self.head(trunk).squeeze(-1)
            return pi, v
        return self.value_head_forward(x)

    def get_model_config(self) -> dict:
        d = super().get_model_config()
        d["model_class"] = self.MODEL_CLASS
        d["n_actions"] = self.n_actions
        return d


def build_graph_transformer_from_config(model_cfg: dict, graph_features: dict) -> GraphTransformerV:
    """Helper used by the checkpoint loader and YAML-driven trainers.

    `model_cfg` keys subset of {state_size, num_classes, d_model, n_layers,
    n_heads, ffn_dim, dropout, output_dim, use_relation_bias, use_dist_bias,
    activation, inference_chunk_size, model_class, n_actions}.
    """
    mc = model_cfg.get("model_class", "GraphTransformerV")
    common = dict(
        graph_features=graph_features,
        d_model=int(model_cfg.get("d_model", 256)),
        n_layers=int(model_cfg.get("n_layers", 4)),
        n_heads=int(model_cfg.get("n_heads", 8)),
        ffn_dim=int(model_cfg["ffn_dim"]) if model_cfg.get("ffn_dim") is not None else None,
        dropout=float(model_cfg.get("dropout", 0.0)),
        use_relation_bias=bool(model_cfg.get("use_relation_bias", True)),
        use_dist_bias=bool(model_cfg.get("use_dist_bias", True)),
        activation=str(model_cfg.get("activation", "gelu")),
        inference_chunk_size=model_cfg.get("inference_chunk_size", 4096),
    )
    if mc == "GraphTransformerVPi":
        return GraphTransformerVPi(
            n_actions=int(model_cfg.get("n_actions", 24)),
            **common,
        )
    return GraphTransformerV(
        output_dim=int(model_cfg.get("output_dim", 1)),
        **common,
    )
