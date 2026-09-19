"""Architecture zoo for the tetraminx Q-head comparison matrix.

Every model here presents the SAME interface so the search stack and the TPU port
need no per-architecture special-casing:

    forward(states int (B, 88)) -> (B, 24)     the all-neighbours Q head
    value(states)               -> (B,)        only if has_value_head
    get_model_config()          -> dict        round-trips through build_model()

Trunk parameter names are kept identical across the ResMLP variants
(`embedding.` / `input_stack.` / `res_blocks.`) so checkpoints warm-start into each
other, and the 24-wide head is called `head` in the ResMLP family so an existing
`ResMLPDistance(output_dim=24)` checkpoint (e.g. tq0) loads straight into the
dual-head variant.

Architectures
  resmlp      ResMLPDistance trunk + 24-wide head            (what tq0/tq2 are)
  gflownet    cayley.gflow_model.ResMLPGFlowNet, policy_head used as Q
  transformer PieceTransformer over the 50 physical pieces derived by
              scripts/50_derive_piece_layout.py
  latent_relational
              50 physical pieces -> learned latent bottleneck -> action queries;
              a standalone relational Q model with no inference-time teacher
  generator_isab
              ResMLP Q base + zero-initialised generator-conditioned correction;
              bidirectional induced attention keeps all physical-piece tokens

`az_head=True` adds a scalar value head on the shared trunk. Note this only changes
BEAM results through multi-task interference on the trunk, unless inference actually
uses the value head (Q shortlist -> V rerank), so both modes must be evaluated.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from cayley.gflow_model import ResMLPGFlowNet
from cayley.model import ResBlock

ARCHS = ("resmlp", "gflownet", "transformer", "latent_relational", "generator_isab")


# --------------------------------------------------------------------------
def _resmlp_trunk(state_size, num_classes, hidden_dims, num_res_blocks, embed_dim):
    embedding = nn.Embedding(num_classes, embed_dim)
    layers, prev = [], state_size * embed_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
        prev = h
    return embedding, nn.Sequential(*layers), nn.ModuleList(
        [ResBlock(prev) for _ in range(num_res_blocks)]), prev


class ResMLPQ(nn.Module):
    """ResMLPDistance-compatible trunk with a 24-wide Q head and an optional value head."""

    def __init__(self, state_size=88, num_classes=88, hidden_dims=(2048, 512),
                 num_res_blocks=2, embed_dim=16, n_actions=24, az_head=False):
        super().__init__()
        self.state_size, self.num_classes = state_size, num_classes
        self.hidden_dims, self.num_res_blocks = tuple(hidden_dims), num_res_blocks
        self.embed_dim, self.output_dim = embed_dim, n_actions
        self.has_value_head = bool(az_head)
        self.embedding, self.input_stack, self.res_blocks, prev = _resmlp_trunk(
            state_size, num_classes, hidden_dims, num_res_blocks, embed_dim)
        self.head = nn.Linear(prev, n_actions)              # name matches ResMLPDistance
        self.value_head = nn.Linear(prev, 1) if az_head else None
        self.return_value = False   # training sets this; inference leaves it off

    def trunk(self, x):
        h = self.embedding(x.long()).flatten(start_dim=-2).to(self.input_stack[0].weight.dtype)
        h = self.input_stack(h)
        for b in self.res_blocks:
            h = b(h)
        return h

    def forward(self, x):
        h = self.trunk(x)
        q = self.head(h)
        if self.return_value and self.value_head is not None:
            return q, self.value_head(h).squeeze(-1)
        return q

    def value(self, x):
        if self.value_head is None:
            raise RuntimeError("model has no value head")
        return self.value_head(self.trunk(x)).squeeze(-1)

    def get_model_config(self):
        return {"arch": "resmlp", "state_size": self.state_size, "num_classes": self.num_classes,
                "hidden_dims": list(self.hidden_dims), "num_res_blocks": self.num_res_blocks,
                "embed_dim": self.embed_dim, "n_actions": self.output_dim,
                "az_head": self.has_value_head}


class GFlowNetQ(nn.Module):
    """ResMLPGFlowNet with `policy_head` read as the Q head.

    The net is architecturally dual-head already (policy + value + log_Z), so
    `az_head` here controls whether the value head is TRAINED and exposed, not
    whether it exists -- keeping the cell comparable to the ResMLP pair.
    """

    def __init__(self, state_size=88, num_classes=88, hidden_dims=(2048, 512),
                 num_res_blocks=2, embed_dim=16, n_actions=24, az_head=False):
        super().__init__()
        self.net = ResMLPGFlowNet(state_size=state_size, num_classes=num_classes,
                                  hidden_dims=tuple(hidden_dims), num_res_blocks=num_res_blocks,
                                  encoding="embedding", embed_dim=embed_dim,
                                  n_actions=n_actions, inference_chunk_size=None)
        self.state_size, self.num_classes = state_size, num_classes
        self.hidden_dims, self.num_res_blocks = tuple(hidden_dims), num_res_blocks
        self.embed_dim, self.output_dim = embed_dim, n_actions
        self.has_value_head = bool(az_head)
        self.return_value = False

    def forward(self, x):
        h = self.net.trunk(x)
        q = self.net.policy_head(h)
        if self.return_value and self.has_value_head:
            return q, self.net.value_head(h).squeeze(-1)
        return q

    def value(self, x):
        if not self.has_value_head:
            raise RuntimeError("model has no value head")
        return self.net.value_head(self.net.trunk(x)).squeeze(-1)

    def get_model_config(self):
        return {"arch": "gflownet", "state_size": self.state_size, "num_classes": self.num_classes,
                "hidden_dims": list(self.hidden_dims), "num_res_blocks": self.num_res_blocks,
                "embed_dim": self.embed_dim, "n_actions": self.output_dim,
                "az_head": self.has_value_head}


# --------------------------------------------------------------------------
class _SelfAttn(nn.Module):
    """Fused-QKV self-attention through F.scaled_dot_product_attention.

    Deliberately NOT nn.MultiheadAttention. In training mode MHA takes the unfused
    math path, which materialises AND saves-for-backward a (B, H, T, T) score
    tensor per layer. Measured on an A100 (7680 rows, T=51, 4 layers): forward was
    66 ms but the train step 484 ms -- backward alone 6.3x the forward, where ~2x
    is normal. Same failure family as CLAUDE.md Rule 27.

    Parameter NAMES are kept identical to nn.MultiheadAttention
    (`in_proj_weight`, `in_proj_bias`, `out_proj.*`) so existing checkpoints and
    the JAX loader in jax_model.py both keep working unchanged.
    """

    def __init__(self, d_model, nhead, attn_impl="sdpa"):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        if attn_impl not in ("sdpa", "einsum"):
            raise ValueError(f"attn_impl must be 'sdpa' or 'einsum', got {attn_impl!r}")
        self.attn_impl = attn_impl
        self.d_model, self.nhead = d_model, nhead
        self.head_dim = d_model // nhead
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d_model, d_model))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.in_proj_weight)      # matches MHA's reset
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        if self.attn_impl == "einsum":
            return self._forward_einsum(qkv, B, T)
        qkv = qkv.view(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        a = F.scaled_dot_product_attention(q, k, v)        # no mask -> flash path
        return self.out_proj(a.transpose(1, 2).reshape(B, T, self.d_model))

    def _forward_einsum(self, qkv, B, T):
        """Same maths, but the heads never leave dim 2. TPU-motivated.

        The SDPA path above costs 25.1 ms fwd+bwd per layer on a v6e at our
        shapes; this costs 14.9 (1.69x). The saving is in the BACKWARD -- the
        permute itself is only 0.06 ms forward, but reversing it around SDPA's
        saved tensors is not. On a GPU the SDPA path wins (flash attention), so
        this is opt-in rather than the new default.

        The softmax is taken in float32 on purpose. torch_tpu's fused SDPA
        accumulates internally in float32; a naive bf16 einsum does not, and
        that alone measured 1.5e-2 relative error against a float64 reference
        versus SDPA's 8.2e-3. Promoting only the softmax -- elementwise on
        (B,H,T,T), so nearly free -- closes the gap without paying for an fp32
        matmul. Both paths sit at ~4.6e-3 in fp32 regardless: that is the MXU's
        default 1-pass bf16 multiply, not a property of either formulation.
        """
        q, k, v = qkv.view(B, T, 3, self.nhead, self.head_dim).unbind(2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bthd,bshd->bhts", q, k) * scale
        probs = scores.float().softmax(dim=-1).to(v.dtype)
        a = torch.einsum("bhts,bshd->bthd", probs, v)
        return self.out_proj(a.reshape(B, T, self.d_model))


class _CrossAttn(nn.Module):
    """Multi-head cross-attention with fused KV projection and SDPA.

    Query and key/value token counts differ in both latent compression (8 <- 50)
    and action decoding (24 <- 8), so this cannot reuse `_SelfAttn` directly.
    """

    def __init__(self, d_model, nhead):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model, self.nhead = d_model, nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.kv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, key_value, attn_mask=None):
        b, tq, _ = query.shape
        tk = key_value.size(1)
        q = self.q_proj(query).view(b, tq, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(key_value).view(
            b, tk, 2, self.nhead, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(a.transpose(1, 2).reshape(b, tq, self.d_model))


class _EncoderBlock(nn.Module):
    """Pre-norm block, unmasked self-attention so SDPA takes the fused path."""

    def __init__(self, d_model, nhead, ff_dim, dropout, activation="silu",
                 attn_impl="sdpa"):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _SelfAttn(d_model, nhead, attn_impl=attn_impl)
        self.norm2 = nn.LayerNorm(d_model)
        act = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[activation]()
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), act,
                                nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        return x + self.dropout(self.ff(self.norm2(x)))


class _CrossBlock(nn.Module):
    """Pre-norm residual cross-attention followed by a query-token FF sublayer."""

    def __init__(self, d_model, nhead, ff_dim, dropout, activation="silu"):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.key_value_norm = nn.LayerNorm(d_model)
        self.attn = _CrossAttn(d_model, nhead)
        self.ff_norm = nn.LayerNorm(d_model)
        act = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}[activation]()
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), act,
                                nn.Dropout(dropout), nn.Linear(ff_dim, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, attn_mask=None):
        query = query + self.dropout(self.attn(
            self.query_norm(query), self.key_value_norm(key_value), attn_mask=attn_mask))
        return query + self.dropout(self.ff(self.ff_norm(query)))


def _folded_piece_tokens(
    x, *, piece_positions, piece_mask, piece_types, piece_indices,
    local_value_embedding, piece_projection, piece_position_embedding,
    piece_type_embedding, num_pieces, max_piece_size, num_classes, d_model,
):
    """Encode physical pieces without materialising `(B, P, K, D)`.

    This is shared by the full PieceTransformer and the latent model while leaving
    every parameter name unchanged, so existing PieceTransformer checkpoints remain
    compatible.
    """
    b = x.size(0)
    vals = x.long().index_select(1, piece_positions.reshape(-1)).view(
        b, num_pieces, max_piece_size)
    W = piece_projection.weight.view(d_model, max_piece_size, d_model)
    E = local_value_embedding.weight.view(max_piece_size, num_classes, d_model)
    table = torch.einsum("ojd,jvd->jvo", W, E)
    h = x.new_zeros((b, num_pieces, d_model), dtype=table.dtype)
    for j in range(max_piece_size):
        oh = F.one_hot(vals[:, :, j].reshape(-1), num_classes).to(table.dtype)
        contrib = (oh @ table[j]).view(b, num_pieces, d_model)
        h = h + contrib * piece_mask[:, j].view(1, -1, 1).to(table.dtype)
    h = h + piece_projection.bias
    h = h + piece_position_embedding(piece_indices).unsqueeze(0)
    h = h + piece_type_embedding(piece_types).unsqueeze(0)
    return h


class PieceTransformerQ(nn.Module):
    """One token per physical piece; the token carries the stickers currently on it.

    Layout comes from `data/piece_layout.json` (scripts/50_derive_piece_layout.py),
    which derives the piece partition from the generators and self-verifies it as a
    block system. For tetraminx that is 50 pieces (4 x size-3, 30 x size-2,
    16 x size-1), max_piece_size 3, so the value table is 3 x 88 rows.
    """

    def __init__(self, layout_path, state_size=88, num_classes=88, n_actions=24,
                 d_model=256, nhead=8, num_layers=4, ff_dim=1024, dropout=0.0,
                 activation="silu", az_head=False, attn_impl="sdpa"):
        super().__init__()
        self.attn_impl = attn_impl
        layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        self.layout_path = str(layout_path)
        self.state_size, self.num_classes, self.output_dim = state_size, num_classes, n_actions
        self.d_model, self.nhead, self.num_layers, self.ff_dim = d_model, nhead, num_layers, ff_dim
        self.activation = activation
        self.has_value_head = bool(az_head)
        self.num_pieces = int(layout["num_pieces"])
        self.max_piece_size = int(layout["max_piece_size"])
        self.num_piece_types = int(layout["num_piece_types"])
        self.register_buffer("piece_positions",
                             torch.tensor(layout["piece_positions"], dtype=torch.int64), persistent=False)
        self.register_buffer("piece_mask",
                             torch.tensor(layout["piece_mask"], dtype=torch.bool), persistent=False)
        self.register_buffer("piece_types",
                             torch.tensor(layout["piece_types"], dtype=torch.int64), persistent=False)
        self.register_buffer("piece_indices", torch.arange(self.num_pieces), persistent=False)
        self.register_buffer("local_offsets",
                             torch.arange(self.max_piece_size) * num_classes, persistent=False)
        self.local_value_embedding = nn.Embedding(self.max_piece_size * num_classes, d_model)
        self.piece_projection = nn.Linear(self.max_piece_size * d_model, d_model)
        self.piece_position_embedding = nn.Embedding(self.num_pieces, d_model)
        self.piece_type_embedding = nn.Embedding(self.num_piece_types, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_EncoderBlock(d_model, nhead, ff_dim, dropout,
                                                   activation, attn_impl=attn_impl)
                                     for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_actions)
        self.value_head = nn.Linear(d_model, 1) if az_head else None
        self.return_value = False

    def trunk(self, x):
        b = x.size(0)
        # FOLDED input stage: push piece_projection into the value table, so a token
        # is sum_j table[j, v_j] * mask_j + bias instead of concat-then-project.
        # Mathematically identical (same parameters, gradients flow through the
        # einsum) but it never materialises the (B, P, K, D) tensor, which is what
        # made this OOM: at B=14848 rows that intermediate alone is 2.3 GB in fp32
        # and dragged a 16 GB card into WDDM paging (measured 2026-07-28: 2.4 h
        # without completing one epoch). Same form as the JAX port.
        # ONE-HOT MATMUL rather than index_select. A gather's backward is a
        # scatter-add, and here 384k gradients per slot contend for an 88-row
        # table -- measured on an A100 (7680 rows): 472 ms/step all-trainable vs
        # 158 ms with this stage frozen, i.e. the embedding gradient alone was
        # 67 pct of the whole step. Recast as (N, C) @ (C, D), both directions
        # become dense GEMMs and the atomics disappear. C is only 88, so the
        # extra forward cost is ~8.6 GFLOP per slot against a ~2.5 TFLOP step.
        h = _folded_piece_tokens(
            x, piece_positions=self.piece_positions, piece_mask=self.piece_mask,
            piece_types=self.piece_types, piece_indices=self.piece_indices,
            local_value_embedding=self.local_value_embedding,
            piece_projection=self.piece_projection,
            piece_position_embedding=self.piece_position_embedding,
            piece_type_embedding=self.piece_type_embedding,
            num_pieces=self.num_pieces, max_piece_size=self.max_piece_size,
            num_classes=self.num_classes, d_model=self.d_model,
        )
        h = torch.cat((self.cls_token.expand(b, -1, -1).to(h.dtype), h), dim=1)
        h = self.dropout(self.input_norm(h))
        for blk in self.blocks:
            h = blk(h)
        return self.output_norm(h[:, 0])

    def forward(self, x):
        h = self.trunk(x)
        q = self.head(h)
        if self.return_value and self.value_head is not None:
            return q, self.value_head(h).squeeze(-1)
        return q

    def value(self, x):
        if self.value_head is None:
            raise RuntimeError("model has no value head")
        return self.value_head(self.trunk(x)).squeeze(-1)

    def get_model_config(self):
        return {"arch": "transformer", "layout_path": self.layout_path,
                "state_size": self.state_size, "num_classes": self.num_classes,
                "n_actions": self.output_dim, "d_model": self.d_model, "nhead": self.nhead,
                "num_layers": self.num_layers, "ff_dim": self.ff_dim,
                "activation": self.activation, "az_head": self.has_value_head,
                "attn_impl": self.attn_impl}


class LatentRelationalQ(nn.Module):
    """Standalone piece-relational Q model with an early learned bottleneck.

    Fifty physical piece tokens are read once by a small learned latent set. Repeated
    relational reasoning then operates on the latent tokens, and 24 learned action
    queries decode generator-specific advantages. A centred dueling decomposition
    preserves the absolute parent calibration required by global beam top-k.
    """

    def __init__(self, layout_path, state_size=88, num_classes=88, n_actions=24,
                 d_model=96, nhead=4, num_latents=8, num_layers=2, ff_dim=192,
                 action_ff_dim=192, dropout=0.0, activation="silu", az_head=False):
        super().__init__()
        if az_head:
            raise ValueError(
                "latent_relational uses an internal dueling value baseline; "
                "an auxiliary AZ value head is not supported")
        if num_latents < 1:
            raise ValueError(f"num_latents must be positive, got {num_latents}")
        if num_layers < 0:
            raise ValueError(f"num_layers must be non-negative, got {num_layers}")
        layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        self.layout_path = str(layout_path)
        self.state_size, self.num_classes, self.output_dim = state_size, num_classes, n_actions
        self.d_model, self.nhead = d_model, nhead
        self.num_latents, self.num_layers = num_latents, num_layers
        self.ff_dim, self.action_ff_dim = ff_dim, action_ff_dim
        self.dropout_rate, self.activation = dropout, activation
        self.has_value_head = False
        self.return_value = False

        self.num_pieces = int(layout["num_pieces"])
        self.max_piece_size = int(layout["max_piece_size"])
        self.num_piece_types = int(layout["num_piece_types"])
        self.register_buffer("piece_positions",
                             torch.tensor(layout["piece_positions"], dtype=torch.int64),
                             persistent=False)
        self.register_buffer("piece_mask",
                             torch.tensor(layout["piece_mask"], dtype=torch.bool),
                             persistent=False)
        self.register_buffer("piece_types",
                             torch.tensor(layout["piece_types"], dtype=torch.int64),
                             persistent=False)
        self.register_buffer("piece_indices", torch.arange(self.num_pieces), persistent=False)

        self.local_value_embedding = nn.Embedding(self.max_piece_size * num_classes, d_model)
        self.piece_projection = nn.Linear(self.max_piece_size * d_model, d_model)
        self.piece_position_embedding = nn.Embedding(self.num_pieces, d_model)
        self.piece_type_embedding = nn.Embedding(self.num_piece_types, d_model)
        self.piece_dropout = nn.Dropout(dropout)

        self.latent_queries = nn.Parameter(torch.empty(1, num_latents, d_model))
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)
        self.compressor = _CrossBlock(d_model, nhead, ff_dim, dropout, activation)
        self.latent_blocks = nn.ModuleList([
            _EncoderBlock(d_model, nhead, ff_dim, dropout, activation)
            for _ in range(num_layers)
        ])
        self.latent_norm = nn.LayerNorm(d_model)

        self.action_queries = nn.Parameter(torch.empty(1, n_actions, d_model))
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)
        self.action_decoder = _CrossBlock(
            d_model, nhead, action_ff_dim, dropout, activation)
        self.action_norm = nn.LayerNorm(d_model)
        self.advantage_head = nn.Linear(d_model, 1)
        self.state_value_head = nn.Linear(d_model, 1)

    def trunk(self, x):
        b = x.size(0)
        pieces = _folded_piece_tokens(
            x, piece_positions=self.piece_positions, piece_mask=self.piece_mask,
            piece_types=self.piece_types, piece_indices=self.piece_indices,
            local_value_embedding=self.local_value_embedding,
            piece_projection=self.piece_projection,
            piece_position_embedding=self.piece_position_embedding,
            piece_type_embedding=self.piece_type_embedding,
            num_pieces=self.num_pieces, max_piece_size=self.max_piece_size,
            num_classes=self.num_classes, d_model=self.d_model,
        )
        pieces = self.piece_dropout(pieces)
        latents = self.latent_queries.expand(b, -1, -1).to(pieces.dtype)
        latents = self.compressor(latents, pieces)
        for block in self.latent_blocks:
            latents = block(latents)
        return self.latent_norm(latents)

    def q_components(self, x):
        """Return `(value, centred_advantage)` for calibration diagnostics."""
        latents = self.trunk(x)
        b = latents.size(0)
        actions = self.action_queries.expand(b, -1, -1).to(latents.dtype)
        actions = self.action_decoder(actions, latents)
        advantage = self.advantage_head(self.action_norm(actions)).squeeze(-1)
        centred_advantage = advantage - advantage.mean(dim=1, keepdim=True)
        value = self.state_value_head(latents.mean(dim=1)).squeeze(-1)
        return value, centred_advantage

    def forward(self, x):
        value, centred_advantage = self.q_components(x)
        return value.unsqueeze(1) + centred_advantage

    def value(self, x):
        raise RuntimeError(
            "latent_relational has no separately trained value head; use its 24 Q outputs")

    def get_model_config(self):
        return {
            "arch": "latent_relational", "layout_path": self.layout_path,
            "state_size": self.state_size, "num_classes": self.num_classes,
            "n_actions": self.output_dim, "d_model": self.d_model,
            "nhead": self.nhead, "num_latents": self.num_latents,
            "num_layers": self.num_layers, "ff_dim": self.ff_dim,
            "action_ff_dim": self.action_ff_dim, "dropout": self.dropout_rate,
            "activation": self.activation, "az_head": False,
        }


class _InducedPieceBlock(nn.Module):
    """Low-rank relational mixer that returns information to every piece.

    This is the two-stage induced set-attention pattern rather than a one-way
    Perceiver-style pool: inducing tokens first read all pieces, then every piece
    reads the inducing tokens.  Piece identity is therefore retained across depth.
    """

    def __init__(self, d_model, nhead, num_latents, ff_dim, dropout, activation):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.empty(1, num_latents, d_model))
        nn.init.normal_(self.inducing_points, mean=0.0, std=0.02)
        self.piece_to_latent = _CrossBlock(
            d_model, nhead, ff_dim, dropout, activation)
        self.latent_to_piece = _CrossBlock(
            d_model, nhead, ff_dim, dropout, activation)

    def forward(self, pieces):
        latents = self.inducing_points.expand(pieces.size(0), -1, -1).to(pieces.dtype)
        latents = self.piece_to_latent(latents, pieces)
        pieces = self.latent_to_piece(pieces, latents)
        return pieces, latents


def _resolve_generator_file(layout_path: Path, layout: dict, generator_path) -> Path:
    """Resolve puzzle_info.json portably from a Windows- or POSIX-authored layout."""
    raw = generator_path if generator_path is not None else layout.get("puzzle_file")
    if raw is None:
        raise ValueError("generator_isab needs generator_path or layout.puzzle_file")
    portable = Path(str(raw).replace("\\", "/"))
    candidates = (portable, layout_path.parent / portable.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not resolve generator file {raw!r}; tried "
        + ", ".join(str(p) for p in candidates))


def _derive_generator_geometry(layout_path, generator_path=None):
    """Return exact action-to-piece incidence and facelet permutations.

    A generator is stored in the same gather convention used by the puzzle code:
    child[destination] = parent[generator[destination]].  The physical-piece layout
    is a verified block system, so all facelets of one destination piece must come
    from one source piece.  Assertions here turn any convention/layout mismatch into
    an immediate construction failure rather than corrupted training data.
    """
    layout_path = Path(layout_path)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    puzzle_path = _resolve_generator_file(layout_path, layout, generator_path)
    puzzle = json.loads(puzzle_path.read_text(encoding="utf-8"))
    move_names = list(layout.get("move_names", puzzle["generators"].keys()))
    positions = torch.tensor(layout["piece_positions"], dtype=torch.int64)
    mask = torch.tensor(layout["piece_mask"], dtype=torch.bool)
    num_pieces = int(layout["num_pieces"])
    state_size = int(layout["state_size"])

    facelet_to_piece = torch.full((state_size,), -1, dtype=torch.int64)
    for piece in range(num_pieces):
        live = positions[piece][mask[piece]]
        if bool((facelet_to_piece[live] >= 0).any()):
            raise ValueError("piece layout assigns a facelet to multiple pieces")
        facelet_to_piece[live] = piece
    if bool((facelet_to_piece < 0).any()):
        raise ValueError("piece layout does not cover every facelet")

    permutations = torch.tensor(
        [puzzle["generators"][name] for name in move_names], dtype=torch.int64)
    if permutations.shape != (len(move_names), state_size):
        raise ValueError(f"generator shape mismatch: {tuple(permutations.shape)}")

    source_pieces = torch.empty((len(move_names), num_pieces), dtype=torch.int64)
    moved_pieces = torch.zeros((len(move_names), num_pieces), dtype=torch.bool)
    for action, permutation in enumerate(permutations):
        for piece in range(num_pieces):
            destination_facelets = positions[piece][mask[piece]]
            source_facelets = permutation[destination_facelets]
            sources = torch.unique(facelet_to_piece[source_facelets])
            if sources.numel() != 1:
                raise ValueError(
                    f"generator {move_names[action]!r} breaks piece block {piece}")
            source_pieces[action, piece] = sources.item()
            moved_pieces[action, piece] = bool(
                (source_facelets != destination_facelets).any())
        if not bool(moved_pieces[action].any()):
            raise ValueError(f"generator {move_names[action]!r} moves no physical pieces")

    moved_facelets = permutations.ne(torch.arange(state_size).unsqueeze(0))
    return {
        "layout": layout,
        "puzzle_path": puzzle_path,
        "move_names": move_names,
        "permutations": permutations,
        "source_pieces": source_pieces,
        "moved_pieces": moved_pieces,
        "moved_facelets": moved_facelets,
    }


class GeneratorISABQ(nn.Module):
    """ResMLP base plus a generator-aware bidirectional relational correction.

    The base tensor names deliberately match :class:`ResMLPQ`, so a trained tq0
    checkpoint loads without translation.  ``correction_head`` is zero-initialised;
    immediately after warm-start the complete model is therefore exactly the proven
    ResMLP scorer.  The new branch can learn improvements without first relearning
    absolute Q calibration.
    """

    _BASE_MODULES = ("embedding", "input_stack", "res_blocks", "head")

    def __init__(self, layout_path, generator_path=None, state_size=88, num_classes=88,
                 n_actions=24, hidden_dims=(2048, 512), num_res_blocks=2,
                 embed_dim=16, d_model=128, nhead=4, num_latents=12,
                 num_layers=2, ff_dim=256, action_ff_dim=256, dropout=0.0,
                 activation="silu", az_head=False):
        super().__init__()
        if az_head:
            raise ValueError("generator_isab does not support an auxiliary AZ head")
        if num_latents < 1:
            raise ValueError(f"num_latents must be positive, got {num_latents}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        geometry = _derive_generator_geometry(layout_path, generator_path)
        layout = geometry["layout"]
        if len(geometry["move_names"]) != n_actions:
            raise ValueError(
                f"layout has {len(geometry['move_names'])} actions, expected {n_actions}")

        self.layout_path = str(layout_path)
        self.generator_path = None if generator_path is None else str(generator_path)
        self.state_size, self.num_classes, self.output_dim = state_size, num_classes, n_actions
        self.hidden_dims, self.num_res_blocks = tuple(hidden_dims), num_res_blocks
        self.embed_dim = embed_dim
        self.d_model, self.nhead = d_model, nhead
        self.num_latents, self.num_layers = num_latents, num_layers
        self.ff_dim, self.action_ff_dim = ff_dim, action_ff_dim
        self.dropout_rate, self.activation = dropout, activation
        self.has_value_head = False
        self.return_value = False

        # Exact ResMLPQ names and shapes: warm-start is a direct state-dict load.
        self.embedding, self.input_stack, self.res_blocks, prev = _resmlp_trunk(
            state_size, num_classes, hidden_dims, num_res_blocks, embed_dim)
        self.head = nn.Linear(prev, n_actions)

        self.num_pieces = int(layout["num_pieces"])
        self.max_piece_size = int(layout["max_piece_size"])
        self.num_piece_types = int(layout["num_piece_types"])
        self.register_buffer("piece_positions",
                             torch.tensor(layout["piece_positions"], dtype=torch.int64),
                             persistent=False)
        self.register_buffer("piece_mask",
                             torch.tensor(layout["piece_mask"], dtype=torch.bool),
                             persistent=False)
        self.register_buffer("piece_types",
                             torch.tensor(layout["piece_types"], dtype=torch.int64),
                             persistent=False)
        self.register_buffer("piece_indices", torch.arange(self.num_pieces), persistent=False)
        self.register_buffer("action_source_pieces", geometry["source_pieces"], persistent=False)
        self.register_buffer("action_piece_mask", geometry["moved_pieces"], persistent=False)
        self.register_buffer("generator_permutations", geometry["permutations"], persistent=False)
        self.register_buffer("generator_facelet_mask", geometry["moved_facelets"], persistent=False)
        self.register_buffer("facelet_indices", torch.arange(state_size), persistent=False)

        self.local_value_embedding = nn.Embedding(self.max_piece_size * num_classes, d_model)
        self.piece_projection = nn.Linear(self.max_piece_size * d_model, d_model)
        self.piece_position_embedding = nn.Embedding(self.num_pieces, d_model)
        self.piece_type_embedding = nn.Embedding(self.num_piece_types, d_model)
        self.piece_input_norm = nn.LayerNorm(d_model)
        self.piece_dropout = nn.Dropout(dropout)

        self.induced_blocks = nn.ModuleList([
            _InducedPieceBlock(
                d_model, nhead, num_latents, ff_dim, dropout, activation)
            for _ in range(num_layers)
        ])
        self.piece_output_norm = nn.LayerNorm(d_model)
        self.latent_output_norm = nn.LayerNorm(d_model)

        # Factorised descriptor of the exact facelet permutation.  Unlike a moved-set
        # mask it distinguishes inverse moves and within-piece orientation changes.
        self.generator_source_embedding = nn.Embedding(state_size, d_model)
        self.generator_destination_embedding = nn.Embedding(state_size, d_model)
        self.action_seed = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.action_seed, mean=0.0, std=0.02)
        self.action_decoder = _CrossBlock(
            d_model, nhead, action_ff_dim, dropout, activation)
        self.global_projection = nn.Linear(d_model, d_model)
        self.action_norm = nn.LayerNorm(d_model)
        self.correction_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def base_trunk(self, x):
        h = self.embedding(x.long()).flatten(start_dim=-2).to(self.input_stack[0].weight.dtype)
        h = self.input_stack(h)
        for block in self.res_blocks:
            h = block(h)
        return h

    def base_q(self, x):
        return self.head(self.base_trunk(x))

    def relational_trunk(self, x):
        pieces = _folded_piece_tokens(
            x, piece_positions=self.piece_positions, piece_mask=self.piece_mask,
            piece_types=self.piece_types, piece_indices=self.piece_indices,
            local_value_embedding=self.local_value_embedding,
            piece_projection=self.piece_projection,
            piece_position_embedding=self.piece_position_embedding,
            piece_type_embedding=self.piece_type_embedding,
            num_pieces=self.num_pieces, max_piece_size=self.max_piece_size,
            num_classes=self.num_classes, d_model=self.d_model,
        )
        pieces = self.piece_dropout(self.piece_input_norm(pieces))
        latents = None
        for block in self.induced_blocks:
            pieces, latents = block(pieces)
        return self.piece_output_norm(pieces), self.latent_output_norm(latents)

    def generator_queries(self, dtype):
        src = self.generator_source_embedding(self.generator_permutations)
        dst = self.generator_destination_embedding(self.facelet_indices).unsqueeze(0)
        live = self.generator_facelet_mask.unsqueeze(-1).to(src.dtype)
        geometry = (src * dst * live).sum(dim=1)
        geometry = geometry / live.sum(dim=1).clamp_min(1.0)
        return (self.action_seed.expand(1, self.output_dim, -1)
                + geometry.unsqueeze(0)).to(dtype)

    def correction(self, x):
        pieces, latents = self.relational_trunk(x)
        actions = self.generator_queries(pieces.dtype).expand(x.size(0), -1, -1)
        actions = self.action_decoder(
            actions, pieces, attn_mask=self.action_piece_mask)
        actions = actions + self.global_projection(latents.mean(dim=1)).unsqueeze(1)
        return self.correction_head(self.action_norm(actions)).squeeze(-1)

    def forward(self, x):
        return self.base_q(x) + self.correction(x)

    def base_parameters(self):
        for module_name in self._BASE_MODULES:
            yield from getattr(self, module_name).parameters()

    def freeze_base(self):
        for parameter in self.base_parameters():
            parameter.requires_grad_(False)

    def unfreeze_base(self):
        for parameter in self.base_parameters():
            parameter.requires_grad_(True)

    def value(self, x):
        raise RuntimeError("generator_isab has no separately trained value head")

    def get_model_config(self):
        return {
            "arch": "generator_isab", "layout_path": self.layout_path,
            "generator_path": self.generator_path,
            "state_size": self.state_size, "num_classes": self.num_classes,
            "n_actions": self.output_dim, "hidden_dims": list(self.hidden_dims),
            "num_res_blocks": self.num_res_blocks, "embed_dim": self.embed_dim,
            "d_model": self.d_model, "nhead": self.nhead,
            "num_latents": self.num_latents, "num_layers": self.num_layers,
            "ff_dim": self.ff_dim, "action_ff_dim": self.action_ff_dim,
            "dropout": self.dropout_rate, "activation": self.activation,
            "az_head": False,
        }


# --------------------------------------------------------------------------
def build_model(arch: str, az_head: bool = False, layout_path=None, **kw) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")
    if arch == "resmlp":
        return ResMLPQ(az_head=az_head, **kw)
    if arch == "gflownet":
        return GFlowNetQ(az_head=az_head, **kw)
    if layout_path is None:
        raise ValueError(f"{arch} needs layout_path")
    if arch == "transformer":
        return PieceTransformerQ(layout_path=layout_path, az_head=az_head, **kw)
    if arch == "latent_relational":
        return LatentRelationalQ(layout_path=layout_path, az_head=az_head, **kw)
    return GeneratorISABQ(layout_path=layout_path, az_head=az_head, **kw)


def model_from_config(cfg: dict) -> nn.Module:
    cfg = dict(cfg)
    arch = cfg.pop("arch")
    az = cfg.pop("az_head", False)
    cfg.pop("model_class", None)
    return build_model(arch, az_head=az, **cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GFlowNetTBQ(nn.Module):
    """A trajectory-balance-trained GFlowNet presented through the common Q interface.

    The other cells expose `forward(states) -> (B, n_actions)` where LOWER is better,
    and the beam takes a GLOBAL top-B across parents. A GFN gives neither directly,
    so the mapping is:

        Q(s, a) = -log F(s) - log P_B(a | s)

    * `P_B(a|s)` is the BACKWARD policy. Trajectories are generated forward FROM the
      solved state, so "backward" is the direction that returns toward solved --
      i.e. P_B is already a policy over the 24 moves at the PARENT, exactly our Q
      shape, one forward per parent.
    * `-log P_B` alone is per-state normalised, so it can rank a parent's own children
      but is meaningless ACROSS parents -- which would break a global top-B. Adding
      `-log F(s)`, constant in `a`, leaves the within-parent order untouched while
      making scores comparable between parents: higher flow = nearer solved = lower
      score. `log F(s) = -log P_F(stop|s)` comes from the same forward, so the Q
      amortisation is preserved.

    Trained by 53_train_gfn_tb.py; this class is inference-only.
    """

    def __init__(self, policy, s0_preimages, n_actions=24):
        super().__init__()
        self.policy = policy
        self.output_dim = int(n_actions)
        self.has_value_head = False
        self.return_value = False
        self.register_buffer("s0_preimages", s0_preimages.long(), persistent=True)

    def forward(self, x):
        bwd_logits, fwd_logits = self.policy(x)
        # mask forward actions that would step INTO the solved state (no such edge)
        mask = (x.unsqueeze(1) == self.s0_preimages.unsqueeze(0)).all(dim=2)   # (B, A)
        f = fwd_logits.clone()
        f[:, : self.output_dim] = f[:, : self.output_dim].masked_fill(mask, float("-inf"))
        log_pf = torch.log_softmax(f, dim=-1)
        log_flow = -log_pf[:, -1]                      # log F(s) = -log P_F(stop|s)
        log_pb = torch.log_softmax(bwd_logits, dim=-1)
        return -log_flow.unsqueeze(1) - log_pb

    def get_model_config(self):
        cfg = {"arch": "gfn_tb", "n_actions": self.output_dim, "az_head": False}
        cfg.update(getattr(self.policy, "_build_cfg", {}))
        return cfg


class ValueHeadOnly(nn.Module):
    """Expose a dual-head model's VALUE head as a scalar V scorer.

    The beam's V path wants `forward(states) -> (B,)` and dispatches on
    `output_dim == 1`. Cells 2 and 4 carry a trained value head that inference has
    so far never used -- the beam reads their Q head and the value head only acts
    through trunk interference. This wrapper makes the megaminx arrangement
    testable here: on megaminx the AZ win came through the V head used AS the
    scorer (`m_az_v4_v_only.pt`), not through the policy head.

    Note `taz_v1` already tested that recipe on tetraminx with the OLD training
    (Bellman-V + policy CE on floor paths, ResMLP trunk) and it was a wash --
    434 vs tv0's 437 on 13 pids. This is a different model: a transformer trunk
    whose value head was trained against walk depth plus exact BFS anchors while
    a sparse-Q head shared the trunk.
    """

    def __init__(self, base):
        super().__init__()
        if not getattr(base, "has_value_head", False):
            raise ValueError("base model has no value head")
        self.base = base
        self.output_dim = 1
        self.has_value_head = True
        self.return_value = False

    def forward(self, x):
        return self.base.value(x)

    def get_model_config(self):
        cfg = dict(self.base.get_model_config())
        cfg["value_head_only"] = True
        return cfg


class BlendedQ(nn.Module):
    """Output-space ensemble: average K models' Q scores at every beam step.

    The complement to weight-space souping (`scripts/54_model_soup.py`). Souping
    collapses K checkpoints into ONE model, so it is free at inference; this averages
    K live forwards, so a K-blend at width B costs about what one model costs at K*B.
    That is the bar it has to clear -- on tetraminx, width converts compute into moves
    better than anything else measured (1M -> 4M = -13 on the 15-pid set), so a blend
    only earns its place once the width curve has flattened.

    The two are NOT the same bet and can fail independently: souping needs the
    checkpoints to share a loss basin (a bad soup is incoherent), while blending needs
    nothing of the sort but pays K forwards forever.

    Averaging is meaningful here only because every cell-4 checkpoint predicts
    distance-to-solved under the same sparse-Q objective, so the scores share units.
    Do NOT blend across objectives or across symmetry frames -- raw scores from
    different frames are not comparable (the megaminx V-across-frames rule).

    A constant per-model offset cancels (every candidate in a step gets it), so only
    differences in SCALE matter; `weights` is there for a greedy/weighted soup-style
    search if one model turns out to dominate.
    """

    def __init__(self, models, weights=None):
        super().__init__()
        if not models:
            raise ValueError("BlendedQ needs at least one model")
        self.models = nn.ModuleList(models)
        w = list(weights) if weights is not None else [1.0] * len(models)
        if len(w) != len(models):
            raise ValueError(f"{len(w)} weights for {len(models)} models")
        tot = float(sum(w))
        if tot <= 0:
            raise ValueError("blend weights must sum to > 0")
        self.register_buffer("weights", torch.tensor([x / tot for x in w], dtype=torch.float32))
        # Q is blended; V is taken from the FIRST member that has a value head rather
        # than averaged, because the members' value heads are trained separately and are
        # not calibrated to a common scale -- averaging them would mix two different
        # notions of "distance from here" into the consistency term. This mirrors
        # `jax_model.apply_qv`, which is what the deployed TPU stack actually runs; the
        # PyTorch side used to hard-code has_value_head=False, so --blend silently
        # disabled --qv-consistency here while the TPU path kept it (dual-code-path
        # drift, same class of bug as the streaming step body).
        self._v_member = next(
            (i for i, m in enumerate(self.models)
             if getattr(getattr(m, "_orig_mod", m), "has_value_head", False)), None)
        self.has_value_head = self._v_member is not None
        self.return_value = False
        # The solver decides Q-head vs scalar-V mode from `output_dim` (30_solve.py:275).
        # Without forwarding it the blend loads as a scalar V and dies on a shape
        # mismatch deep in the beam step, so mirror the members' head width here.
        first = self.models[0]
        self.output_dim = int(getattr(first, "output_dim",
                                      getattr(first, "n_actions", 1)))
        for m in self.models[1:]:
            other = int(getattr(m, "output_dim", getattr(m, "n_actions", 1)))
            if other != self.output_dim:
                raise ValueError(
                    f"blend members disagree on head width: {self.output_dim} vs {other}")

    def forward(self, x):
        want_v = bool(self.return_value) and self._v_member is not None
        out, v = None, None
        for i, (m, w) in enumerate(zip(self.models, self.weights)):
            base = getattr(m, "_orig_mod", m)
            take_v = want_v and i == self._v_member
            if take_v:
                prev, base.return_value = base.return_value, True
            q = m(x)
            if take_v:
                base.return_value = prev
            if isinstance(q, tuple):
                q, vv = q[0], q[1]
                if take_v:
                    v = vv
            q = q.float() * w
            out = q if out is None else out + q
        if want_v:
            if v is None:
                raise RuntimeError("blend value member returned no value output")
            return out, v
        return out

    def get_model_config(self):
        cfg = dict(self.models[0].get_model_config())
        cfg["blend_size"] = len(self.models)
        return cfg
