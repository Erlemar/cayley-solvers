"""Pure-JAX inference for ResMLPDistance (IHES picture cube).

The E6 distance model used by the cube beam search is:
  Embedding(num_classes=72, embed_dim=16)
  -> flatten -> Linear(in=1152, hidden_dims[0]) -> LayerNorm -> ReLU
  -> Linear(hidden_dims[0], hidden_dims[1]) -> LayerNorm -> ReLU
  -> ResBlock x num_res_blocks (Linear->LN->ReLU->Linear->LN->residual->ReLU)
  -> Linear(hidden_dims[-1], output_dim)

E6 (V, Bellman-refined): hidden=(1024, 256), num_res_blocks=1, output_dim=1,
~1.6M params. Code is shared verbatim with the megaminx solver stack
(megaminx/kaggle_notebooks/tpu_beam_spmd_jax/jax_model.py).

Functions:
  load_params_from_pt(path, hidden_dims, num_res_blocks=2)
  apply(params, x, dtype=jnp.float32)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


def _strip_orig_mod(sd: dict) -> dict:
    """Strip torch.compile's '_orig_mod.' key prefix if present."""
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def load_params_from_pt(
    pt_path: str | Path,
    hidden_dims: tuple[int, ...],
    num_res_blocks: int = 2,
) -> dict[str, Any]:
    """Load a PyTorch ResMLPDistance .pt and convert to a JAX params dict.

    Linear weights are transposed (PyTorch (out, in) -> JAX (in, out) so that
    `h @ w` is the canonical matmul). All params are returned as jax.numpy
    arrays in float32. Cast at the call site (apply(..., dtype=jnp.bfloat16)).
    """
    import torch
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = _strip_orig_mod(sd)

    def t(name: str) -> jnp.ndarray:
        return jnp.asarray(sd[name].float().numpy(), dtype=jnp.float32)

    params: dict[str, Any] = {
        "embed": t("embedding.weight"),  # (num_classes, embed_dim)
        "input_stack": [],
        "res_blocks": [],
        "head_w": jnp.transpose(t("head.weight"), (1, 0)),
        "head_b": t("head.bias"),
    }
    # input_stack layout: Linear, LN, ReLU per hidden_dim. Each "step" is at
    # indices [3*level, 3*level+1, (3*level+2 is ReLU, no params)].
    for level in range(len(hidden_dims)):
        lin_idx = 3 * level
        ln_idx = 3 * level + 1
        params["input_stack"].append({
            "lin_w": jnp.transpose(t(f"input_stack.{lin_idx}.weight"), (1, 0)),
            "lin_b": t(f"input_stack.{lin_idx}.bias"),
            "ln_gamma": t(f"input_stack.{ln_idx}.weight"),
            "ln_beta": t(f"input_stack.{ln_idx}.bias"),
        })
    for rb in range(num_res_blocks):
        params["res_blocks"].append({
            "lin1_w": jnp.transpose(t(f"res_blocks.{rb}.lin1.weight"), (1, 0)),
            "lin1_b": t(f"res_blocks.{rb}.lin1.bias"),
            "ln1_gamma": t(f"res_blocks.{rb}.ln1.weight"),
            "ln1_beta": t(f"res_blocks.{rb}.ln1.bias"),
            "lin2_w": jnp.transpose(t(f"res_blocks.{rb}.lin2.weight"), (1, 0)),
            "lin2_b": t(f"res_blocks.{rb}.lin2.bias"),
            "ln2_gamma": t(f"res_blocks.{rb}.ln2.weight"),
            "ln2_beta": t(f"res_blocks.{rb}.ln2.bias"),
        })
    return params


def _layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                eps: float = 1e-5) -> jnp.ndarray:
    """PyTorch nn.LayerNorm semantics over the last dim."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    x_norm = (x - mean) * jax.lax.rsqrt(var + eps)
    return x_norm * gamma + beta


def apply(params: dict[str, Any], x: jnp.ndarray,
          dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Forward pass. Dispatches ResMLP vs PieceTransformer on key presence.

    x: (B, state_size) integer state.
    dtype: compute dtype. Embedding/output are cast to this.

    Returns (B,) for output_dim=1 (V model) or (B, output_dim) otherwise.
    """
    if "members" in params:    # output-space blend of several models
        acc = None
        for p, w in zip(params["members"], params["weights"]):
            o = apply(p, x, dtype=dtype).astype(jnp.float32) * jnp.float32(w)
            acc = o if acc is None else acc + o
        return acc.astype(dtype)
    if "table" in params:      # PieceTransformer params carry the folded value table
        return apply_piece_transformer(
            params, x, params["piece_positions"], params["piece_mask"],
            params["piece_types"], dtype=dtype)
    # Embedding lookup (params['embed'] is (num_classes, embed_dim) float32).
    embed = params["embed"]
    e = embed[x.astype(jnp.int32)]  # (B, state_size, embed_dim) float32
    h = e.reshape(e.shape[0], -1).astype(dtype)  # (B, state_size * embed_dim)

    for layer in params["input_stack"]:
        w = layer["lin_w"].astype(dtype)
        b = layer["lin_b"].astype(dtype)
        gamma = layer["ln_gamma"].astype(dtype)
        beta = layer["ln_beta"].astype(dtype)
        h = h @ w + b
        h = _layer_norm(h, gamma, beta)
        h = jax.nn.relu(h)

    for rb in params["res_blocks"]:
        skip = h
        h1 = h @ rb["lin1_w"].astype(dtype) + rb["lin1_b"].astype(dtype)
        h1 = _layer_norm(h1, rb["ln1_gamma"].astype(dtype), rb["ln1_beta"].astype(dtype))
        h1 = jax.nn.relu(h1)
        h2 = h1 @ rb["lin2_w"].astype(dtype) + rb["lin2_b"].astype(dtype)
        h2 = _layer_norm(h2, rb["ln2_gamma"].astype(dtype), rb["ln2_beta"].astype(dtype))
        h = jax.nn.relu(skip + h2)

    out = h @ params["head_w"].astype(dtype) + params["head_b"].astype(dtype)
    if out.shape[-1] == 1:
        out = jnp.squeeze(out, axis=-1)
    return out


def num_params(params: dict[str, Any]) -> int:
    """Count parameters (for sanity)."""
    if "members" in params:
        return sum(num_params(p) for p in params["members"])
    if "table" in params:
        return sum(int(v.size) for k, v in params.items()
                   if hasattr(v, "size") and k not in ("piece_positions", "piece_mask",
                                                       "piece_types")) + sum(
            int(v.size) for blk in params["blocks"] for v in blk.values())
    n = int(params["embed"].size)
    for layer in params["input_stack"]:
        n += sum(int(v.size) for v in layer.values())
    for rb in params["res_blocks"]:
        n += sum(int(v.size) for v in rb.values())
    n += int(params["head_w"].size) + int(params["head_b"].size)
    return n


# ---------------------------------------------------------------------------
# PieceTransformer (tetraminx architecture-comparison matrix)
# ---------------------------------------------------------------------------
def load_piece_transformer_params_from_pt(pt_path: str | Path,
                                          layout_path: str | Path | None = None) -> dict[str, Any]:
    """Convert a tetraminx.models.PieceTransformerQ checkpoint to JAX params.

    The input stage is FOLDED: piece_projection is pushed into the value table, so
    token p is  sum_j table[j, state[pos[p,j]]] * mask[p,j] + bias  instead of
    concat-then-project. This is exact (verified max abs diff 6e-7 on GPU) and it
    is REQUIRED here, not an optimisation -- the unfolded (B, P, K, D) intermediate
    is ~10 GB at TPU beam batch sizes, where the folded (B, P, D) is ~3 GB.
    """
    import torch
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = _strip_orig_mod(ckpt.get("state_dict", ckpt))
    cfg = ckpt.get("model_config", {})

    def t(name):
        return jnp.asarray(sd[name].float().numpy(), dtype=jnp.float32)

    d_model = int(cfg.get("d_model", 256))
    n_layers = int(cfg.get("num_layers", 4))
    n_head = int(cfg.get("nhead", 8))
    emb = t("local_value_embedding.weight")            # (K*C, D)
    proj_w = t("piece_projection.weight")              # (D, K*D)
    proj_b = t("piece_projection.bias")                # (D,)
    K = proj_w.shape[1] // d_model
    C = emb.shape[0] // K
    # table[j, v, :] = W_j @ E[j*C + v]
    W = proj_w.reshape(d_model, K, d_model)            # (out, slot, in)
    E = emb.reshape(K, C, d_model)                     # (slot, value, in)
    table = jnp.einsum("ojd,jvd->jvo", W, E)           # (K, C, D)

    static = t("piece_position_embedding.weight")      # (P, D) indexed by arange(P)
    types = jnp.asarray(ckpt["model_config"].get("piece_types", []), dtype=jnp.int32) \
        if "piece_types" in ckpt.get("model_config", {}) else None
    blocks = []
    for i in range(n_layers):
        blocks.append({
            "n1_g": t(f"blocks.{i}.norm1.weight"), "n1_b": t(f"blocks.{i}.norm1.bias"),
            "qkv_w": jnp.transpose(t(f"blocks.{i}.attn.in_proj_weight"), (1, 0)),
            "qkv_b": t(f"blocks.{i}.attn.in_proj_bias"),
            "o_w": jnp.transpose(t(f"blocks.{i}.attn.out_proj.weight"), (1, 0)),
            "o_b": t(f"blocks.{i}.attn.out_proj.bias"),
            "n2_g": t(f"blocks.{i}.norm2.weight"), "n2_b": t(f"blocks.{i}.norm2.bias"),
            "ff1_w": jnp.transpose(t(f"blocks.{i}.ff.0.weight"), (1, 0)),
            "ff1_b": t(f"blocks.{i}.ff.0.bias"),
            "ff2_w": jnp.transpose(t(f"blocks.{i}.ff.3.weight"), (1, 0)),
            "ff2_b": t(f"blocks.{i}.ff.3.bias"),
        })
    # Piece layout travels WITH the params. The beam kernel closes over the params
    # dict (it is not a jit argument), so Python ints and non-array entries here are
    # closure constants and trace fine -- which is what lets apply() dispatch without
    # touching jax_beam_spmd_v_only.py at all.
    import json as _json
    if layout_path is None:
        layout_path = Path(pt_path).parent / "piece_layout.json"
    lay = _json.loads(Path(layout_path).read_text(encoding="utf-8"))
    return {
        "kind": "piece_transformer",
        "piece_positions": jnp.asarray(lay["piece_positions"], dtype=jnp.int32),
        "piece_mask": jnp.asarray(lay["piece_mask"], dtype=bool),
        "piece_types": jnp.asarray(lay["piece_types"], dtype=jnp.int32),
        "table": table, "proj_b": proj_b,
        "pos_emb": static, "type_emb": t("piece_type_embedding.weight"),
        "cls": t("cls_token").reshape(1, 1, d_model),
        "in_g": t("input_norm.weight"), "in_b": t("input_norm.bias"),
        "blocks": blocks,
        "out_g": t("output_norm.weight"), "out_b": t("output_norm.bias"),
        "head_w": jnp.transpose(t("head.weight"), (1, 0)), "head_b": t("head.bias"),
        "n_head": n_head, "d_model": d_model,
        # AZ value head, when the checkpoint has one. It is nn.Linear(d_model, 1) over
        # the SAME pooled tensor the Q head reads, so V costs one extra matmul on an
        # already-computed activation -- not a second trunk pass. That is what makes
        # `--qv-consistency` free here exactly as it is in PyTorch.
        **({"value_w": jnp.transpose(t("value_head.weight"), (1, 0)),
            "value_b": t("value_head.bias")} if "value_head.weight" in sd else {}),
    }


def apply_piece_transformer(params, x, piece_positions, piece_mask, piece_types,
                            dtype=jnp.float32, want_value=False):
    """x (B, S) int -> (B, n_actions), or ((B, n_actions), (B,)) if want_value.

    `want_value` reads the AZ value head off the SAME pooled activation as the Q head,
    so both come from ONE trunk pass. Requesting it costs a (B, d_model) @ (d_model, 1)
    matmul, nothing more.
    """
    B = x.shape[0]
    P, K = piece_positions.shape
    D = params["d_model"]
    vals = jnp.take(x.astype(jnp.int32), piece_positions.reshape(-1), axis=1).reshape(B, P, K)
    h = jnp.zeros((B, P, D), dtype=dtype)
    tbl = params["table"].astype(dtype)                       # (K, C, D)
    for j in range(K):                                        # unrolled: K is 3
        contrib = jnp.take(tbl[j], vals[:, :, j], axis=0)     # (B, P, D)
        h = h + contrib * piece_mask[None, :, j, None].astype(dtype)
    h = h + params["proj_b"].astype(dtype)
    h = h + params["pos_emb"].astype(dtype)[None]
    h = h + jnp.take(params["type_emb"].astype(dtype), piece_types, axis=0)[None]
    h = jnp.concatenate([jnp.broadcast_to(params["cls"].astype(dtype), (B, 1, D)), h], axis=1)
    h = _layer_norm(h, params["in_g"].astype(dtype), params["in_b"].astype(dtype))
    T = P + 1
    nh = params["n_head"]
    dh = D // nh
    for blk in params["blocks"]:
        z = _layer_norm(h, blk["n1_g"].astype(dtype), blk["n1_b"].astype(dtype))
        qkv = z @ blk["qkv_w"].astype(dtype) + blk["qkv_b"].astype(dtype)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, nh, dh).transpose(0, 2, 1, 3)
        att = jax.nn.softmax((q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(jnp.asarray(dh, dtype)),
                             axis=-1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        h = h + (o @ blk["o_w"].astype(dtype) + blk["o_b"].astype(dtype))
        z = _layer_norm(h, blk["n2_g"].astype(dtype), blk["n2_b"].astype(dtype))
        f = z @ blk["ff1_w"].astype(dtype) + blk["ff1_b"].astype(dtype)
        f = f * jax.nn.sigmoid(f)                              # SiLU
        h = h + (f @ blk["ff2_w"].astype(dtype) + blk["ff2_b"].astype(dtype))
    pooled = _layer_norm(h[:, 0], params["out_g"].astype(dtype), params["out_b"].astype(dtype))
    q = pooled @ params["head_w"].astype(dtype) + params["head_b"].astype(dtype)
    if not want_value:
        return q
    if "value_w" not in params:
        raise ValueError("want_value=True but the checkpoint has no AZ value head")
    v = (pooled @ params["value_w"].astype(dtype) + params["value_b"].astype(dtype))
    return q, jnp.squeeze(v, axis=-1)


def make_blend(members: list[dict[str, Any]],
               weights: list[float] | None = None) -> dict[str, Any]:
    """Output-space ensemble of Q models, scored jointly at every beam step.

    Measured on the local PyTorch beam (2026-08-02, ep1000 @1M, hd=1): blending the
    PieceTransformer with the ResMLP+AZ at 0.8/0.2 gave 433 vs 435 for the transformer
    alone, at 1.07x wall -- the ResMLP is ~1/17 of the transformer's forward cost, so a
    blend is nearly free, unlike a K-way blend of transformers which costs Kx.

    Averaging is only meaningful because both heads predict distance-to-solved under the
    same sparse-Q objective; the two were measured to share a scale almost exactly
    (std ratio 1.000, means 13.15 vs 13.20). Do NOT blend across objectives, or across
    symmetry frames -- raw scores from different frames are not comparable.
    """
    if not members:
        raise ValueError("make_blend needs at least one member")
    w = list(weights) if weights is not None else [1.0] * len(members)
    if len(w) != len(members):
        raise ValueError(f"{len(w)} weights for {len(members)} members")
    tot = float(sum(w))
    if tot <= 0:
        raise ValueError("blend weights must sum to > 0")
    return {"kind": "blend", "members": members, "weights": [x / tot for x in w]}


def has_value_head(params: dict[str, Any]) -> bool:
    """Can apply_qv() serve a parent value for this params tree?"""
    if "members" in params:
        return any(has_value_head(p) for p in params["members"])
    return "value_w" in params


def apply_qv(params: dict[str, Any], x: jnp.ndarray,
             dtype: jnp.dtype = jnp.float32) -> tuple[jnp.ndarray, jnp.ndarray]:
    """-> (Q (B, n_actions), V (B,)) from ONE trunk pass.

    For a blend the Q side is the weighted average as usual, but V is taken from the
    FIRST member that has a value head rather than averaged: the members' value heads
    are trained separately and are not calibrated to a common scale, so averaging them
    would mix two different notions of "distance from here" into the consistency term.
    """
    if "members" in params:
        q = None
        v = None
        for p, w in zip(params["members"], params["weights"]):
            if v is None and has_value_head(p):
                qi, v = apply_qv(p, x, dtype=dtype)
            else:
                qi = apply(p, x, dtype=dtype)
            qi = qi.astype(jnp.float32) * jnp.float32(w)
            q = qi if q is None else q + qi
        if v is None:
            raise ValueError("no blend member has a value head")
        return q.astype(dtype), v
    if "table" not in params:
        raise ValueError("apply_qv currently supports PieceTransformer params only")
    return apply_piece_transformer(
        params, x, params["piece_positions"], params["piece_mask"],
        params["piece_types"], dtype=dtype, want_value=True)
