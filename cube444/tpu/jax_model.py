"""Pure-JAX inference for ResMLPDistance.

CUBE444 PORT: adds ONE-HOT encoding alongside megaminx's embedding.

The megaminx model is embedding-encoded (Embedding(num_classes=120, embed_dim=16)
-> flatten(1920) -> ...). The cube444 V is a COLOR cube with only 6 symbols, so it
uses `encoding="onehot"`: there is NO embedding table, the input is
`flatten(one_hot(x, 6))` of width state_size*6 = 576, fed straight into
`input_stack.0` (which is (hidden, 576), vs megaminx's (hidden, 1920)).

`load_params_from_pt` auto-detects: if the checkpoint has `embedding.weight` it is
an embedding model; otherwise it is one-hot and `num_classes` is inferred from
`input_stack.0.weight` (in_dim / state_size). `apply` branches on which key set is
present. The ResBlock / head logic is identical for both.

Model shape after the encoder is the same "flatten -> Linear -> LN -> ReLU ..."
for both encodings, so only the first-layer input and the presence of the embed
table differ.

Functions:
  load_params_from_pt(path, hidden_dims, num_res_blocks=2, state_size=None)
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
    state_size: int | None = None,
) -> dict[str, Any]:
    """Load a PyTorch ResMLPDistance .pt and convert to a JAX params dict.

    Linear weights are transposed (PyTorch (out, in) -> JAX (in, out) so that
    `h @ w` is the canonical matmul). All params are returned as jax.numpy
    arrays in float32. Cast at the call site (apply(..., dtype=jnp.bfloat16)).

    Auto-detects encoding: a checkpoint with `embedding.weight` is embedding-
    encoded; otherwise it is one-hot and `num_classes` is inferred from
    `input_stack.0.weight`'s in_dim / state_size (state_size required for onehot).
    """
    import torch
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = _strip_orig_mod(sd)

    def t(name: str) -> jnp.ndarray:
        return jnp.asarray(sd[name].float().numpy(), dtype=jnp.float32)

    is_embedding = "embedding.weight" in sd
    params: dict[str, Any] = {
        "encoding": "embedding" if is_embedding else "onehot",
        "input_stack": [],
        "res_blocks": [],
        "head_w": jnp.transpose(t("head.weight"), (1, 0)),
        "head_b": t("head.bias"),
    }
    if is_embedding:
        params["embed"] = t("embedding.weight")  # (num_classes, embed_dim)
    else:
        in_dim = int(sd["input_stack.0.weight"].shape[1])  # (hidden, state_size*num_classes)
        if state_size is None:
            raise ValueError("one-hot checkpoint requires state_size to infer num_classes")
        if in_dim % state_size != 0:
            raise ValueError(
                f"onehot in_dim {in_dim} not divisible by state_size {state_size}")
        params["num_classes"] = in_dim // state_size
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
    """Forward pass.

    x: (B, state_size) integer state.
    dtype: compute dtype. Embedding/output are cast to this.

    Returns (B,) for output_dim=1 (V model) or (B, output_dim) otherwise.
    """
    # Encoder: embedding lookup OR one-hot. Both flatten to (B, state_size * feat).
    if params.get("encoding", "embedding") == "onehot":
        # F.one_hot(x, C).flatten(start_dim=-2): layout [s0c0..s0c(C-1), s1c0..].
        # jax.nn.one_hot matches this ordering exactly.
        oh = jax.nn.one_hot(x.astype(jnp.int32), params["num_classes"], dtype=dtype)
        h = oh.reshape(oh.shape[0], -1)  # (B, state_size * num_classes)
    else:
        embed = params["embed"]  # (num_classes, embed_dim) float32
        e = embed[x.astype(jnp.int32)]  # (B, state_size, embed_dim)
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
    n = int(params["embed"].size) if "embed" in params else 0
    for layer in params["input_stack"]:
        n += sum(int(v.size) for v in layer.values())
    for rb in params["res_blocks"]:
        n += sum(int(v.size) for v in rb.values())
    n += int(params["head_w"].size) + int(params["head_b"].size)
    return n
