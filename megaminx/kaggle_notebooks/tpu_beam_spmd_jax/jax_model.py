"""Pure-JAX inference for ResMLPDistance.

The model used in the megaminx solver stack is:
  Embedding(num_classes=120, embed_dim=16)
  -> flatten -> Linear(in=1920, hidden_dims[0]) -> LayerNorm -> ReLU
  -> Linear(hidden_dims[0], hidden_dims[1]) -> LayerNorm -> ReLU
  -> ResBlock x num_res_blocks (Linear->LN->ReLU->Linear->LN->residual->ReLU)
  -> Linear(hidden_dims[-1], output_dim)

Three variants the beam search uses:
  m_curr_v3 (V teacher):       hidden=(2048, 512),  output_dim=1   ~6M params
  m23_v2    (Q shortlister):   hidden=(2048, 1024), output_dim=24  ~12M params
  m_pi_v2   (policy):          hidden=(2048, 512),  output_dim=24  ~6M params

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
    """Forward pass.

    x: (B, state_size) integer state.
    dtype: compute dtype. Embedding/output are cast to this.

    Returns (B,) for output_dim=1 (V model) or (B, output_dim) otherwise.
    """
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
    n = int(params["embed"].size)
    for layer in params["input_stack"]:
        n += sum(int(v.size) for v in layer.values())
    for rb in params["res_blocks"]:
        n += sum(int(v.size) for v in rb.values())
    n += int(params["head_w"].size) + int(params["head_b"].size)
    return n
