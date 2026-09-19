"""Pure-JAX inference for the cube555 `ResMLPQ` 30-wide Q head.

Ported from `tetraminx/kaggle_notebooks/tpu_beam_tetraminx/jax_model.py`, with the
PieceTransformer branch DROPPED -- cube555 has no transformer checkpoint, so the file
carries only the ResMLP family plus the blend/qv plumbing the beam engine imports.

THE MODEL (`cube555/src/cube555/models.py`, 24,757,807 params):

    Embedding(num_classes=150, embed_dim=24)      # picture cube: classes == state_size
      -> flatten                                  # (B, 150*24 = 3600)
      -> Linear(3600, 1024) -> LayerNorm -> ReLU  # input_stack, ONE level
      -> ResBlock x 10                            # lin1 -> ln1 -> ReLU -> lin2 -> ln2
                                                  #   -> ReLU(skip + h)
      -> q_head Linear(1024, 30)                  # one score per generator
      -> v_head Linear(1024, 1)                   # auxiliary, used by qv_consistency

WHY A Q HEAD IS THE WHOLE POINT ON TPU. A scalar-V beam step costs `B x 30` forwards
(score every child); this head scores all 30 children from ONE forward on the parent,
so a step costs `B`. That 30x is what converts directly into beam width, which is the
lever on this puzzle.

SHAPES COME FROM THE CHECKPOINT, not from the call site. Every cube555 checkpoint
embeds `model_config`, so `load_params_from_pt` reads d_model / num_res_blocks /
encoding from the file rather than taking them as arguments that can silently
disagree with the weights. This is deliberately different from the tetraminx loader
(which takes `hidden_dims=` / `num_res_blocks=`): there, a mismatch raises a KeyError
on a missing layer; here it would be a shape error deep in a matmul.

ENCODING. `num_classes` is 150, so one-hot is a 22,500-dim input (44.1M params,
2x slower for no measured gain -- APPROACH.md s2). The shipped checkpoints are all
`encoding="embedding"`. The one-hot branch is kept only so an old checkpoint loads.

Functions the beam engine imports by name:
    apply(params, x, dtype)       -> (B, 30)
    apply_qv(params, x, dtype)    -> ((B, 30), (B,))
    make_blend(members, weights)
    has_value_head(params)
    num_params(params)
    load_params_from_pt(path)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

STATE_SIZE = 150
N_GENERATORS = 30
NUM_CLASSES = 150


def _strip_orig_mod(sd: dict) -> dict:
    """Strip torch.compile's '_orig_mod.' key prefix if present."""
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def load_params_from_pt(pt_path: str | Path) -> dict[str, Any]:
    """Load a cube555 ResMLPQ .pt and convert to a JAX params dict.

    Linear weights are transposed (PyTorch (out, in) -> JAX (in, out)) so that
    `h @ w` is the canonical matmul. Everything is returned as float32; cast at the
    call site via `apply(..., dtype=jnp.bfloat16)`.
    """
    import torch

    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sd = _strip_orig_mod(ckpt["model"] if "model" in ckpt else
                         ckpt.get("state_dict", ckpt))
    mc = dict(ckpt.get("model_config", {}))
    mc.pop("model_class", None)

    encoding = mc.get("encoding", "embedding")
    d_model = int(mc.get("d_model", 1024))
    n_rb = int(mc.get("num_res_blocks", 10))
    out_dim = int(mc.get("output_dim", N_GENERATORS))
    state_size = int(mc.get("state_size", STATE_SIZE))
    num_classes = int(mc.get("num_classes", NUM_CLASSES))

    def t(name: str) -> jnp.ndarray:
        return jnp.asarray(sd[name].float().numpy(), dtype=jnp.float32)

    params: dict[str, Any] = {
        "encoding": encoding,
        "state_size": state_size,
        "num_classes": num_classes,
        "d_model": d_model,
        "output_dim": out_dim,
        "input_stack": [],
        "res_blocks": [],
        # `head_*` keeps the tetraminx engine's naming so `q_mode` detection
        # (`int(head_w.shape[-1]) == n_gen`) works unchanged.
        "head_w": jnp.transpose(t("q_head.weight"), (1, 0)),
        "head_b": t("q_head.bias"),
    }
    if encoding == "embedding":
        params["embed"] = t("embedding.weight")          # (num_classes, embed_dim)

    # input_stack is nn.Sequential(Linear, LayerNorm, ReLU): one Linear at index 0,
    # one LayerNorm at index 1. The tetraminx model had one such triple per
    # hidden_dim; ResMLPQ has exactly one, driven by d_model.
    params["input_stack"].append({
        "lin_w": jnp.transpose(t("input_stack.0.weight"), (1, 0)),
        "lin_b": t("input_stack.0.bias"),
        "ln_gamma": t("input_stack.1.weight"),
        "ln_beta": t("input_stack.1.bias"),
    })
    for rb in range(n_rb):
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
    if "v_head.weight" in sd:
        params["value_w"] = jnp.transpose(t("v_head.weight"), (1, 0))
        params["value_b"] = t("v_head.bias")
    return params


def _layer_norm(x: jnp.ndarray, gamma: jnp.ndarray, beta: jnp.ndarray,
                eps: float = 1e-5) -> jnp.ndarray:
    """PyTorch nn.LayerNorm semantics over the last dim."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * gamma + beta


def _trunk(params: dict[str, Any], x: jnp.ndarray, dtype) -> jnp.ndarray:
    """Shared trunk -> (B, d_model). Both heads read this, so qv is one pass."""
    if params.get("encoding", "embedding") == "onehot":
        oh = jax.nn.one_hot(x.astype(jnp.int32), params["num_classes"], dtype=dtype)
        h = oh.reshape(oh.shape[0], -1)
    else:
        # x is uint8 (0..149 -- see the state-dtype note in jax_beam_spmd_v_only.py).
        # int8 would wrap classes 128..149 negative and index the embedding from the
        # far end without erroring, which is why the cast target is explicit here.
        e = params["embed"][x.astype(jnp.int32)]          # (B, S, embed_dim) f32
        h = e.reshape(e.shape[0], -1).astype(dtype)

    for layer in params["input_stack"]:
        h = h @ layer["lin_w"].astype(dtype) + layer["lin_b"].astype(dtype)
        h = _layer_norm(h, layer["ln_gamma"].astype(dtype),
                        layer["ln_beta"].astype(dtype))
        h = jax.nn.relu(h)

    for rb in params["res_blocks"]:
        skip = h
        h1 = h @ rb["lin1_w"].astype(dtype) + rb["lin1_b"].astype(dtype)
        h1 = _layer_norm(h1, rb["ln1_gamma"].astype(dtype), rb["ln1_beta"].astype(dtype))
        h1 = jax.nn.relu(h1)
        h2 = h1 @ rb["lin2_w"].astype(dtype) + rb["lin2_b"].astype(dtype)
        h2 = _layer_norm(h2, rb["ln2_gamma"].astype(dtype), rb["ln2_beta"].astype(dtype))
        h = jax.nn.relu(skip + h2)
    return h


def apply(params: dict[str, Any], x: jnp.ndarray,
          dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    """Forward pass. x: (B, state_size) integer state. Returns (B, output_dim).

    output_dim is 30 here, so this is NOT squeezed -- the squeeze-to-(B,) branch of
    the tetraminx original only fired for a scalar V head, which this file does not
    ship. The engine's `q_mode` path expects the (B, n_gen) shape.
    """
    if "members" in params:                      # output-space blend
        acc = None
        for p, w in zip(params["members"], params["weights"]):
            o = apply(p, x, dtype=dtype).astype(jnp.float32) * jnp.float32(w)
            acc = o if acc is None else acc + o
        return acc.astype(dtype)
    h = _trunk(params, x, dtype)
    out = h @ params["head_w"].astype(dtype) + params["head_b"].astype(dtype)
    if out.shape[-1] == 1:
        out = jnp.squeeze(out, axis=-1)
    return out


def apply_qv(params: dict[str, Any], x: jnp.ndarray,
             dtype: jnp.dtype = jnp.float32) -> tuple[jnp.ndarray, jnp.ndarray]:
    """-> (Q (B, n_gen), V (B,)) from ONE trunk pass.

    That single pass is what makes `qv_consistency` free rather than a second forward.
    For a blend the Q side is the weighted average, but V is taken from the FIRST
    member that has a value head rather than averaged: value heads are trained
    separately and are not calibrated to a common scale.
    """
    if "members" in params:
        q = v = None
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
    if "value_w" not in params:
        raise ValueError("apply_qv needs a checkpoint with v_head.* (az_head=True)")
    h = _trunk(params, x, dtype)
    q = h @ params["head_w"].astype(dtype) + params["head_b"].astype(dtype)
    v = h @ params["value_w"].astype(dtype) + params["value_b"].astype(dtype)
    return q, jnp.squeeze(v, axis=-1)


def make_blend(members: list[dict[str, Any]],
               weights: list[float] | None = None) -> dict[str, Any]:
    """Output-space ensemble of Q models, scored jointly at every beam step.

    Legitimate here only because every cube555 checkpoint is the same architecture
    warm-started from the same pretrained parent, so the members share a distance
    scale by construction. Do NOT blend across symmetry frames -- raw scores from
    different frames are not comparable.

    Note RESULTS.md s2 measures the checkpoints as COMPLEMENTARY (min-merge over six
    arms: 24/24 vs 18/24 for the best single one). That is a per-pid min over separate
    runs, which is not the same thing as averaging scores inside one beam step; this
    blend is the untested variant, so it defaults off in the notebook.
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


def num_params(params: dict[str, Any]) -> int:
    """Count parameters (for sanity against the documented 24,757,807)."""
    if "members" in params:
        return sum(num_params(p) for p in params["members"])
    n = int(params["embed"].size) if "embed" in params else 0
    for layer in params["input_stack"]:
        n += sum(int(v.size) for v in layer.values())
    for rb in params["res_blocks"]:
        n += sum(int(v.size) for v in rb.values())
    n += int(params["head_w"].size) + int(params["head_b"].size)
    if "value_w" in params:
        n += int(params["value_w"].size) + int(params["value_b"].size)
    return n
