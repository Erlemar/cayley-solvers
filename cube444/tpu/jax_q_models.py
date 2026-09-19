"""Pure-JAX inference for the cube444 Q heads: PieceTransformer (s3) and PairQMLP (mlp_x16).

Ports the two models shipped in `cube444_inference/solver/unified_training/models.py` so the
TPU beam can run the exact `run_best.sh` scorer:

    Q = (1 - w) * Q_piece_transformer + w * Q_pair_qmlp        (w = 0.4)

Both heads emit 24 values per state, one per action, with Q(s,a) = d(apply(s,a)). That gives
the beam two usable contracts for free:

  * Q-native  -- score B parents, take a GLOBAL top-alpha*B over all (parent, action) pairs
                 (see [[qshort_global_vs_per_parent]]: a flat argsort, never per-parent).
  * V-like    -- min_a Q(s,a) = d(s) - 1, i.e. a drop-in scalar scorer for the existing
                 V-only kernel. 24x more forwards, but no beam changes.

Layout note (gotcha 1): num_classes is 6, NOT state_size. Every torch constructor in the
bundle defaults it to state_size, which silently builds a 96-way embedding and a different,
worse model. The shapes here are pinned to the shipped checkpoints instead of re-derived.

Functions:
    cube4_layout()                              -> positions (56,3), mask (56,3), types (56,)
    load_piece_transformer(pt_path)             -> params dict
    load_pair_qmlp(pt_path)                     -> params dict
    apply_piece_transformer(params, x, dtype)   -> (B, 24) float32
    apply_pair_qmlp(params, x, dtype)           -> (B, 24) float32
    apply_blend(tr, mlp, x, mlp_weight, dtype)  -> (B, 24) float32
    export_npz(pt_path, kind, out_path)         -> torch-free params for the Kaggle kernel
    load_npz(path, kind)                        -> params dict
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

NUM_CLASSES = 6
STATE_SIZE = 96
NUM_ACTIONS = 24
D_MODEL = 256
N_HEAD = 8
N_LAYERS = 4
FF_DIM = 1024
MAX_PIECE_SIZE = 3
LN_EPS = 1e-5
BN_EPS = 1e-5


def cube4_layout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 56-piece cube4 layout: 8 corners, 24 wings, 24 centres.

    Transcribed from models.py::_cube4_layout. Centres are KEPT -- doc 04 insisted on that
    for an unreduced-cube scorer, and the shipped checkpoint was trained with them.
    """
    corners = np.array([
        [0, 51, 64], [3, 35, 48], [12, 16, 67], [15, 19, 32],
        [28, 79, 80], [31, 44, 83], [47, 60, 95], [63, 76, 92],
    ], dtype=np.int32)
    wings = np.array([
        [1, 50, 0], [2, 49, 0], [4, 65, 0], [7, 34, 0],
        [8, 66, 0], [11, 33, 0], [13, 17, 0], [14, 18, 0],
        [20, 71, 0], [23, 36, 0], [24, 75, 0], [27, 40, 0],
        [29, 81, 0], [30, 82, 0], [39, 52, 0], [43, 56, 0],
        [45, 87, 0], [46, 91, 0], [55, 68, 0], [59, 72, 0],
        [61, 94, 0], [62, 93, 0], [77, 88, 0], [78, 84, 0],
    ], dtype=np.int32)
    centers = np.array(
        [[face * 16 + offset, 0, 0] for face in range(6) for offset in (5, 6, 9, 10)],
        dtype=np.int32,
    )
    positions = np.concatenate((corners, wings, centers))
    mask = np.concatenate((
        np.ones((8, 3), dtype=bool),
        np.tile(np.array([True, True, False]), (24, 1)),
        np.tile(np.array([True, False, False]), (24, 1)),
    ))
    types = np.concatenate((
        np.zeros(8, dtype=np.int32),
        np.ones(24, dtype=np.int32),
        np.full(24, 2, dtype=np.int32),
    ))
    return positions, mask, types


def _strip_orig_mod(sd: dict) -> dict:
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def _load_state_dict(pt_path: str | Path) -> dict:
    import torch

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    return _strip_orig_mod(sd)


# --------------------------------------------------------------------------------------
# PieceTransformer (s3 / s1L_4k / orig_transformer -- identical architecture)
# --------------------------------------------------------------------------------------

def load_piece_transformer(pt_path: str | Path) -> dict[str, Any]:
    """Convert a shipped PieceTransformer .pth into a JAX params dict.

    torch nn.Linear stores (out, in) and computes x @ W.T; we transpose once here so the
    JAX side is the canonical `h @ w`. nn.MultiheadAttention packs q/k/v into a single
    (3*d, d) in_proj_weight -- split it here rather than at every forward.
    """
    sd = _load_state_dict(pt_path)
    n = lambda k: np.asarray(sd[k].detach().cpu().float().numpy())  # noqa: E731

    params: dict[str, Any] = {
        "cls_token": n("cls_token").reshape(1, 1, D_MODEL),
        "local_value_embedding": n("local_value_embedding.weight"),
        "piece_projection_w": n("piece_projection.weight").T,
        "piece_projection_b": n("piece_projection.bias"),
        "piece_position_embedding": n("piece_position_embedding.weight"),
        "piece_type_embedding": n("piece_type_embedding.weight"),
        "input_norm_w": n("input_norm.weight"),
        "input_norm_b": n("input_norm.bias"),
        "output_norm_w": n("output_norm.weight"),
        "output_norm_b": n("output_norm.bias"),
        "output_layer_w": n("output_layer.weight").T,
        "output_layer_b": n("output_layer.bias"),
        "blocks": [],
    }
    for i in range(N_LAYERS):
        p = f"blocks.{i}."
        in_w = n(p + "attn.in_proj_weight")          # (3d, d)
        in_b = n(p + "attn.in_proj_bias")            # (3d,)
        wq, wk, wv = np.split(in_w, 3, axis=0)
        bq, bk, bv = np.split(in_b, 3, axis=0)
        params["blocks"].append({
            "norm1_w": n(p + "norm1.weight"), "norm1_b": n(p + "norm1.bias"),
            "wq": wq.T, "bq": bq, "wk": wk.T, "bk": bk, "wv": wv.T, "bv": bv,
            "out_w": n(p + "attn.out_proj.weight").T, "out_b": n(p + "attn.out_proj.bias"),
            "norm2_w": n(p + "norm2.weight"), "norm2_b": n(p + "norm2.bias"),
            "ff1_w": n(p + "ff.0.weight").T, "ff1_b": n(p + "ff.0.bias"),
            "ff2_w": n(p + "ff.3.weight").T, "ff2_b": n(p + "ff.3.bias"),
        })
    positions, mask, types = cube4_layout()
    params["piece_positions"] = positions.reshape(-1)
    params["piece_mask"] = mask.astype(np.float32).reshape(1, 56, MAX_PIECE_SIZE, 1)
    params["piece_types"] = types
    return params


def _layer_norm(x, w, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + LN_EPS) * w + b


def _attention(x, blk, dtype):
    """Pre-norm MHA, batch_first, no mask -- matches nn.MultiheadAttention exactly."""
    b, t, _ = x.shape
    head_dim = D_MODEL // N_HEAD
    q = (x @ blk["wq"].astype(dtype) + blk["bq"].astype(dtype))
    k = (x @ blk["wk"].astype(dtype) + blk["bk"].astype(dtype))
    v = (x @ blk["wv"].astype(dtype) + blk["bv"].astype(dtype))
    shape = (b, t, N_HEAD, head_dim)
    q = jnp.transpose(q.reshape(shape), (0, 2, 1, 3))
    k = jnp.transpose(k.reshape(shape), (0, 2, 1, 3))
    v = jnp.transpose(v.reshape(shape), (0, 2, 1, 3))
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(
        jnp.asarray(head_dim, dtype=jnp.float32)).astype(dtype)
    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(dtype)
    out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(b, t, D_MODEL)
    return out @ blk["out_w"].astype(dtype) + blk["out_b"].astype(dtype)


def apply_piece_transformer(params, states, dtype=jnp.float32):
    """states: (B, 96) integer colours in [0, 6). Returns (B, 24) float32 Q values."""
    b = states.shape[0]
    values = jnp.take(states, params["piece_positions"], axis=1).reshape(
        b, 56, MAX_PIECE_SIZE)

    # Folded input stage. The naive form embeds to (B, 56, 3, d_model), masks, then
    # matmuls the concatenated 768-wide vector through piece_projection. That
    # intermediate is 168 rows per state -- 45 GB at a 524,288-wide shard, which is
    # what OOM'd a 16 GB v5e chip. piece_projection is linear, so fold slot j's
    # 256-block of it into slot j's 6 embedding rows ONCE per forward:
    #     T[j] = emb[6j : 6j+6] @ W_proj[256j : 256j+256]      -> (3, 6, d_model)
    # then gather and sum. Exact (not an approximation), no 768-wide intermediate,
    # and peak memory drops ~3x. Weights are untouched, so checkpoints still load
    # strict=True.
    emb = params["local_value_embedding"].astype(dtype)
    w_proj = params["piece_projection_w"].astype(dtype)
    folded = jnp.stack([
        emb[j * NUM_CLASSES:(j + 1) * NUM_CLASSES]
        @ w_proj[j * D_MODEL:(j + 1) * D_MODEL]
        for j in range(MAX_PIECE_SIZE)
    ])                                                    # (3, 6, d_model)
    mask = params["piece_mask"].astype(dtype).reshape(1, 56, MAX_PIECE_SIZE)
    h = params["piece_projection_b"].astype(dtype).reshape(1, 1, D_MODEL)
    h = jnp.broadcast_to(h, (b, 56, D_MODEL))
    for j in range(MAX_PIECE_SIZE):
        h = h + mask[:, :, j:j + 1] * jnp.take(folded[j], values[:, :, j], axis=0)
    h = h + params["piece_position_embedding"].astype(dtype).reshape(1, 56, D_MODEL)
    h = h + jnp.take(
        params["piece_type_embedding"].astype(dtype), params["piece_types"], axis=0
    ).reshape(1, 56, D_MODEL)
    cls = jnp.broadcast_to(params["cls_token"].astype(dtype), (b, 1, D_MODEL))
    h = jnp.concatenate((cls, h), axis=1)
    h = _layer_norm(h, params["input_norm_w"].astype(dtype),
                    params["input_norm_b"].astype(dtype))
    for blk in params["blocks"]:
        att = _attention(
            _layer_norm(h, blk["norm1_w"].astype(dtype), blk["norm1_b"].astype(dtype)),
            blk, dtype)
        h = h + att
        ff = _layer_norm(h, blk["norm2_w"].astype(dtype), blk["norm2_b"].astype(dtype))
        ff = jax.nn.relu(ff @ blk["ff1_w"].astype(dtype) + blk["ff1_b"].astype(dtype))
        h = h + (ff @ blk["ff2_w"].astype(dtype) + blk["ff2_b"].astype(dtype))
    pooled = h[:, 0]
    pooled = _layer_norm(pooled, params["output_norm_w"].astype(dtype),
                         params["output_norm_b"].astype(dtype))
    out = pooled @ params["output_layer_w"].astype(dtype) + params[
        "output_layer_b"].astype(dtype)
    return out.astype(jnp.float32)


# --------------------------------------------------------------------------------------
# PairQMLP (mlp_x16) -- 47M, BatchNorm in EVAL mode (running stats, never batch stats)
# --------------------------------------------------------------------------------------

def load_pair_qmlp(pt_path: str | Path) -> dict[str, Any]:
    """Fold each BatchNorm1d into a per-channel (scale, shift).

    In eval mode BN is y = (x - mean)/sqrt(var + eps) * w + b, which is affine, so folding
    it once here removes a division from every forward and keeps the TPU step lean. This is
    exact, not an approximation -- but it is only valid in eval mode.
    """
    sd = _load_state_dict(pt_path)
    n = lambda k: np.asarray(sd[k].detach().cpu().float().numpy())  # noqa: E731

    def bn(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        w, b = n(prefix + ".weight"), n(prefix + ".bias")
        mean, var = n(prefix + ".running_mean"), n(prefix + ".running_var")
        scale = w / np.sqrt(var + BN_EPS)
        return scale, b - mean * scale

    params: dict[str, Any] = {
        # PositionClassLinear already stores (state_size*num_classes, out) -- no transpose.
        "input_w": n("input_layer.weight"),
        "input_b": n("input_layer.bias"),
        "input_bn": bn("input_bn"),
        "hidden_w": n("hidden.weight").T,
        "hidden_b": n("hidden.bias"),
        "hidden_bn": bn("hidden_bn"),
        "out_w": n("output_layer.weight").T,
        "out_b": n("output_layer.bias"),
        "blocks": [],
    }
    i = 0
    while f"blocks.{i}.linear1.weight" in sd:
        p = f"blocks.{i}."
        params["blocks"].append({
            "l1_w": n(p + "linear1.weight").T, "l1_b": n(p + "linear1.bias"),
            "bn1": bn(p + "bn1"),
            "l2_w": n(p + "linear2.weight").T, "l2_b": n(p + "linear2.bias"),
            "bn2": bn(p + "bn2"),
        })
        i += 1
    params["position_offsets"] = (np.arange(STATE_SIZE) * NUM_CLASSES).astype(np.int32)
    return params


def apply_pair_qmlp(params, states, dtype=jnp.float32):
    """states: (B, 96) integer colours in [0, 6). Returns (B, 24) float32 Q values.

    PositionClassLinear is an embedding_bag(sum) over one token per position, which is
    exactly a gather-and-sum of 96 rows -- far cheaper than materialising a 576-wide
    one-hot and matmul'ing it.
    """
    idx = states.astype(jnp.int32) + params["position_offsets"].reshape(1, -1)
    h = jnp.sum(jnp.take(params["input_w"].astype(dtype), idx, axis=0), axis=1)
    h = h + params["input_b"].astype(dtype)
    s, t = params["input_bn"]
    h = jax.nn.relu(h * s.astype(dtype) + t.astype(dtype))
    h = h @ params["hidden_w"].astype(dtype) + params["hidden_b"].astype(dtype)
    s, t = params["hidden_bn"]
    h = jax.nn.relu(h * s.astype(dtype) + t.astype(dtype))
    for blk in params["blocks"]:
        z = h @ blk["l1_w"].astype(dtype) + blk["l1_b"].astype(dtype)
        s, t = blk["bn1"]
        z = jax.nn.relu(z * s.astype(dtype) + t.astype(dtype))
        z = z @ blk["l2_w"].astype(dtype) + blk["l2_b"].astype(dtype)
        s, t = blk["bn2"]
        z = z * s.astype(dtype) + t.astype(dtype)
        h = jax.nn.relu(h + z)
    out = h @ params["out_w"].astype(dtype) + params["out_b"].astype(dtype)
    return out.astype(jnp.float32)


# --------------------------------------------------------------------------------------
# Blend + serialisation
# --------------------------------------------------------------------------------------

def apply_blend(tr_params, mlp_params, states, mlp_weight=0.4, dtype=jnp.float32,
                mlp_dtype=None):
    """The run_best.sh scorer: 0.6 * transformer + 0.4 * mlp_x16, per action.

    `mlp_dtype` defaults to `dtype`. Measured on 64 real states, bf16 argmin
    agreement vs fp32: transformer 0.9688, mlp_x16 0.9375, blend 0.9062 -- the 47M
    ResMLP is the dominant error source, not the transformer (whose folded input
    stage removed the 768-wide bf16 matmul). Running the MLP in fp32 while the
    transformer stays bf16 buys most of the ordering back for a fraction of the
    fp32 cost, since the MLP is the cheaper of the two per state.
    """
    q = (1.0 - mlp_weight) * apply_piece_transformer(tr_params, states, dtype)
    if mlp_weight > 0.0:
        q = q + mlp_weight * apply_pair_qmlp(
            mlp_params, states, dtype if mlp_dtype is None else mlp_dtype)
    return q


def _flatten(params: dict, prefix: str = "") -> dict[str, np.ndarray]:
    flat: dict[str, np.ndarray] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name + "/"))
        elif isinstance(value, (list, tuple)) and value and isinstance(value[0], dict):
            for i, item in enumerate(value):
                flat.update(_flatten(item, f"{name}/{i}/"))
        elif isinstance(value, tuple):
            for i, item in enumerate(value):
                flat[f"{name}/{i}"] = np.asarray(item)
        else:
            flat[name] = np.asarray(value)
    return flat


def _unflatten(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split("/")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def fix(node):
        if not isinstance(node, dict):
            return node
        keys = list(node)
        if keys and all(k.isdigit() for k in keys):
            return [fix(node[k]) for k in sorted(keys, key=int)]
        return {k: fix(v) for k, v in node.items()}

    out = fix(root)
    for key in ("input_bn", "hidden_bn"):
        if isinstance(out.get(key), list):
            out[key] = tuple(out[key])
    for blk in out.get("blocks", []) or []:
        for key in ("bn1", "bn2"):
            if isinstance(blk.get(key), list):
                blk[key] = tuple(blk[key])
    return out


def export_npz(pt_path: str | Path, kind: str, out_path: str | Path) -> Path:
    """Convert a .pth to a torch-free .npz so the Kaggle TPU kernel needs no torch."""
    loader = {"transformer": load_piece_transformer, "mlp": load_pair_qmlp}[kind]
    flat = _flatten(loader(pt_path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **flat)
    return out_path


def load_npz(path: str | Path, kind: str) -> dict[str, Any]:
    with np.load(path) as data:
        params = _unflatten({k: data[k] for k in data.files})
    if kind == "transformer":
        params["piece_positions"] = params["piece_positions"].astype(np.int32)
        params["piece_types"] = params["piece_types"].astype(np.int32)
    return params
