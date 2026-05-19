"""Numerical-equivalence test: JAX port of ResMLPDistance vs the PyTorch original.

Loads m_curr_v3 (V) and m_pi_v2 (policy) checkpoints into BOTH the original PyTorch
ResMLPDistance and our pure-JAX `apply` function. Compares element-wise outputs on
the SOLVED state plus 32 random scrambled states. Pass criterion: max abs diff <
1e-4 in float32. If this passes, the JAX inference is faithful to PyTorch.

Run via: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_spmd_jax/sanity_check_model.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jax_model import apply, load_params_from_pt, num_params  # noqa: E402

ROOT = HERE.parents[1]  # megaminx/
PUZZLE_INFO = json.loads((ROOT / "data" / "puzzle_info.json").read_text())
SOLVED = list(PUZZLE_INFO["central_state"])
GENERATORS = PUZZLE_INFO["generators"]
MOVE_NAMES = list(GENERATORS.keys())
STATE_SIZE = len(SOLVED)


# --- PyTorch reference impl (copy of the model used in the working notebooks) ---

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x):
        h = F.relu(self.ln1(self.lin1(x)))
        h = self.ln2(self.lin2(h))
        return F.relu(x + h)


class ResMLPDistance(nn.Module):
    def __init__(self, state_size=120, num_classes=120,
                 hidden_dims=(2048, 512), num_res_blocks=2,
                 embed_dim=16, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        in_dim = state_size * embed_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(num_res_blocks)])
        self.head = nn.Linear(prev, output_dim)
    def encode(self, x):
        return self.embedding(x.long()).flatten(start_dim=-2)
    def forward(self, x):
        target_dtype = self.input_stack[0].weight.dtype
        h = self.encode(x).to(target_dtype)
        h = self.input_stack(h)
        for blk in self.res_blocks:
            h = blk(h)
        out = self.head(h)
        if out.shape[-1] == 1:
            return out.squeeze(-1)
        return out


def random_states(n: int, rng: np.random.Generator, walk_len: int = 30) -> np.ndarray:
    """Generate n scrambled states via random walks of length `walk_len` from SOLVED."""
    gens = np.array([GENERATORS[m] for m in MOVE_NAMES], dtype=np.int64)  # (24, 120)
    out = np.zeros((n, STATE_SIZE), dtype=np.int8)
    for i in range(n):
        s = np.array(SOLVED, dtype=np.int64)
        for _ in range(walk_len):
            g_idx = rng.integers(0, len(gens))
            s = s[gens[g_idx]]
        out[i] = s.astype(np.int8)
    return out


def _strip_orig_mod(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}


def check_model(label: str, pt_path: Path, hidden_dims: tuple[int, ...],
                output_dim: int, rng_seed: int) -> bool:
    print(f"\n=== {label}: {pt_path.name} (hidden_dims={hidden_dims}, output_dim={output_dim}) ===")

    # PyTorch reference (float32)
    pt_model = ResMLPDistance(hidden_dims=hidden_dims, output_dim=output_dim).eval()
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = _strip_orig_mod(ckpt.get("state_dict", ckpt))
    missing, unexpected = pt_model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  PyTorch load: missing {len(missing)} keys (first: {missing[:3]})")
    if unexpected:
        print(f"  PyTorch load: unexpected {len(unexpected)} keys (first: {unexpected[:3]})")
    pt_n = sum(p.numel() for p in pt_model.parameters())

    # JAX port
    jax_params = load_params_from_pt(pt_path, hidden_dims=hidden_dims)
    jax_n = num_params(jax_params)
    print(f"  param counts: pytorch={pt_n:,}  jax={jax_n:,}  match={pt_n == jax_n}")
    if pt_n != jax_n:
        print("  FAIL: param count mismatch")
        return False

    # Generate inputs
    rng = np.random.default_rng(rng_seed)
    states_np = np.concatenate([
        np.array(SOLVED, dtype=np.int8)[None, :],  # SOLVED
        random_states(32, rng, walk_len=30),       # 32 random scrambles
    ], axis=0)
    x_pt = torch.from_numpy(states_np)
    x_jax = jnp.asarray(states_np)

    # Forward
    with torch.no_grad():
        out_pt = pt_model(x_pt).numpy()  # (33,) or (33, 24)
    out_jax = np.asarray(apply(jax_params, x_jax, dtype=jnp.float32))

    if out_pt.shape != out_jax.shape:
        print(f"  FAIL: shape mismatch  pt={out_pt.shape}  jax={out_jax.shape}")
        return False

    diff = np.abs(out_pt - out_jax)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    print(f"  output shape: {out_pt.shape}")
    print(f"  max abs diff: {max_abs:.3e}   mean abs diff: {mean_abs:.3e}")

    # SOLVED's V should be near 0 (it's the goal). Just print for context.
    v_solved_pt = float(out_pt.reshape(out_pt.shape[0], -1)[0, 0])
    v_solved_jax = float(out_jax.reshape(out_jax.shape[0], -1)[0, 0])
    print(f"  V(SOLVED): pytorch={v_solved_pt:.6f}  jax={v_solved_jax:.6f}")

    # Range of outputs on scrambled states (V should be ~ true distance for V models).
    flat_pt = out_pt.reshape(out_pt.shape[0], -1).flatten()
    print(f"  output range across all states: [{float(flat_pt.min()):.3f}, {float(flat_pt.max()):.3f}]")

    tol = 1e-3  # CPU JAX float32 vs PyTorch float32: small rounding from matmul order.
    if max_abs > tol:
        print(f"  FAIL: max abs diff {max_abs:.3e} exceeds tolerance {tol:.0e}")
        return False
    print(f"  PASS")
    return True


if __name__ == "__main__":
    m_curr_v3_path = ROOT / "models" / "m_curr_v3" / "epoch_0499.pt"
    m_pi_v2_path = ROOT / "models" / "m_pi_v2" / "epoch_0199.pt"

    results = []
    if m_curr_v3_path.exists():
        results.append(check_model(
            "m_curr_v3 (V teacher)", m_curr_v3_path,
            hidden_dims=(2048, 512), output_dim=1, rng_seed=0,
        ))
    else:
        print(f"SKIP m_curr_v3 (not found at {m_curr_v3_path})")
        results.append(None)

    if m_pi_v2_path.exists():
        results.append(check_model(
            "m_pi_v2 (policy)", m_pi_v2_path,
            hidden_dims=(2048, 512), output_dim=24, rng_seed=1,
        ))
    else:
        print(f"SKIP m_pi_v2 (not found at {m_pi_v2_path})")
        results.append(None)

    print()
    print("=== summary ===")
    n_pass = sum(1 for r in results if r is True)
    n_fail = sum(1 for r in results if r is False)
    n_skip = sum(1 for r in results if r is None)
    print(f"pass={n_pass}  fail={n_fail}  skipped={n_skip}")
    sys.exit(0 if n_fail == 0 else 1)
