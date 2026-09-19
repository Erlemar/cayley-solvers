"""Numerical parity: JAX one-hot forward == PyTorch ResMLPDistance.

    /c/Users/and-l/cayley/.venv/Scripts/python.exe cube444/tpu/test_parity.py

The single most load-bearing check in the TPU port. If the JAX forward does not
match PyTorch on random states, every V score in the beam is wrong and the whole
run is silently garbage. Runs on CPU; no TPU needed.
"""
import os
import sys
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "True"
import numpy as np
import torch
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

TPU_DIR = Path(__file__).resolve().parent
PROJECT = TPU_DIR.parent
sys.path.insert(0, str(TPU_DIR))
sys.path.insert(0, str(PROJECT.parent / "src"))

from jax_model import apply as jax_apply, load_params_from_pt, num_params  # noqa: E402
from cayley.search import load_model_checkpoint  # noqa: E402

CKPT = PROJECT / "models" / "c_bells2" / "epoch_0399.pt"
STATE_SIZE = 96
NUM_CLASSES = 6


def main() -> int:
    print(f"checkpoint: {CKPT}")
    # PyTorch reference
    tmodel = load_model_checkpoint(CKPT, device="cpu", dtype=torch.float32).eval()

    # JAX
    jparams = load_params_from_pt(CKPT, hidden_dims=(2048, 512), num_res_blocks=2,
                                  state_size=STATE_SIZE)
    print(f"jax encoding={jparams['encoding']} num_classes={jparams.get('num_classes')} "
          f"params={num_params(jparams):,}")

    rng = np.random.default_rng(0)
    # mix of solved, near-solved, and random color states
    solved = np.array(__import__("json").load(
        open(PROJECT / "data" / "puzzle_info.json", encoding="utf-8"))["central_state"],
        dtype=np.int64)
    batch = [solved]
    for _ in range(2000):
        batch.append(rng.integers(0, NUM_CLASSES, STATE_SIZE))
    X = np.stack(batch).astype(np.int64)

    with torch.no_grad():
        vt = tmodel(torch.from_numpy(X)).squeeze(-1).float().numpy()

    # JAX fp32 and bf16 (bf16 is what the beam actually runs)
    vj32 = np.asarray(jax_apply(jparams, jnp.asarray(X), dtype=jnp.float32))
    vjbf = np.asarray(jax_apply(jparams, jnp.asarray(X), dtype=jnp.bfloat16)).astype(np.float32)

    d32 = np.abs(vt - vj32)
    dbf = np.abs(vt - vjbf)
    print(f"\nfp32 JAX vs PyTorch:  max|d|={d32.max():.2e}  mean|d|={d32.mean():.2e}")
    print(f"bf16 JAX vs PyTorch:  max|d|={dbf.max():.3f}  mean|d|={dbf.mean():.3f}")
    print(f"V(solved): torch={vt[0]:+.4f}  jax_fp32={vj32[0]:+.4f}  jax_bf16={vjbf[0]:+.4f}")

    # ranking agreement is what the beam actually depends on (topk ordering)
    order_t = np.argsort(vt)
    order_j = np.argsort(vjbf)
    # Spearman-ish: fraction of adjacent pairs in the same order
    kt = np.mean((vt[order_t[:-1]] <= vt[order_t[1:]]))  # sanity (=1)
    # check top-100 smallest set overlap (beam keeps smallest-V)
    top_t = set(np.argsort(vt)[:100].tolist())
    top_j = set(np.argsort(vjbf)[:100].tolist())
    overlap = len(top_t & top_j) / 100
    print(f"\ntop-100-smallest-V set overlap (bf16): {overlap:.2%}")

    ok = d32.max() < 1e-3 and overlap >= 0.95
    print("\n" + ("PARITY OK" if ok else "PARITY FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
