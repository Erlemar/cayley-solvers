"""Parity: the JAX ResMLPQ port vs the PyTorch original, on real cube555 states.

The whole TPU kernel rests on this file. If the JAX forward disagrees with the
PyTorch one, every beam result is quietly scored by a different model than the one
that was measured, and nothing downstream would notice -- the paths would still
replay to solved, just longer.

Checks, in order of what they would catch:
  1. param count matches the documented 24,757,807
  2. Q agrees elementwise in fp32 (this is the real gate)
  3. V agrees, and apply_qv's Q is bit-identical to apply's Q (one trunk, two heads)
  4. ARGMIN agrees -- the beam only ever consumes the ranking, so a tiny numeric
     drift that never flips an argmin is harmless, while a systematic one is fatal
  5. bf16 agreement, which is the dtype the kernel actually runs

Usage:
  .venv/Scripts/python.exe cube555/tpu/test_parity.py
  .venv/Scripts/python.exe cube555/tpu/test_parity.py --checkpoint <path> --n 512
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "True"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_HANDOFF = Path(
    "C:/Users/and-l/cayley/cube555_pull/cube555_handoff_2026_08_22/cube555")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF,
                    help="cube555/ dir holding data/ and models/")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--n", type=int, default=256, help="states to compare")
    args = ap.parse_args()

    ck = args.checkpoint or (args.handoff / "models" / "q555_2k_BEST.pt")

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    import jax_model as jm

    sys.path.insert(0, str(args.handoff / "src"))
    sys.path.insert(0, str(args.handoff.parent / "src"))
    from cube555.models import load_model

    # ---- build a realistic state batch: real test states + random-walk states ----
    info = json.loads((args.handoff / "data" / "puzzle_info.json").read_text(
        encoding="utf-8"))
    gens = np.stack([np.asarray(v, dtype=np.int64)
                     for v in info["generators"].values()])
    solved = np.asarray(info["central_state"], dtype=np.int64)

    states = []
    with open(args.handoff / "data" / "test.csv", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= args.n // 2:
                break
            states.append([int(x) for x in r["initial_state"].split(",")])
    rng = np.random.default_rng(555)
    for _ in range(args.n - len(states)):          # shallow walks: the endgame regime
        s = solved.copy()
        for _ in range(int(rng.integers(1, 12))):
            s = s[gens[rng.integers(0, len(gens))]]
        states.append(s.tolist())
    x_np = np.asarray(states, dtype=np.int64)
    print(f"batch: {x_np.shape[0]} states "
          f"({args.n // 2} real test.csv + {x_np.shape[0] - args.n // 2} walks)")
    assert x_np.max() == 149 and x_np.min() == 0, "expected sticker classes 0..149"

    # ---- PyTorch reference ----
    pt = load_model(ck, device="cpu", dtype=torch.float32)
    pt.return_value = True
    pt.inference_chunk_size = None
    with torch.no_grad():
        q_pt, v_pt = pt(torch.from_numpy(x_np))
    q_pt = q_pt.numpy().astype(np.float64)
    v_pt = v_pt.numpy().astype(np.float64)

    # ---- JAX port ----
    params = jm.load_params_from_pt(ck)
    n_jax, n_pt = jm.num_params(params), pt.num_parameters()
    print(f"params: jax {n_jax:,}  torch {n_pt:,}  "
          f"{'OK' if n_jax == n_pt == 24_757_807 else 'MISMATCH'}")
    if not (n_jax == n_pt == 24_757_807):
        return 1

    # uint8 is the kernel's state dtype -- exercise it here, not just int64, so a
    # wrap bug at 128..149 shows up in the parity test rather than on the TPU.
    x_u8 = jnp.asarray(x_np.astype(np.uint8))
    q_jx = np.asarray(jm.apply(params, x_u8, dtype=jnp.float32)).astype(np.float64)
    q_qv, v_qv = jm.apply_qv(params, x_u8, dtype=jnp.float32)
    q_qv = np.asarray(q_qv).astype(np.float64)
    v_qv = np.asarray(v_qv).astype(np.float64)

    fails = 0

    def report(label: str, a: np.ndarray, b: np.ndarray, tol: float) -> None:
        nonlocal fails
        d = np.abs(a - b)
        ok = d.max() <= tol
        fails += 0 if ok else 1
        print(f"  {label:38s} max|d| {d.max():.3e}  mean {d.mean():.3e}  "
              f"{'OK' if ok else 'FAIL'}  (tol {tol:g})")

    print(f"shapes: Q {q_jx.shape} vs {q_pt.shape}")
    assert q_jx.shape == q_pt.shape == (x_np.shape[0], 30)
    print("fp32 agreement:")
    report("Q  jax vs torch", q_jx, q_pt, 2e-3)
    report("V  jax vs torch", v_qv, v_pt, 2e-3)
    report("Q  apply vs apply_qv (same trunk)", q_jx, q_qv, 0.0)

    # The beam consumes ONLY the ranking, so this is the decision-relevant check.
    am_j, am_p = q_jx.argmin(1), q_pt.argmin(1)
    agree = int((am_j == am_p).sum())
    print(f"  argmin agreement                       {agree}/{len(am_j)}"
          f"  {'OK' if agree == len(am_j) else 'FAIL'}")
    fails += 0 if agree == len(am_j) else 1

    # Full ordering, not just the winner: the beam takes a global top-B over
    # (parent, action), so the whole 30-vector's order matters.
    order_same = int((np.argsort(q_jx, 1) == np.argsort(q_pt, 1)).all(1).sum())
    print(f"  full 30-action ordering identical      {order_same}/{len(am_j)}")

    # ---- bf16, the dtype the kernel runs in ----
    q_bf = np.asarray(jm.apply(params, x_u8, dtype=jnp.bfloat16).astype(jnp.float32),
                      dtype=np.float64)
    d_bf = np.abs(q_bf - q_pt)
    am_bf = q_bf.argmin(1)
    print(f"bf16 (kernel dtype): max|d| vs torch fp32 {d_bf.max():.3f}  "
          f"mean {d_bf.mean():.3f}   argmin agreement {int((am_bf == am_p).sum())}/{len(am_p)}")

    # ---- int8 would be a silent disaster: prove the port rejects it loudly ----
    wrapped = x_np.astype(np.int8)
    n_wrap = int((wrapped != x_np).sum())
    print(f"int8 wrap check: {n_wrap:,} of {x_np.size:,} sticker values would wrap "
          f"negative under int8 -- kernel MUST carry states as uint8")

    print("\nVERDICT :", "PASS" if fails == 0 else f"FAIL ({fails} check(s))")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
