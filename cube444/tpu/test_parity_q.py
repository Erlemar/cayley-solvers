"""JAX vs PyTorch parity for the cube444 Q heads.

Run this after ANY change to jax_q_models.py. A JAX reimplementation that silently differs
from the torch model is the exact failure mode [[dual_codepath_drift]] describes: the beam
still runs, still returns verified paths, and is just quietly worse. The gate that matters
is not the numeric delta but ARGMIN AGREEMENT -- the beam only consumes the ordering.

Usage:
  .venv/Scripts/python.exe cube444/tpu/test_parity_q.py [--n 64] [--bundle <path>]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (PROJECT / "cube444" / "kaggle_inference" / "cube444_inference" / "solver")


def load_states(bundle: Path, n: int) -> np.ndarray:
    with open(bundle / "test.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:n]
    return np.array(
        [[int(v) for v in r["initial_state"].split(",")] for r in rows], dtype=np.int32
    )


def torch_reference(bundle: Path, states: np.ndarray):
    import json
    import torch

    sys.path.insert(0, str(bundle))
    from pilgrim.factory import build_model_from_info

    out = {}
    for tag, sub in (("transformer", "s3"), ("mlp", "mlp_x16")):
        info = json.loads((bundle / "models" / sub / "model.json").read_text(encoding="utf-8"))
        model = build_model_from_info(
            info, num_classes=6, state_size=96, output_dim=24
        )
        sd = torch.load(bundle / "models" / sub / "model.pth",
                        map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        model.load_state_dict(sd, strict=True)   # strict: a silent shape skip is a bug
        model.eval()
        with torch.inference_mode():
            out[tag] = model(torch.from_numpy(states).long()).float().numpy()
    return out


def report(name: str, ref: np.ndarray, got: np.ndarray) -> bool:
    adiff = np.abs(ref - got)
    scale = np.maximum(np.abs(ref).max(), 1e-9)
    argmin_same = float((ref.argmin(1) == got.argmin(1)).mean())
    order_same = float((np.argsort(ref, 1) == np.argsort(got, 1)).all(1).mean())
    print(f"  {name:34s} max|d|={adiff.max():.3e}  rel={adiff.max()/scale:.3e}  "
          f"argmin_agree={argmin_same:.4f}  full_order_agree={order_same:.4f}")
    return argmin_same == 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = ap.parse_args()

    import jax.numpy as jnp

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import jax_q_models as jqm

    states = load_states(args.bundle, args.n)
    print(f"parity on {states.shape[0]} real test states from {args.bundle / 'test.csv'}")

    ref = torch_reference(args.bundle, states)
    tr = jqm.load_piece_transformer(args.bundle / "models" / "s3" / "model.pth")
    mlp = jqm.load_pair_qmlp(args.bundle / "models" / "mlp_x16" / "model.pth")
    x = jnp.asarray(states)

    ok = True
    print("\nfp32 (must be exact to numerical noise):")
    ok &= report("PieceTransformer s3",
                 ref["transformer"], np.asarray(jqm.apply_piece_transformer(tr, x)))
    ok &= report("PairQMLP mlp_x16",
                 ref["mlp"], np.asarray(jqm.apply_pair_qmlp(mlp, x)))
    blend_ref = 0.6 * ref["transformer"] + 0.4 * ref["mlp"]
    ok &= report("blend 0.6*TR + 0.4*MLP",
                 blend_ref, np.asarray(jqm.apply_blend(tr, mlp, x, 0.4)))

    print("\nbf16 (the TPU compute dtype -- drift here is expected, ordering is what matters):")
    report("PieceTransformer s3 bf16",
           ref["transformer"],
           np.asarray(jqm.apply_piece_transformer(tr, x, jnp.bfloat16)))
    report("PairQMLP mlp_x16 bf16",
           ref["mlp"], np.asarray(jqm.apply_pair_qmlp(mlp, x, jnp.bfloat16)))
    report("blend bf16",
           blend_ref, np.asarray(jqm.apply_blend(tr, mlp, x, 0.4, jnp.bfloat16)))
    report("blend MIXED tr=bf16 mlp=fp32",
           blend_ref,
           np.asarray(jqm.apply_blend(tr, mlp, x, 0.4, jnp.bfloat16, jnp.float32)))

    print("\nnpz round-trip (the torch-free path the Kaggle kernel uses):")
    tmp = Path(__file__).resolve().parent / "_parity_npz"
    jqm.export_npz(args.bundle / "models" / "s3" / "model.pth", "transformer",
                   tmp / "s3.npz")
    jqm.export_npz(args.bundle / "models" / "mlp_x16" / "model.pth", "mlp",
                   tmp / "mlp.npz")
    tr2 = jqm.load_npz(tmp / "s3.npz", "transformer")
    mlp2 = jqm.load_npz(tmp / "mlp.npz", "mlp")
    ok &= report("s3 via npz",
                 ref["transformer"], np.asarray(jqm.apply_piece_transformer(tr2, x)))
    ok &= report("mlp_x16 via npz",
                 ref["mlp"], np.asarray(jqm.apply_pair_qmlp(mlp2, x)))

    print("\nV-like contract: min_a Q(s,a) should equal d(s) - 1")
    q = np.asarray(jqm.apply_piece_transformer(tr, x))
    print(f"  min_a Q  mean={q.min(1).mean():.3f}  range=[{q.min():.2f}, {q.max():.2f}]")

    print("\nPARITY " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
