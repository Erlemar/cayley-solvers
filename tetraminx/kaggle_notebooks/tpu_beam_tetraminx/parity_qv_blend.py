"""JAX-vs-PyTorch parity for the value head, the blend, and the consistency term.

`cpu_smoke.py` validates kernel shapes/packing but hard-codes the ResMLP loader, so it
cannot load a transformer checkpoint. This checks the thing that actually matters for
the new features: that the JAX path reproduces the PyTorch numbers that were MEASURED,
so a TPU run of `--qv-consistency` / `--blend` is testing the same function the local
beam tested (ep1000 @1M hd=1: 435 -> 431 for consistency, 435 -> 433 for blend).

Checks, on real scrambled states:
  1. Q parity                 jax apply()      vs torch model(x)
  2. V parity                 jax apply_qv()   vs torch value head, SAME trunk pass
  3. blend parity             jax make_blend() vs torch BlendedQ
  4. consistency-term parity  Q + lam*|Q - (V-1)| computed both ways

Usage:
  python tetraminx/kaggle_notebooks/tpu_beam_tetraminx/parity_qv_blend.py \
      --transformer <tf.pt> --resmlp <resmlp.pt>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))
sys.path.insert(0, str(PROJECT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--resmlp", default=None)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    args = ap.parse_args()

    import torch
    import jax.numpy as jnp
    from jax_model import (apply, apply_qv, make_blend, has_value_head,
                           load_params_from_pt, load_piece_transformer_params_from_pt)
    from tetraminx.models import model_from_config, BlendedQ
    from tetraminx.puzzle import Tetraminx

    puzzle = Tetraminx.load(args.data_dir / "puzzle_info.json")
    names = list(puzzle.move_names)
    gen = np.array([puzzle.generators[n] for n in names], dtype=np.int64)
    rng = np.random.default_rng(0)
    s = np.tile(np.asarray(puzzle.solved_state, dtype=np.int64), (args.n, 1))
    for _ in range(12):                                    # real scrambles, not noise
        mv = rng.integers(0, len(names), size=args.n)
        s = np.take_along_axis(s, gen[mv], axis=1)
    x_np = s
    x_t = torch.from_numpy(x_np)
    x_j = jnp.asarray(x_np)

    def torch_model(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = model_from_config(dict(ck["model_config"])).eval()
        sd = {k.removeprefix("_orig_mod."): v for k, v in ck.get("state_dict", ck).items()}
        m.load_state_dict(sd)
        return m

    ok = True

    def check(tag, a, b):
        nonlocal ok
        d = float(np.max(np.abs(np.asarray(a, dtype=np.float64)
                                - np.asarray(b, dtype=np.float64))))
        good = d < args.tol
        ok = ok and good
        print(f"  {tag:38s} max|diff| = {d:.3e}   {'OK' if good else 'FAIL'}")

    print("1/4  Q parity (transformer)")
    tf_t = torch_model(args.transformer)
    tf_j = load_piece_transformer_params_from_pt(
        args.transformer, layout_path=args.data_dir / "piece_layout.json")
    with torch.no_grad():
        q_t = tf_t(x_t).float().numpy()
    q_j = np.asarray(apply(tf_j, x_j, dtype=jnp.float32))
    check("Q  jax vs torch", q_j, q_t)

    print("2/4  V parity (same trunk pass)")
    if not has_value_head(tf_j):
        print("  checkpoint has no value head -- skipping V and consistency")
    else:
        base = getattr(tf_t, "_orig_mod", tf_t)
        base.return_value = True
        with torch.no_grad():
            q_t2, v_t = base(x_t)
        base.return_value = False
        q_j2, v_j = apply_qv(tf_j, x_j, dtype=jnp.float32)
        check("Q  from apply_qv vs torch", np.asarray(q_j2), q_t2.float().numpy())
        check("V  from apply_qv vs torch", np.asarray(v_j), v_t.float().numpy())

        print("4/4  consistency term")
        lam = args.lam
        exp_t = (v_t.float().numpy() - 1.0)[:, None]
        sc_t = q_t2.float().numpy() + lam * np.abs(q_t2.float().numpy() - exp_t)
        qj = np.asarray(q_j2)
        exp_j = (np.asarray(v_j) - 1.0)[:, None]
        sc_j = qj + lam * np.abs(qj - exp_j)
        check(f"score = Q + {lam}*|Q-(V-1)|", sc_j, sc_t)

    if args.resmlp:
        print("3/4  blend parity (transformer x resmlp)")
        rm_t = torch_model(args.resmlp)
        cfg = torch.load(args.resmlp, map_location="cpu", weights_only=False)["model_config"]
        hid = tuple(cfg.get("hidden_dims", (2048, 512)))
        rm_j = load_params_from_pt(args.resmlp, hidden_dims=hid,
                                   num_res_blocks=int(cfg.get("num_res_blocks", 2)))
        with torch.no_grad():
            b_t = BlendedQ([tf_t, rm_t], weights=[0.8, 0.2]).eval()(x_t).float().numpy()
        b_j = np.asarray(apply(make_blend([tf_j, rm_j], [0.8, 0.2]), x_j, dtype=jnp.float32))
        check("blend 0.8/0.2 jax vs torch", b_j, b_t)

    print("\nPARITY OK" if ok else "\nPARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
