"""Export an IHES PieceTransformer checkpoint to the torch-free npz of the 256M TPU kernel.

Writes the npz (key layout of the cube444 kernel's jax_q_models.export_npz, see
kaggle_notebooks/tpu_beam_ihes_tf/ihes_jax_q_models.py) and then PROVES it: the JAX
forward on the exported file is compared with the PyTorch model on random-walk states of
every depth plus real test states. Fails (exit 1) unless outputs agree to 1e-3 and the
per-state argmin agrees on >= 99.9 percent of rows.

    python scripts/87_export_ihes_tf_npz.py --checkpoint models/ihes_tf_a/epoch_1000.pt \
        --out exports/ihes_tf_a_e1000.npz
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "tetraminx" / "src"))
sys.path.insert(0, str(PROJECT / "kaggle_notebooks" / "tpu_beam_ihes_tf"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Export + parity-check an IHES transformer npz.")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--layout", type=Path, default=PROJECT / "data" / "ihes_piece_layout.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--probe", type=int, default=512, help="random-walk probe states")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import torch
    import ihes_jax_q_models as iq
    from tetraminx.models import model_from_config

    out = iq.export_npz(args.checkpoint, args.layout, args.out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    params = jax.tree_util.tree_map(jnp.asarray, iq.load_npz(out))   # as the kernel does
    meta = {k: int(v) for k, v in params["meta"].items()}
    print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB) sha256 {sha}")
    print(f"meta: {meta}")

    # torch reference, fp32 on CPU
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_config"])
    cfg["layout_path"] = str(args.layout)
    model = model_from_config(cfg).eval()
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["state_dict"].items()})

    info = json.loads((PROJECT / "data" / "puzzle_info.json").read_text(encoding="utf-8"))
    gens = np.array(list(info["generators"].values()), dtype=np.int64)
    solved = np.array(info["central_state"], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    probes = []
    for i in range(args.probe):
        s = solved.copy()
        for m in rng.integers(0, gens.shape[0], size=int(rng.integers(0, 31))):
            s = s[gens[m]]
        probes.append(s)
    with open(PROJECT / "data" / "test.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows[:: max(1, len(rows) // 128)]:
        probes.append(np.array([int(x) for x in r["initial_state"].split(",")], dtype=np.int64))
    x = np.stack(probes)

    with torch.no_grad():
        model.return_value = bool(meta.get("n_layers", 0) and "value_w" in params)
        out_t = model(torch.from_numpy(x))
        if isinstance(out_t, tuple):
            q_t, v_t = out_t[0].numpy(), out_t[1].numpy()
        else:
            q_t, v_t = out_t.numpy(), None
    xj = jnp.asarray(x.astype(np.int32))
    q_full = np.asarray(iq.apply_piece_transformer(params, xj))
    q_mixed = np.asarray(iq.apply_piece_transformer_mixed(params, xj))
    ok = True
    for name, q in (("full", q_full), ("mixed", q_mixed)):
        diff = float(np.abs(q - q_t).max())
        agree = float((q.argmin(1) == q_t.argmin(1)).mean())
        print(f"JAX {name:5s} vs torch: max|dQ| {diff:.2e}  argmin agreement {agree:.4f} "
              f"over {x.shape[0]} states")
        ok &= diff < 1e-3 and agree >= 0.999
    if v_t is not None:
        v_j = np.asarray(iq.apply_value(params, xj))
        dv = float(np.abs(v_j - v_t).max())
        print(f"JAX value vs torch: max|dV| {dv:.2e}")
        ok &= dv < 1e-3
    print("PARITY: PASS" if ok else "PARITY: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
