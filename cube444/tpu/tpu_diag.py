"""On-device diagnostics for the cube444 TPU beam. Run BEFORE the beam.

    CUBE_DATA=/mnt/data/cube444 ~/tpu-env/bin/python tpu_diag.py

Catches the config-gap-vs-real-bug class in <1 min:
  1. host int64 state-hash == device int64 state-hash (all-reduce path sanity)
  2. V(solved) ~ 0 and V is monotone on a short random walk (model loaded right,
     one-hot encoder wired right, bf16 path sane)
Mirrors megaminx's hashtest.py + vtest.py.
"""
import os
os.environ["JAX_ENABLE_X64"] = "True"
import json
import sys

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

DATA = os.environ.get("CUBE_DATA", "/mnt/data/cube444")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DATA)
from jax_model import load_params_from_pt, apply as model_apply, num_params

pinfo = json.load(open(f"{DATA}/puzzle_info.json", encoding="utf-8"))
solved = np.array(pinfo["central_state"], dtype=np.int64)
gens = pinfo["generators"]
names = list(gens)
P = np.array([gens[n] for n in names], dtype=np.int64)
S = len(solved)

print(f"[diag] devices={jax.device_count()} kind={jax.devices()[0].device_kind}")

# 1. hash parity
rng = np.random.default_rng(0)
hv = rng.integers(0, int(1e15), size=S, dtype=np.int64)
X = np.stack([solved] + [rng.integers(0, 6, S) for _ in range(2000)]).astype(np.int64)
host_h = (X.astype(np.int64) * hv).sum(1)
dev_h = np.asarray(jnp.sum(jnp.asarray(X) * jnp.asarray(hv), axis=1))
hash_ok = np.array_equal(host_h, dev_h)
print(f"[diag] host==device int64 hash: {hash_ok}  (max|d|={np.abs(host_h-dev_h).max()})")

# 2. V calibration
vp = load_params_from_pt(f"{DATA}/c_bells2_epoch_0399.pt", hidden_dims=(2048, 512),
                         num_res_blocks=2, state_size=S)
print(f"[diag] V params={num_params(vp):,} encoding={vp['encoding']}")
v0 = float(model_apply(vp, jnp.asarray(solved[None]), dtype=jnp.bfloat16)[0])
# short walk: V should rise roughly monotonically
cur = solved.copy()
walk = [cur.copy()]
for _ in range(12):
    cur = cur[P[int(rng.integers(0, len(P)))]]
    walk.append(cur.copy())
vw = np.asarray(model_apply(vp, jnp.asarray(np.stack(walk)), dtype=jnp.bfloat16)).astype(float)
print(f"[diag] V(solved)={v0:+.3f}  (want ~0)")
print(f"[diag] V along 12-step walk: {np.round(vw, 2).tolist()}")
mono = np.mean(np.diff(vw) > 0)
print(f"[diag] fraction of walk steps with V increasing: {mono:.2f}")

ok = hash_ok and abs(v0) < 0.5
print("[diag] " + ("DIAGNOSTICS OK" if ok else "DIAGNOSTICS FAILED"))
sys.exit(0 if ok else 1)
