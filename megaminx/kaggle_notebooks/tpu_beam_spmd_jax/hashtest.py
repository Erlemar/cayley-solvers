import os
os.environ["JAX_ENABLE_X64"] = "True"
import json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

rng = np.random.default_rng(0)
hv = rng.integers(0, int(1e15), size=120, dtype=np.int64)  # make_hash_vec(120, seed=0)
hv_d = jnp.asarray(hv)

p = json.load(open("/mnt/data/v6e/puzzle_info.json"))
v0 = np.array(p["central_state"], dtype=np.int8)


def host_hash(st):
    return int((st.astype(np.int64) * hv).sum())          # numpy int64, wraps mod 2^64


def dev_hash(st):
    return int(jnp.sum(jnp.asarray(st).astype(jnp.int64) * hv_d))  # device int64


print("x64:", jax.config.jax_enable_x64, "device:", jax.devices()[0].device_kind)
prod = jnp.asarray(v0).astype(jnp.int64) * hv_d
print("prod.dtype:", prod.dtype, "sum.dtype:", jnp.sum(prod).dtype)
hh, dh = host_hash(v0), dev_hash(v0)
print(f"V0    host={hh}  dev={dh}  match={hh == dh}")
for s in range(3):
    st = v0.copy()
    np.random.default_rng(s + 1).shuffle(st)
    hh, dh = host_hash(st), dev_hash(st)
    print(f"rand{s} host={hh}  dev={dh}  match={hh == dh}")
