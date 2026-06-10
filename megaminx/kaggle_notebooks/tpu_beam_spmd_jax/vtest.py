import os
os.environ["JAX_ENABLE_X64"] = "True"
import json
import csv
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, "/mnt/data/v6e")
from jax_model import apply, load_params_from_pt

v = load_params_from_pt("/mnt/data/v6e/m_az_v4_v_only.pt", hidden_dims=(2048, 512))
p = json.load(open("/mnt/data/v6e/puzzle_info.json"))
v0 = np.array(p["central_state"], dtype=np.int64)
gens = p["generators"]
names = list(gens.keys())


def V(st, dt=jnp.bfloat16):
    return float(apply(v, jnp.asarray(st[None]), dt)[0])


print("device", jax.devices()[0].device_kind)
print("V(V0) bf16=%.3f fp32=%.3f  (expect ~0)" % (V(v0), V(v0, jnp.float32)))
st = v0.copy()
rng = np.random.default_rng(0)
for d in range(7):
    print("V(d=%d)=%.3f" % (d, V(st)))
    m = int(rng.integers(0, len(names)))
    g = gens[names[m]]
    st = np.array([st[x] for x in g], dtype=np.int64)  # move: new[i]=old[g[i]]
rows = list(csv.DictReader(open("/mnt/data/v6e/test.csv")))
for pid in [0, 300, 1000]:
    s = np.array([int(x) for x in rows[pid]["initial_state"].split(",")], dtype=np.int64)
    print("V(pid%d)=%.3f" % (pid, V(s)))
