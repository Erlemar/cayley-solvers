"""Piece-by-piece finiteness probe for gfn_train_tpu (single device, tiny model)."""
import sys
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

sys.path.insert(0, ".")
import gfn_train_tpu as T


def fin(name, x):
    x = np.asarray(x, dtype=np.float64)
    print(f"{name}: finite={np.isfinite(x).all()} min={np.nanmin(x):.4g}"
          f" max={np.nanmax(x):.4g} nan={np.isnan(x).sum()}", flush=True)


names, gens, inv_idx, solved, pre = T.load_env("/home/and-l/v6e/puzzle_info.json")
nmax, eps, batch = 60, 0.1, 8
sample_traj, loss_fn, greedy_rollout, _ = T.make_fns(gens, inv_idx, pre, solved, nmax, eps)

key = jax.random.key(0)
key, mk = jax.random.split(key)
model = T.GFNPolicy(120, 120, 24, 256, 16, 6, mk)
params, static = eqx.partition(model, eqx.is_array)

key, sk = jax.random.split(key)
states, actions = sample_traj(sk, model, batch)
print("states", states.shape, states.dtype, "actions", actions.shape, flush=True)
print("actions range", int(actions.min()), int(actions.max()), flush=True)

# manual loss pieces
tp1, b, ssz = states.shape
flat = states.reshape(tp1 * b, ssz)
bwd, fwd = jax.vmap(model)(flat)
fin("bwd_logits", bwd)
fin("fwd_logits", fwd)
mask = (flat[:, None, :] == jnp.asarray(pre)[None, :, :]).all(-1)
print("mask rate", float(mask.mean()), flush=True)
full_mask = jnp.concatenate([mask, jnp.zeros((mask.shape[0], 1), bool)], axis=1)
log_pf = jax.nn.log_softmax(jnp.where(full_mask, T.NEG_INF, fwd), axis=-1)
fin("log_pf", log_pf)
log_pb = jax.nn.log_softmax(bwd, axis=-1)
fin("log_pb", log_pb)
log_pf = log_pf.reshape(tp1, b, -1)
log_pb = log_pb.reshape(tp1, b, -1)
log_flows = -log_pf[..., -1]
fin("log_flows", log_flows)
pf_t = jnp.take_along_axis(log_pf[:-1], actions[..., None], axis=-1)[..., 0]
fin("pf_taken", pf_t)
pb_t = jnp.take_along_axis(log_pb[1:], jnp.asarray(inv_idx)[actions][..., None], axis=-1)[..., 0]
fin("pb_taken", pb_t)
zeros = jnp.zeros((1, b))
pre_f = jnp.concatenate([zeros, jnp.cumsum(pf_t, axis=0)], axis=0)
pre_b = jnp.concatenate([zeros, jnp.cumsum(pb_t, axis=0)], axis=0)
residual = T.TRUE_LOG_Z + pre_f - pre_b - log_flows
fin("residual", residual)

(loss, metrics), grads = jax.value_and_grad(
    lambda p: loss_fn(p, static, jnp.array(T.TRUE_LOG_Z), states, actions,
                      jnp.array(0.0), jnp.array(85.0)), has_aux=True)(params)
print("loss", float(loss), "metrics", np.asarray(metrics), flush=True)
gnorm = optax.global_norm(grads)
print("grad global norm", float(gnorm), flush=True)
for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
    a = np.asarray(leaf)
    if not np.isfinite(a).all():
        print("NONFINITE GRAD:", jax.tree_util.keystr(path), a.shape,
              "nan", np.isnan(a).sum(), "inf", np.isinf(a).sum(), flush=True)
print("probe done", flush=True)
