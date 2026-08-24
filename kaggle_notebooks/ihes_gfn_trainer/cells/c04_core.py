# --- GFN core: environment, model, sampling, loss ---
# Self-contained port of Morozov et al. 2026 for a permutation Cayley graph.
import json, math
from pathlib import Path
from functools import partial
import numpy as np
import jax, jax.numpy as jnp
import equinox as eqx

NEG_INF = -1e30

# ---- environment (reversed-graph GFN construction, paper section 3.2) ----
def load_env(puzzle_info_path):
    info = json.load(open(puzzle_info_path, encoding="utf-8"))
    names = list(info["generators"].keys())
    gens = np.array([info["generators"][n] for n in names], dtype=np.int32)   # (A,S)
    solved = np.array(info["central_state"], dtype=np.int32)
    assert (solved == np.arange(len(solved))).all(), "solved must be identity"
    inv = lambda n: n[1:] if n.startswith("-") else "-" + n
    inv_idx = np.array([names.index(inv(n)) for n in names], dtype=np.int32)
    pre = gens[inv_idx].copy()                 # preimage of solved under each fwd action
    return gens, inv_idx, solved, pre

# ---- model: ResMLP trunk + backward/forward/stop logits (shared backbone) ----
class GFNPolicy(eqx.Module):
    emb: jax.Array
    inp: eqx.nn.Linear
    fcs: list
    lns: list
    outp: eqx.nn.Linear
    n_actions: int = eqx.field(static=True)
    def __init__(self, state_size, num_classes, n_actions, hidden, emb_dim, n_blocks, key):
        ks = jax.random.split(key, n_blocks + 3)
        self.n_actions = n_actions
        self.emb = jax.random.normal(ks[0], (num_classes, emb_dim)) * 0.02
        self.inp = eqx.nn.Linear(state_size * emb_dim, hidden, key=ks[1])
        self.fcs = [eqx.nn.Linear(hidden, hidden, key=ks[2 + i]) for i in range(n_blocks)]
        self.lns = [eqx.nn.LayerNorm((hidden,)) for _ in range(n_blocks)]
        self.outp = eqx.nn.Linear(hidden, n_actions + n_actions + 1, key=ks[-1])
    def __call__(self, x):
        h0 = self.inp(self.emb[x].reshape(-1)); h = h0
        for fc, ln in zip(self.fcs, self.lns):
            h = jax.nn.relu(fc(ln(h)) + h)
        out = self.outp(h + h0)
        return out[:self.n_actions], out[self.n_actions:]   # backward (A,), forward (A+1,)

# ---- sampling + loss, closed over the (device) environment tensors ----
def make_fns(gens, inv_idx, pre, solved, nmax, eps, true_log_z):
    gens_j, inv_j = jnp.asarray(gens), jnp.asarray(inv_idx)
    pre_j, solved_j = jnp.asarray(pre), jnp.asarray(solved)
    A = gens.shape[0]

    def entry_mask(s):                      # (N,A) True where a fwd move re-enters solved
        return (s[:, None, :] == pre_j[None]).all(-1)
    def apply_a(s, a):                      # apply gen a to each row
        return jnp.take_along_axis(s, gens_j[a], axis=1)

    def sample(key, model, batch):          # on-policy fwd trajectories from solved
        s0 = jnp.tile(solved_j[None], (batch, 1))
        def step(carry, _):
            k, s = carry
            _, fwd = jax.vmap(model)(s); logits = fwd[:, :A]
            k, ke, ka = jax.random.split(k, 3)
            logits = jnp.where(jax.random.bernoulli(ke, eps, (batch,))[:, None], 0.0, logits)
            logits = jnp.where(entry_mask(s), NEG_INF, logits)
            a = jax.random.categorical(ka, logits)
            return (k, apply_a(s, a)), (apply_a(s, a), a)
        (_, _), (st, ac) = jax.lax.scan(step, (key, s0), None, length=nmax)
        return jnp.concatenate([s0[None], st], 0), ac

    def loss_fn(params, static, states, actions, lam, flow_shift):
        model = eqx.combine(params, static)
        T, b, S = states.shape; flat = states.reshape(T * b, S)
        bwd, fwd = jax.vmap(model)(flat)
        mask = entry_mask(flat)
        full = jnp.concatenate([mask, jnp.zeros((mask.shape[0], 1), bool)], 1)
        log_pf = jax.nn.log_softmax(jnp.where(full, NEG_INF, fwd), -1).reshape(T, b, -1)
        log_pb = jax.nn.log_softmax(bwd, -1).reshape(T, b, -1)
        log_flows = -log_pf[..., -1]                       # F(s) = 1/P_F(stop|s)
        pf_t = jnp.take_along_axis(log_pf[:-1], actions[..., None], -1)[..., 0]
        pb_t = jnp.take_along_axis(log_pb[1:], inv_j[actions][..., None], -1)[..., 0]
        z = jnp.zeros((1, b))
        pre_f = jnp.concatenate([z, jnp.cumsum(pf_t, 0)], 0)
        pre_b = jnp.concatenate([z, jnp.cumsum(pb_t, 0)], 0)
        residual = true_log_z + pre_f - pre_b - log_flows   # prefix trajectory balance
        tb = jnp.mean(residual ** 2)
        # exact paper flow reg, float32 fixed-shift factorization (lnZ 56 fits fp32):
        lse = jax.scipy.special.logsumexp(log_flows[1:], 0)
        reg = lam * jnp.mean(jnp.exp(jnp.minimum(lse - flow_shift, 55.0)))
        return tb + reg, jnp.array([tb, reg, jnp.mean(log_flows[1]), jnp.mean(log_flows[-1])])

    def greedy(model, starts, max_steps):    # argmax P_B rollout (for a cheap eval)
        def step(carry, _):
            s, done, ln = carry
            a = jnp.argmax(jax.vmap(model)(s)[0], -1)
            s2 = jnp.where(done[:, None], s, apply_a(s, a))
            done2 = done | (s2 == solved_j[None]).all(-1)
            return (s2, done2, ln + (~done)), None
        d0 = (starts == solved_j[None]).all(-1)
        (s, d, ln), _ = jax.lax.scan(step, (starts, d0, jnp.zeros(starts.shape[0], jnp.int32)),
                                     None, length=max_steps)
        return d, ln

    return sample, loss_fn, greedy, apply_a

print("core defined")
