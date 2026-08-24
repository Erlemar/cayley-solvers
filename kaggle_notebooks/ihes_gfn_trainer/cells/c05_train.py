# --- Training loop (pmap data-parallel over the 8 TPU cores) ---
import os, time, optax

# puzzle_info.json ships in the companion dataset. Kaggle mount paths vary
# (/kaggle/input/<slug>/ vs /kaggle/input/datasets/<owner>/<slug>/), so glob for it.
import glob
cands = (glob.glob("/kaggle/input/**/puzzle_info.json", recursive=True))
assert cands, "puzzle_info.json not found - attach the ihes-gfn-checkpoint dataset"
PUZZLE = cands[0]
print("puzzle:", PUZZLE)

gens, inv_idx, solved, pre = load_env(PUZZLE)
true_log_z = math.log(IHES_ORDER)
ndev = jax.device_count(); b_local = CFG["batch_global"] // ndev
sample, loss_fn, greedy, apply_a = make_fns(gens, inv_idx, pre, solved, CFG["nmax"],
                                            CFG["eps_explore"], true_log_z)
print(f"devices={ndev} b_local={b_local} lnZ={true_log_z:.4f} actions={gens.shape[0]}")

key = jax.random.key(CFG["seed"]); key, mk = jax.random.split(key)
model = GFNPolicy(len(solved), len(solved), gens.shape[0], CFG["hidden"],
                  CFG["emb_dim"], CFG["blocks"], mk)
params, static = eqx.partition(model, eqx.is_array)
opt = optax.chain(optax.clip_by_global_norm(CFG["clip"]),
                  optax.adamw(CFG["lr"], weight_decay=1e-5))
opt_state = opt.init(params)

def loss_wrap(p, s, a, lam, sh):
    return loss_fn(p, static, s, a, lam, sh)

@partial(jax.pmap, axis_name="dp", static_broadcasted_argnums=())
def train_step(params, opt_state, key, lam, shift):
    key, sk = jax.random.split(key)
    st, ac = sample(sk, eqx.combine(params, static), b_local)
    (loss, met), g = jax.value_and_grad(loss_wrap, has_aux=True)(params, st, ac, lam, shift)
    g = jax.lax.pmean(g, "dp"); met = jax.lax.pmean(met, "dp")
    upd, opt_state = opt.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, key, met

# Replicate across the 8 cores by broadcasting a leading device axis; pmap then
# distributes it (version-agnostic - avoids the deprecated device_put_replicated).
def rep(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(jnp.asarray(x), (ndev,) + jnp.asarray(x).shape), tree)
P, O = rep(params), rep(opt_state)
keys = jax.random.split(jax.random.fold_in(key, 1), ndev)
lam_r, sh_r = rep(jnp.float32(CFG["reg_coef"])), rep(jnp.float32(0.0))

def save(tag, P):
    ph = jax.tree_util.tree_map(lambda x: x[0], P)
    m = eqx.combine(ph, static)
    eqx.tree_serialise_leaves(f"{CFG['save_dir']}/{tag}.eqx",
                              {"m": ph, "z": jnp.float32(true_log_z)})
    json.dump({"task": "ihes", "hidden": CFG["hidden"], "blocks": CFG["blocks"],
               "emb_dim": CFG["emb_dim"], "lnZ": true_log_z},
              open(f"{CFG['save_dir']}/config.json", "w"))

t0 = time.time()
for it in range(CFG["iters"]):
    P, O, keys, met = train_step(P, O, keys, lam_r, sh_r)
    if (it + 1) % CFG["log_every"] == 0:
        m = np.asarray(met[0])
        print(f"iter {it+1:>7} tb={m[0]:.3g} reg={m[1]:.3g} logF_d1={m[2]:.3g} "
              f"logF_last={m[3]:.3g}  {(it+1)/(time.time()-t0):.1f} it/s", flush=True)
    if (it + 1) % CFG["ckpt_every"] == 0:
        save("train", P); print("  checkpoint saved")
save("train", P)
print(f"done in {(time.time()-t0)/3600:.2f} h -> /kaggle/working/train.eqx")
