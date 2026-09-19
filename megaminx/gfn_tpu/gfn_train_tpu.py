"""GFN-pathfinding trainer for FULL megaminx on TPU (JAX/equinox, pmap data-parallel).

Port of Morozov et al. 2026 (arXiv:2603.01786) at rubik3-scale budget, with the
recipe improvements validated locally 2026-07-17/18 (see
megaminx/gfn_pathfinding_analysis_2026-07-17.md):
  - embedding state encoding (120 classes x 16, vs one-hot 14400)
  - eps-explore 0.1 (anti mode-collapse; mandatory)
  - learnable logZ initialized at the exact group order (Schreier-Sims verified)
  - AUTO two-stage flow regularization: stage 1 runs lambda=0 (pure TB relaxation;
    the reg is provably inert during the flow climb and float32-unsafe there);
    when the EMA of the TB loss drops below --tb-switch, lambda is self-sized as
    reg_target / exp(logF_d1_at_snap) and a safety flow-clip is set. This is the
    v2 two-stage recipe, automated for an unattended run.
  - snap + periodic immutable checkpoints (transient-peak lesson), resumable
    (params + opt state + host state json) for flex-VM death.

Run (on the TPU VM):
  ~/tpu-env/bin/python gfn_train_tpu.py --batch-global 4096 --nmax 60 \
      --iters 1000000 --save-dir /mnt/data/gfn_ckpt \
      --puzzle-info /mnt/data/v6e/puzzle_info.json --test-csv /mnt/data/v6e/test.csv

ASCII-only output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

# Megaminx sticker group order (Schreier-Sims on the 24 generators, 2026-07-17).
MEGAMINX_ORDER = 100669616553523347122516032313645505168688116411019768627200000000000
TRUE_LOG_Z = math.log(MEGAMINX_ORDER)  # 156.582460
IHES_ORDER = 2125922464947725402112000  # 18-gen 72-perm, lnZ 56.0162 (cube-scale)

NEG_INF = -1e30


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def build_env(names, gens, inv_idx, solved, num_classes):
    """Construct an env from raw arrays. Handles BOTH permutation puzzles
    (megaminx: solved == identity, num_classes == state_size) and COLOR puzzles
    (cubes: solved is a coloring, num_classes == 6). The generator inverse check
    is on the PERMUTATIONS (via arange, coloring-independent); the preimage check
    is on the coloring."""
    gens = np.asarray(gens, dtype=np.int32)
    inv_idx = np.asarray(inv_idx, dtype=np.int32)
    solved = np.asarray(solved, dtype=np.int32)
    n = gens.shape[1]
    ar = np.arange(n, dtype=np.int32)
    # perm-composition identity: gen j then gen inv_idx[j] returns any state
    for j in range(len(gens)):
        assert (gens[j][gens[inv_idx[j]]] == ar).all(), f"perm inverse fail {j}"
    # coloring preimage: pre[j] is the state one fwd-step INTO solved via action j
    pre = solved[gens[inv_idx]].copy()
    for j in range(len(gens)):
        assert (pre[j][gens[j]] == solved).all(), f"coloring preimage fail {j}"
    return names, gens, inv_idx, solved, pre, num_classes


def load_env(puzzle_info_path: str):
    with open(puzzle_info_path, encoding="utf-8") as f:
        info = json.load(f)
    names = list(info["generators"].keys())
    gens = [info["generators"][n] for n in names]
    inv_name = lambda n: n[1:] if n.startswith("-") else "-" + n
    inv_idx = [names.index(inv_name(n)) for n in names]
    solved = info["central_state"]
    return build_env(names, gens, inv_idx, solved, num_classes=len(solved))


def load_cube_env(spec_path: str, task: str):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)[task]
    names = [f"m{i}" for i in range(len(spec["gens"]))]
    env = build_env(names, spec["gens"], spec["inv_idx"], spec["solved"],
                    num_classes=spec["num_classes"])
    return env, spec["log_z"]


# ---------------------------------------------------------------------------
# Model (paper arch: 6 single-linear residual blocks; embedding input)
# ---------------------------------------------------------------------------


class GFNPolicy(eqx.Module):
    emb: jax.Array  # (num_classes, emb_dim)
    inp: eqx.nn.Linear
    fcs: list
    lns: list
    outp: eqx.nn.Linear
    n_actions: int = eqx.field(static=True)

    def __init__(self, state_size, num_classes, n_actions, hidden, emb_dim, n_blocks, key):
        keys = jax.random.split(key, n_blocks + 3)
        self.n_actions = n_actions
        self.emb = jax.random.normal(keys[0], (num_classes, emb_dim)) * 0.02
        self.inp = eqx.nn.Linear(state_size * emb_dim, hidden, key=keys[1])
        self.fcs = [eqx.nn.Linear(hidden, hidden, key=keys[2 + i]) for i in range(n_blocks)]
        self.lns = [eqx.nn.LayerNorm(shape=(hidden,)) for _ in range(n_blocks)]
        self.outp = eqx.nn.Linear(hidden, n_actions + n_actions + 1, key=keys[-1])

    def __call__(self, x):  # x: (state_size,) int
        h0 = self.inp(self.emb[x].reshape(-1))
        h = h0
        for fc, ln in zip(self.fcs, self.lns):
            h = jax.nn.relu(fc(ln(h)) + h)
        out = self.outp(h + h0)
        return out[: self.n_actions], out[self.n_actions:]  # bwd (A,), fwd (A+1,)


# ---------------------------------------------------------------------------
# Sampling + loss (per-device shard)
# ---------------------------------------------------------------------------


def make_fns(gens, inv_idx, pre, solved, nmax, eps, reg2_coef=0.0, paper_reg=False,
             true_log_z=TRUE_LOG_Z):
    gens_j = jnp.asarray(gens)
    inv_j = jnp.asarray(inv_idx)
    pre_j = jnp.asarray(pre)
    solved_j = jnp.asarray(solved)
    n_act = gens.shape[0]

    def entry_mask(s):  # s (B,S) -> (B,A) True where fwd action enters solved
        return (s[:, None, :] == pre_j[None, :, :]).all(-1)

    def apply_actions(s, a):  # (B,S),(B,) -> (B,S)
        return jnp.take_along_axis(s, gens_j[a], axis=1)

    def sample_trajectories(key, model, batch):
        s0 = jnp.tile(solved_j[None, :], (batch, 1))

        def step(carry, _):
            k, s = carry
            _, fwd = jax.vmap(model)(s)
            logits = fwd[:, :n_act]
            k, ke, ka = jax.random.split(k, 3)
            explore = jax.random.bernoulli(ke, eps, (batch,))
            logits = jnp.where(explore[:, None], 0.0, logits)
            logits = jnp.where(entry_mask(s), NEG_INF, logits)
            a = jax.random.categorical(ka, logits)
            s2 = apply_actions(s, a)
            return (k, s2), (s2, a)

        (_, _), (states, actions) = jax.lax.scan(step, (key, s0), None, length=nmax)
        states = jnp.concatenate([s0[None], states], axis=0)  # (T+1,B,S)
        return states, actions  # actions (T,B)

    def loss_fn(params, static, log_z, states, actions, lam, flow_clip,
                flow_shift, fixed_z=False):
        model = eqx.combine(params, static)
        tp1, b, ssz = states.shape
        flat = states.reshape(tp1 * b, ssz)
        bwd, fwd = jax.vmap(model)(flat)
        mask = entry_mask(flat)
        full_mask = jnp.concatenate([mask, jnp.zeros((mask.shape[0], 1), bool)], axis=1)
        log_pf = jax.nn.log_softmax(jnp.where(full_mask, NEG_INF, fwd), axis=-1)
        log_pb = jax.nn.log_softmax(bwd, axis=-1)
        log_pf = log_pf.reshape(tp1, b, -1)
        log_pb = log_pb.reshape(tp1, b, -1)
        log_flows = -log_pf[..., -1]  # (T+1,B)

        pf_t = jnp.take_along_axis(log_pf[:-1], actions[..., None], axis=-1)[..., 0]
        pb_t = jnp.take_along_axis(log_pb[1:], inv_j[actions][..., None], axis=-1)[..., 0]
        zeros = jnp.zeros((1, b))
        pre_f = jnp.concatenate([zeros, jnp.cumsum(pf_t, axis=0)], axis=0)
        pre_b = jnp.concatenate([zeros, jnp.cumsum(pb_t, axis=0)], axis=0)

        lz = true_log_z if fixed_z else log_z
        residual = lz + pre_f - pre_b - log_flows
        tb = jnp.mean(residual**2)
        # flow reg: inert in stage 1 (lam=0); stage 2 lambda is self-sized so the
        # clip (set ~30 units above the snap flow level) never binds in normal
        # operation -- it only guards float32 exp overflow on transient outliers.
        if paper_reg:
            # EXACT paper reg (train.py): reg_coef * mean_B(exp(logsumexp_T(logF[1:]))).
            # = lam * mean_B(sum_T F_t). Fixed-shift factorization for float32:
            #   lam_paper*mean(exp(LSE)) == (lam_paper*e^shift)*mean(exp(LSE-shift)).
            # shift=0 for cubes (LSE ~ lnZ <= 45, exp fits float32); shift~110 for
            # megaminx (LSE ~ 160 overflows). reg-coef = lam_paper*e^shift.
            lse_t = jax.scipy.special.logsumexp(log_flows[1:], axis=0)  # (B,)
            reg = lam * jnp.mean(jnp.exp(jnp.minimum(lse_t - flow_shift, 55.0)))
        else:
            # Two-term float32 homolog (2026-07-19): linear-in-logF prices out
            # the degenerate uniform+flat-at-Z TB solution; exp(logF/2) adds an
            # exponential near-root/deep pressure differential. Verdict at 430k:
            # reshapes flows as designed but produces NO deployable ordering.
            reg = lam * jnp.mean(jnp.minimum(log_flows[1:], flow_clip))
            if reg2_coef > 0.0:
                reg = reg + reg2_coef * jnp.mean(
                    jnp.exp(jnp.minimum(log_flows[1:] * 0.5, 60.0)))
        loss = tb + reg
        metrics = jnp.array([
            tb, reg, jnp.sqrt(jnp.mean(residual**2)),
            jnp.mean(log_flows[1]), jnp.mean(log_flows[-1]),
        ])
        return loss, metrics

    def greedy_rollout(model, starts, max_steps):
        def step(carry, _):
            s, done, ln = carry
            bwd, _ = jax.vmap(model)(s)
            a = jnp.argmax(bwd, axis=-1)
            s2 = apply_actions(s, a)
            s2 = jnp.where(done[:, None], s, s2)
            ln = ln + (~done)
            done2 = done | (s2 == solved_j[None, :]).all(-1)
            return (s2, done2, ln), None

        done0 = (starts == solved_j[None, :]).all(-1)
        (s, done, ln), _ = jax.lax.scan(
            step, (starts, done0, jnp.zeros(starts.shape[0], jnp.int32)), None,
            length=max_steps)
        return done, ln

    return sample_trajectories, loss_fn, greedy_rollout, apply_actions


# ---------------------------------------------------------------------------
# Eval-state generation (host, numpy)
# ---------------------------------------------------------------------------


def bfs_exact_states(gens, solved, max_depth, sample_per_depth, rng):
    seen = {solved.tobytes()}
    frontier = solved[None, :].astype(np.uint8)
    out = {}
    for d in range(1, max_depth + 1):
        children = frontier[:, gens].reshape(-1, frontier.shape[1])
        children = np.unique(children, axis=0)
        keep = np.fromiter((c.tobytes() not in seen for c in children), bool, len(children))
        frontier = children[keep]
        for c in frontier:
            seen.add(c.tobytes())
        idx = rng.choice(len(frontier), size=min(sample_per_depth, len(frontier)),
                         replace=False)
        out[d] = frontier[idx].astype(np.int32)
    return out


def walk_states(gens, inv_idx, solved, depths, n, rng):
    out = {}
    s = np.tile(solved[None, :], (n, 1))
    prev = np.full(n, -1)
    for d in range(1, max(depths) + 1):
        a = rng.integers(0, len(gens), n)
        bad = (prev >= 0) & (a == inv_idx[np.clip(prev, 0, None)])
        while bad.any():
            a[bad] = rng.integers(0, len(gens), bad.sum())
            bad = (prev >= 0) & (a == inv_idx[np.clip(prev, 0, None)])
        s = np.take_along_axis(s, gens[a], axis=1)
        prev = a
        if d in depths:
            out[d] = s.copy().astype(np.int32)
    return out


def load_pid_states(test_csv, pids):
    with open(test_csv, encoding="utf-8") as f:
        rows = {r["initial_state_id"]: r for r in csv.DictReader(f)}
    out = []
    for p in pids:
        r = rows[str(p)]
        key = [k for k in r if k != "initial_state_id"][0]
        raw = r[key].replace("[", "").replace("]", "")
        sep = ";" if ";" in raw else ","
        out.append([int(x) for x in raw.split(sep)])
    return np.array(out, dtype=np.int32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-global", type=int, default=4096)
    ap.add_argument("--nmax", type=int, default=60)
    ap.add_argument("--iters", type=int, default=1000000)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--emb-dim", type=int, default=16)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--logz-lr-mult", type=float, default=10.0)
    ap.add_argument("--clip", type=float, default=100.0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--tb-switch", type=float, default=1.0,
                    help="EMA tb threshold for the snap milestone checkpoint")
    ap.add_argument("--reg-coef", type=float, default=0.05,
                    help="lambda for the linear-in-logF flow reg, active from step 0")
    ap.add_argument("--reg2-coef", type=float, default=1e-29,
                    help="lambda for the exp(logF/2) near-root ordering reg")
    ap.add_argument("--paper-reg", action="store_true",
                    help="EXACT paper objective via fixed-shift factorization +"
                         " fixed true Z. reg-coef is lam' = lam*e^shift")
    ap.add_argument("--flow-shift", type=float, default=100.0,
                    help="fixed shift for paper-reg; 0 for cubes (float32-safe at"
                         " lnZ<=45), ~100 for megaminx. reg-coef = lam_paper*e^shift")
    ap.add_argument("--task", default="megaminx",
                    choices=["megaminx", "rubik2", "rubik3", "ihes"])
    ap.add_argument("--cube-specs", default="", help="cube_specs.json for cube tasks")
    ap.add_argument("--test-npy", default="", help="cube test set (N,S) colorings")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=10000)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--immutable-every", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--load", default="", help="checkpoint dir to resume from")
    ap.add_argument("--puzzle-info", default="")
    ap.add_argument("--test-csv", default="")
    ap.add_argument("--eval-pids", default="0,1,2,3,5,8,10,15,20,30,50,100,200")
    args = ap.parse_args()

    ndev = jax.device_count()
    assert args.batch_global % ndev == 0
    b_local = args.batch_global // ndev

    if args.task == "megaminx":
        names, gens, inv_idx, solved, pre, num_classes = load_env(args.puzzle_info)
        true_log_z = TRUE_LOG_Z
    elif args.task == "ihes":
        names, gens, inv_idx, solved, pre, num_classes = load_env(args.puzzle_info)
        true_log_z = math.log(IHES_ORDER)
    else:
        (names, gens, inv_idx, solved, pre, num_classes), true_log_z = load_cube_env(
            args.cube_specs, args.task)
    print(f"devices={ndev} ({jax.devices()[0].device_kind}) b_local={b_local}"
          f" task={args.task} true_lnZ={true_log_z:.4f} num_classes={num_classes}",
          flush=True)

    sample_traj, loss_fn, greedy_rollout, _ = make_fns(
        gens, inv_idx, pre, solved, args.nmax, args.eps, reg2_coef=args.reg2_coef,
        paper_reg=args.paper_reg, true_log_z=true_log_z)

    key = jax.random.key(args.seed)
    key, mk = jax.random.split(key)
    model = GFNPolicy(len(solved), num_classes, len(gens), args.hidden,
                      args.emb_dim, args.blocks, mk)
    params, static = eqx.partition(model, eqx.is_array)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"model params: {n_params}", flush=True)

    train_tree = {"m": params, "z": jnp.array(float(true_log_z))}
    labels = {"m": jax.tree_util.tree_map(lambda _: "m", params), "z": "z"}
    opt = optax.chain(
        optax.clip_by_global_norm(args.clip),
        optax.multi_transform(
            {"m": optax.adamw(args.lr, weight_decay=1e-5),
             # plain adam for logZ: weight decay would drag the scalar toward 0
             "z": optax.adam(args.lr * args.logz_lr_mult)},
            labels),
    )
    opt_state = opt.init(train_tree)

    # host state (stage / lambda / step) for resume
    host_defaults = {"step": 0, "stage": 1, "lam": args.reg_coef,
                     "flow_clip": 170.0, "flow_shift": args.flow_shift,
                     "ema_tb": None}
    host = dict(host_defaults)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.load:
        ld = Path(args.load)
        train_tree = eqx.tree_deserialise_leaves(ld / "train.eqx", train_tree)
        opt_state = eqx.tree_deserialise_leaves(ld / "opt.eqx", opt_state)
        with open(ld / "host.json", encoding="utf-8") as f:
            host = {**host_defaults, **json.load(f)}
        print(f"resumed from {ld} at step {host['step']} stage {host['stage']}"
              f" lam {host['lam']:.3e}", flush=True)

    cfg = {**vars(args), "n_params": int(n_params), "ndev": ndev}
    print("resolved config:", json.dumps(cfg, sort_keys=True), flush=True)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # eval assets
    rng = np.random.default_rng(args.seed)
    exact = bfs_exact_states(gens, solved.astype(np.uint8), 4, 96, rng)
    wdepths = [8, 14, 20, 25, 28] if args.task == "ihes" else [10, 20, 30, 40, 60]
    walks = walk_states(gens, inv_idx, solved, wdepths, 96, rng)
    if args.test_npy:  # cube test set (colorings): sample 64 for in-loop greedy eval
        full = np.load(args.test_npy).astype(np.int32)
        idx = rng.choice(len(full), size=min(64, len(full)), replace=False)
        pid_states = full[idx]
        pid_list = list(range(len(pid_states)))
        print(f"eval assets: exact d1-4 x96, walks x96, cube-test x{len(pid_states)}"
              f" (of {len(full)})", flush=True)
    else:
        pid_list = [int(x) for x in args.eval_pids.split(",") if x] if args.test_csv else []
        pid_states = load_pid_states(args.test_csv, pid_list) if pid_list else None
        print(f"eval assets: exact d1-4 x96, walks x96, pids {pid_list}", flush=True)

    def loss_wrapped(tree, states, actions, lam, flow_clip, flow_shift):
        return loss_fn(tree["m"], static, tree["z"], states, actions, lam,
                       flow_clip, flow_shift, fixed_z=args.paper_reg)

    @partial(jax.pmap, axis_name="dp")
    def train_step(tree, opt_state, key, lam, flow_clip, flow_shift):
        key, sk = jax.random.split(key)
        model = eqx.combine(tree["m"], static)
        states, actions = sample_traj(sk, model, b_local)
        (loss, metrics), grads = jax.value_and_grad(loss_wrapped, has_aux=True)(
            tree, states, actions, lam, flow_clip, flow_shift)
        grads = jax.lax.pmean(grads, "dp")
        metrics = jax.lax.pmean(metrics, "dp")
        updates, opt_state = opt.update(grads, opt_state, tree)
        tree = optax.apply_updates(tree, updates)
        return tree, opt_state, key, metrics

    @partial(jax.jit, static_argnums=2)
    def eval_greedy(tree, starts, max_steps):
        model = eqx.combine(tree["m"], static)
        return greedy_rollout(model, starts, max_steps)

    rep = lambda t: jax.device_put_replicated(t, jax.devices())
    tree_r = rep(train_tree)
    opt_r = rep(opt_state)
    keys = jax.random.split(jax.random.fold_in(key, 1), ndev)
    lam_r = rep(jnp.array(host["lam"], jnp.float32))
    clip_r = rep(jnp.array(host["flow_clip"], jnp.float32))
    shift_r = rep(jnp.array(host["flow_shift"], jnp.float32))

    def save(tag):
        tree_h = jax.tree_util.tree_map(lambda x: x[0], tree_r)
        opt_h = jax.tree_util.tree_map(lambda x: x[0], opt_r)
        d = save_dir / tag
        d.mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(d / "train.eqx", tree_h)
        eqx.tree_serialise_leaves(d / "opt.eqx", opt_h)
        with open(d / "host.json", "w", encoding="utf-8") as f:
            json.dump(host, f)

    def run_eval(step):
        t0 = time.time()
        tree_h = jax.tree_util.tree_map(lambda x: x[0], tree_r)
        lines = []
        # single max_steps everywhere -> one compile per batch shape
        for d, st in sorted(exact.items()):
            done, ln = eval_greedy(tree_h, jnp.asarray(st), 150)
            opt_frac = float(((ln == d) & done).mean())
            lines.append(f"exact d={d}: greedy {float(done.mean()):.2f}"
                         f" optimal {opt_frac:.2f}")
        for d, st in sorted(walks.items()):
            done, ln = eval_greedy(tree_h, jnp.asarray(st), 150)
            ml = float(ln[done].mean()) if bool(done.any()) else -1.0
            lines.append(f"walk d={d}: greedy {float(done.mean()):.2f} len {ml:.1f}")
        if pid_states is not None:
            done, ln = eval_greedy(tree_h, jnp.asarray(pid_states), 150)
            if args.test_npy:  # cube test set: summarize rate + mean len
                ml = float(ln[done].mean()) if bool(done.any()) else -1.0
                lines.append(f"test greedy solved {int(done.sum())}/{len(pid_list)}"
                             f" mean_len {ml:.1f}")
            else:
                solved_pids = [(pid_list[i], int(ln[i])) for i in range(len(pid_list))
                               if bool(done[i])]
                lines.append(f"pids greedy solved: {solved_pids}")
        print(f"[eval @ {step}] ({time.time() - t0:.0f}s)\n    "
              + "\n    ".join(lines), flush=True)

    print("compiling + first step...", flush=True)
    t_start = time.time()
    t_last = t_start
    ema = host["ema_tb"]
    for step in range(host["step"], args.iters):
        tree_r, opt_r, keys, metrics = train_step(tree_r, opt_r, keys, lam_r,
                                                  clip_r, shift_r)
        host["step"] = step + 1
        if (step + 1) % args.log_every == 0:
            m = np.asarray(metrics[0])
            tb, reg, rms, lf1, lfl = (float(x) for x in m)
            lz = float(np.asarray(tree_r["z"][0]))
            ema = tb if ema is None else 0.9 * ema + 0.1 * tb
            host["ema_tb"] = ema
            now = time.time()
            ips = args.log_every / (now - t_last)
            t_last = now
            print(f"iter {step + 1} tb={tb:.4g} reg={reg:.4g} rms={rms:.4g}"
                  f" logF_d1={lf1:.4g} logF_last={lfl:.4g} logZ={lz:.2f}"
                  f" lam={host['lam']:.3e} it_per_s={ips:.2f}", flush=True)
            if host["stage"] == 1 and ema is not None and ema < args.tb_switch:
                # single-stage recipe: reg is already active; the "snap" is now
                # just a milestone checkpoint (transient-peak lesson).
                host.update(stage=2)
                print(f"[tb snap] at iter {step + 1}: logF_d1={lf1:.3f}"
                      f" logZ={lz:.2f}", flush=True)
                save("ckpt_snap")
        if (step + 1) % args.eval_every == 0:
            run_eval(step + 1)
        if (step + 1) % args.ckpt_every == 0:
            save("ckpt_latest")
        if (step + 1) % args.immutable_every == 0:
            save(f"ckpt_{step + 1:07d}")

    save("ckpt_final")
    run_eval(args.iters)
    print(f"done in {(time.time() - t_start) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
