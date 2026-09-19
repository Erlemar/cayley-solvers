"""Offline policy-beam eval for a gfn_train_tpu checkpoint (hybrid: model forward
on TPU, beam selection host-side numpy). ASCII output.

Usage:
  ~/tpu-env/bin/python gfn_beam_eval.py --ckpt /home/and-l/gfn_ckpt/ckpt_latest \
      --puzzle-info ~/v6e/puzzle_info.json --test-csv ~/v6e/test.csv \
      --walk-depths 6,10,15,20,30,40 --n-walks 10 --width 256 --pids 0,1,2,3,5 \
      --pid-width 1024 --max-steps 120
"""
import argparse
import json
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

sys.path.insert(0, ".")
import gfn_train_tpu as T


def beam_solve(fwd_fn, gens, solved, start, width, max_steps):
    beam = start[None, :].astype(np.int32)  # (W,S)
    scores = np.zeros(1, dtype=np.float64)
    solved_b = solved.astype(np.int32)
    for step in range(max_steps):
        w = beam.shape[0]
        bwd = np.asarray(fwd_fn(jnp.asarray(beam)))  # (w, A)
        logp = bwd - np.logaddexp.reduce(bwd, axis=1, keepdims=True)
        cand_scores = (scores[:, None] + logp).reshape(-1)
        children = beam[:, gens].reshape(-1, beam.shape[1])  # (w*A, S)
        hit = (children == solved_b[None, :]).all(1)
        if hit.any():
            return True, step + 1
        # dedup: keep best score per state
        order = np.argsort(-cand_scores, kind="stable")
        ch_o = children[order]
        view = ch_o.view([("", ch_o.dtype)] * ch_o.shape[1]).ravel()
        _, first = np.unique(view, return_index=True)
        keep = order[np.sort(first)]
        if keep.shape[0] > width:
            top = np.argpartition(-cand_scores[keep], width)[:width]
            keep = keep[top]
        beam = children[keep]
        scores = cand_scores[keep]
    return False, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--puzzle-info", default="")
    ap.add_argument("--cube-specs", default="")
    ap.add_argument("--test-csv", default="")
    ap.add_argument("--test-npy", default="", help="cube test set for --pids-npy")
    ap.add_argument("--walk-depths", default="6,10,15,20,30,40")
    ap.add_argument("--n-walks", type=int, default=10)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--pids", default="")
    ap.add_argument("--pids-npy", default="", help="comma indices into --test-npy")
    ap.add_argument("--pid-width", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from pathlib import Path

    ckpt = Path(args.ckpt)
    cfg = json.load(open(ckpt.parent / "config.json", encoding="utf-8"))
    task = cfg.get("task", "megaminx")
    if task in ("megaminx", "ihes"):
        names, gens, inv_idx, solved, pre, num_classes = T.load_env(args.puzzle_info)
    else:
        (names, gens, inv_idx, solved, pre, num_classes), _ = T.load_cube_env(
            args.cube_specs, task)

    key = jax.random.key(0)
    model = T.GFNPolicy(len(solved), num_classes, len(gens), cfg["hidden"],
                        cfg["emb_dim"], cfg["blocks"], key)
    params, static = eqx.partition(model, eqx.is_array)
    tree = {"m": params, "z": jnp.array(T.TRUE_LOG_Z)}
    tree = eqx.tree_deserialise_leaves(str(ckpt / "train.eqx"), tree)
    model = eqx.combine(tree["m"], static)
    with open(ckpt / "host.json", encoding="utf-8") as f:
        host = json.load(f)
    print(f"loaded ckpt at step {host['step']}, logZ={float(np.asarray(tree['z'])):.2f}")

    @jax.jit
    def fwd_fn(x):
        bwd, _ = jax.vmap(model)(x)
        return bwd

    rng = np.random.default_rng(args.seed)
    depths = [int(x) for x in args.walk_depths.split(",") if x]
    walks = T.walk_states(gens, inv_idx, solved, depths, args.n_walks, rng)
    for d in depths:
        t0 = time.time()
        sn, tot = 0, 0
        for i in range(args.n_walks):
            ok, ln = beam_solve(fwd_fn, gens, solved, walks[d][i], args.width,
                                min(args.max_steps, 2 * d + 30))
            sn += int(ok)
            tot += ln if ok else 0
        ml = tot / sn if sn else -1
        print(f"walk d={d}: beam{args.width} rate={sn}/{args.n_walks}"
              f" len={ml:.1f} ({time.time() - t0:.0f}s)", flush=True)

    if args.pids and args.test_csv:
        pids = [int(x) for x in args.pids.split(",") if x]
        st = T.load_pid_states(args.test_csv, pids)
        for p, s in zip(pids, st):
            t0 = time.time()
            ok, ln = beam_solve(fwd_fn, gens, solved, s, args.pid_width, args.max_steps)
            print(f"pid {p}: beam{args.pid_width} found={ok} len={ln}"
                  f" ({time.time() - t0:.0f}s)", flush=True)

    # cube test set: beam over a random subset, report Table-1-style avg length
    if args.test_npy:
        full = np.load(args.test_npy).astype(np.int32)
        n = int(args.pids_npy) if args.pids_npy else min(50, len(full))
        idx = rng.choice(len(full), size=min(n, len(full)), replace=False)
        sn, tot = 0, 0
        t0 = time.time()
        for i in idx:
            ok, ln = beam_solve(fwd_fn, gens, solved, full[i], args.pid_width,
                                args.max_steps)
            sn += int(ok)
            tot += ln if ok else 0
        print(f"cube-test beam{args.pid_width} n={len(idx)}: solve_rate={sn/len(idx):.3f}"
              f" avg_len={tot/sn if sn else -1:.2f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
