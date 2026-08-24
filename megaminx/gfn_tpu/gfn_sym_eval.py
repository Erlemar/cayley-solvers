"""Sym-ensemble beam eval for a GFN checkpoint on IHES.

For each pid, beam-solve the base scramble plus K random cube-symmetry frames
(s_rot = P . s . P^-1) and take the per-pid MINIMUM solved length. The beam
navigates each rotated instance differently, so min-over-frames shaves moves
(the production sym-ensemble trick). Reports base vs sym-min, and compares to a
reference submission CSV.

Usage (on the TPU VM):
  ~/tpu-env/bin/python gfn_sym_eval.py --ckpt ~/ihes_long_l5e8/ckpt_latest \
     --puzzle-info ~/v6e/puzzle_info.json --test-csv ~/v6e/test.csv \
     --sym-file ~/v6e/cube_symmetries.npy --sym-k 2 \
     --pids 25,50,75,...,1000 --width 16384 --max-steps 45
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

import sys
sys.path.insert(0, ".")
import gfn_train_tpu as T
from gfn_beam_eval import beam_solve


def apply_symmetry(s, P, P_inv):
    # P . s . P^-1  ->  new[p] = P[s[P_inv[p]]]
    return P[s[P_inv]]


def make_device_beam(model, gens, solved, fwd_chunk=131072):
    """Device-resident policy beam for large widths (up to ~1M). Memory-efficient:
    scores all W*A candidates (a cheap (W*A,) float vector), top_k, then builds
    ONLY the W selected child states (never materializes W*A*S). No dedup: at this
    puzzle's branching mid-beam collisions are rare, so effective width ~ W. Solve
    is detected among the top-W survivors (a solving move scores near-max, so it
    survives width-1M pruning in practice)."""
    gens_j = jnp.asarray(gens.astype(np.int32))          # (A,S)
    solved_j = jnp.asarray(solved.astype(np.int32))      # (S,)
    A, S = gens_j.shape

    gens8 = gens_j.astype(jnp.int8)
    solved8 = solved_j.astype(jnp.int8)

    def fwd_chunked(beam_i8):
        # beam stored int8; model wants int32. Chunk to bound activation memory.
        outs = []
        for i in range(0, beam_i8.shape[0], fwd_chunk):
            outs.append(jax.vmap(model)(beam_i8[i:i+fwd_chunk].astype(jnp.int32))[0])
        return jnp.concatenate(outs, axis=0)

    from functools import partial

    @partial(jax.jit, static_argnums=2)
    def step_fn(beam, scores, width):
        # beam: (W,S) int8; scores: (W,) f32
        bwd = fwd_chunked(beam)                              # (W,A)
        logp = jax.nn.log_softmax(bwd, axis=-1)
        cand = (scores[:, None] + logp).reshape(-1)         # (W*A,) f32
        topval, top = jax.lax.top_k(cand, width)            # (width,)
        parent = top // A
        move = top % A
        # build only the width selected children (int8): new[i]=beam[parent[i]][gens[move[i]]]
        new_beam = jnp.take_along_axis(beam[parent], gens8[move], axis=1)   # (width,S) int8
        hit = (new_beam == solved8[None, :]).all(-1).any()
        return new_beam, topval, hit

    def solve(start, width, max_steps):
        beam = jnp.asarray(start[None, :].astype(np.int8))   # (1,S) int8
        scores = jnp.zeros(1)
        for step in range(max_steps):
            k = int(min(width, beam.shape[0] * A))
            beam, scores, hit = step_fn(beam, scores, k)
            if bool(hit):
                return True, step + 1
        return False, -1

    return solve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--puzzle-info", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--sym-file", required=True, help="cube_symmetries.npy (48,72)")
    ap.add_argument("--sym-k", type=int, default=2, help="random non-identity syms per pid")
    ap.add_argument("--sym-seed", type=int, default=0)
    ap.add_argument("--pids", required=True)
    ap.add_argument("--width", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=45)
    ap.add_argument("--ref-csv", default="", help="submission CSV to compare against")
    ap.add_argument("--device-beam", action="store_true",
                    help="use device-resident beam (needed for width > ~65536)")
    ap.add_argument("--fwd-chunk", type=int, default=131072)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg = json.load(open(ckpt.parent / "config.json", encoding="utf-8"))
    names, gens, inv_idx, solved, pre, num_classes = T.load_env(args.puzzle_info)

    model = T.GFNPolicy(len(solved), num_classes, len(gens), cfg["hidden"],
                        cfg["emb_dim"], cfg["blocks"], jax.random.key(0))
    params, static = eqx.partition(model, eqx.is_array)
    tree = {"m": params, "z": jnp.array(T.TRUE_LOG_Z)}
    tree = eqx.tree_deserialise_leaves(str(ckpt / "train.eqx"), tree)
    model = eqx.combine(tree["m"], static)

    @jax.jit
    def fwd_fn(x):
        bwd, _ = jax.vmap(model)(x)
        return bwd

    dev_solve = make_device_beam(model, gens, solved, fwd_chunk=args.fwd_chunk) \
        if args.device_beam else None

    def solve_one(st):
        if args.device_beam:
            return dev_solve(st.astype(np.int32), args.width, args.max_steps)
        return beam_solve(fwd_fn, gens, solved, st.astype(np.int32),
                          args.width, args.max_steps)

    sym = np.load(args.sym_file).astype(np.int64)         # (48,72)
    sym_inv = np.empty_like(sym)
    for i in range(sym.shape[0]):
        sym_inv[i][sym[i]] = np.arange(sym.shape[1])
    # identity is row where sym[i]==arange; find non-identity indices
    ar = np.arange(sym.shape[1])
    nonid = [i for i in range(sym.shape[0]) if not np.array_equal(sym[i], ar)]
    rng = np.random.default_rng(args.sym_seed)

    pids = [int(x) for x in args.pids.split(",") if x]
    states = T.load_pid_states(args.test_csv, pids)       # (P,72)

    ref = {}
    if args.ref_csv:
        for row in csv.reader(open(args.ref_csv)):
            if row[0].lower() in ("id", "initial_state_id"):
                continue
            mv = row[1].strip()
            ref[int(row[0])] = (mv.count(".") + 1 if "." in mv else len(mv.split())) if mv else 0

    print(f"sym-ensemble: base + {args.sym_k} random syms (of {len(nonid)} non-id), "
          f"width {args.width}", flush=True)
    base_tot = symmin_tot = ref_tot = n_ok = 0
    base_win = sym_win = 0
    for p, s in zip(pids, states):
        frames = [None] + list(rng.choice(nonid, size=args.sym_k, replace=False))
        lens = []
        t0 = time.time()
        for fr in frames:
            st = s if fr is None else apply_symmetry(s, sym[fr], sym_inv[fr])
            ok, ln = solve_one(st)
            lens.append(ln if ok else 10**6)
        base = lens[0]
        symmin = min(lens)
        rl = ref.get(p)
        tag = ""
        if base < 10**6:
            base_tot += base; n_ok += 1
            symmin_tot += symmin
            if rl is not None:
                ref_tot += rl
                if symmin < rl:
                    sym_win += 1; tag = " SYM<REF"
                if base < rl:
                    base_win += 1
        print(f"pid {p}: base={base if base<10**6 else 'X'} "
              f"sym_min={symmin if symmin<10**6 else 'X'} "
              f"ref={rl} frames={[f if f is not None else 'id' for f in frames]}"
              f" ({time.time()-t0:.0f}s){tag}", flush=True)

    if n_ok:
        print(f"\n=== {n_ok} solved pids ===", flush=True)
        print(f"base total {base_tot} (avg {base_tot/n_ok:.2f})", flush=True)
        print(f"sym-min total {symmin_tot} (avg {symmin_tot/n_ok:.2f})  "
              f"[sym saved {base_tot-symmin_tot} over base]", flush=True)
        if ref_tot:
            print(f"ref total {ref_tot} (avg {ref_tot/n_ok:.2f})", flush=True)
            print(f"per-pid min(ref, sym): sym beats ref on {sym_win} pids; "
                  f"base beats ref on {base_win}", flush=True)


if __name__ == "__main__":
    main()
