#!/usr/bin/env python3
"""Standalone v6e-4 driver for the SPMD V+qshort megaminx beam.

Adapted from the proven Kaggle v5e-8 notebook (build_notebook.py). Differences:
  - all inputs read from $MM_DATA (default /mnt/data/v6e), not /kaggle/input
  - world_size / devices are dynamic (len(jax.devices()) == 4 on v6e-4)
  - per-pid tree memmap written under --tree-dir (default /mnt/data/trees)
  - single shard, sym_pos=0 (identity rotation), no NISS  -- ceiling-finding /
    smoke config. Sym-ensemble + merge can be layered on later.

The kernel itself (jax_beam_spmd_v_qshort.py) needs NO changes for 4 ranks except
the backpointer rebalance (25/2/5) already applied -> B_local up to 2^25 = 33.5M,
i.e. B_global up to 134M (128Mi) on 4 ranks.

Usage:
  python gcp_beam_v6e.py --b-global 16777216 --start-pid 300 --end-pid 301 \
      --num-steps 70 --alpha 2
"""
from __future__ import annotations

import os

# These MUST be set before `import jax`: the kernel's int64 state hashes need
# x64 mode, and the beam wants 95% of HBM (the default 0.75 caps beam width).
# Matches the Kaggle notebook's setup cell.
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import json
import sys
import time

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# jax-version compat: the kernel calls jax.lax.pcast(x, ("cores",), to="varying"),
# an API name from the Kaggle TPU jax build. jax 0.6.2 (the v6e-compatible release)
# exposes the same shard_map op as jax.lax.pvary(x, axis). Every kernel call site
# uses to="varying" along a single axis, so this maps exactly.
if not hasattr(jax.lax, "pcast"):
    def _pcast_to_pvary(x, axes, to="varying"):
        axis = axes[0] if isinstance(axes, (tuple, list)) else axes
        return jax.lax.pvary(x, axis)
    jax.lax.pcast = _pcast_to_pvary

DATA = os.environ.get("MM_DATA", "/mnt/data/v6e")
sys.path.insert(0, DATA)

from jax_model import load_params_from_pt, num_params  # noqa: E402
from jax_beam_spmd_v_qshort import (  # noqa: E402
    beam_solve_v_qshort_spmd_packed,
    make_mesh,
    build_solved_neighborhood,
)


def make_hash_vec(state_size: int, seed: int = 0) -> np.ndarray:
    """int64 hash vector (inlined from jax_beam.py, identical RNG/range/dtype)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, int(1e15), size=state_size, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-global", type=int, default=16 * 1024 * 1024)
    ap.add_argument("--start-pid", type=int, default=300)
    ap.add_argument("--end-pid", type=int, default=301)
    ap.add_argument("--num-steps", type=int, default=90)
    ap.add_argument("--alpha", type=int, default=2)  # qshort: top-alpha children/parent
    ap.add_argument("--internal-bs", type=int, default=16384)
    ap.add_argument("--parent-chunk", type=int, default=131072)
    ap.add_argument("--nbhd-radius", type=int, default=5)
    ap.add_argument("--profile-skip", type=str, default="")  # gen|qfwd|vfwd|packbuild|a2a|argsort
    ap.add_argument("--progress-every", type=int, default=5)
    ap.add_argument("--no-meta", action="store_true")  # disable meta-materialize (idea 4)
    ap.add_argument("--tree-dir", type=str, default="/mnt/data/trees")
    ap.add_argument("--out", type=str, default="/mnt/data/out/results.json")
    args = ap.parse_args()

    devices = jax.devices()
    n_dev = len(devices)
    print(f"[cfg] devices={n_dev} kind={devices[0].device_kind} "
          f"b_global={args.b_global:,} num_steps={args.num_steps} alpha={args.alpha}",
          flush=True)
    assert args.b_global % n_dev == 0, f"b_global must divide by {n_dev}"
    b_local = args.b_global // n_dev
    k_per_peer = (args.alpha * b_local) // n_dev
    assert b_local % args.parent_chunk == 0, \
        f"parent_chunk={args.parent_chunk} must divide b_local={b_local}"
    print(f"[cfg] b_local={b_local:,} k_per_peer={k_per_peer:,} "
          f"n_chunks/step={b_local // args.parent_chunk}", flush=True)

    # --- puzzle ---
    pinfo = json.load(open(f"{DATA}/puzzle_info.json", encoding="utf-8"))
    generators = pinfo["generators"]
    solved = tuple(pinfo["central_state"])
    move_names = list(generators.keys())
    n_gen = len(move_names)
    state_size = len(solved)
    all_moves_np = np.array([generators[n] for n in move_names], dtype=np.int32)
    all_moves = jnp.asarray(all_moves_np)
    v0_np = np.array(solved, dtype=np.int8)
    v0 = jnp.asarray(v0_np)
    print(f"[puzzle] n_gen={n_gen} state_size={state_size}", flush=True)

    # inverse-move index (for the solved-neighborhood early stop)
    inv_move_idx = np.full(n_gen, -1, dtype=np.int32)
    nm2i = {n: i for i, n in enumerate(move_names)}
    for i, nm in enumerate(move_names):
        inv = nm[1:] if nm.startswith("-") else "-" + nm
        inv_move_idx[i] = nm2i[inv]
    assert (inv_move_idx >= 0).all()

    # --- scrambles ---
    rows = list(csv.DictReader(open(f"{DATA}/test.csv", encoding="utf-8")))
    all_states = {pid: [int(x) for x in rows[pid]["initial_state"].split(",")]
                  for pid in range(len(rows))}
    print(f"[scrambles] {len(all_states)} pids loaded", flush=True)

    # --- models (torch only used to read the .pt state dicts) ---
    print("[load] V (m_az_v4_v_only) ...", flush=True)
    v_params = load_params_from_pt(f"{DATA}/m_az_v4_v_only.pt", hidden_dims=(2048, 512))
    print(f"[load]   V params {num_params(v_params):,}", flush=True)
    print("[load] Q (m23_v3_az_v4_sym) ...", flush=True)
    q_params = load_params_from_pt(f"{DATA}/m23_v3_az_v4_sym_epoch_0199.pt",
                                   hidden_dims=(2048, 1024), num_res_blocks=3)
    print(f"[load]   Q params {num_params(q_params):,}", flush=True)

    # --- hashes / mesh ---
    hash_vec_np = make_hash_vec(state_size, seed=0)
    hash_vec = jnp.asarray(hash_vec_np)
    owner_rng = np.random.default_rng(12345)
    owner_hash_vec = jnp.asarray(
        owner_rng.integers(0, np.iinfo(np.uint32).max, size=state_size, dtype=np.uint32))
    mesh = make_mesh(devices)
    print(f"[mesh] {mesh}", flush=True)

    # --- solved neighborhood (K1-ball early stop), built once ---
    print(f"[nbhd] building radius={args.nbhd_radius} ...", flush=True)
    t_nb = time.time()
    nbhd_hash, nbhd_suflen, nbhd_sufmv = build_solved_neighborhood(
        all_moves_np, inv_move_idx, hash_vec_np, v0_np, args.nbhd_radius)
    print(f"[nbhd]   M={nbhd_hash.shape[0]:,} ({time.time() - t_nb:.1f}s)", flush=True)

    os.makedirs(args.tree_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def verify(initial_state, path_idx):
        cur = list(initial_state)
        for m in path_idx:
            g = generators[move_names[m]]
            cur = [cur[x] for x in g]
        return tuple(cur) == solved

    results = []
    for pid in range(args.start_pid, args.end_pid):
        s0 = all_states[pid]
        tree_path = f"{args.tree_dir}/tree_pid{pid}.u32"
        t0 = time.time()
        r = beam_solve_v_qshort_spmd_packed(
            list(s0), v_params, q_params, args.alpha,
            all_moves, v0, hash_vec, mesh,
            B_local=b_local, K_per_peer=k_per_peer,
            n_gen=n_gen, state_size=state_size,
            num_steps=args.num_steps, dtype=jnp.bfloat16,
            internal_bs=args.internal_bs,
            tree_path=tree_path, parent_chunk=args.parent_chunk,
            pack_v_score=True, progress_every=args.progress_every,
            owner_hash_vec=owner_hash_vec,
            nbhd_hash_sorted=nbhd_hash, nbhd_suffix_len=nbhd_suflen,
            nbhd_suffix_moves=nbhd_sufmv,
            use_meta_materialize=not args.no_meta, alpha_req=1.5, lean_merge=True,
            profile_skip=args.profile_skip,
        )
        wall = time.time() - t0
        # sym_pos=0 -> path is already in original move space
        ok = verify(s0, r["path_idx"]) if r["found"] else False
        print(f"[solve] pid={pid} found={r['found']} len={r.get('path_len', -1)} "
              f"verify={ok} via={r.get('via', '-')} found_step={r.get('found_step', -1)} "
              f"wall={wall:.1f}s", flush=True)
        results.append({
            "pid": pid, "b_global": args.b_global,
            "found": bool(r["found"]), "verify": bool(ok),
            "path_len": int(r.get("path_len", -1)),
            "path_idx": list(map(int, r.get("path_idx", []))),
            "found_step": int(r.get("found_step", -1)),
            "wall_s": round(wall, 1),
        })
        if os.path.exists(tree_path):
            try:
                os.unlink(tree_path)
            except OSError:
                pass
        json.dump(results, open(args.out, "w", encoding="utf-8"), indent=0)

    print(f"[done] {len(results)} pids -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
