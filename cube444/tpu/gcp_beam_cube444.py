#!/usr/bin/env python3
"""V-only SPMD beam driver for the CayleyPy 4x4x4 COLOR cube, with the
24-rotation sym-ensemble built in.

Derived from megaminx's gcp_beam_v_only.py. Differences:
  * loads a ONE-HOT ResMLPDistance (num_classes inferred from the checkpoint)
  * NO NISS (the state is a coloring, not a permutation -- inverse is undefined)
  * sym-ensemble is the COLOR-CUBE form: solve `pi_R[s[R]]` in each rotation
    frame, translate the returned move word back through move_relabel_inv, and
    re-verify against the ORIGINAL state before accepting.

Frames are prefix-nested (frames(4) is a prefix of frames(8)) so a K-sweep and a
shard split are both trivially safe -- identical convention to cube444/scripts/
05_solve.py.

Every accepted path is replayed against central_state on the host; a sym
translation bug can therefore never produce a bad submission -- it can only lose
a solve.

Config via argv: --b-global --start-pid --end-pid --num-steps --alpha
--internal-bs --parent-chunk (0=non-streaming) --sym-ensemble --sym-seed
--sym-positions lo:hi --v-checkpoint --hidden-dims --num-res-blocks --out.
"""
from __future__ import annotations

import os

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

# Persistent compilation cache. The beam step_fn is rebuilt (fresh closure) on
# every run_beam call, so JAX's in-memory jit cache MISSES across pids/frames and
# each call re-compiles (~90 s for 2^20 on v6e-8). The persistent cache is keyed
# by the computation's HLO hash (shapes + ops), NOT the Python callable identity,
# so a fresh jit that lowers to identical HLO HITS the on-disk cache: the first
# beam call compiles (~90 s, written to disk), every later call loads in ~seconds.
# Without this, a full 1043-pid sym-4 run is ~5 days of almost-pure recompile.
_CACHE_DIR = os.environ.get("JAX_CACHE_DIR", "/tmp/jax_cube_cache")
jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

# jax-version compat: kernel calls jax.lax.pcast(...); newer jax exposes pvary.
if not hasattr(jax.lax, "pcast"):
    def _pcast_to_pvary(x, axes, to="varying"):
        axis = axes[0] if isinstance(axes, (tuple, list)) else axes
        return jax.lax.pvary(x, axis)
    jax.lax.pcast = _pcast_to_pvary

DATA = os.environ.get("CUBE_DATA", "/mnt/data/cube444")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DATA)

from jax_model import load_params_from_pt, num_params  # noqa: E402
from jax_beam_spmd_v_only import (  # noqa: E402
    beam_solve_v_only_spmd_packed,
    make_mesh,
)


def make_hash_vec(state_size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, int(1e15), size=state_size, dtype=np.int64)


def canonical_frames(k_sym: int, seed: int) -> list[int]:
    """Identity first, then the first K-1 of a seeded permutation of rotations 1..23.
    Prefix-nested; matches cube444/scripts/05_solve.py exactly."""
    order = [0] + [int(x) for x in np.random.default_rng(seed).permutation(np.arange(1, 24))]
    return order[: max(1, min(k_sym, 24))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b-global", type=int, default=1048576)
    ap.add_argument("--start-pid", type=int, default=0)
    ap.add_argument("--end-pid", type=int, default=1)
    ap.add_argument("--pids", type=str, default="",
                    help="explicit comma-separated pid list; overrides --start/--end-pid "
                         "(for targeted rescue of specific puzzles)")
    ap.add_argument("--num-steps", type=int, default=120)
    ap.add_argument("--alpha", type=int, default=2)
    ap.add_argument("--internal-bs", type=int, default=16384)
    ap.add_argument("--parent-chunk", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--tree-dir", type=str, default="/tmp/cube_trees")
    ap.add_argument("--out", type=str, default="/tmp/cube_out/results.json")
    ap.add_argument("--puzzle-info", type=str, default="")
    ap.add_argument("--test-csv", type=str, default="")
    ap.add_argument("--sym-dir", type=str, default="",
                    help="dir with rotations_24.npy / color_maps_24.npy / "
                         "move_relabel_inv_24.npy; defaults to CUBE_DATA")
    ap.add_argument("--v-checkpoint", type=str, default="c_bells2_epoch_0399.pt")
    ap.add_argument("--hidden-dims", type=str, default="2048,512")
    ap.add_argument("--num-res-blocks", type=int, default=2)
    ap.add_argument("--sym-ensemble", type=int, default=1)
    ap.add_argument("--sym-seed", type=int, default=0)
    ap.add_argument("--sym-positions", type=str, default="",
                    help="'lo:hi' slice of the canonical frame list for this shard")
    ap.add_argument("--stop-at-frame-solve", action="store_true",
                    help="stop trying further frames for a pid once one frame solves it")
    args = ap.parse_args()

    devices = jax.devices()
    n_dev = len(devices)
    assert args.b_global % n_dev == 0, f"b_global {args.b_global} not divisible by n_dev {n_dev}"
    b_local = args.b_global // n_dev
    k_per_peer = (args.alpha * b_local) // n_dev
    parent_chunk = args.parent_chunk if args.parent_chunk > 0 else None
    if parent_chunk is not None:
        assert b_local % parent_chunk == 0
    print(f"[cfg] devices={n_dev} kind={devices[0].device_kind} b_global={args.b_global:,} "
          f"b_local={b_local:,} k_per_peer={k_per_peer:,} steps={args.num_steps} "
          f"parent_chunk={parent_chunk}", flush=True)

    # --- puzzle ---
    puzzle_info = args.puzzle_info or f"{DATA}/puzzle_info.json"
    pinfo = json.load(open(puzzle_info, encoding="utf-8"))
    generators = pinfo["generators"]
    solved = tuple(pinfo["central_state"])
    move_names = list(generators.keys())
    n_gen = len(move_names)
    state_size = len(solved)
    all_moves = jnp.asarray(np.array([generators[n] for n in move_names], dtype=np.int32))
    v0 = jnp.asarray(np.array(solved, dtype=np.int8))
    central = np.array(solved, dtype=np.int64)
    G = {n: np.array(v, dtype=np.int64) for n, v in generators.items()}

    # --- sym tables ---
    sym_dir = args.sym_dir or DATA
    rot = np.load(f"{sym_dir}/rotations_24.npy")
    cmap = np.load(f"{sym_dir}/color_maps_24.npy")
    inv_relabel = np.load(f"{sym_dir}/move_relabel_inv_24.npy")
    frames = canonical_frames(args.sym_ensemble, args.sym_seed)
    if args.sym_positions:
        lo, hi = (int(x) for x in args.sym_positions.split(":"))
        frames = frames[lo:hi]
    print(f"[sym] K_SYM={args.sym_ensemble} seed={args.sym_seed} frames={frames}", flush=True)

    # --- scrambles ---
    test_csv = args.test_csv or f"{DATA}/test.csv"
    rows = list(csv.DictReader(open(test_csv, encoding="utf-8")))
    records = []
    for row_idx, row in enumerate(rows):
        rid = int(row.get("initial_state_id", row.get("id", row_idx)))
        s = [int(x) for x in row.get("initial_state", "").split(",") if x.strip()]
        if len(s) != state_size:
            raise ValueError(f"{test_csv} pid {rid}: parsed {len(s)} values, expected {state_size}")
        records.append((rid, s))
    if args.pids:
        want = {int(x) for x in args.pids.split(",") if x.strip()}
        selected = [(pid, s) for pid, s in records if pid in want]
        missing = want - {pid for pid, _ in selected}
        if missing:
            raise ValueError(f"--pids not found in test csv: {sorted(missing)[:5]}")
        print(f"[pids] explicit list: {len(selected)} pids", flush=True)
    else:
        selected = [(pid, s) for pid, s in records if args.start_pid <= pid < args.end_pid]
    if not selected:
        raise ValueError(f"no rows selected (start={args.start_pid} end={args.end_pid} "
                         f"pids={'set' if args.pids else 'none'})")

    # --- model / hashes / mesh ---
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x)
    print(f"[load] V ({args.v_checkpoint}, hidden={hidden_dims}, rb={args.num_res_blocks}) ...",
          flush=True)
    v_params = load_params_from_pt(
        f"{DATA}/{args.v_checkpoint}", hidden_dims=hidden_dims,
        num_res_blocks=args.num_res_blocks, state_size=state_size)
    print(f"[load]   V {num_params(v_params):,}  encoding={v_params['encoding']}", flush=True)
    hash_vec = jnp.asarray(make_hash_vec(state_size, 0))
    owner_hash_vec = jnp.asarray(np.random.default_rng(12345).integers(
        0, np.iinfo(np.uint32).max, size=state_size, dtype=np.uint32))
    mesh = make_mesh(devices)

    os.makedirs(args.tree_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    def verify(initial_state, path_idx):
        cur = np.array(initial_state, dtype=np.int64)
        for m in path_idx:
            cur = cur[G[move_names[m]]]
        return np.array_equal(cur, central)

    def run_beam(state, tag):
        t0 = time.time()
        r = beam_solve_v_only_spmd_packed(
            list(state), v_params, all_moves, v0, hash_vec, mesh,
            B_local=b_local, K_per_peer=k_per_peer, n_gen=n_gen, state_size=state_size,
            num_steps=args.num_steps, dtype=jnp.bfloat16, internal_bs=args.internal_bs,
            tree_path=f"{args.tree_dir}/tree_{tag}.u32", parent_chunk=parent_chunk,
            pack_v_score=True, progress_every=args.progress_every, owner_hash_vec=owner_hash_vec)
        return r, time.time() - t0

    results = []
    for pid, s0 in selected:
        s0 = np.array(s0, dtype=np.int64)
        best = None  # (len, path_idx, frame)
        for k in frames:
            s_k = cmap[k][s0[rot[k]]].tolist()  # sym(s, R_k) = pi_R[s[R]]
            r, wall = run_beam(s_k, f"pid{pid}_k{k}")
            if not r["found"]:
                print(f"[solve] pid={pid} frame={k} found=False wall={wall:.0f}s", flush=True)
                continue
            # translate the frame-k word back to the original frame
            path = [int(inv_relabel[k, m]) for m in r["path_idx"]]
            ok = verify(s0, path)
            print(f"[solve] pid={pid} frame={k} found=True len={len(path)} verify={ok} "
                  f"wall={wall:.0f}s", flush=True)
            if ok and (best is None or len(path) < best[0]):
                best = (len(path), path, k)
            if ok and args.stop_at_frame_solve:
                break

        if best is not None:
            best_len, best_path, best_k = best
            print(f"[pid {pid}] BEST len={best_len} via frame {best_k}", flush=True)
            results.append({"pid": pid, "found": True, "path_len": best_len,
                            "path_idx": best_path, "frame": int(best_k), "verify": True})
        else:
            print(f"[pid {pid}] NO verified solve", flush=True)
            results.append({"pid": pid, "found": False, "path_len": -1,
                            "path_idx": [], "verify": False})
        json.dump(results, open(args.out, "w", encoding="utf-8"), indent=0)

    n_ok = sum(1 for r in results if r["verify"])
    print(f"[done] {n_ok}/{len(results)} verified -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
