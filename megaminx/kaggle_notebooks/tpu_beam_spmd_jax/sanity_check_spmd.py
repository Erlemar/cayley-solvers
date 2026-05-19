"""SPMD sanity check: forces JAX to expose 8 logical CPU devices via
XLA_FLAGS=--xla_force_host_platform_device_count=8, then runs the
shard_map'd beam search at tiny scale (B_local=128).

The KEY check: any path returned with found=True MUST verify when applied to
the initial state. False positives would replicate the torch_xla SPMD bug.

Run via: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_spmd_jax/sanity_check_spmd.py
"""
from __future__ import annotations

import os
# IMPORTANT: must set BEFORE importing jax.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_ENABLE_X64", "true")  # int64 hash_vec needs x64 on CPU.

import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jax_beam import make_hash_vec  # noqa: E402
from jax_beam_spmd import beam_solve_qshort_spmd, make_mesh  # noqa: E402
from jax_model import load_params_from_pt  # noqa: E402

ROOT = HERE.parents[1]
PUZZLE_INFO = json.loads((ROOT / "data" / "puzzle_info.json").read_text())
GENERATORS = PUZZLE_INFO["generators"]
MOVE_NAMES = list(GENERATORS.keys())
SOLVED = tuple(PUZZLE_INFO["central_state"])
N_GEN = len(MOVE_NAMES)
STATE_SIZE = len(SOLVED)


def verify_path(initial_state, path_idx):
    cur = list(initial_state)
    for m in path_idx:
        gen = GENERATORS[MOVE_NAMES[m]]
        cur = [cur[g] for g in gen]
    return tuple(cur) == SOLVED


def main():
    print(f"JAX {jax.__version__}  devices: {jax.devices()}")
    devices = jax.devices()
    if len(devices) != 8:
        print(f"FAIL: expected 8 devices via XLA_FLAGS hack, got {len(devices)}")
        return 1

    mesh = make_mesh(devices)
    print(f"mesh: {mesh}")

    m_curr_v3_path = ROOT / "models" / "m_curr_v3" / "epoch_0499.pt"
    m_pi_v2_path = ROOT / "models" / "m_pi_v2" / "epoch_0199.pt"
    if not m_curr_v3_path.exists() or not m_pi_v2_path.exists():
        print("SKIP: required checkpoints not found locally")
        return 1
    print("loading V/teacher ...")
    teacher_params = load_params_from_pt(m_curr_v3_path, hidden_dims=(2048, 512))
    print("loading policy + Q stand-in ...")
    policy_params = load_params_from_pt(m_pi_v2_path, hidden_dims=(2048, 512))
    student_params = policy_params  # m_pi_v2 stand-in for m23_v2 (24-output shape match).

    all_moves_np = np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32)
    all_moves = jnp.asarray(all_moves_np)
    V0 = jnp.asarray(SOLVED, dtype=jnp.int8)
    hash_vec_np = make_hash_vec(STATE_SIZE, seed=0)
    hash_vec = jnp.asarray(hash_vec_np)  # int64

    rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
    test_pids = [0, 1, 2, 3]

    cfg = dict(
        B_local=128, K_per_peer=32,
        n_gen=N_GEN, state_size=STATE_SIZE,
        num_steps=20, lambda_policy=0.05,
        dtype=jnp.float32,  # CPU JAX: float32 (bf16 on CPU is buggy/slow).
        internal_bs=64,  # chunked forward, must divide B_local (128/64=2 chunks).
    )
    print(f"config: B_global={cfg['B_local']*8}, B_local={cfg['B_local']}, "
          f"K_per_peer={cfg['K_per_peer']}, num_steps={cfg['num_steps']}")

    n_runs = 0
    n_found = 0
    n_verify_ok = 0
    n_false_positive = 0
    for pid in test_pids:
        s0 = [int(x) for x in rows[pid]["initial_state"].split(",")]
        t0 = time.time()
        try:
            r = beam_solve_qshort_spmd(
                s0, teacher_params, student_params, policy_params,
                all_moves, V0, hash_vec, mesh,
                **cfg,
            )
        except Exception as e:
            import traceback
            print(f"pid {pid}: EXCEPTION {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        wall = time.time() - t0
        n_runs += 1
        verify = False
        if r["found"]:
            verify = verify_path(s0, r["path_idx"])
            n_found += 1
            if verify:
                n_verify_ok += 1
            else:
                n_false_positive += 1
        first = r.get("first_iter_s")
        print(f"pid {pid:>3}  found={r['found']}  step={r['found_step']:>2}  "
              f"path_len={r['path_len']:>2}  verify={verify}  "
              f"wall={wall:6.1f}s  first_iter={first}")

    print()
    print(f"runs: {n_runs}  found: {n_found}  verify_ok: {n_verify_ok}  "
          f"false_positives: {n_false_positive}")

    # SPMD-specific success criterion: NO false positives.
    if n_false_positive > 0:
        print("FAIL: SPMD produced false positives (path doesn't verify) - "
              "regression of the torch_xla SPMD bug.")
        return 1
    print("Plumbing check: PASS (no false positives)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
