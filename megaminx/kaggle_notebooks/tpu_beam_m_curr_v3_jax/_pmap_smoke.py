"""Local validation: wrap jax_beam.beam_solve_qshort_jit in jax.pmap and run on
8 CPU virtual devices. Pure plumbing check — confirms the pmap composition
with the closure-captured static args works before pushing to Kaggle.
"""
from __future__ import annotations

import os
# Must be set BEFORE jax import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
SPMD_DIR = HERE.parent / "tpu_beam_spmd_jax"
sys.path.insert(0, str(SPMD_DIR))

from jax_beam import beam_solve_qshort_jit, make_hash_vec  # noqa: E402
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


def reconstruct_path(tree_idx, tree_move, fs, fp):
    if fs < 0:
        return []
    path = []
    pos = int(fp)
    for j in range(fs, -1, -1):
        m = int(tree_move[j, pos])
        path.append(m)
        pos = int(tree_idx[j, pos])
    path.reverse()
    while path and path[0] < 0:
        path.pop(0)
    return path


def main():
    print(f"JAX {jax.__version__}, devices = {jax.devices()}")
    assert len(jax.devices()) == 8, "expected 8 emulated CPU devices"

    m_curr = ROOT / "models" / "m_curr_v3" / "epoch_0499.pt"
    m_pi = ROOT / "models" / "m_pi_v2" / "epoch_0199.pt"
    if not m_curr.exists() or not m_pi.exists():
        print("SKIP: required checkpoints not found locally")
        return 1

    print("loading params ...")
    teacher_params = load_params_from_pt(m_curr, hidden_dims=(2048, 512))
    policy_params = load_params_from_pt(m_pi, hidden_dims=(2048, 512))
    student_params = policy_params  # local stand-in for m23_v2

    all_moves = jnp.asarray(np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32))
    V0 = jnp.asarray(SOLVED, dtype=jnp.int8)
    hash_vec = jnp.asarray(make_hash_vec(STATE_SIZE, seed=0))

    # Static config — baked into the pmapped closure via positional bind.
    B = 1024            # tiny per-device beam for the local smoke
    NUM_STEPS = 20
    INTERNAL_BS = 256   # B / 4
    ALPHA = 2
    LAMBDA_POLICY = 0.05

    # in_axes: init_state has leading device axis (sharded across 8 cores);
    # everything else is replicated (broadcast to all devices). Without this,
    # pmap defaults to sharding ALL inputs along axis 0 — which fails for
    # params dicts whose leaf arrays don't have an 8-divisible leading axis.
    @partial(jax.pmap, axis_name="cores",
             in_axes=(0, None, None, None, None, None, None))
    def pmap_beam(init_state, t_params, s_params, p_params,
                  all_moves_p, V0_p, hash_vec_p):
        return beam_solve_qshort_jit(
            init_state, t_params, s_params, p_params,
            all_moves_p, V0_p, hash_vec_p, jnp.float32(LAMBDA_POLICY),
            B=B, alpha=ALPHA, n_gen=N_GEN, state_size=STATE_SIZE,
            num_steps=NUM_STEPS, internal_bs=INTERNAL_BS,
            dtype=jnp.float32, use_chunked_v=True, use_chunked_q=True,
        )

    rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
    pids = list(range(8))
    init_states_np = np.stack(
        [np.array([int(x) for x in rows[p]["initial_state"].split(",")], dtype=np.int8) for p in pids],
        axis=0,
    )
    init_states = jnp.asarray(init_states_np)
    print(f"init_states.shape = {init_states.shape}")

    t0 = time.time()
    states_f, tree_idx, tree_move, mv_log, found_step, found_pos = pmap_beam(
        init_states, teacher_params, student_params, policy_params,
        all_moves, V0, hash_vec,
    )
    jax.block_until_ready(found_step)
    wall = time.time() - t0
    print(f"pmap_beam ran in {wall:.1f}s (includes first-call compile)")

    fs_h = np.asarray(found_step)        # (8,)
    fp_h = np.asarray(found_pos)         # (8,)
    tree_idx_h = np.asarray(tree_idx)    # (8, NUM_STEPS, B)
    tree_move_h = np.asarray(tree_move)  # (8, NUM_STEPS, B)

    n_found = 0
    n_verify = 0
    for i, pid in enumerate(pids):
        fs = int(fs_h[i])
        fp = int(fp_h[i])
        path = reconstruct_path(tree_idx_h[i], tree_move_h[i], fs, fp)
        init = init_states_np[i].tolist()
        verify = verify_path(init, path) if path else False
        if fs >= 0:
            n_found += 1
            if verify:
                n_verify += 1
        print(f"  pid {pid}: fs={fs:>3} path_len={len(path):>3} verify={verify}")

    print()
    print(f"runs: 8  found: {n_found}  verify_ok: {n_verify}  false_pos: {n_found - n_verify}")
    return 0 if n_found == 0 or n_verify == n_found else 1


if __name__ == "__main__":
    sys.exit(main())
