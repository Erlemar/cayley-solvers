"""Local sanity check for the single-device JAX beam search.

Runs `beam_solve_qshort` on a few pids at SMALL B (1024) on CPU. We use
m_curr_v3 as V/teacher and m_pi_v2 as a stand-in for both the Q-shortlister
(m23_v2, not available locally) and the policy term. Path quality is NOT
expected to be production-grade, but:

  - The algorithm must execute without exceptions / shape mismatches.
  - If `found=True`, applying the returned path to the initial state MUST
    reproduce SOLVED.
  - The min_V trajectory should monotonically improve (informational).

If both run cleanly with reproducible verification, the JIT'd beam loop is
plumbed correctly and we can move to the SPMD version with confidence.

Run via: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_spmd_jax/sanity_check_beam.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jax_beam import beam_solve_qshort, make_hash_vec  # noqa: E402
from jax_model import load_params_from_pt  # noqa: E402

ROOT = HERE.parents[1]
PUZZLE_INFO = json.loads((ROOT / "data" / "puzzle_info.json").read_text())
GENERATORS = PUZZLE_INFO["generators"]
MOVE_NAMES = list(GENERATORS.keys())
SOLVED = tuple(PUZZLE_INFO["central_state"])
N_GEN = len(MOVE_NAMES)
STATE_SIZE = len(SOLVED)


def verify_path(initial_state: list[int], path_idx: list[int]) -> bool:
    cur = list(initial_state)
    for m in path_idx:
        gen = GENERATORS[MOVE_NAMES[m]]
        cur = [cur[g] for g in gen]
    return tuple(cur) == SOLVED


def main():
    print(f"JAX {jax.__version__}  devices: {jax.devices()}")

    m_curr_v3_path = ROOT / "models" / "m_curr_v3" / "epoch_0499.pt"
    m_pi_v2_path = ROOT / "models" / "m_pi_v2" / "epoch_0199.pt"
    if not m_curr_v3_path.exists() or not m_pi_v2_path.exists():
        print("SKIP: required model checkpoints not found locally")
        return 1

    print("loading V/teacher (m_curr_v3) ...")
    teacher_params = load_params_from_pt(m_curr_v3_path, hidden_dims=(2048, 512))
    print("loading policy + Q stand-in (m_pi_v2) ...")
    policy_params = load_params_from_pt(m_pi_v2_path, hidden_dims=(2048, 512))
    # Use m_pi_v2 as both policy AND a stand-in for the qshort student (m23_v2
    # is in the Kaggle dataset, not local). The 24-output shape matches, but
    # the values won't be Q-distillation calibrated — so solve quality will be
    # poor. This test is for plumbing correctness, not solve rate.
    student_params = policy_params

    all_moves_np = np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32)
    all_moves = jnp.asarray(all_moves_np)
    V0 = jnp.asarray(SOLVED, dtype=jnp.int8)
    hash_vec_np = make_hash_vec(STATE_SIZE, seed=0)
    hash_vec = jnp.asarray(hash_vec_np)

    rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
    test_pids = [0, 1, 2, 3]

    # Tiny config: B=1024, alpha=2, 30 steps. CPU JAX float32.
    cfg = dict(
        B=1024, alpha=2, n_gen=N_GEN, state_size=STATE_SIZE,
        num_steps=30, internal_bs=1024, dtype=jnp.float32,
        use_chunked_v=False, use_chunked_q=False, lambda_policy=0.05,
    )
    print(f"config: {cfg}")

    n_found = 0
    n_verify_ok = 0
    n_runs = 0
    for pid in test_pids:
        s0 = [int(x) for x in rows[pid]["initial_state"].split(",")]
        t0 = time.time()
        try:
            r = beam_solve_qshort(
                s0, teacher_params, student_params, policy_params,
                all_moves, V0, hash_vec, **cfg,
            )
        except Exception as e:
            print(f"pid {pid}: EXCEPTION {type(e).__name__}: {e}")
            return 1
        wall = time.time() - t0
        n_runs += 1

        verify = False
        if r["found"]:
            verify = verify_path(s0, r["path_idx"])
            n_found += 1
            if verify:
                n_verify_ok += 1

        last_v = r["min_v_trajectory"][-1] if r["min_v_trajectory"] else None
        print(f"pid {pid:>3}  found={r['found']}  step={r['found_step']:>2}  "
              f"path_len={r['path_len']:>2}  verify={verify}  "
              f"wall={wall:6.1f}s  last_min_V={last_v}")

    print()
    print(f"runs: {n_runs}  found: {n_found}  verify_ok: {n_verify_ok}")
    print("Plumbing check: PASS"
          if n_runs == len(test_pids) and (n_found == 0 or n_verify_ok == n_found)
          else "Plumbing check: FAIL — some 'found' paths did not verify")

    return 0 if (n_runs == len(test_pids)
                 and (n_found == 0 or n_verify_ok == n_found)) else 1


if __name__ == "__main__":
    sys.exit(main())
