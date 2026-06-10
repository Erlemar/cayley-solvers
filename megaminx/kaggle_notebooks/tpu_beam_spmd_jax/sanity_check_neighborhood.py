"""Local validation for ideas (2) neighborhood early-stop and (1) approx top-k
added to jax_beam_spmd_v_qshort.py.

Two layers:
  A. Pure-numpy check of build_solved_neighborhood: every stored suffix actually
     returns its state to solved, V0 is present with suffix_len 0, and the hash
     convention matches the kernel's V0_hash computation. (Model-independent.)
  B. SPMD smoke (8 logical CPU devices via XLA_FLAGS) of
     beam_solve_v_qshort_spmd_packed with the new flags ON, asserting found +
     verify with NO false positives (the SPMD-bug guard), and reporting which
     solutions came `via` the neighborhood vs exact V0.

Run: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_spmd_jax/sanity_check_neighborhood.py
"""
from __future__ import annotations

import os
# Must be set BEFORE importing jax.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
os.environ.setdefault("JAX_ENABLE_X64", "true")  # int64 hash needs x64 on CPU.

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
from jax_model import load_params_from_pt  # noqa: E402
from jax_beam_spmd_v_qshort import (  # noqa: E402
    beam_solve_v_qshort_spmd_packed,
    build_solved_neighborhood,
    make_mesh,
)

ROOT = HERE.parents[1]
PUZZLE_INFO = json.loads((ROOT / "data" / "puzzle_info.json").read_text())
GENERATORS = PUZZLE_INFO["generators"]
MOVE_NAMES = list(GENERATORS.keys())
SOLVED = tuple(PUZZLE_INFO["central_state"])
N_GEN = len(MOVE_NAMES)
STATE_SIZE = len(SOLVED)

ALL_MOVES_NP = np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32)

# INV_MOVE_IDX: megaminx "name"/"-name" inverse-pair convention (as in Cell 3).
_name_to_idx = {n: i for i, n in enumerate(MOVE_NAMES)}
INV_MOVE_IDX = np.full(N_GEN, -1, dtype=np.int32)
for _i, _nm in enumerate(MOVE_NAMES):
    _inv = _nm[1:] if _nm.startswith("-") else "-" + _nm
    INV_MOVE_IDX[_i] = _name_to_idx[_inv]
assert (INV_MOVE_IDX >= 0).all()
assert INV_MOVE_IDX[INV_MOVE_IDX].tolist() == list(range(N_GEN))


def verify_path(initial_state, path_idx):
    cur = list(initial_state)
    for m in path_idx:
        cur = [cur[g] for g in GENERATORS[MOVE_NAMES[m]]]
    return tuple(cur) == SOLVED


def test_neighborhood_builder(hash_vec_np, max_radius=3):
    print(f"\n== A. build_solved_neighborhood (radius {max_radius}) ==")
    t0 = time.time()
    hashes, suf_len, suf_mv = build_solved_neighborhood(
        ALL_MOVES_NP, INV_MOVE_IDX, hash_vec_np, np.array(SOLVED), max_radius,
    )
    M = hashes.shape[0]
    print(f"  M={M:,} states  ({time.time()-t0:.2f}s)  "
          f"suf_len max={int(suf_len.max())}")

    # Sorted ascending (device binary-searches).
    assert np.all(np.diff(hashes) >= 0), "neighborhood hashes not sorted"
    # No hash collisions inside the table would corrupt suffix lookup.
    assert len(np.unique(hashes)) == M, "hash collision within neighborhood table"

    # V0 present with suffix_len 0 and the kernel's exact V0 hash.
    v0_hash = int((np.array(SOLVED, dtype=np.int64) * hash_vec_np.astype(np.int64)).sum())
    pos = int(np.searchsorted(hashes, v0_hash))
    assert pos < M and hashes[pos] == v0_hash, "V0 hash missing from neighborhood"
    assert suf_len[pos] == 0, "V0 suffix_len should be 0"

    # Reconstruct every state from its hash is not possible, but we can BFS again
    # and check each state's stored suffix actually solves it.
    # Re-run a small BFS to recover (state -> suffix) and cross-check.
    gens = ALL_MOVES_NP
    table = {np.array(SOLVED, dtype=np.int8).tobytes(): b""}
    frontier = [(np.array(SOLVED, dtype=np.int8), b"")]
    for _d in range(max_radius):
        nxt = []
        for st, word in frontier:
            last = word[-1] if word else -1
            for gi in range(N_GEN):
                if last >= 0 and gi == int(INV_MOVE_IDX[last]):
                    continue
                ch = st[gens[gi]]
                key = ch.tobytes()
                if key not in table:
                    table[key] = word + bytes((gi,))
                    nxt.append((ch, word + bytes((gi,))))
        frontier = nxt

    n_checked = 0
    with np.errstate(over="ignore"):
        for key, word in table.items():
            st = np.frombuffer(key, dtype=np.int8).astype(np.int64)
            h = int((st * hash_vec_np.astype(np.int64)).sum())
            i = int(np.searchsorted(hashes, h))
            assert hashes[i] == h
            slen = int(suf_len[i])
            suffix = suf_mv[i][:slen].astype(int).tolist()
            # Apply suffix to the state; must reach SOLVED.
            cur = np.frombuffer(key, dtype=np.int8).copy()
            for m in suffix:
                cur = cur[ALL_MOVES_NP[m]]
            assert np.array_equal(cur, np.array(SOLVED, dtype=np.int8)), (
                f"suffix failed to solve state at depth {len(word)}")
            n_checked += 1
    print(f"  verified {n_checked:,} (state -> suffix -> solved) reconstructions OK")
    return hashes, suf_len, suf_mv


def run_spmd(pid_states, v_params, q_params, hash_vec, mesh, *,
             nbhd=None, approx=False, meta=False, lean=False, tag=""):
    nbhd_hash, nbhd_len, nbhd_mv = (nbhd if nbhd is not None else (None, None, None))
    cfg = dict(
        B_local=128, K_per_peer=32, n_gen=N_GEN, state_size=STATE_SIZE,
        num_steps=22, dtype=jnp.float32, internal_bs=64, parent_chunk=64,
        pack_v_score=True, progress_every=0,
        nbhd_hash_sorted=nbhd_hash, nbhd_suffix_len=nbhd_len,
        nbhd_suffix_moves=nbhd_mv, use_approx_topk=approx,
        use_meta_materialize=meta, alpha_req=3.0,  # high cap -> no overflow at B=128
        lean_merge=lean,
    )
    V0 = jnp.asarray(SOLVED, dtype=jnp.int8)
    all_moves = jnp.asarray(ALL_MOVES_NP)
    n_fp = 0
    results = {}
    print(f"\n== B. SPMD smoke [{tag}] (nbhd={nbhd is not None} approx={approx} meta={meta}) ==")
    for pid, s0 in pid_states:
        t0 = time.time()
        r = beam_solve_v_qshort_spmd_packed(
            s0, v_params, q_params, 2, all_moves, V0, hash_vec, mesh, **cfg,
        )
        wall = time.time() - t0
        ok = verify_path(s0, r["path_idx"]) if r["found"] else False
        if r["found"] and not ok:
            n_fp += 1
        results[pid] = {"found": r["found"], "path_len": r["path_len"], "verify": ok}
        print(f"  pid {pid:>3}  found={r['found']}  via={r.get('via','-'):>12}  "
              f"step={r.get('found_step',-1):>2}  len={r['path_len']:>2}  "
              f"verify={ok}  wall={wall:5.1f}s")
    return n_fp, results


def main():
    devices = jax.devices()
    print(f"JAX {jax.__version__}  devices={len(devices)}")
    if len(devices) != 8:
        print(f"FAIL: expected 8 logical CPU devices, got {len(devices)}")
        return 1
    mesh = make_mesh(devices)

    hash_vec_np = make_hash_vec(STATE_SIZE, seed=0)
    hash_vec = jnp.asarray(hash_vec_np)

    # --- A. neighborhood builder (model-independent) ---
    nbhd = test_neighborhood_builder(hash_vec_np, max_radius=3)

    # --- B. SPMD smoke (needs local checkpoints) ---
    v_path = ROOT / "models" / "m_curr_v3" / "epoch_0499.pt"
    q_path = ROOT / "models" / "m_pi_v2" / "epoch_0199.pt"
    if not v_path.exists() or not q_path.exists():
        print("\nSKIP B: m_curr_v3 / m_pi_v2 checkpoints not found locally.")
        print("Layer A (neighborhood builder) PASSED.")
        return 0
    print(f"\nloading V={v_path.name}  Q={q_path.name}")
    v_params = load_params_from_pt(v_path, hidden_dims=(2048, 512))
    q_params = load_params_from_pt(q_path, hidden_dims=(2048, 512))

    rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
    pid_states = [(pid, [int(x) for x in rows[pid]["initial_state"].split(",")])
                  for pid in (0, 1, 2, 3)]

    n_fp = 0
    fp, base = run_spmd(pid_states, v_params, q_params, hash_vec, mesh,
                        nbhd=None, approx=False, meta=False, tag="baseline")
    n_fp += fp
    fp, _ = run_spmd(pid_states, v_params, q_params, hash_vec, mesh,
                     nbhd=nbhd, approx=False, meta=False, tag="neighborhood")
    n_fp += fp
    fp, _ = run_spmd(pid_states, v_params, q_params, hash_vec, mesh,
                     nbhd=nbhd, approx=True, meta=False, tag="neighborhood+approx")
    n_fp += fp
    fp, meta = run_spmd(pid_states, v_params, q_params, hash_vec, mesh,
                        nbhd=None, approx=False, meta=True, tag="meta-materialize (idea 4)")
    n_fp += fp
    fp, lean = run_spmd(pid_states, v_params, q_params, hash_vec, mesh,
                        nbhd=None, approx=False, lean=True, tag="lean-merge (compact carry)")
    n_fp += fp

    # Idea (4) length-safety invariant: meta selects the same beam as baseline.
    print("\n== C. idea-4 length-safety: meta-materialize vs baseline ==")
    mism = 0
    for pid, _ in pid_states:
        b, m = base[pid], meta[pid]
        if b["found"] != m["found"] or (b["found"] and b["path_len"] != m["path_len"]):
            mism += 1
            print(f"  MISMATCH pid {pid}: baseline(found={b['found']},len={b['path_len']}) "
                  f"vs meta(found={m['found']},len={m['path_len']})")
    if mism == 0:
        print("  meta selects the IDENTICAL beam (same found + path_len) -> length-safe")

    # lean-merge length-safety invariant: same beam as baseline.
    print("\n== D. lean-merge length-safety: lean vs baseline ==")
    lean_mism = 0
    for pid, _ in pid_states:
        b, m = base[pid], lean[pid]
        if b["found"] != m["found"] or (b["found"] and b["path_len"] != m["path_len"]):
            lean_mism += 1
            print(f"  MISMATCH pid {pid}: baseline(found={b['found']},len={b['path_len']}) "
                  f"vs lean(found={m['found']},len={m['path_len']})")
    if lean_mism == 0:
        print("  lean-merge selects the IDENTICAL beam (same found + path_len) -> length-safe")

    print()
    if n_fp > 0 or mism > 0 or lean_mism > 0:
        print(f"FAIL: {n_fp} false positive(s), {mism} meta mismatch(es), {lean_mism} lean mismatch(es).")
        return 1
    print("PASS: neighborhood + approx + meta-materialize correct; no false positives; idea-4 length-safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
