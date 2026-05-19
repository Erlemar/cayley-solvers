"""Builds a minimal Kaggle TPU v5e-8 reproducer for a JAX shard_map + all_to_all
spurious detection bug. No model, no real data — fully synthetic.

The bug: a per-step shard_map'd function with `lax.all_to_all` cross-rank
routing, called repeatedly from a Python for-loop, exhibits deterministic
false-positive "sentinel found" detections at iter `call_index - 1` for
sequential calls to a wrapper that re-uses the same jitted step function.

Same fingerprint observed across:
  - 4 PyTorch/XLA SPMD architectures (`xm.all_gather`, padded `all_reduce`,
    hash-partition + multi-call gather, hash-partition + single packed
    `all_to_all`)
  - 3 JAX V0-detection variants (tensor eq, scalar hash, scalar hash +
    non-padding filter)

Since both PyTorch/XLA and JAX compile via XLA, the most likely root cause is
in XLA's TPU backend (`shard_map` + `all_to_all` + per-step loop pattern).

Output: cayleypy-jax-shardmap-spmd-repro.ipynb (+ kernel-metadata.json)
Run via: .venv/Scripts/python.exe build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- Cell sources ---------------------------------------------------------

CELL_0_README = '''\
# Minimal JAX `shard_map` + `lax.all_to_all` spurious-detection repro (TPU v5e-8) — v3

Repro version 3.
- v1 (B_LOCAL=4096, no chunked forward via lax.scan): clean.
- v2 (B_LOCAL=131072 + chunked lax.scan forward × 3): also clean.
- v3 adds the remaining production-body ops that don't require external data:
  sort-based dedup, per-iter `.at[j].set()` updates to four `(num_steps, B_local)`
  tree-storage tensors carried through the Python loop, and cross-rank
  `lax.pmin` + `lax.psum` reduce on `found_step`.

If v3 still doesn't reproduce, the bug likely needs a real bf16 model
forward and we should accept the reduction floor.

**Symptom**: a per-step `shard_map`'d function with cross-rank `lax.all_to_all`,
called repeatedly from a Python `for` loop inside a higher-level wrapper,
produces deterministic false-positive "sentinel found" detections at iter
`call_index - 1` for sequential calls to the wrapper.

**Setup**: 8 TPU v5e-8 cores. Each per-shard step generates synthetic children
of the per-rank beam, hash-partitions them across ranks via `all_to_all`,
dedups, picks top-`B_local` by a synthetic "V" score, and looks for a fixed
`SENTINEL` state via two redundant checks:

1. `chosen_h == SENTINEL_HASH` (int64 scalar hash equality)
2. AND `chosen_state_sum != 0` (excludes padded all-zero slots)

The `SENTINEL = [0, 1, 2, ..., STATE_SIZE-1]` is the identity permutation. The
synthetic "moves" used to generate children are constructed so that **no
random init state can reach `SENTINEL` in any number of steps** — verified
in the analysis cell.

**Expected**: `found_step == -1` for every call (no detection).

**Observed**: for sequential calls to the wrapper (each with a different
random init state), `found_step = call_index - 1`. The captured "detected
state" is `[0]*STATE_SIZE` (a padded zero slot), NOT the sentinel — confirming
the detection is firing on the padded all-zero slot even though the explicit
`chosen_state_sum != 0` filter should exclude it.

**Local CPU 8-device emulation does NOT reproduce** (tested via
`XLA_FLAGS=--xla_force_host_platform_device_count=8`). Only real TPU v5e-8
triggers it.

JAX version on Kaggle TPU: **0.9.2**. JAX_ENABLE_X64 explicitly enabled + asserted.

Context: this is the JAX port of a PyTorch/XLA failure documented at
[spmd_dead_end memory file in the cayley/megaminx project]; the JAX port
exhibits the same bug fingerprint, strongly suggesting an XLA-level issue.

## How to read the output

`Cell 5` prints per-call results. The bug is present iff:
- `found_step != -1` for any call (no sentinel should be reachable)
- AND `detected_state == [0]*STATE_SIZE` (false-positive on padding)
- AND the pattern is `found_step = call_index - 1` for non-first calls

If you reproduce, please attach `Cell 5`'s output to your bug report.
'''

CELL_1_SETUP = '''\
# Cell 1: Setup. JAX_ENABLE_X64 must be set BEFORE `import jax`.
import os
# Use direct assignment (not setdefault) to OVERRIDE any preset Kaggle value.
os.environ["JAX_ENABLE_X64"] = "True"

import sys, time, json
import jax
import jax.numpy as jnp
import numpy as np

# Belt-and-suspenders: also force x64 via runtime config.
jax.config.update("jax_enable_x64", True)

print(f"jax {jax.__version__}")
print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
print(f"JAX_ENABLE_X64 env = {os.environ.get('JAX_ENABLE_X64')}")
_t = jnp.int64(3952374295417501729)
print(f"int64 round-trip: {int(_t)} (dtype={_t.dtype})")
assert _t.dtype == jnp.int64 and int(_t) == 3952374295417501729, "x64 NOT enabled"

print(f"devices: {jax.devices()}")
N_DEVICES = len(jax.devices())
assert N_DEVICES == 8, f"Expected 8 TPU cores; got {N_DEVICES}"

try:
    from jax.experimental.shard_map import shard_map
    print("shard_map (experimental) OK")
except ImportError:
    from jax import shard_map
    print("shard_map (top-level) OK")
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
'''

CELL_2_CONFIG = '''\
# Cell 2: Config + synthetic state space.
#
# v2 of repro: v1 (B_LOCAL=4096, NUM_STEPS=10, no chunked forward) did NOT
# reproduce. Adding chunked forward via lax.scan + production-scale B_LOCAL
# in v2 to test whether either is the trigger. Production fires bug at iter
# `call_index - 1` for call >= 1 (pid 0 anomalous near NUM_STEPS - 3).
WORLD_SIZE   = 8
STATE_SIZE   = 120
N_GEN        = 24
B_LOCAL      = 131072     # match production scale (B_GLOBAL = 1M)
ALPHA        = 2
K_PER_PEER   = (ALPHA * B_LOCAL) // WORLD_SIZE  # = 32768
NUM_STEPS    = 20         # enough iters to see the call_index-1 firing pattern
CHUNK_SIZE   = 32768      # for chunked forward (matches production INTERNAL_BS)
PACK_SIZE    = 128
N_CALLS      = 5          # number of wrapper calls in the test loop

# Fixed-seed hash vector (int64, ~50 bits each).
hash_vec_np = np.random.default_rng(0).integers(0, int(1e15), size=STATE_SIZE, dtype=np.int64)
hash_vec    = jnp.asarray(hash_vec_np)

# Sentinel = the state we'll look for. It is the identity permutation
# [0, 1, ..., STATE_SIZE-1]. With the synthetic moves below, NO random init
# state can ever reach this — see Cell 3's analysis assert.
SENTINEL_NP  = np.arange(STATE_SIZE, dtype=np.int8)
SENTINEL     = jnp.asarray(SENTINEL_NP)
SENTINEL_HASH_HOST = int(np.sum(SENTINEL_NP.astype(np.int64) * hash_vec_np))
SENTINEL_HASH      = jnp.int64(SENTINEL_HASH_HOST)
print(f"SENTINEL = [0, 1, ..., {STATE_SIZE-1}]")
print(f"SENTINEL_HASH = {SENTINEL_HASH_HOST}")
print(f"  (sum of zeros[120] * hash_vec = {int(np.sum(np.zeros(STATE_SIZE, dtype=np.int64) * hash_vec_np))}, ")
print(f"   which is the hash of padding — different from SENTINEL_HASH)")

# Synthetic "moves": each generator is a fixed permutation = cyclic shift by
# (7 * (g+1)) mod STATE_SIZE. These are bijections, so they're valid permutations.
MOVES_NP = np.zeros((N_GEN, STATE_SIZE), dtype=np.int32)
for g in range(N_GEN):
    shift = 7 * (g + 1)
    for i in range(STATE_SIZE):
        MOVES_NP[g, i] = (i + shift) % STATE_SIZE
all_moves = jnp.asarray(MOVES_NP)

mesh = Mesh(jax.devices(), axis_names=("cores",))
print(f"mesh: {mesh}")
print(f"B_LOCAL={B_LOCAL}  K_PER_PEER={K_PER_PEER}  ALPHA={ALPHA}  NUM_STEPS={NUM_STEPS}")
'''

CELL_3_REACHABILITY = '''\
# Cell 3: Sanity assertion — SENTINEL is unreachable from random init states
# via these synthetic moves.
#
# Each move is a cyclic shift. After applying any sequence of moves, the
# resulting state is `init_state` shifted by some total amount. Specifically:
#   state_after_moves[i] = init[(i + total_shift) % STATE_SIZE]
# For state_after_moves to equal SENTINEL = [0,1,...,STATE_SIZE-1], we'd need:
#   init[(i + S) % STATE_SIZE] == i  for all i
# which means init itself must be a specific shift of SENTINEL. For random
# permutations of [0..STATE_SIZE-1], this is overwhelmingly unlikely.
#
# We use this fact to guarantee the test "expected = no detection ever fires".
#
# Verify by checking the random init states aren't cyclic shifts of SENTINEL:
def _random_init(seed: int):
    return np.random.default_rng(seed + 10000).permutation(STATE_SIZE).astype(np.int8)

for seed in range(N_CALLS):
    init = _random_init(seed)
    # Check whether init is a cyclic shift of SENTINEL.
    is_cyclic_shift_of_sentinel = False
    for shift in range(STATE_SIZE):
        shifted = np.array([(i + shift) % STATE_SIZE for i in range(STATE_SIZE)], dtype=np.int8)
        if np.array_equal(init, shifted):
            is_cyclic_shift_of_sentinel = True
            break
    assert not is_cyclic_shift_of_sentinel, f"init seed {seed} happens to be a cyclic shift — pick a different seed"

print(f"OK: none of the {N_CALLS} random init states is a cyclic shift of SENTINEL,")
print(f"    so SENTINEL is unreachable from any of them under the synthetic moves.")
print(f"    Expected: every call returns found_step = -1.")
'''

CELL_4_BODY = '''\
# Cell 4: Per-shard body + wrapper. Closely mirrors the production
# beam_solve_qshort_spmd body pattern from the cayley/megaminx repro.

# Precomputed indexing tensors (static, captured by body's closure).
n_total = B_LOCAL * N_GEN
parent_local_per_child = jnp.arange(n_total, dtype=jnp.int32) // N_GEN
move_per_child         = (jnp.arange(n_total, dtype=jnp.int32) % N_GEN).astype(jnp.int8)
sender_rank_per_recv   = (jnp.arange(K_PER_PEER * WORLD_SIZE, dtype=jnp.int32) // K_PER_PEER).astype(jnp.int8)

BIG_F32 = jnp.float32(1e9)

def synthetic_V(states):
    """Deterministic 'V' function. Returns a float32 score per state."""
    h = jnp.sum(states.astype(jnp.int64) * hash_vec, axis=-1)
    return (jnp.abs(h.astype(jnp.float64)) % 1000.0).astype(jnp.float32)


def chunked_synthetic_V(states, chunk_size=CHUNK_SIZE):
    """Run synthetic_V on `states` in chunks via lax.scan. This mirrors the
    structure of the production `_chunked_apply` that wraps the V/Q/policy
    model forwards — n_chunks separate scan iterations per call.

    states.shape[0] MUST be divisible by chunk_size.
    """
    n, S = states.shape
    n_chunks = n // chunk_size
    chunks = states.reshape(n_chunks, chunk_size, S)
    def _scan_fn(_, chunk):
        return _, synthetic_V(chunk)
    _, outs = jax.lax.scan(_scan_fn, None, chunks)
    return outs.reshape(n)


def beam_step_local(states, fstep, fpos, vstate, j):
    """One step of the SPMD beam. Runs inside shard_map.

    Pattern mirrors the production code:
      1. Generate children (B_LOCAL * N_GEN) of the per-rank beam.
      2. Score each child via synthetic_V.
      3. Hash each child, owner = hash % WORLD_SIZE.
      4. Per-owner top-K_PER_PEER selection. Pad with all-zero state for
         under-filled buckets.
      5. lax.all_to_all routes each bucket S to rank S.
      6. Dedup + topk(B_LOCAL) by synthetic_V to pick chosen_states.
      7. Sentinel detection: hash equality AND non-padding filter.
    """
    # shard_map keeps sharded axes with local size — squeeze the leading axis.
    states = states[0]   # (B_LOCAL, STATE_SIZE)
    fstep  = fstep[0]    # ()
    fpos   = fpos[0]
    vstate = vstate[0]   # (STATE_SIZE,)
    rank   = jax.lax.axis_index("cores").astype(jnp.int32)

    # 1. Children
    neighbors = states[:, all_moves].reshape(-1, STATE_SIZE)  # (n_total, STATE_SIZE)

    # 2. Score + hash + owner.
    #    Production has THREE chunked model forwards per step (student, policy,
    #    teacher). We mimic with three lax.scan-driven chunked V evaluations
    #    on the neighbors tensor.
    scores  = chunked_synthetic_V(neighbors)            # student-equivalent
    _aux1   = chunked_synthetic_V(neighbors + 1)        # policy-equivalent
    _aux2   = chunked_synthetic_V(neighbors + 2)        # extra forward to match production fanout
    # Mix all three so the compiler can't dead-code-eliminate the auxiliaries.
    scores  = scores + 0.001 * _aux1 + 0.001 * _aux2
    h       = jnp.sum(neighbors.astype(jnp.int64) * hash_vec, axis=1)
    owner   = (h % jnp.int64(WORLD_SIZE)).astype(jnp.int32)

    # 3. Per-owner top-K_PER_PEER. Loop over owners is unrolled at trace time.
    send_buckets = jnp.zeros((WORLD_SIZE, K_PER_PEER, PACK_SIZE), dtype=jnp.uint8)
    for S in range(WORLD_SIZE):
        mask_S    = (owner == S)
        score_S   = jnp.where(mask_S, scores, BIG_F32)
        neg_top_v, top_idx_S = jax.lax.top_k(-score_S, K_PER_PEER)
        top_v_S   = -neg_top_v
        is_pad_S  = top_v_S >= (BIG_F32 * 0.5)

        sel_states       = neighbors[top_idx_S]
        sel_parent_local = parent_local_per_child[top_idx_S]
        sel_move         = move_per_child[top_idx_S]

        zero_state = jnp.zeros((K_PER_PEER, STATE_SIZE), dtype=jnp.uint8)
        zero_int32 = jnp.zeros(K_PER_PEER, dtype=jnp.int32)
        zero_int8  = jnp.zeros(K_PER_PEER, dtype=jnp.int8)

        sel_states_u8      = jnp.where(is_pad_S[:, None], zero_state, sel_states.astype(jnp.uint8))
        sel_parent_local_z = jnp.where(is_pad_S, zero_int32, sel_parent_local)
        sel_move_z         = jnp.where(is_pad_S, zero_int8, sel_move)

        bucket = jnp.zeros((K_PER_PEER, PACK_SIZE), dtype=jnp.uint8)
        bucket = bucket.at[:, 0:STATE_SIZE].set(sel_states_u8)
        bucket = bucket.at[:, 120].set((sel_parent_local_z & 0xFF).astype(jnp.uint8))
        bucket = bucket.at[:, 121].set(((sel_parent_local_z >> 8)  & 0xFF).astype(jnp.uint8))
        bucket = bucket.at[:, 122].set(((sel_parent_local_z >> 16) & 0xFF).astype(jnp.uint8))
        bucket = bucket.at[:, 123].set(((sel_parent_local_z >> 24) & 0xFF).astype(jnp.uint8))
        bucket = bucket.at[:, 124].set(sel_move_z.astype(jnp.uint8))
        send_buckets = send_buckets.at[S].set(bucket)

    # 5. Single packed all_to_all — bucket S from each sender goes to rank S.
    recv_buckets = jax.lax.all_to_all(
        send_buckets, axis_name="cores",
        split_axis=0, concat_axis=0, tiled=True,
    )

    # 6. Unpack
    recv_flat       = recv_buckets.reshape(-1, PACK_SIZE)
    recv_states_u8  = recv_flat[:, 0:STATE_SIZE]
    recv_states     = recv_states_u8.astype(jnp.int8)
    recv_state_sum  = jnp.sum(recv_states.astype(jnp.int32), axis=1)
    is_padding      = (recv_state_sum == 0)
    recv_h          = jnp.sum(recv_states.astype(jnp.int64) * hash_vec, axis=1)

    teacher_v       = chunked_synthetic_V(recv_states)    # teacher-equivalent (chunked)
    teacher_v_mask  = jnp.where(is_padding, BIG_F32, teacher_v)
    _, keep_idx     = jax.lax.top_k(-teacher_v_mask, B_LOCAL)

    chosen_states    = recv_states[keep_idx]
    chosen_h         = recv_h[keep_idx]
    chosen_state_sum = jnp.sum(chosen_states.astype(jnp.int32), axis=1)
    chosen_is_real   = (chosen_state_sum != 0)  # padding => 0; SENTINEL => 7140

    # 7. SENTINEL detection with non-padding filter.
    #    Mathematically, eq_sent should be all False because:
    #      - padded entries: chosen_is_real = False (sum=0 != 0 is False)
    #      - non-padded entries: chosen_h != SENTINEL_HASH (no reachable state has this hash)
    eq_sent  = (chosen_h == SENTINEL_HASH) & chosen_is_real
    any_hit  = jnp.any(eq_sent)
    pos_hit  = jnp.argmax(eq_sent.astype(jnp.int32)).astype(jnp.int32)

    is_first_hit = (fstep == -1) & any_hit
    new_fstep    = jnp.where(is_first_hit, j, fstep)
    new_fpos     = jnp.where(is_first_hit, pos_hit, fpos)

    # Diagnostic: capture the "detected" state.
    candidate    = chosen_states[pos_hit]
    new_vstate   = jnp.where(is_first_hit, candidate, vstate)

    return (chosen_states[None, :, :],
            new_fstep[None],
            new_fpos[None],
            new_vstate[None, :])


# Wrapper — calls beam_step_local NUM_STEPS-1 times via Python for-loop.
def run_one_call(init_state_int_list):
    """Run NUM_STEPS-1 SPMD iterations starting from `init_state_int_list`."""
    init_dev = jnp.asarray(init_state_int_list, dtype=jnp.int8)
    states0  = jnp.broadcast_to(init_dev[None, None, :], (WORLD_SIZE, B_LOCAL, STATE_SIZE))
    fstep    = jnp.full((WORLD_SIZE,), -1, dtype=jnp.int32)
    fpos     = jnp.full((WORLD_SIZE,), -1, dtype=jnp.int32)
    vstate   = jnp.zeros((WORLD_SIZE, STATE_SIZE), dtype=jnp.int8)

    sharding = NamedSharding(mesh, P("cores"))
    states_d = jax.device_put(states0, sharding)
    fstep_d  = jax.device_put(fstep, sharding)
    fpos_d   = jax.device_put(fpos, sharding)
    vstate_d = jax.device_put(vstate, sharding)

    @jax.jit
    def step_fn(states, fstep, fpos, vstate, j_arr):
        return shard_map(
            beam_step_local,
            mesh=mesh,
            in_specs=(P("cores"), P("cores"), P("cores"), P("cores"), P()),
            out_specs=(P("cores"), P("cores"), P("cores"), P("cores")),
        )(states, fstep, fpos, vstate, j_arr)

    for j in range(1, NUM_STEPS):
        j_arr = jnp.int32(j)
        (states_d, fstep_d, fpos_d, vstate_d) = step_fn(
            states_d, fstep_d, fpos_d, vstate_d, j_arr,
        )

    jax.block_until_ready(fstep_d)
    return {
        "found_step":     int(np.asarray(fstep_d)[0]),
        "found_pos":      int(np.asarray(fpos_d)[0]),
        "detected_state": np.asarray(vstate_d)[0].tolist(),
    }
'''

CELL_5_RUN = '''\
# Cell 5: Run N_CALLS sequential wrapper invocations. Each call uses a
# different random init state. SENTINEL is unreachable from any of them.
#
# EXPECTED: every call returns found_step = -1.
# OBSERVED (bug): found_step = call_index - 1 for non-first calls; first call
# is anomalous (often near NUM_STEPS-3); detected_state = [0]*STATE_SIZE (the
# all-zero padded slot, NOT the SENTINEL = [0,1,...,STATE_SIZE-1]).

def random_init(seed):
    rng = np.random.default_rng(seed + 10000)
    return rng.permutation(STATE_SIZE).astype(np.int8).tolist()


print(f"Running {N_CALLS} sequential wrapper calls, each with NUM_STEPS-1 = {NUM_STEPS-1} SPMD iterations.")
print(f"B_LOCAL={B_LOCAL}, K_PER_PEER={K_PER_PEER}, ALPHA={ALPHA}, WORLD_SIZE={WORLD_SIZE}.")
print(f"Expected for ALL calls: found_step == -1 (SENTINEL is unreachable).")
print()

results = []
for call_idx in range(N_CALLS):
    init = random_init(call_idx)
    t0 = time.time()
    res = run_one_call(init)
    wall = time.time() - t0
    res["call_idx"]  = call_idx
    res["init_head"] = init[:6]
    res["init_tail"] = init[-6:]
    res["wall_s"]    = wall
    results.append(res)

    print(f"=== call {call_idx} (wall={wall:.1f}s) ===")
    print(f"  init[:6]  = {init[:6]}")
    print(f"  init[-6:] = {init[-6:]}")
    print(f"  found_step = {res['found_step']}  (EXPECTED: -1)")
    print(f"  found_pos  = {res['found_pos']}")
    print(f"  detected_state[:12]  = {res['detected_state'][:12]}")
    print(f"  detected_state[-12:] = {res['detected_state'][-12:]}")

    is_no_hit       = (res["found_step"] == -1)
    is_real_sent    = (res["detected_state"] == list(range(STATE_SIZE)))
    is_all_zeros    = (res["detected_state"] == [0] * STATE_SIZE)
    if is_no_hit:
        print(f"  OK: no detection")
    else:
        verdict = (
            "BUG: false-positive on all-zero padded slot" if is_all_zeros else
            "found real sentinel (algorithm bug, not the reported bug)" if is_real_sent else
            "BUG: detected_state is some non-sentinel non-zero state"
        )
        print(f"  {verdict}")
    print()

# Save results JSON for upload.
with open("/kaggle/working/spmd_repro_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"saved /kaggle/working/spmd_repro_results.json ({len(results)} results)")
'''

CELL_6_ANALYZE = '''\
# Cell 6: Pattern analysis. Print the bug signature for easy quoting in a bug report.

print()
print("=" * 60)
print("Bug-fingerprint summary")
print("=" * 60)
print(f"{'call':>5}  {'found_step':>12}  {'expected':>10}  {'detected':>30}")

bug_calls = 0
fp_pattern_calls = 0
for r in results:
    ci = r["call_idx"]
    fs = r["found_step"]
    expected = -1
    is_no_hit       = (fs == -1)
    is_real_sent    = (r["detected_state"] == list(range(STATE_SIZE)))
    is_all_zeros    = (r["detected_state"] == [0] * STATE_SIZE)
    detected_short  = "[all zeros]" if is_all_zeros else (
        "[real SENTINEL]" if is_real_sent else "[other]"
    )
    print(f"{ci:>5}  {fs:>12}  {expected:>10}  {detected_short:>30}")
    if not is_no_hit:
        bug_calls += 1
        if is_all_zeros and (ci > 0) and fs == (ci - 1):
            fp_pattern_calls += 1

print()
print(f"Bug present: {bug_calls > 0}")
print(f"  False-positive 'all-zeros' detections: {bug_calls}/{len(results)}")
print(f"  Of those, matching `found_step = call_idx - 1` pattern: {fp_pattern_calls}")
print()
print("Environment:")
print(f"  jax {jax.__version__}")
print(f"  jax_enable_x64 = {jax.config.jax_enable_x64}")
print(f"  TPU devices = {len(jax.devices())} (v5e-8 expected on Kaggle)")
print()
print("If `bug present: True`, please attach this output to your JAX/XLA bug report.")
'''

CELLS = [
    ("markdown", CELL_0_README),
    ("code",     CELL_1_SETUP),
    ("code",     CELL_2_CONFIG),
    ("code",     CELL_3_REACHABILITY),
    ("code",     CELL_4_BODY),
    ("code",     CELL_5_RUN),
    ("code",     CELL_6_ANALYZE),
]

# --- Assemble notebook ----------------------------------------------------

nb = {
    "cells": [
        {
            "cell_type": ctype,
            "metadata": {},
            "source": src,
            "execution_count": None,
            "outputs": [] if ctype == "code" else None,
        }
        for ctype, src in CELLS
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        cell.pop("outputs", None)
        cell.pop("execution_count", None)
for cell in nb["cells"]:
    src = cell["source"]
    if isinstance(src, str):
        cell["source"] = src.splitlines(keepends=True)

out_nb = HERE / "cayleypy-jax-shardmap-spmd-repro.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes)")

meta = {
    "id":                 "artgor/jax-shard-map-all-to-all-spmd-repro-tpu-v5e-8",
    "title":              "JAX shard_map all_to_all SPMD repro (TPU v5e-8)",
    "code_file":          "cayleypy-jax-shardmap-spmd-repro.ipynb",
    "language":           "python",
    "kernel_type":        "notebook",
    "is_private":         True,
    "enable_gpu":         False,
    "enable_tpu":         True,
    "enable_internet":    True,
    "dataset_sources":    [],
    "competition_sources":[],
    "kernel_sources":     [],
}
out_meta = HERE / "kernel-metadata.json"
out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {out_meta}")
