"""Builds the JAX SPMD shared-beam Kaggle notebook (.ipynb).

This is the JAX replacement for the failed `tpu_beam_spmd_8m` (torch_xla),
which hit a stale-buffer / all_gather alignment bug across 4 architectures.
The JAX version uses:
  * shard_map + lax.all_to_all (single packed tensor, no multi-call alignment risk)
  * Python for-loop OUTSIDE the shard_map body (jax #26148 workaround for
    fori_loop+manual collectives crash)
  * Pure-functional inference (no buffer aliasing across pid runs)

Reads jax_model.py / jax_beam.py / jax_beam_spmd.py as the source of truth
and inlines them as notebook cells. Locally-validated on 8-CPU-device
emulation: pid 1 found and verifies, zero false positives.

Output: cayleypy-tpu-beam-spmd-jax.ipynb
Run via: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_spmd_jax/build_notebook.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # megaminx/
HERE = Path(__file__).resolve().parent

# --- Inputs ---------------------------------------------------------------

puzzle_info = json.loads((ROOT / "data" / "puzzle_info.json").read_text())

rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
ALL_PID_STATES = {}
for pid in range(len(rows)):
    s = [int(x) for x in rows[pid]["initial_state"].split(",")]
    ALL_PID_STATES[pid] = s
N_TOTAL_PIDS = len(ALL_PID_STATES)


def _strip_module_imports(src: str, kill_imports: list[str]) -> str:
    """Remove `from X import Y` lines for the named modules X, so the file body
    can be inlined into a notebook cell where all symbols are global."""
    lines = src.splitlines(keepends=True)
    out = []
    for line in lines:
        stripped = line.strip()
        skip = False
        for mod in kill_imports:
            if (stripped.startswith(f"from {mod} import")
                    or stripped == f"import {mod}"
                    or stripped.startswith(f"import {mod} ")):
                skip = True
                break
        if not skip:
            out.append(line)
    return "".join(out)


def _file_to_cell(path: Path, kill_imports: list[str], rename_apply: bool) -> str:
    """Read a .py file, strip cross-module imports, optionally rename the
    bare `apply` (from jax_model) to `model_apply` for use in later cells."""
    src = path.read_text(encoding="utf-8")
    src = _strip_module_imports(src, kill_imports)
    if rename_apply:
        # Rename the public `apply` symbol to `model_apply` so beam cells can
        # use it without ambiguity. Match `def apply(` at column 0.
        src = re.sub(r"^def apply\(", "def model_apply(", src, flags=re.MULTILINE)
    return src


# --- Cell sources ---------------------------------------------------------

CELL_0_README = '''\
# Megaminx beam search on Kaggle TPU v5e-8 - B=4M shared beam (JAX SPMD, v7 production)

Single B=4M shared beam across all 8 TPU cores via `shard_map` + `lax.all_to_all`.
B_LOCAL = 512K per rank (hash-partitioned by state).

## Status: working as of v6 (verified 3/3 pids end-to-end)

v6 (diagnostic build, 2026-05-16) confirmed correctness end-to-end at B=1M on
pids 0/1/2 with paths verifying after walkback. v7 scales to B=4M (the original
target) for production runs.

## What changed vs. v1-v5

v1-v5 produced false-positive "found" paths that did not verify. Root cause
was localized in v6: the in-body cross-rank `pmin/psum` reduce was not
correctly propagating the global winner to all ranks' per-rank carry slots
on TPU v5e-8. Walkback then used a wrong-rank tree and recovered an invalid
path. Fix: skip the in-body reduce, pull all per-rank arrays to host, do
cross-rank winner selection there. See [[spmd-jax-recovery]] memory file.

The Python beam loop sits OUTSIDE the `shard_map`'d step (jax-ml/jax #26148:
`fori_loop` with manual collectives inside `shard_map` crashes).

## Production stack

m_curr_v3 V teacher (6M params) + m23_v2 sym-aware Q-shortlister (12M, alpha=2)
+ m_pi_v2 policy term (lambda=0.05) + B=4M shared beam, NUM_STEPS=120.

## When to use this notebook

For the ~50-100 hardest pids that the data-parallel B=1M sweeps can't crack.
For easy/medium pids, use the existing `cayleypy-megaminx-beam-m-curr-v3`
data-parallel notebook (8x throughput).

## How to run

1. Fork on Kaggle ("Copy and Edit").
2. Dataset `artgor/megaminx-tpu-artifacts` auto-attaches.
3. Edit Cell 1 (CONFIG) to set pid range. Default: smoke pids 0-7 at K=1.
4. Switch accelerator to TPU v5e-8 in the sidebar.
5. Run all cells.
6. Download `/kaggle/working/jax_spmd_submission.csv` after completion.

## Wall projection

Per-pid: ~250-400s (slower than B=1M data-parallel's 150s due to cross-rank
collectives + larger effective beam). Plan for ~8-pid smoke in ~1h after first-pid
compile (~30 min). For 50-pid rescue runs: budget 7-8h.

## Memory budget per rank (B_LOCAL=512K)

| component | size |
|---|---|
| local states (int8) | 60 MB |
| children buffer (B_LOCAL * 24 * 120 int8) | 1.44 GB |
| send/recv all_to_all buffers (uint8, 128 PACK) | ~256 MB |
| tree storage (NUM_STEPS * B_LOCAL * (4+1+1) bytes) | ~360 MB |
| model weights (V + Q + policy in bf16) | ~1 GB |
| **total per rank (peak)** | **~3.5 GB** |

TPU v5e-8 has 16 GB HBM per chip / 2 cores per chip => ~7 GB per rank after XLA
workspace. ~3.5 GB peak fits with headroom. If OOM, reduce B_GLOBAL to 2M.
'''

PUZZLE_INFO_LITERAL = json.dumps(puzzle_info, separators=(",", ":"))
ALL_PID_STATES_LITERAL = json.dumps(ALL_PID_STATES, separators=(",", ":"))

CELL_1_CONFIG = f'''\
# ============================================================================
# CONFIG - EDIT THIS CELL, then Run All
# ============================================================================
#
# This is the JAX SPMD shared-beam notebook. ONE pid at a time across all
# 8 TPU cores; beam state hash-partitioned (B_LOCAL = B_GLOBAL/8 per rank).
#
# Knobs:
#   START_PID, END_PID  - pid range (half-open). Default: 8-pid smoke.
#   B_GLOBAL            - 4M default; reduce to 2M if OOM.
#   K_SYM               - sym rotations per pid (1 default). K=2 doubles wall.
#
# Recommended kernel scopes (TPU v5e-8):
#   smoke:   START=0,   END=8,    K=1  -> ~1h total (40min compile + ~20s/pid x 8)
#   medium:  START=0,   END=30,   K=1  -> ~3h
#   rescue:  hardest-50 list, K=1       -> ~5-6h
# ============================================================================

START_PID = 0
END_PID = 8   # v8: 8-pid smoke at B=16M (ceiling test).
K_SYM = 1     # number of rotations when ROT_IDX_OVERRIDE is None.
ROT_IDX_OVERRIDE = None  # int in [0, 360) to force a SPECIFIC rotation index
                         # and run K_SYM=1 with just that rotation. None = use
                         # K_SYM auto-selection. Use this to split a K_SYM=N
                         # run across N kernels: each kernel sets a different
                         # override value (all with the same SYM_SEED so they
                         # collectively cover the same rotations).

B_GLOBAL = 8 * 1024 * 1024     # 8M shared beam (v9 — dropped from v8's 16M after
                               # runtime OOM on a 240 MB alloc with 176 MB free,
                               # i.e. ~15.8 GB used out of 16 GB HBM at B=16M).
                               # 8M projection: ~5 GB linear + ~3-4 GB XLA
                               # workspace = ~8 GB per rank. Comfortable.
ALPHA_QSHORT = 2               # Q-shortlist multiplier; aB_LOCAL = alpha * B_LOCAL.
LAMBDA_POLICY = 0.05           # policy term weight in student score.
INTERNAL_BS = 32768            # chunked-forward chunk size for V/Q model.
NUM_STEPS = 120                # depth of beam search.

# ============================================================================
# Do not edit below this line
# ============================================================================
'''

CELL_2_SETUP = '''\
# Cell 2: Setup. MUST run BEFORE any jax import.
import sys, os, time, json, subprocess

# Enable int64 dtype (default JAX is int32; we need int64 for state hashing).
# Use os.environ[...] = "True" (not setdefault) so we OVERRIDE any prior value
# Kaggle may have set. Capital T also matters — JAX accepts both but be explicit.
os.environ["JAX_ENABLE_X64"] = "True"
# Suppress noisy WARNING in TPU runtime.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, x):
        for s in self.streams:
            s.write(x); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_LOG = open("/kaggle/working/run.log", "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)

try:
    import psutil
    print(f"CPU cores: {psutil.cpu_count()}  (logical={psutil.cpu_count(logical=True)})")
    vm = psutil.virtual_memory()
    print(f"RAM total: {vm.total/1e9:.1f} GB  available: {vm.available/1e9:.1f} GB")
except Exception as e:
    print("psutil unavailable:", e)

# Probe JAX version + shard_map availability. Decision gate: if shard_map is
# missing the notebook stops here with a clear message.
import jax
import jax.numpy as jnp

# Belt-and-suspenders: also force x64 via the runtime config in case the env
# var didn't take effect early enough.
jax.config.update("jax_enable_x64", True)

print(f"jax {jax.__version__}")
print(f"jax_enable_x64 = {jax.config.jax_enable_x64}")
print(f"JAX_ENABLE_X64 env = {os.environ.get('JAX_ENABLE_X64')}")
_test_i64 = jnp.int64(3952374295417501729)
print(f"jnp.int64(3.95e18) = {int(_test_i64)} (dtype={_test_i64.dtype})  "
      f"expect 3952374295417501729 int64")
assert _test_i64.dtype == jnp.int64, f"x64 NOT enabled: int64 became {_test_i64.dtype}"
assert int(_test_i64) == 3952374295417501729, f"int64 truncated: {int(_test_i64)}"
try:
    from jax.experimental.shard_map import shard_map  # noqa: F401
    print("shard_map (experimental) OK")
except ImportError:
    try:
        from jax import shard_map  # JAX 0.5+ promoted location
        print("shard_map (top-level) OK")
    except ImportError as e:
        raise SystemExit(f"FAIL: shard_map not available in this JAX install: {e}")
from jax.sharding import Mesh, PartitionSpec, NamedSharding
print("Mesh, PartitionSpec, NamedSharding OK")

devices = jax.devices()
print(f"jax.devices(): {devices}")
N_DEVICES = len(devices)
print(f"N_DEVICES = {N_DEVICES}")
assert N_DEVICES == 8, f"Expected 8 TPU cores on v5e-8, got {N_DEVICES}"

assert START_PID < END_PID, f"START_PID ({START_PID}) >= END_PID ({END_PID})"
assert K_SYM in (1, 2, 4), f"K_SYM must be 1/2/4, got {K_SYM}"
assert B_GLOBAL % N_DEVICES == 0, f"B_GLOBAL ({B_GLOBAL}) must be divisible by {N_DEVICES}"
B_LOCAL = B_GLOBAL // N_DEVICES
K_PER_PEER = (ALPHA_QSHORT * B_LOCAL) // N_DEVICES
N_PIDS = END_PID - START_PID
print(f"\\n== CONFIG ==")
print(f"pid range: [{START_PID}, {END_PID})  ({N_PIDS} pids)")
print(f"K_SYM: {K_SYM}")
print(f"B_GLOBAL: {B_GLOBAL:,}  B_LOCAL: {B_LOCAL:,}  K_PER_PEER: {K_PER_PEER:,}")
print(f"alpha: {ALPHA_QSHORT}  lambda_policy: {LAMBDA_POLICY}")
print(f"NUM_STEPS: {NUM_STEPS}  INTERNAL_BS: {INTERNAL_BS}")

_proj_compile_h = 0.6
_proj_compute_h = (N_PIDS * K_SYM) * 320 / 3600
_proj_total_h = _proj_compile_h + _proj_compute_h
print(f"estimated wall: {_proj_compile_h:.1f}h compile + {_proj_compute_h:.1f}h compute = {_proj_total_h:.1f}h")
if _proj_total_h > 8.5:
    print(f"WARNING: estimated wall {_proj_total_h:.1f}h exceeds 8.5h target.")
'''

CELL_3_PUZZLE = f'''\
# Cell 3: Puzzle data + all pid states embedded; slice via CONFIG.
PUZZLE_INFO = json.loads({PUZZLE_INFO_LITERAL!r})
GENERATORS = PUZZLE_INFO["generators"]
SOLVED = tuple(PUZZLE_INFO["central_state"])
MOVE_NAMES = list(GENERATORS.keys())
N_GEN = len(MOVE_NAMES)
STATE_SIZE = len(SOLVED)
print(f"N_GEN={{N_GEN}}  STATE_SIZE={{STATE_SIZE}}")

import numpy as np
all_moves_np = np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32)
all_moves = jnp.asarray(all_moves_np)
V0_np = np.array(SOLVED, dtype=np.int8)
V0 = jnp.asarray(V0_np)
print("all_moves shape", tuple(all_moves.shape), "  V0 shape", tuple(V0.shape))

_ALL_PID_STATES = json.loads({ALL_PID_STATES_LITERAL!r})
ALL_PID_STATES = {{int(k): v for k, v in _ALL_PID_STATES.items()}}
TEST_PIDS = list(range(START_PID, END_PID))
PID_STATES = {{pid: ALL_PID_STATES[pid] for pid in TEST_PIDS}}
print(f"selected {{len(TEST_PIDS)}} pids: [{{TEST_PIDS[0]}}..{{TEST_PIDS[-1]}}]")
'''

CELL_4_LOAD_PT = '''\
# Cell 4: Load m_curr_v3 (V teacher), m23_v2 (Q shortlister), m_pi_v2 (policy)
# from the Kaggle dataset and convert each to a JAX params dict.
import torch
from pathlib import Path

dataset_root = None
for cand in [
    Path("/kaggle/input/megaminx-tpu-artifacts"),
    Path("/kaggle/input/datasets/artgor/megaminx-tpu-artifacts"),
]:
    if cand.exists():
        dataset_root = cand
        break
if dataset_root is None:
    listing = list(Path("/kaggle/input").rglob("*.pt"))[:20]
    raise SystemExit(f"megaminx-tpu-artifacts not found. /kaggle/input: {listing}")
print("dataset:", dataset_root)

m_curr_v3_path = dataset_root / "m_curr_v3_epoch_0499.pt"
m23v2_path = dataset_root / "m23_v2_epoch_0499.pt"
m_pi_path = dataset_root / "m_pi_v2_epoch_0199.pt"
for p in (m_curr_v3_path, m23v2_path, m_pi_path):
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}")

def _strip_orig_mod(sd):
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}
'''

CELL_5_MODEL = '''\
# Cell 5: Pure-JAX inference for ResMLPDistance (load_params_from_pt + model_apply).
'''  # body inlined from jax_model.py below

CELL_6_BEAM_SINGLE = '''\
# Cell 6: Single-device JAX beam search. Used as a fallback / smoke test;
# the SPMD shared-beam version is the production path on this notebook.
'''  # body inlined from jax_beam.py below

CELL_7_SPMD = '''\
# Cell 7: SPMD shared-beam (shard_map + all_to_all). The production path.
'''  # body inlined from jax_beam_spmd.py below

CELL_8_SYM = '''\
# Cell 8: Sym ensemble setup (K_SYM rotations per pid).
SYM_SEED = 0

dataset_root_for_rot = None
for cand in [
    Path("/kaggle/input/megaminx-tpu-artifacts"),
    Path("/kaggle/input/datasets/artgor/megaminx-tpu-artifacts"),
]:
    if cand.exists():
        dataset_root_for_rot = cand
        break
rotations_path = dataset_root_for_rot / "rotations.npy"
rotations_np = np.load(rotations_path)
print(f"rotations: {rotations_np.shape} dtype={rotations_np.dtype}")

identity_arr = np.arange(STATE_SIZE, dtype=rotations_np.dtype)
identity_idx = None
for i in range(rotations_np.shape[0]):
    if np.array_equal(rotations_np[i], identity_arr):
        identity_idx = i
        break
assert identity_idx is not None

rng_sym = np.random.default_rng(SYM_SEED)
other_idxs = [i for i in range(rotations_np.shape[0]) if i != identity_idx]
# Compute the K_SYM auto-selection list FIRST (deterministic from SYM_SEED).
# This is what an in-kernel K_SYM run would use. We print it so callers
# splitting K_SYM across multiple kernels know which indices to plug into
# ROT_IDX_OVERRIDE.
auto_chosen = [identity_idx]
if K_SYM > 1:
    auto_chosen += list(rng_sym.choice(other_idxs, size=K_SYM - 1, replace=False).tolist())
print(f"K_SYM={K_SYM}, SYM_SEED={SYM_SEED} auto-selection: {auto_chosen}")

if ROT_IDX_OVERRIDE is not None:
    assert 0 <= int(ROT_IDX_OVERRIDE) < rotations_np.shape[0], (
        f"ROT_IDX_OVERRIDE={ROT_IDX_OVERRIDE} out of range [0, {rotations_np.shape[0]})")
    chosen_rot_idxs = [int(ROT_IDX_OVERRIDE)]
    print(f"ROT_IDX_OVERRIDE set -> using ONLY rotation idx {ROT_IDX_OVERRIDE} "
          f"(K_SYM auto-list ignored for this kernel)")
else:
    chosen_rot_idxs = auto_chosen
print(f"chosen rotation idxs for this kernel: {chosen_rot_idxs}")

ROTATIONS = []
move_name_to_idx = {n: i for i, n in enumerate(MOVE_NAMES)}
perm_to_name = {tuple(v): n for n, v in GENERATORS.items()}

for rot_idx in chosen_rot_idxs:
    R = tuple(int(x) for x in rotations_np[rot_idx])
    R_inv = tuple(int(x) for x in np.argsort(rotations_np[rot_idx]))
    conj_idx = [-1] * N_GEN
    for nm in MOVE_NAMES:
        g = GENERATORS[nm]
        conj_perm = tuple(R_inv[g[R[i]]] for i in range(STATE_SIZE))
        if conj_perm not in perm_to_name:
            raise ValueError(f"rotation {rot_idx} not a symmetry")
        conj_idx[move_name_to_idx[nm]] = move_name_to_idx[perm_to_name[conj_perm]]
    ROTATIONS.append({"rot_idx": rot_idx, "R": R, "R_inv": R_inv, "conj_idx": conj_idx})

def apply_rotation_state(state, R, R_inv):
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))

PID_ROT_PAIRS = []
for pid in TEST_PIDS:
    for ri in range(len(ROTATIONS)):
        PID_ROT_PAIRS.append((pid, ri))
print(f"total (pid, rotation) pairs to solve: {len(PID_ROT_PAIRS)}")
'''

CELL_9_LOAD_PARAMS = '''\
# Cell 9: Load all three model checkpoints, build JAX params dicts.
print("loading m_curr_v3 (V teacher) ...")
teacher_params = load_params_from_pt(m_curr_v3_path, hidden_dims=(2048, 512))
print(f"  V params: {num_params(teacher_params):,}")
print("loading m23_v2 (Q shortlister) ...")
student_params = load_params_from_pt(m23v2_path, hidden_dims=(2048, 1024))
print(f"  Q params: {num_params(student_params):,}")
print("loading m_pi_v2 (policy) ...")
policy_params = load_params_from_pt(m_pi_path, hidden_dims=(2048, 512))
print(f"  policy params: {num_params(policy_params):,}")

# Hash vector + V0 hash (int64).
hash_vec_np = make_hash_vec(STATE_SIZE, seed=0)
hash_vec = jnp.asarray(hash_vec_np)

# Mesh + sharding.
mesh = make_mesh(devices)
print(f"mesh: {mesh}")
'''

CELL_10_SOLVE_LOOP = '''\
# Cell 10: Per-pid loop. Each (pid, rotation) pair runs through the SPMD beam.
import csv as _csv

results_list = []

def _verify(initial_state, path_idx_orig):
    cur = list(initial_state)
    for m in path_idx_orig:
        gen = GENERATORS[MOVE_NAMES[m]]
        cur = [cur[g] for g in gen]
    return tuple(cur) == SOLVED

t_pipeline = time.time()
for i, (pid, rot_i) in enumerate(PID_ROT_PAIRS):
    rot = ROTATIONS[rot_i]
    s0 = PID_STATES[pid]
    s_to_solve = apply_rotation_state(s0, rot["R"], rot["R_inv"])

    t0 = time.time()
    try:
        r = beam_solve_qshort_spmd(
            list(s_to_solve), teacher_params, student_params, policy_params,
            all_moves, V0, hash_vec, mesh,
            B_local=B_LOCAL, K_per_peer=K_PER_PEER,
            n_gen=N_GEN, state_size=STATE_SIZE,
            num_steps=NUM_STEPS, lambda_policy=LAMBDA_POLICY,
            dtype=jnp.bfloat16,
            internal_bs=INTERNAL_BS,
        )
    except Exception as e:
        import traceback
        print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid} rot={rot_i} EXCEPTION: {e}")
        traceback.print_exc()
        results_list.append({
            "pid": pid, "rot_idx": rot_i, "found": False, "verify_ok": False,
            "path_len": 0, "path_idx_orig": [], "wall_s": time.time() - t0,
            "error": str(e),
        })
        continue
    wall = time.time() - t0

    path_orig = None
    verify_ok = False
    if r["found"]:
        conj_idx = rot["conj_idx"]
        path_orig = [conj_idx[m] for m in r["path_idx"]]
        verify_ok = _verify(s0, path_orig)

    rec = {
        "pid": pid, "rot_idx": rot_i,
        "found": r["found"], "verify_ok": verify_ok,
        "path_len": len(path_orig) if path_orig is not None else 0,
        "path_idx_orig": path_orig if verify_ok else [],
        "path_idx_rotated": r.get("path_idx", []),
        "wall_s": wall,
        "found_step": r.get("found_step", -1),
        "first_iter_s": r.get("first_iter_s"),
        "min_v_trajectory": r.get("min_v_trajectory", []),
        "detected_state_first12": r.get("detected_state_first12"),
        "detected_state_last12": r.get("detected_state_last12"),
        "detected_is_V0": r.get("detected_is_V0"),
    }
    results_list.append(rec)
    last_v = (rec["min_v_trajectory"] or [None])[-1]
    fs = r.get("found_step", -1)
    mvt = r.get("min_v_trajectory", [])
    minv_at_fs = mvt[fs] if 0 <= fs < len(mvt) else None
    minv_at_fs1 = mvt[fs+1] if 0 <= fs+1 < len(mvt) else None
    print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid:>4} rot={rot_i} "
          f"found={r['found']} verify={verify_ok} path_len={rec['path_len']:>3} "
          f"wall={wall:.1f}s first={rec['first_iter_s']} last_min_V={last_v}", flush=True)
    # Concise per-pid line. Per-rank summary stays in jax_spmd_final.json for archival.
    if r.get("found"):
        n_hit_ranks = sum(1 for rr in (r.get("per_rank_summary") or []) if rr["fstep"] >= 0)
        print(f"  winner_rank={r.get('found_pos_rank')}  fs={fs}  "
              f"n_local_hit_ranks={n_hit_ranks}  "
              f"winner_is_V0={r.get('detected_is_V0')}  "
              f"expected_owner={r.get('expected_owner_rank')}",
              flush=True)

    # Per-iteration partial save (recovers from kernel kill).
    with open("/kaggle/working/jax_spmd_partial.json", "w") as f:
        json.dump(results_list, f)

with open("/kaggle/working/jax_spmd_final.json", "w") as f:
    json.dump(results_list, f, indent=2)
print(f"\\n== pipeline done in {time.time() - t_pipeline:.1f}s, {len(results_list)} pairs ==")
'''

CELL_11_AGGREGATE = '''\
# Cell 11: Aggregate per-pid min-over-rotations, write submission CSV.
by_pid = {}
for r in results_list:
    by_pid.setdefault(r["pid"], []).append(r)

print()
print("== per-pid (min over K rotations) ==")
print(f"{'pid':>6}  {'best_rot':>8}  {'best_path':>9}  {'found/K':>10}  {'wall':>9}")
final_per_pid = {}
total_path = 0
n_solved = 0
for pid in sorted(by_pid.keys()):
    rs = by_pid[pid]
    valid = [r for r in rs if r["verify_ok"]]
    if valid:
        best = min(valid, key=lambda r: r["path_len"])
        final_per_pid[pid] = {"found": True, "best_path_len": best["path_len"],
                              "best_path_idx": best["path_idx_orig"],
                              "best_rot_idx": best["rot_idx"]}
        total_path += best["path_len"]
        n_solved += 1
        wall_max = max(r["wall_s"] for r in rs)
        print(f"{pid:>6}  {best['rot_idx']:>8}  {best['path_len']:>9}  "
              f"{len(valid):>4}/{len(rs):<4}  {wall_max:>9.1f}")
    else:
        final_per_pid[pid] = {"found": False}
        wall_max = max(r["wall_s"] for r in rs)
        print(f"{pid:>6}  {'-':>8}  {'-':>9}  {'0':>4}/{len(rs):<4}  {wall_max:>9.1f}")

print()
print(f"== summary ==")
print(f"solved (any rotation): {n_solved}/{len(by_pid)}")
print(f"total_path (best per pid): {total_path}")

csv_path = "/kaggle/working/jax_spmd_submission.csv"
with open(csv_path, "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["initial_state_id", "path"])
    for pid in sorted(by_pid.keys()):
        info = final_per_pid[pid]
        if info["found"]:
            names = [MOVE_NAMES[m] for m in info["best_path_idx"]]
            w.writerow([pid, ".".join(names)])
        else:
            w.writerow([pid, ""])
print(f"wrote {csv_path}")
'''

# --- Inline the .py files as cell content -------------------------------

model_src = _file_to_cell(HERE / "jax_model.py",
                          kill_imports=["jax_model"],
                          rename_apply=True)
beam_single_src = _file_to_cell(HERE / "jax_beam.py",
                                kill_imports=["jax_model", "jax_beam"],
                                rename_apply=False)
beam_spmd_src = _file_to_cell(HERE / "jax_beam_spmd.py",
                              kill_imports=["jax_model", "jax_beam", "jax_beam_spmd"],
                              rename_apply=False)

CELL_5_MODEL = CELL_5_MODEL + "\n" + model_src
CELL_6_BEAM_SINGLE = CELL_6_BEAM_SINGLE + "\n" + beam_single_src
CELL_7_SPMD = CELL_7_SPMD + "\n" + beam_spmd_src

CELLS = [
    ("markdown", CELL_0_README),
    ("code", CELL_1_CONFIG),
    ("code", CELL_2_SETUP),
    ("code", CELL_3_PUZZLE),
    ("code", CELL_4_LOAD_PT),
    ("code", CELL_5_MODEL),
    ("code", CELL_6_BEAM_SINGLE),
    ("code", CELL_7_SPMD),
    ("code", CELL_8_SYM),
    ("code", CELL_9_LOAD_PARAMS),
    ("code", CELL_10_SOLVE_LOOP),
    ("code", CELL_11_AGGREGATE),
]

# --- Notebook assembly ----------------------------------------------------

nb = {
    "cells": [
        {
            "cell_type": ctype,
            "metadata": {},
            "source": code,
            "execution_count": None,
            "outputs": [] if ctype == "code" else None,
        }
        for ctype, code in CELLS
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

out_nb = HERE / "cayleypy-tpu-beam-spmd-jax.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes, "
      f"{len(ALL_PID_STATES)} pid states embedded)")

meta = {
    "id": "artgor/cayleypy-megaminx-beam-spmd-jax",
    "title": "cayleypy megaminx beam SPMD JAX",
    "code_file": "cayleypy-tpu-beam-spmd-jax.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": True,
    "enable_internet": True,
    "dataset_sources": ["artgor/megaminx-tpu-artifacts"],
    "competition_sources": [],
    "kernel_sources": [],
}
out_meta = HERE / "kernel-metadata.json"
out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {out_meta}")
