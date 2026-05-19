"""Builds the SHAREABLE AZ v4 V-only SPMD beam at B=32M (Phase A+B+C+D).

Companion to `tpu_beam_az_v4_v_only_jax_shareable/` (B=8M production):
this notebook quadruples the beam to 32M for higher-quality search at the
cost of ~50% more wall per pid. Same AZ v4 V model, same V-only stack
(no qshort), same multi-collaborator workflow.

Built on the Phase A+B+C+D optimization stack:
  * Phase A: dropped per-child index arrays, donated step_fn carry.
  * Phase B: packed uint32 backpointers, host np.memmap tree, early stop.
  * Phase C: streamed child generation via lax.scan over parent chunks
             (mandatory: full neighbors materialization OOMs at 32M).
  * Phase D: bf16 V score packed into bucket bytes 125-126, skip
             receive-side V re-run.

Defaults verified at 32M alpha=2 on Kaggle TPU v5e-8: pid 0 / 55 moves /
verify_ok=True / ~57 min wall (vs 8M shareable's pid 0 / ~40 min).

Reuses jax_model.py + jax_beam.py + jax_beam_spmd_v_only.py from
../tpu_beam_spmd_jax/ -- single source of truth.

Output: cayleypy-megaminx-beam-az-v4-32m-shareable-jax.ipynb
Run via: .venv/Scripts/python.exe build_notebook.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # megaminx/
HERE = Path(__file__).resolve().parent
SPMD_DIR = ROOT / "kaggle_notebooks" / "tpu_beam_spmd_jax"

puzzle_info = json.loads((ROOT / "data" / "puzzle_info.json").read_text())

rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
ALL_PID_STATES = {}
for pid in range(len(rows)):
    s = [int(x) for x in rows[pid]["initial_state"].split(",")]
    ALL_PID_STATES[pid] = s
N_TOTAL_PIDS = len(ALL_PID_STATES)


def _strip_module_imports(src: str, kill_imports: list[str]) -> str:
    lines = src.splitlines(keepends=True)
    out = []
    for line in lines:
        stripped = line.strip()
        skip = any(
            stripped.startswith(f"from {mod} import")
            or stripped == f"import {mod}"
            or stripped.startswith(f"import {mod} ")
            for mod in kill_imports
        )
        if not skip:
            out.append(line)
    return "".join(out)


def _file_to_cell(path: Path, kill_imports: list[str], rename_apply: bool) -> str:
    src = path.read_text(encoding="utf-8")
    src = _strip_module_imports(src, kill_imports)
    if rename_apply:
        src = re.sub(r"^def apply\(", "def model_apply(", src, flags=re.MULTILINE)
    return src


# --- Cell sources ---------------------------------------------------------

CELL_0_README = '''\
# Megaminx beam search on Kaggle TPU v5e-8 - AZ v4 V-only shareable (JAX SPMD, B=32M)

Multi-collaborator TPU beam search for the
[CayleyPy Megaminx Kaggle competition](https://www.kaggle.com/competitions/cayley-py-megaminx).
**4x larger beam than the 8M shareable** for higher-quality search.
Each collaborator runs a chunk of pids on their own TPU quota; results
union-merged via per-pid min.

**Algorithm:** SPMD shared-beam at B_GLOBAL=32M, all 8 TPU v5e-8 cores
cooperating on one pid at a time. V model ranks ALL B_LOCAL * N_GEN
children per step (no qshort prefilter). 32M fits HBM via:
  * **Phase A** -- dropped per-child index arrays, donated step_fn carry.
  * **Phase B** -- packed uint32 backpointers, host np.memmap tree,
    early stop on first V0 hit.
  * **Phase C** -- streamed child generation via lax.scan over parent
    chunks (full neighbors materialization OOMs at 32M).
  * **Phase D** -- bf16 V score packed into the all-to-all bucket;
    receive side skips V re-run.
  * **One-sort receive dedup** (single argsort + sorted-order top-k).
  * **uint32 owner hash** for cheaper child-side bucket routing.

**Stack:** AZ v4 V-only (6M params, `m_az_v4_v_only.pt`).

## When to use this vs the 8M shareable

| metric | 8M shareable | this (32M) |
|---|---|---|
| per-pid wall (steady state) | ~40 min | ~55-60 min |
| weekly TPU quota fits | ~30 pids | ~20 pids |
| beam size | 8,388,608 | 33,554,432 |
| HBM peak / rank | ~5 GB | ~13 GB |
| memmap (per pid, /kaggle/working) | none | ~15 GB (auto-deleted) |
| expected quality on hard pids | baseline | shorter paths |

Use this notebook when you want **better paths on hard pids**; use the 8M
shareable when you need **maximum throughput**.

## Per-pid wall (TPU v5e-8, B_GLOBAL=32M, alpha=2)

| measure | value |
|---|---|
| compile (first pid) | ~50-140s |
| per-step (steady) | ~50-60s |
| per-pid wall (50-step solve) | ~50 min |
| per-pid wall (max num_steps=120) | ~115 min |

## Recommended scopes per kernel

| scope (N_PIDS x N_POSITIONS) | wall (incl. compile) |
|---|---|
| 4 pids x 1 position (smoke)        | ~4 h |
| 6 pids x 1 position                | ~6 h |
| 8 pids x 1 position                | ~8 h (close to 9h kernel kill) |
| 4 pids x 2 positions               | ~8 h (close to 9h kernel kill) |
| 2 pids x 4 positions               | ~8 h (one shard of K=8 split) |

The K=8 split lets two collaborators jointly run a K_SYM=8 ensemble on
2 hard pids each in their 9h budget (vs 1 pid solo).

## Sym ensemble split (K_SYM=8 across two kernels)

The notebook separates **K_SYM** (the full deterministic ensemble size)
from **SYM_POSITIONS** (which positions of that ensemble this kernel
actually runs). Both shards generate the SAME canonical K_SYM-rotation
list (seed=0) and just slice it differently.

To run K_SYM=8 across two kernels on the same `[START_PID, END_PID)`:

| kernel | K_SYM | SYM_POSITIONS | output CSV |
|---|---|---|---|
| A | 8 | `range(0, 4)` | `share_jax_submission_k8_sym0_3.csv` |
| B | 8 | `range(4, 8)` | `share_jax_submission_k8_sym4_7.csv` |

After both finish, min-merging is a per-pid `min(path_len)` over the two
CSVs (associative: `min(sym0..3, sym4..7) == min(sym0..7)`). Locally:

```powershell
.venv/Scripts/python.exe megaminx/scripts/16_merge_rescue.py `
    --base megaminx/submissions/current_best.csv `
    --rescue share_jax_submission_k8_sym0_3.csv `
    --out  tmp_after_shardA.csv
.venv/Scripts/python.exe megaminx/scripts/16_merge_rescue.py `
    --base tmp_after_shardA.csv `
    --rescue share_jax_submission_k8_sym4_7.csv `
    --out  merged_k8.csv
```

## How to run as a collaborator

1. **Fork** this notebook on Kaggle ("Copy and Edit").
2. The dataset `artgor/megaminx-tpu-artifacts` auto-attaches; must contain
   `m_az_v4_v_only.pt`.
3. **Edit Cell 1 (CONFIG)** to set `START_PID`, `END_PID`, `K_SYM`,
   `SYM_POSITIONS`. If you're pairing with another collaborator on a K=8
   split, agree on the pid range first, then take complementary
   `SYM_POSITIONS` slices.
4. **Switch accelerator to TPU v5e-8** in the right sidebar.
5. Run all cells.
6. Per-iteration partial saves -- even if Kaggle kills mid-run, every
   completed (pid, sym_pos) pair is in
   `/kaggle/working/share_jax_partial_<SLICE_TAG>.json`.
7. After completion, download
   `/kaggle/working/share_jax_submission_<SLICE_TAG>.csv` (and the matching
   `_final_<SLICE_TAG>.json` if convenient) and send back to the owner
   for cross-shard min-merging.

## Notes for collaborators

- The 9h Kaggle TPU kernel kill is real; keep
  `N_PIDS * len(SYM_POSITIONS) <= 8` for safety (the projected wall in
  Cell 2 output will warn at >8.5h).
- Weekly TPU quota is 20h per account.
- /kaggle/working tree memmap files (~15 GB per active pid) auto-delete
  in the wrapper's `finally` block.
- `verify=True` in the per-pid output line is the integrity check
  (path applied to init state produces V0). Should always be True for
  found pids; report if you see verify=False.
- **Both shards of a K=8 split MUST use the same `K_SYM`** (and the same
  hardcoded `SYM_SEED=0`), otherwise their position indices refer to
  different rotations and the merge is meaningless.
'''

PUZZLE_INFO_LITERAL = json.dumps(puzzle_info, separators=(",", ":"))
ALL_PID_STATES_LITERAL = json.dumps(ALL_PID_STATES, separators=(",", ":"))

CELL_1_CONFIG = f'''\
# ============================================================================
# CONFIG - EDIT THIS CELL, then Run All
# ============================================================================
#
# Test set has 1001 pids (0..1000), ordered by random-walk length
# (pid 0 = easiest scramble, pid 1000 = hardest).
#
# Knobs:
#   START_PID, END_PID  - pid range to solve (half-open).
#   K_SYM               - FULL target sym-ensemble size: 1 / 2 / 4 / 8.
#                         Defines the deterministic canonical K_SYM-rotation
#                         list (seed=0). Both shards of a split run MUST set
#                         the same K_SYM, otherwise their position indices
#                         refer to different rotations.
#   SYM_POSITIONS       - which positions of the K_SYM ensemble THIS kernel
#                         actually runs (a range or list of ints in
#                         [0, K_SYM)). Lets one kernel run a slice and
#                         another run the rest; min-merge their CSVs.
#                         Memory does NOT scale with len(SYM_POSITIONS).
#
# Split recipes (B=32M, ~55 min/pid/position; keep N_PIDS*N_POSITIONS <= 8):
#
#   K=8 across two kernels (recommended for hardest pids):
#     Kernel A:  K_SYM=8, SYM_POSITIONS=range(0, 4)
#     Kernel B:  K_SYM=8, SYM_POSITIONS=range(4, 8)
#     Then min-merge the two share_jax_submission_k8_sym*.csv locally.
#
#   K=4 in one kernel (smaller ensemble, no split):
#     K_SYM=4, SYM_POSITIONS=range(0, 4)
#
#   Smoke / identity only:
#     K_SYM=1, SYM_POSITIONS=range(0, 1)
# ============================================================================

START_PID = 0
END_PID = 4
K_SYM = 8
SYM_POSITIONS = range(0, 4)   # kernel A; kernel B uses range(4, 8)

# Verified at B=32M alpha=2 on TPU v5e-8:
#   pid 0 -> 55 moves, ~55 min wall, verify_ok=True.
B_GLOBAL = 32 * 1024 * 1024
ALPHA_QSHORT = 2              # receive-side 2x oversampling for quality
INTERNAL_BS = 16384
NUM_STEPS = 120

PARENT_CHUNK = 65536          # B_LOCAL / PARENT_CHUNK = 64 chunks per step
PACK_V_SCORE = True           # bf16 score packed into bucket; skip recv V re-run
PROGRESS_EVERY = 5            # print per-step timing every N steps

# ============================================================================
# Do not edit below this line
# ============================================================================
'''

CELL_2_SETUP = '''\
# Cell 2: Setup. JAX_ENABLE_X64 + XLA_PYTHON_CLIENT_MEM_FRACTION must be set
# BEFORE `import jax`. Phase A: raise the JAX/XLA memory fraction from the
# default 0.75 to 0.95 -- frees ~3 GB of headroom at the cost of less
# scheduler slack.
import sys, os, time, json
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
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

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
print(f"jax {jax.__version__}  x64={jax.config.jax_enable_x64}")
_t = jnp.int64(3952374295417501729)
assert _t.dtype == jnp.int64 and int(_t) == 3952374295417501729, "x64 NOT enabled"

try:
    from jax.experimental.shard_map import shard_map  # noqa: F401
except ImportError:
    from jax import shard_map  # JAX 0.5+
from jax.sharding import Mesh, PartitionSpec, NamedSharding  # noqa: F401

devices = jax.devices()
print(f"devices: {devices}")
N_DEVICES = len(devices)
assert N_DEVICES == 8, f"Expected 8 TPU cores; got {N_DEVICES}. Switch accelerator to TPU v5e-8."

assert 0 <= START_PID < END_PID <= ''' + str(N_TOTAL_PIDS) + ''', \\
    f"invalid pid range: START_PID={START_PID}, END_PID={END_PID}"
assert K_SYM in (1, 2, 4, 8), f"K_SYM must be 1/2/4/8; got {K_SYM}"
SYM_POSITIONS = list(SYM_POSITIONS)
assert len(SYM_POSITIONS) >= 1, "SYM_POSITIONS must be non-empty"
assert all(isinstance(p, int) and 0 <= p < K_SYM for p in SYM_POSITIONS), \\
    f"SYM_POSITIONS={SYM_POSITIONS} contains values outside [0, K_SYM={K_SYM})"
assert len(set(SYM_POSITIONS)) == len(SYM_POSITIONS), \\
    f"SYM_POSITIONS has duplicate positions: {SYM_POSITIONS}"
N_LOCAL_SYM = len(SYM_POSITIONS)
assert B_GLOBAL % N_DEVICES == 0
B_LOCAL = B_GLOBAL // N_DEVICES
K_PER_PEER = (ALPHA_QSHORT * B_LOCAL) // N_DEVICES
assert B_LOCAL % PARENT_CHUNK == 0, \\
    f"PARENT_CHUNK={PARENT_CHUNK} must divide B_LOCAL={B_LOCAL}"
N_PIDS = END_PID - START_PID
print(f"\\n== CONFIG ==")
print(f"pid range: [{START_PID}, {END_PID})  ({N_PIDS} pids)")
print(f"K_SYM (full ensemble): {K_SYM}")
print(f"SYM_POSITIONS (this kernel): {SYM_POSITIONS}  ({N_LOCAL_SYM} positions)")
print(f"B_GLOBAL: {B_GLOBAL:,}  B_LOCAL: {B_LOCAL:,}  K_PER_PEER: {K_PER_PEER:,}")
print(f"NUM_STEPS: {NUM_STEPS}  alpha: {ALPHA_QSHORT}")
print(f"PARENT_CHUNK: {PARENT_CHUNK:,}  n_chunks_per_step: {B_LOCAL // PARENT_CHUNK}")

# V-only is ~6x qshort per-step compute. Streaming at 32M adds scan overhead.
_per_pid = {1*1024*1024: 450, 4*1024*1024: 1300, 8*1024*1024: 2600,
            16*1024*1024: 5200, 32*1024*1024: 10800}.get(B_GLOBAL, 10800)
_n_pairs = N_PIDS * N_LOCAL_SYM
_proj = 0.1 + (_n_pairs * _per_pid) / 3600
print(f"estimated wall: {_proj:.1f}h (compile + {_n_pairs} pairs × ~{_per_pid}s)")
if _proj > 8.5:
    print(f"WARNING: estimated wall {_proj:.1f}h exceeds 8.5h target.")
    print(f"  Kaggle kills TPU kernels around 9h. Consider reducing pid range")
    print(f"  or shrinking SYM_POSITIONS (and running the rest in a second kernel).")
'''

CELL_3_PUZZLE = f'''\
# Cell 3: Puzzle + all 1001 pid states embedded.
PUZZLE_INFO = json.loads({PUZZLE_INFO_LITERAL!r})
GENERATORS = PUZZLE_INFO["generators"]
SOLVED = tuple(PUZZLE_INFO["central_state"])
MOVE_NAMES = list(GENERATORS.keys())
N_GEN = len(MOVE_NAMES)
STATE_SIZE = len(SOLVED)
print(f"N_GEN={{N_GEN}}  STATE_SIZE={{STATE_SIZE}}")

all_moves_np = np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32)
all_moves = jnp.asarray(all_moves_np)
V0_np = np.array(SOLVED, dtype=np.int8)
V0 = jnp.asarray(V0_np)

_ALL_PID_STATES = json.loads({ALL_PID_STATES_LITERAL!r})
ALL_PID_STATES = {{int(k): v for k, v in _ALL_PID_STATES.items()}}
TEST_PIDS = list(range(START_PID, END_PID))
PID_STATES = {{pid: ALL_PID_STATES[pid] for pid in TEST_PIDS}}
print(f"selected {{len(TEST_PIDS)}} pids: [{{TEST_PIDS[0]}}..{{TEST_PIDS[-1]}}]")
'''

CELL_4_LOAD_PT = '''\
# Cell 4: Load m_az_v4_v_only from the Kaggle dataset.
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
    raise SystemExit("megaminx-tpu-artifacts dataset not found in /kaggle/input. "
                     "If you forked this notebook, the dataset should auto-attach. "
                     "Check the right sidebar -> Data -> Input.")
print("dataset:", dataset_root)

v_path = dataset_root / "m_az_v4_v_only.pt"
if not v_path.exists():
    raise SystemExit(
        f"missing artifact: {v_path}\\n"
        f"The dataset must include m_az_v4_v_only.pt (added 2026-05-11 with the\\n"
        f"AZ v4 V-only breakthrough). If you forked an older notebook copy, re-attach\\n"
        f"the latest version of artgor/megaminx-tpu-artifacts via the right sidebar."
    )
print(f"V model: {v_path}")
'''

CELL_5_MODEL = '''\
# Cell 5: Pure-JAX inference (inlined from jax_model.py).
'''

CELL_6_BEAM_SINGLE = '''\
# Cell 6: Single-device JAX helpers (inlined from jax_beam.py — used for the
# step-0 V eval on the 24 first-move children, which is too small to chunk).
'''

CELL_7_SPMD_V_ONLY = '''\
# Cell 7: SPMD V-only shared-beam (inlined from jax_beam_spmd_v_only.py).
# Same all_to_all + host-side cross-rank reduce as the qshort body, but V is
# computed on ALL children (no qshort prefilter).
'''

CELL_8_SYM = '''\
# Cell 8: Sym ensemble setup. Deterministic K_SYM rotations from SYM_SEED=0.
SYM_SEED = 0

rotations_path = dataset_root / "rotations.npy"
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

# Deterministic full K_SYM-rotation list. Identity at position 0, then
# K_SYM-1 distinct non-identity rotations drawn with seed=0. Both shards
# of a split run hit this same code path with the same K_SYM, so they
# agree on what every position 0..K_SYM-1 means.
auto_chosen = [identity_idx]
if K_SYM > 1:
    auto_chosen += list(rng_sym.choice(other_idxs, size=K_SYM - 1, replace=False).tolist())
print(f"K_SYM={K_SYM} SYM_SEED={SYM_SEED} full auto-selection: {auto_chosen}")

# This kernel runs only the slice asked for in Cell 1.
chosen_rot_idxs = [auto_chosen[p] for p in SYM_POSITIONS]
print(f"this kernel sym positions: {SYM_POSITIONS}")
print(f"this kernel rotation idxs: {chosen_rot_idxs}")

ROTATIONS = []
move_name_to_idx = {n: i for i, n in enumerate(MOVE_NAMES)}
perm_to_name = {tuple(v): n for n, v in GENERATORS.items()}

for sym_pos, rot_idx in zip(SYM_POSITIONS, chosen_rot_idxs):
    R = tuple(int(x) for x in rotations_np[rot_idx])
    R_inv = tuple(int(x) for x in np.argsort(rotations_np[rot_idx]))
    conj_idx = [-1] * N_GEN
    for nm in MOVE_NAMES:
        g = GENERATORS[nm]
        conj_perm = tuple(R_inv[g[R[i]]] for i in range(STATE_SIZE))
        if conj_perm not in perm_to_name:
            raise ValueError(f"rotation {rot_idx} not a symmetry")
        conj_idx[move_name_to_idx[nm]] = move_name_to_idx[perm_to_name[conj_perm]]
    ROTATIONS.append({
        "sym_pos": sym_pos,           # position in the full K_SYM ensemble
        "rot_idx": rot_idx,           # absolute idx into rotations.npy
        "R": R, "R_inv": R_inv, "conj_idx": conj_idx,
    })

def apply_rotation_state(state, R, R_inv):
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))

PID_ROT_PAIRS = []
for pid in TEST_PIDS:
    for ri in range(len(ROTATIONS)):
        PID_ROT_PAIRS.append((pid, ri))
print(f"total (pid, rotation) pairs to solve: {len(PID_ROT_PAIRS)}")
'''

CELL_9_LOAD_PARAMS = '''\
# Cell 9: Load V checkpoint, build JAX params dict, set up mesh.
print("loading AZ v4 V (V teacher, V-only) ...")
v_params = load_params_from_pt(v_path, hidden_dims=(2048, 512))
print(f"  V params: {num_params(v_params):,}")

hash_vec_np = make_hash_vec(STATE_SIZE, seed=0)
hash_vec = jnp.asarray(hash_vec_np)

# Cheaper uint32 hash used only for owner routing (not for dedup / V0).
# Owner partition needs uniform 0..world_size-1, not collision resistance.
_owner_rng = np.random.default_rng(12345)
owner_hash_vec_np = _owner_rng.integers(
    0, np.iinfo(np.uint32).max, size=STATE_SIZE, dtype=np.uint32,
)
owner_hash_vec = jnp.asarray(owner_hash_vec_np)

mesh = make_mesh(devices)
print(f"mesh: {mesh}")
'''

CELL_10_SOLVE_LOOP = '''\
# Cell 10: SPMD V-only solve loop — 1 pid at a time, all 8 cores cooperating.
# Shard-aware output filenames so two collaborators can download artifacts
# without colliding.
SLICE_TAG = f"k{K_SYM}_sym{min(SYM_POSITIONS)}_{max(SYM_POSITIONS)}"
PARTIAL_PATH = f"/kaggle/working/share_jax_partial_{SLICE_TAG}.json"
FINAL_PATH   = f"/kaggle/working/share_jax_final_{SLICE_TAG}.json"
print(f"SLICE_TAG: {SLICE_TAG}")
print(f"partial: {PARTIAL_PATH}")
print(f"final:   {FINAL_PATH}")

def _verify(initial_state, path_idx_orig):
    cur = list(initial_state)
    for m in path_idx_orig:
        gen = GENERATORS[MOVE_NAMES[m]]
        cur = [cur[g] for g in gen]
    return tuple(cur) == SOLVED

results_list = []
t_pipeline = time.time()
for i, (pid, rot_i) in enumerate(PID_ROT_PAIRS):
    rot = ROTATIONS[rot_i]
    sym_pos = rot["sym_pos"]
    rot_idx_abs = rot["rot_idx"]
    s0 = PID_STATES[pid]
    s_to_solve = apply_rotation_state(s0, rot["R"], rot["R_inv"])

    # Phase B: per-pid memmap tree (auto-deleted in the wrapper's finally).
    # Tagged by sym_pos rather than local i so concurrent same-pid different-pos
    # runs across forks remain unambiguous in logs.
    tree_path = f"/kaggle/working/tree_pid{pid}_sym{sym_pos}.u32"

    t0 = time.time()
    try:
        r = beam_solve_v_only_spmd_packed(
            list(s_to_solve), v_params,
            all_moves, V0, hash_vec, mesh,
            B_local=B_LOCAL, K_per_peer=K_PER_PEER,
            n_gen=N_GEN, state_size=STATE_SIZE,
            num_steps=NUM_STEPS, dtype=jnp.bfloat16,
            internal_bs=INTERNAL_BS,
            tree_path=tree_path,
            parent_chunk=PARENT_CHUNK,
            pack_v_score=PACK_V_SCORE,
            progress_every=PROGRESS_EVERY,
            owner_hash_vec=owner_hash_vec,
        )
    except Exception as e:
        import traceback
        print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid} sym_pos={sym_pos} EXCEPTION: {e}")
        traceback.print_exc()
        results_list.append({
            "pid": pid,
            "sym_pos": sym_pos,
            "rot_idx": rot_idx_abs,
            "found": False, "verify_ok": False,
            "path_len": 0, "path_idx_orig": [],
            "wall_s": time.time() - t0, "error": str(e),
        })
        if os.path.exists(tree_path):
            try: os.unlink(tree_path)
            except OSError: pass
        continue
    wall = time.time() - t0

    path_orig = None
    verify_ok = False
    if r["found"]:
        conj_idx = rot["conj_idx"]
        path_orig = [conj_idx[m] for m in r["path_idx"]]
        verify_ok = _verify(s0, path_orig)

    rec = {
        "pid": pid,
        "sym_pos": sym_pos,           # logical position in the full K_SYM ensemble
        "rot_idx": rot_idx_abs,       # absolute idx into rotations.npy
        "found": r["found"], "verify_ok": verify_ok,
        "path_len": len(path_orig) if path_orig is not None else 0,
        "path_idx_orig": path_orig if verify_ok else [],
        "path_idx_rotated": r.get("path_idx", []),
        "wall_s": wall,
        "found_step": r.get("found_step", -1),
        "first_iter_s": r.get("first_iter_s"),
        "last_completed_step": r.get("last_completed_step"),
    }
    results_list.append(rec)
    print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid:>4} sym_pos={sym_pos} rot_idx={rot_idx_abs} "
          f"found={r['found']} verify={verify_ok} path_len={rec['path_len']:>3} "
          f"wall={wall:.1f}s  found_step={r.get('found_step', -1)} "
          f"last_step={r.get('last_completed_step')}",
          flush=True)

    with open(PARTIAL_PATH, "w") as f:
        json.dump(results_list, f)

with open(FINAL_PATH, "w") as f:
    json.dump(results_list, f, indent=2)
print(f"\\n== pipeline done in {time.time() - t_pipeline:.1f}s, {len(results_list)} pairs ==")
'''

CELL_11_AGGREGATE = '''\
# Cell 11: Per-pid min over THIS kernel's slice of positions + submission CSV.
# Owner min-merges this CSV against the other shard's CSV (and current best)
# locally with megaminx/scripts/16_merge_rescue.py.
import csv as _csv

by_pid = {}
for r in results_list:
    by_pid.setdefault(r["pid"], []).append(r)

print()
print(f"== per-pid (min over this kernel's sym positions {SYM_POSITIONS}) ==")
print(f"{'pid':>6}  {'best_pos':>8}  {'best_rot':>8}  {'best_path':>9}  {'found/N':>10}  {'wall':>9}")
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
                              "best_sym_pos": best["sym_pos"],
                              "best_rot_idx": best["rot_idx"]}
        total_path += best["path_len"]
        n_solved += 1
        wall_max = max(r["wall_s"] for r in rs)
        print(f"{pid:>6}  {best['sym_pos']:>8}  {best['rot_idx']:>8}  {best['path_len']:>9}  "
              f"{len(valid):>4}/{len(rs):<4}  {wall_max:>9.1f}")
    else:
        final_per_pid[pid] = {"found": False}
        wall_max = max(r["wall_s"] for r in rs)
        print(f"{pid:>6}  {'-':>8}  {'-':>8}  {'-':>9}  {'0':>4}/{len(rs):<4}  {wall_max:>9.1f}")

print()
print(f"solved (any position in this slice): {n_solved}/{len(by_pid)}")
print(f"total_path (best per pid, this slice only): {total_path}")

csv_path = f"/kaggle/working/share_jax_submission_{SLICE_TAG}.csv"
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
print()
print(f"== Send share_jax_submission_{SLICE_TAG}.csv (and share_jax_final_{SLICE_TAG}.json) ==")
print(f"== back to the kernel owner for cross-shard min-merge.                            ==")
'''

# Inline shared .py files from SPMD_DIR
model_src = _file_to_cell(SPMD_DIR / "jax_model.py",
                          kill_imports=["jax_model"], rename_apply=True)
beam_single_src = _file_to_cell(SPMD_DIR / "jax_beam.py",
                                kill_imports=["jax_model", "jax_beam"], rename_apply=False)
beam_v_only_src = _file_to_cell(SPMD_DIR / "jax_beam_spmd_v_only.py",
                                kill_imports=["jax_model", "jax_beam", "jax_beam_spmd_v_only"],
                                rename_apply=False)

CELL_5_MODEL = CELL_5_MODEL + "\n" + model_src
CELL_6_BEAM_SINGLE = CELL_6_BEAM_SINGLE + "\n" + beam_single_src
CELL_7_SPMD_V_ONLY = CELL_7_SPMD_V_ONLY + "\n" + beam_v_only_src

CELLS = [
    ("markdown", CELL_0_README),
    ("code", CELL_1_CONFIG),
    ("code", CELL_2_SETUP),
    ("code", CELL_3_PUZZLE),
    ("code", CELL_4_LOAD_PT),
    ("code", CELL_5_MODEL),
    ("code", CELL_6_BEAM_SINGLE),
    ("code", CELL_7_SPMD_V_ONLY),
    ("code", CELL_8_SYM),
    ("code", CELL_9_LOAD_PARAMS),
    ("code", CELL_10_SOLVE_LOOP),
    ("code", CELL_11_AGGREGATE),
]

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

out_nb = HERE / "cayleypy-megaminx-beam-az-v4-32m-shareable-jax.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes, "
      f"{len(ALL_PID_STATES)} pid states embedded)")

# Note on slug: the canonical-looking `cayleypy-megaminx-beam-az-v4-32m-shareable-jax`
# is permanently namespace-cached at Kaggle from earlier failed push attempts
# (the initial CPU-push attempts during 64M debugging held the namespace).
# Using `cayleypy-megaminx-32m-shareable` instead -- shorter but functional.
meta = {
    "id": "artgor/cayleypy-megaminx-32m-shareable",
    "title": "Cayleypy Megaminx 32M Shareable",
    "code_file": "cayleypy-megaminx-beam-az-v4-32m-shareable-jax.ipynb",
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
