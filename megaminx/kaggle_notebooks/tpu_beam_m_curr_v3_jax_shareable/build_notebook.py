"""Builds the SHAREABLE m_curr_v3 SPMD shared-beam Kaggle TPU notebook (JAX).

This is the community-collaborator variant of cayleypy-megaminx-beam-m-curr-v3-jax.
Each collaborator forks the kernel, sets their pid chunk + K_SYM rotation,
runs on their TPU quota, and sends back the resulting CSV for min-merge.

Algorithm identical to cayleypy-megaminx-beam-m-curr-v3-jax-tpu-v5e-8 — SPMD
shared B=8M beam via `shard_map` + `lax.all_to_all`, host-side cross-rank
winner selection. Reuses the same source files from ../tpu_beam_spmd_jax/.

Output: cayleypy-megaminx-beam-shareable-jax.ipynb
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
# Megaminx beam search on Kaggle TPU v5e-8 - shareable (JAX SPMD, B=8M)

Multi-collaborator TPU beam search for the
[CayleyPy Megaminx Kaggle competition](https://www.kaggle.com/competitions/cayley-py-megaminx).
Each collaborator runs a different chunk of pids on their own TPU quota,
and we union the results back via per-pid min.

**Algorithm:** SPMD shared-beam. All 8 TPU v5e-8 cores cooperate on a single
B_GLOBAL=8M shared beam (hash-partitioned across ranks, routed via
`lax.all_to_all`), one pid at a time. Validated equivalent to
`cayleypy-megaminx-beam-m-curr-v3-jax-tpu-v5e-8`.

**Stack:** m_curr_v3 V teacher (6M) + m23_v2 Q-shortlister (12M, alpha=2)
+ m_pi_v2 policy (lambda=0.05) + sym-rotation diversity (optional).

## How to run as a collaborator

1. **Fork** this notebook on Kaggle ("Copy and Edit").
2. Dataset `artgor/megaminx-tpu-artifacts` auto-attaches.
3. **Edit Cell 1 (CONFIG)** to set your `START_PID`, `END_PID`, and
   optionally `ROT_IDX_OVERRIDE` (see "Sym-rotation split" below).
4. **Switch accelerator to TPU v5e-8** in the right sidebar.
5. Run all cells.
6. Per-iteration partial saves — even if Kaggle hits the 9h kill mid-run,
   every completed pid is recovered from `/kaggle/working/share_jax_partial.json`.
7. After completion, download `/kaggle/working/share_jax_submission.csv` and
   send to the owner for min-merging.

## Per-pid wall (TPU v5e-8 measured, single-direction)

| beam | wall per beam call |
|---|---|
| B_GLOBAL=1M  | ~80s  |
| B_GLOBAL=4M  | ~200s |
| B_GLOBAL=8M  | ~400s (default for this notebook) |

First-pid compile is ~50s on top of that. Total beam calls per pid =
`K_SYM * (2 if use_niss else 1)`.

## Recommended pid-range scopes (one TPU kernel, B_GLOBAL=8M)

`use_niss=1` roughly doubles wall per (pid, rotation) pair, so adjust scope:

| K_SYM | use_niss | beam calls / pid | safe pid count (under 9h kill) |
|---|---|---|---|
| 1 | 0 | 1 | ~60 pids (default config) |
| 1 | 1 | 2 | ~30 pids |
| 4 | 0 | 4 | ~15 pids |
| 4 | 1 | 8 | ~7 pids |

For full-1001 coverage at the default (K_SYM=1, use_niss=0) you need
~17 collaborators each doing ~60 pids. With `use_niss=1` you need
~33 collaborators each doing ~30 pids.

## NISS — non-inverse scramble solving

When `use_niss=1`, for each (pid, rotation) pair the beam runs TWICE:
once on the rotated state `S`, and once on `invert_state(S)` (the permutation
inverse). The path found in the inverse direction is reverse-then-move-inverted
to produce a valid solution for `S`; we keep the shorter of the two paths.
Costs 2x wall per pair; typical heuristic gain on megaminx is 1-3% extra
moves saved on top of forward-only beam, with the wins concentrated on pids
where the forward beam barely solves or stalls on a plateau. Verified by the
same `_verify` step the forward path goes through, so a buggy inversion can
never silently corrupt the submission.

## Sym-rotation diversity (`K_SYM`)

`K_SYM` = number of symmetry rotations to run per pid. The beam runs once per
rotation (sequentially within the kernel) and the per-pid result is the
shortest verifying path across rotations. `K_SYM` does NOT scale memory —
each rotation's beam is independent, B_GLOBAL=8M memory per rotation. It
DOES scale wall: `K=4` is roughly 4× the wall of `K=1`.

The K specific rotations are deterministic from `SYM_SEED=0`, so two
collaborators with the same `K_SYM` and `SYM_SEED` will run the same rotations.

## Suggested chunking

If you have 1 collaborator with ~7h quota at the default (`K_SYM=1, use_niss=0`):
  - kernel A: `START_PID=0,   END_PID=60`   -> 60 pids, ~7h

If you have 3 collaborators each with ~7h quota at `K_SYM=1, use_niss=0`:
  - collab A: `START_PID=0,   END_PID=60`
  - collab B: `START_PID=60,  END_PID=120`
  - collab C: `START_PID=120, END_PID=180`

If you flip `use_niss=1`, halve the pid window (~30 pids per kernel). For
`K_SYM=4, use_niss=1` the per-kernel wall is roughly 8x the default --
pick `END_PID - START_PID ~= 7`.

## What to send back

Two files from `/kaggle/working/`:
1. `share_jax_submission.csv` — the per-pid best paths in Kaggle's submission format.
2. `share_jax_final.json` — full per-pid metadata (path lengths, walls, found_steps).

Either is sufficient; the owner can reconstruct the other if needed.

## Notes for collaborators

- The 9h Kaggle TPU kernel kill is real; pick a scope that fits under it
  (~60 pids at B=8M K=1, per the table above).
- Kaggle's weekly TPU quota is 20h per account.
- If something looks off, check the per-pid output line — it prints `found`,
  `verify`, `path_len`, and `wall` per pid. `verify=True` is the integrity
  check (the path applied to the init state actually produces V0).
- Compiled JIT cache is per-shape; if you change `B_GLOBAL` or `K_SYM` mid-run
  the next pid pays the ~50s compile penalty again.
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
#   START_PID, END_PID  - pid range to solve (half-open). Pick a chunk that
#                         fits in ~7h wall (see scope table in README).
#   K_SYM               - sym-ensemble rotation count: 1 / 2 / 4 / 8.
#                         K>1 runs the beam K times per pid (sequentially)
#                         and takes the per-pid min. Wall scales linearly
#                         with K. Memory does NOT scale (B=8M fits at any K).
#   use_niss            - 0 = normal behaviour (forward beam only).
#                         1 = also solve invert_state(S) and take the shorter
#                         path. Doubles wall PER (pid, rotation) pair.
#                         Combines multiplicatively with K_SYM: total beam
#                         calls per pid = K_SYM * (2 if use_niss else 1).
#                         Typical heuristic gain: 1-3% extra moves saved.
# ============================================================================

START_PID = 0
END_PID = 8       # change this for production runs
K_SYM = 1         # 1 / 2 / 4 / 8
use_niss = 0      # 0 = forward only (default), 1 = also try inverse direction

B_GLOBAL = 8 * 1024 * 1024   # 8M shared beam (verified ceiling on TPU v5e-8)
ALPHA_QSHORT = 2
LAMBDA_POLICY = 0.05
INTERNAL_BS = 32768
NUM_STEPS = 120

# ============================================================================
# Do not edit below this line
# ============================================================================
'''

CELL_2_SETUP = '''\
# Cell 2: Setup. JAX_ENABLE_X64 must be set BEFORE `import jax`.
import sys, os, time, json
os.environ["JAX_ENABLE_X64"] = "True"
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
assert use_niss in (0, 1), f"use_niss must be 0 or 1; got {use_niss}"
assert B_GLOBAL % N_DEVICES == 0
B_LOCAL = B_GLOBAL // N_DEVICES
K_PER_PEER = (ALPHA_QSHORT * B_LOCAL) // N_DEVICES
N_PIDS = END_PID - START_PID
NISS_FACTOR = 2 if use_niss else 1
print(f"\\n== CONFIG ==")
print(f"pid range: [{START_PID}, {END_PID})  ({N_PIDS} pids)")
print(f"K_SYM: {K_SYM}   use_niss: {use_niss}  (beam calls per pid = K_SYM * {NISS_FACTOR})")
print(f"B_GLOBAL: {B_GLOBAL:,}  B_LOCAL: {B_LOCAL:,}  K_PER_PEER: {K_PER_PEER:,}")
print(f"NUM_STEPS: {NUM_STEPS}  alpha: {ALPHA_QSHORT}  lambda_policy: {LAMBDA_POLICY}")

_per_pid = {1*1024*1024: 80, 4*1024*1024: 200, 8*1024*1024: 400}.get(B_GLOBAL, 400)
_n_pairs = N_PIDS * K_SYM
_n_beam_calls = _n_pairs * NISS_FACTOR
_proj = 0.1 + (_n_beam_calls * _per_pid) / 3600
print(f"estimated wall: {_proj:.1f}h (compile + {_n_beam_calls} beam calls × ~{_per_pid}s)")
if _proj > 8.5:
    print(f"WARNING: estimated wall {_proj:.1f}h exceeds 8.5h target.")
    print(f"  Kaggle kills TPU kernels around 9h. Consider reducing pid range.")
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

# NISS support: per-move-index inverse. Megaminx convention: "name" and "-name"
# are inverse generators. INV_MOVE_IDX[i] = index of inverse of MOVE_NAMES[i].
INV_MOVE_IDX = np.full(N_GEN, -1, dtype=np.int32)
_move_name_to_idx = {{n: i for i, n in enumerate(MOVE_NAMES)}}
for _i, _nm in enumerate(MOVE_NAMES):
    _inv = _nm[1:] if _nm.startswith("-") else "-" + _nm
    if _inv not in _move_name_to_idx:
        raise ValueError(f"missing inverse generator for {{_nm}} (expected {{_inv}})")
    INV_MOVE_IDX[_i] = _move_name_to_idx[_inv]
assert (INV_MOVE_IDX >= 0).all() and INV_MOVE_IDX[INV_MOVE_IDX].tolist() == list(range(N_GEN))
print(f"INV_MOVE_IDX built: {{N_GEN}} move pairs verified involutive")

_ALL_PID_STATES = json.loads({ALL_PID_STATES_LITERAL!r})
ALL_PID_STATES = {{int(k): v for k, v in _ALL_PID_STATES.items()}}
TEST_PIDS = list(range(START_PID, END_PID))
PID_STATES = {{pid: ALL_PID_STATES[pid] for pid in TEST_PIDS}}
print(f"selected {{len(TEST_PIDS)}} pids: [{{TEST_PIDS[0]}}..{{TEST_PIDS[-1]}}]")
'''

CELL_4_LOAD_PT = '''\
# Cell 4: Load m_curr_v3 + m23_v2 + m_pi_v2 from the Kaggle dataset.
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

m_curr_v3_path = dataset_root / "m_curr_v3_epoch_0499.pt"
m23v2_path = dataset_root / "m23_v2_epoch_0499.pt"
m_pi_path = dataset_root / "m_pi_v2_epoch_0199.pt"
for p in (m_curr_v3_path, m23v2_path, m_pi_path):
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}")
'''

CELL_5_MODEL = '''\
# Cell 5: Pure-JAX inference (inlined from jax_model.py).
'''  # body inlined below

CELL_6_BEAM_SINGLE = '''\
# Cell 6: Single-device JAX beam helpers (inlined from jax_beam.py).
'''  # body inlined below

CELL_7_SPMD = '''\
# Cell 7: SPMD shared-beam (inlined from jax_beam_spmd.py).
'''  # body inlined below

CELL_8_SYM = '''\
# Cell 8: Sym ensemble setup. Deterministic K_SYM rotations from SYM_SEED.
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
chosen_rot_idxs = [identity_idx]
if K_SYM > 1:
    chosen_rot_idxs += list(rng_sym.choice(other_idxs, size=K_SYM - 1, replace=False).tolist())
print(f"K_SYM={K_SYM}, SYM_SEED={SYM_SEED} chose rotation idxs: {chosen_rot_idxs}")

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
# Cell 9: Load all three model checkpoints + build JAX params dicts + mesh.
print("loading m_curr_v3 (V teacher) ...")
teacher_params = load_params_from_pt(m_curr_v3_path, hidden_dims=(2048, 512))
print(f"  V params: {num_params(teacher_params):,}")
print("loading m23_v2 (Q shortlister) ...")
student_params = load_params_from_pt(m23v2_path, hidden_dims=(2048, 1024))
print(f"  Q params: {num_params(student_params):,}")
print("loading m_pi_v2 (policy) ...")
policy_params = load_params_from_pt(m_pi_path, hidden_dims=(2048, 512))
print(f"  policy params: {num_params(policy_params):,}")

hash_vec_np = make_hash_vec(STATE_SIZE, seed=0)
hash_vec = jnp.asarray(hash_vec_np)

mesh = make_mesh(devices)
print(f"mesh: {mesh}")
'''

CELL_10_SOLVE_LOOP = '''\
# Cell 10: SPMD solve loop - 1 pid at a time, all 8 cores cooperating.
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
    s0 = PID_STATES[pid]
    s_to_solve = apply_rotation_state(s0, rot["R"], rot["R_inv"])

    t_iter = time.time()

    # --- Forward direction: solve s_to_solve directly ---
    t0 = time.time()
    try:
        r_fwd = beam_solve_qshort_spmd(
            list(s_to_solve), teacher_params, student_params, policy_params,
            all_moves, V0, hash_vec, mesh,
            B_local=B_LOCAL, K_per_peer=K_PER_PEER,
            n_gen=N_GEN, state_size=STATE_SIZE,
            num_steps=NUM_STEPS, lambda_policy=LAMBDA_POLICY,
            dtype=jnp.bfloat16, internal_bs=INTERNAL_BS,
        )
    except Exception as e:
        import traceback
        print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid} rot={rot_i} FWD EXCEPTION: {e}")
        traceback.print_exc()
        results_list.append({
            "pid": pid, "rot_idx": rot_i, "found": False, "verify_ok": False,
            "path_len": 0, "path_idx_orig": [], "wall_s": time.time() - t0,
            "direction": "fwd", "error": str(e),
        })
        continue
    wall_fwd = time.time() - t0

    # --- NISS: also solve invert_state(s_to_solve), invert the path back ---
    r_inv = None
    inv_path_rotated = None
    wall_inv = 0.0
    if use_niss:
        s_inv = np.argsort(np.asarray(s_to_solve, dtype=np.int32)).tolist()
        t1 = time.time()
        try:
            r_inv = beam_solve_qshort_spmd(
                s_inv, teacher_params, student_params, policy_params,
                all_moves, V0, hash_vec, mesh,
                B_local=B_LOCAL, K_per_peer=K_PER_PEER,
                n_gen=N_GEN, state_size=STATE_SIZE,
                num_steps=NUM_STEPS, lambda_policy=LAMBDA_POLICY,
                dtype=jnp.bfloat16, internal_bs=INTERNAL_BS,
            )
        except Exception as e:
            import traceback
            print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid} rot={rot_i} INV EXCEPTION (continuing with fwd only): {e}")
            traceback.print_exc()
            r_inv = {"found": False, "path_idx": [], "found_step": -1}
        wall_inv = time.time() - t1
        if r_inv.get("found"):
            # Path solves s_inv. Invert it (reverse + swap each move with its inverse)
            # to get a path that solves s_to_solve (still in the rotated frame).
            inv_path_rotated = [int(INV_MOVE_IDX[m]) for m in reversed(list(r_inv["path_idx"]))]

    # --- Build candidate list, un-rotate, and verify each before picking min ---
    # Verifying both means a buggy inversion can never drop a valid fwd path:
    # we keep the shorter VERIFIED candidate, falling back to fwd if inv breaks.
    conj_idx = rot["conj_idx"]
    raw_candidates = []
    if r_fwd.get("found"):
        raw_candidates.append(("fwd", list(r_fwd["path_idx"]), r_fwd.get("found_step", -1)))
    if inv_path_rotated is not None:
        raw_candidates.append(("inv", inv_path_rotated, r_inv.get("found_step", -1)))

    verified_candidates = []
    for tag, rot_path, fstep in raw_candidates:
        orig = [conj_idx[m] for m in rot_path]
        if _verify(s0, orig):
            verified_candidates.append((tag, rot_path, orig, fstep))
        else:
            print(f"  WARN pid={pid} rot={rot_i} {tag} path failed verify ({len(orig)} moves) - dropping")

    if verified_candidates:
        direction, best_rot_path, path_orig, found_step = min(
            verified_candidates, key=lambda x: len(x[1])
        )
        found_any = True
        verify_ok = True
    else:
        direction, best_rot_path, path_orig, found_step = "none", [], None, -1
        found_any = False
        verify_ok = False

    wall = time.time() - t_iter
    rec = {
        "pid": pid, "rot_idx": rot_i,
        "found": found_any, "verify_ok": verify_ok,
        "path_len": len(path_orig) if path_orig is not None else 0,
        "path_idx_orig": path_orig if verify_ok else [],
        "path_idx_rotated": best_rot_path,
        "direction": direction,
        "wall_s": wall,
        "wall_fwd_s": wall_fwd,
        "wall_inv_s": wall_inv,
        "found_step": found_step,
        "fwd_found": bool(r_fwd.get("found")),
        "fwd_path_len": len(r_fwd["path_idx"]) if r_fwd.get("found") else 0,
        "inv_found": bool(r_inv.get("found")) if r_inv is not None else None,
        "inv_path_len": len(inv_path_rotated) if inv_path_rotated is not None else 0,
        "first_iter_s": r_fwd.get("first_iter_s"),
    }
    results_list.append(rec)
    _inv_str = f" inv={rec['inv_path_len']:>3}" if use_niss else ""
    print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid:>4} rot={rot_i} "
          f"found={found_any} verify={verify_ok} "
          f"fwd={rec['fwd_path_len']:>3}{_inv_str} -> {direction} {rec['path_len']:>3} "
          f"wall={wall:.1f}s (fwd={wall_fwd:.1f}s inv={wall_inv:.1f}s)",
          flush=True)

    # Per-iteration partial save (kernel-kill recovery).
    with open("/kaggle/working/share_jax_partial.json", "w") as f:
        json.dump(results_list, f)

with open("/kaggle/working/share_jax_final.json", "w") as f:
    json.dump(results_list, f, indent=2)
print(f"\\n== pipeline done in {time.time() - t_pipeline:.1f}s, {len(results_list)} pairs ==")
'''

CELL_11_AGGREGATE = '''\
# Cell 11: Per-pid min over rotations + submission CSV (the file to send back).
import csv as _csv

by_pid = {}
for r in results_list:
    by_pid.setdefault(r["pid"], []).append(r)

print()
print("== per-pid (min over rotations in this kernel) ==")
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
print(f"solved (any rotation): {n_solved}/{len(by_pid)}")
print(f"total_path (best per pid): {total_path}")

csv_path = "/kaggle/working/share_jax_submission.csv"
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
print("== Send `share_jax_submission.csv` (and optionally `share_jax_final.json`) ==")
print("== back to the kernel owner for min-merging.                              ==")
'''

# Inline shared .py files from SPMD_DIR
model_src = _file_to_cell(SPMD_DIR / "jax_model.py",
                          kill_imports=["jax_model"], rename_apply=True)
beam_single_src = _file_to_cell(SPMD_DIR / "jax_beam.py",
                                kill_imports=["jax_model", "jax_beam"], rename_apply=False)
beam_spmd_src = _file_to_cell(SPMD_DIR / "jax_beam_spmd.py",
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

out_nb = HERE / "cayleypy-megaminx-beam-shareable-jax.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes, "
      f"{len(ALL_PID_STATES)} pid states embedded)")

meta = {
    "id": "artgor/cayleypy-megaminx-beam-shareable-jax",
    "title": "Cayleypy Megaminx Beam Shareable JAX",   # plain title -> slug matches `id`,
                                                       # avoids the "tpu-" substring
                                                       # which triggers Kaggle's
                                                       # "Notebook not found" error in
                                                       # CPU-mode pushes.
    "code_file": "cayleypy-megaminx-beam-shareable-jax.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,   # owner flips to False in the Kaggle UI when ready to share
    "enable_gpu": False,
    "enable_tpu": True,   # If you hit "Maximum batch TPU session count of 1
                          # reached" on a re-push, set this to False
                          # temporarily (Kaggle enforces 1 active TPU session
                          # per account; CPU-mode pushes still update the
                          # kernel source). Collaborators who fork this kernel
                          # set their accelerator to TPU v5e-8 in the UI.
    "enable_internet": True,
    "dataset_sources": ["artgor/megaminx-tpu-artifacts"],
    "competition_sources": [],
    "kernel_sources": [],
}
out_meta = HERE / "kernel-metadata.json"
out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {out_meta}")
