"""Builds the m_curr_v3 SPMD shared-beam Kaggle TPU notebook (JAX).

Production version of the SPMD shared-beam approach (sibling of
cayleypy-megaminx-beam-spmd-jax), named after the production V model
(m_curr_v3) for clarity. Same algorithm — 1 pid at a time with all 8 TPU
v5e-8 cores cooperating on a single B_GLOBAL=8M shared beam via
`shard_map` + `lax.all_to_all`, host-side cross-rank winner selection
(see `spmd_jax_recovery.md` for why the in-body reduce is skipped).

Reuses `jax_model.py`, `jax_beam.py`, and `jax_beam_spmd.py` from
../tpu_beam_spmd_jax/ — single source of truth.

Defaults configured for production runs: B_GLOBAL=8M, K_SYM=1, NUM_STEPS=120.
The `ROT_IDX_OVERRIDE` knob lets you split a K_SYM=N sym-ensemble across
N kernels (each running one rotation), then min-merge their CSVs.

Output: cayleypy-megaminx-beam-m-curr-v3-jax.ipynb
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

# --- Inputs ---------------------------------------------------------------

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
    src = path.read_text(encoding="utf-8")
    src = _strip_module_imports(src, kill_imports)
    if rename_apply:
        src = re.sub(r"^def apply\(", "def model_apply(", src, flags=re.MULTILINE)
    return src


# --- Cell sources ---------------------------------------------------------

CELL_0_README = '''\
# Megaminx beam search on Kaggle TPU v5e-8 - m_curr_v3 SPMD shared-beam (JAX)

Production SPMD shared-beam tool for the m_curr_v3 + m23_v2 + m_pi_v2 stack.
8 TPU cores cooperate on a single B_GLOBAL=8M beam (hash-partitioned across
ranks, routed via `lax.all_to_all`). One pid at a time.

## When to use vs. the data-parallel notebook

| this (SPMD shared-beam) | data-parallel (`pmap`) sibling |
|---|---|
| 1 pid, 8 cores cooperating, B up to 8M shared | 8 pids in parallel, B=1M each |
| Better beam diversity per pid | Better wall throughput across pids |
| For hard-tail rescue (top-200 longest paths) | For full-1001 production sweeps |

## Stack

m_curr_v3 (V teacher, 6M) + m23_v2 (Q-shortlister, 12M, alpha=2) + m_pi_v2
(policy, lambda=0.05). Same models the project has used since the 78,029
submission.

## Multi-kernel sym-rotation split

To run K_SYM=N rotations across N kernels for parallelism / quota fit:

1. Set `K_SYM=N` in any kernel; note the printed `K_SYM=N, SYM_SEED=0
   auto-selection: [...]` line — those are the N rotation indices that
   a single K_SYM=N run would have used.
2. Push N separate kernels. Each kernel sets `ROT_IDX_OVERRIDE` to one of
   those indices (same SYM_SEED across kernels so the rotations match).
3. After all kernels finish, min-merge the N output CSVs per pid.

## Kernel scope guidance (TPU v5e-8)

Per-pid wall at B=8M is ~400s steady state, ~50s first-pid compile.

| scope | wall |
|---|---|
| 8 pids smoke | ~1 h |
| top-50 hardest | ~5.5 h |
| top-100 hardest | ~11 h (split across 2 kernels) |
| full-1001 | infeasible at B=8M in this notebook — use data-parallel sibling |

## How to run

1. Fork on Kaggle ("Copy and Edit").
2. Dataset `artgor/megaminx-tpu-artifacts` auto-attaches.
3. Edit Cell 1 (CONFIG) — pid range, K_SYM, ROT_IDX_OVERRIDE.
4. Switch accelerator to TPU v5e-8.
5. Run all cells.
6. Download `/kaggle/working/m_curr_v3_jax_submission.csv` after completion.

## Caveats

- `B_GLOBAL=8M` works on v5e-8 (v9 of the sibling SPMD notebook verified it).
  Going higher (B=16M) OOMs at runtime; the 16 GB HBM per chip is the ceiling.
- Per-iteration partial save means kernel-kill at 9h still recovers all
  completed pids.
'''

PUZZLE_INFO_LITERAL = json.dumps(puzzle_info, separators=(",", ":"))
ALL_PID_STATES_LITERAL = json.dumps(ALL_PID_STATES, separators=(",", ":"))

CELL_1_CONFIG = f'''\
# ============================================================================
# CONFIG - EDIT THIS CELL, then Run All
# ============================================================================
#
# SPMD shared-beam (1 pid at a time, all 8 cores cooperating on B=8M shared).
#
# Main knobs:
#   START_PID, END_PID       - pid range (half-open). Default: 8-pid smoke.
#   K_SYM                    - sym rotations per pid (1, 2, 4, or 8).
#   ROT_IDX_OVERRIDE         - int [0, 360) forces a specific rotation idx;
#                              None = use K_SYM auto-selection.
#                              Use this to split K_SYM=N across N kernels.
# ============================================================================

START_PID = 0
END_PID = 8           # 8-pid smoke
K_SYM = 1
ROT_IDX_OVERRIDE = None  # int [0, 360); None = K_SYM-based auto-selection

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
# Direct assignment (NOT setdefault — Kaggle may pre-set the env var).
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
assert N_DEVICES == 8, f"Expected 8 TPU cores; got {N_DEVICES}"

assert 0 <= START_PID < END_PID <= ''' + str(N_TOTAL_PIDS) + ''', \\
    f"invalid pid range: START_PID={START_PID}, END_PID={END_PID}"
assert K_SYM in (1, 2, 4, 8), f"K_SYM must be 1/2/4/8; got {K_SYM}"
assert B_GLOBAL % N_DEVICES == 0
B_LOCAL = B_GLOBAL // N_DEVICES
K_PER_PEER = (ALPHA_QSHORT * B_LOCAL) // N_DEVICES
N_PIDS = END_PID - START_PID
print(f"\\n== CONFIG ==")
print(f"pid range: [{START_PID}, {END_PID})  ({N_PIDS} pids)")
print(f"K_SYM: {K_SYM}  ROT_IDX_OVERRIDE: {ROT_IDX_OVERRIDE}")
print(f"B_GLOBAL: {B_GLOBAL:,}  B_LOCAL: {B_LOCAL:,}  K_PER_PEER: {K_PER_PEER:,}")
print(f"alpha: {ALPHA_QSHORT}  lambda_policy: {LAMBDA_POLICY}")
print(f"NUM_STEPS: {NUM_STEPS}  INTERNAL_BS: {INTERNAL_BS}")
'''

CELL_3_PUZZLE = f'''\
# Cell 3: Puzzle data + all pid states embedded.
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
# Cell 4: Load m_curr_v3 + m23_v2 + m_pi_v2 .pt files from the Kaggle dataset.
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
    raise SystemExit("megaminx-tpu-artifacts dataset not found in /kaggle/input")
print("dataset:", dataset_root)

m_curr_v3_path = dataset_root / "m_curr_v3_epoch_0499.pt"
m23v2_path = dataset_root / "m23_v2_epoch_0499.pt"
m_pi_path = dataset_root / "m_pi_v2_epoch_0199.pt"
for p in (m_curr_v3_path, m23v2_path, m_pi_path):
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}")
'''

CELL_5_MODEL = '''\
# Cell 5: Pure-JAX inference (inlined from jax_model.py — single source of truth).
'''  # body inlined below

CELL_6_BEAM_SINGLE = '''\
# Cell 6: Single-device JAX beam helpers (inlined from jax_beam.py).
'''  # body inlined below

CELL_7_SPMD = '''\
# Cell 7: SPMD shared-beam (inlined from jax_beam_spmd.py — host-side cross-rank
# winner selection; in-body pmin/psum was buggy on TPU, see spmd_jax_recovery.md).
'''  # body inlined below

CELL_8_SYM = '''\
# Cell 8: Sym ensemble setup with ROT_IDX_OVERRIDE knob for kernel splits.
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

# Print the K_SYM auto-selection FIRST (deterministic from SYM_SEED) so callers
# splitting K_SYM across N kernels know which indices to plug into ROT_IDX_OVERRIDE.
rng_sym = np.random.default_rng(SYM_SEED)
other_idxs = [i for i in range(rotations_np.shape[0]) if i != identity_idx]
auto_chosen = [identity_idx]
if K_SYM > 1:
    auto_chosen += list(rng_sym.choice(other_idxs, size=K_SYM - 1, replace=False).tolist())
print(f"K_SYM={K_SYM}, SYM_SEED={SYM_SEED} auto-selection: {auto_chosen}")

if ROT_IDX_OVERRIDE is not None:
    assert 0 <= int(ROT_IDX_OVERRIDE) < rotations_np.shape[0], (
        f"ROT_IDX_OVERRIDE={ROT_IDX_OVERRIDE} out of range")
    chosen_rot_idxs = [int(ROT_IDX_OVERRIDE)]
    print(f"ROT_IDX_OVERRIDE set -> using ONLY rotation idx {ROT_IDX_OVERRIDE}")
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
# Cell 10: SPMD solve loop — 1 pid at a time, all 8 cores cooperating.
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

    t0 = time.time()
    try:
        r = beam_solve_qshort_spmd(
            list(s_to_solve), teacher_params, student_params, policy_params,
            all_moves, V0, hash_vec, mesh,
            B_local=B_LOCAL, K_per_peer=K_PER_PEER,
            n_gen=N_GEN, state_size=STATE_SIZE,
            num_steps=NUM_STEPS, lambda_policy=LAMBDA_POLICY,
            dtype=jnp.bfloat16, internal_bs=INTERNAL_BS,
        )
    except Exception as e:
        import traceback
        print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid} rot={rot_i} EXCEPTION: {e}")
        traceback.print_exc()
        results_list.append({
            "pid": pid, "rot_idx": rot_i, "found": False, "verify_ok": False,
            "path_len": 0, "path_idx_orig": [], "wall_s": time.time() - t0, "error": str(e),
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
    }
    results_list.append(rec)
    print(f"[{i+1}/{len(PID_ROT_PAIRS)}] pid={pid:>4} rot={rot_i} "
          f"found={r['found']} verify={verify_ok} path_len={rec['path_len']:>3} "
          f"wall={wall:.1f}s  found_step={r.get('found_step', -1)}",
          flush=True)

    # Per-iteration partial save (kernel-kill recovery).
    with open("/kaggle/working/m_curr_v3_jax_partial.json", "w") as f:
        json.dump(results_list, f)

with open("/kaggle/working/m_curr_v3_jax_final.json", "w") as f:
    json.dump(results_list, f, indent=2)
print(f"\\n== pipeline done in {time.time() - t_pipeline:.1f}s, {len(results_list)} pairs ==")
'''

CELL_11_AGGREGATE = '''\
# Cell 11: Per-pid min over rotations + submission CSV.
import csv as _csv

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
print(f"solved (any rotation): {n_solved}/{len(by_pid)}")
print(f"total_path (best per pid): {total_path}")

csv_path = "/kaggle/working/m_curr_v3_jax_submission.csv"
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

# Inline shared .py files from SPMD_DIR (single source of truth).
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

out_nb = HERE / "cayleypy-megaminx-beam-m-curr-v3-jax.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes, "
      f"{len(ALL_PID_STATES)} pid states embedded)")

meta = {
    "id": "artgor/cayleypy-megaminx-beam-m-curr-v3-jax-tpu-v5e-8",
    "title": "cayleypy megaminx beam m_curr_v3 (JAX, TPU v5e-8)",
    "code_file": "cayleypy-megaminx-beam-m-curr-v3-jax.ipynb",
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
