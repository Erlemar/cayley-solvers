"""Builds the SHAREABLE AZ v4 V-only SPMD shared-beam Kaggle TPU notebook (JAX).

Sibling of cayleypy-megaminx-beam-shareable-jax, but built around the
AZ v4 V model (`m_az_v4_v_only.pt`) with NO qshort. AZ v4 V is the first
6M-param model in this project to break sub-89 strat-5 standalone (51/51 /
mean 87.5 vs m05 baseline's 89.4). The breakthrough finding: m23_v2 qshort
REGRESSES AZ v4 V by ~240 moves on strat-5 because m23_v2 was distilled
from m05's V landscape, not AZ's. So this notebook drops qshort entirely
and runs a plain V-only beam.

Algorithm: SPMD shared-beam at B_GLOBAL=8M (same as the m_curr_v3 shareable),
all 8 TPU v5e-8 cores cooperating on one pid at a time. Each step runs V
on ALL B_LOCAL * N_GEN children (not just qshort's top-αB), so per-pid wall
is ~6× the m_curr_v3 shareable at the same B_GLOBAL.

Reuses jax_model.py + jax_beam.py + jax_beam_spmd_v_only.py from
../tpu_beam_spmd_jax/ — single source of truth.

Output: cayleypy-megaminx-beam-az-v4-shareable-jax.ipynb
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
# Megaminx beam search on Kaggle TPU v5e-8 - AZ v4 V-only shareable (JAX SPMD, B=8M)

Multi-collaborator TPU beam search for the
[CayleyPy Megaminx Kaggle competition](https://www.kaggle.com/competitions/cayley-py-megaminx).
Each collaborator runs a different chunk of pids on their own TPU quota,
and we union the results back via per-pid min.

**Algorithm:** SPMD shared-beam. All 8 TPU v5e-8 cores cooperate on a single
B_GLOBAL=8M shared beam (hash-partitioned, routed via `lax.all_to_all`),
one pid at a time. V model ranks ALL B_LOCAL · N_GEN children per step —
no qshort prefilter.

**Stack:** AZ v4 V-only (6M params, `m_az_v4_v_only.pt`).

## Why V-only (no qshort)

The sibling shareable (`cayleypy-megaminx-beam-shareable-jax`) uses
m_curr_v3 V + m23_v2 Q-shortlister (qshort). **m23_v2 was distilled from
m05's V landscape, not AZ v4's.** Strat-5 measurements (2026-05-11):

| stack                                           | total | mean |
|---|---|---|
| AZ v4 V-only                                    | **4,465** | **87.5** ← clean win |
| AZ v4 V + AZ π + m23_v2 qshort                  | 4,621 | 90.6 |
| AZ v4 V + m23_v2 qshort, no policy              | 4,705 | 92.3 |

m23_v2 qshort regresses AZ v4 V by ~240 moves. So this notebook drops
qshort and uses AZ v4 V to rank ALL B_LOCAL · N_GEN neighbors directly.

V-only has ~6× the per-step compute of qshort at the same B (V on
B · N_GEN children vs. student/policy on B + teacher on αB), so per-pid
wall at B=8M is ~6× the m_curr_v3 shareable.

## Per-pid wall (TPU v5e-8 projected, B_GLOBAL=8M)

| beam | wall per pid |
|---|---|
| B_GLOBAL = 1M | ~400-500s |
| B_GLOBAL = 4M | ~1200-1500s |
| B_GLOBAL = 8M | ~2400-2800s (default — ~40-45 min per pid) |

First-pid compile is ~1 min on top of that.

## Recommended pid-range scopes (one TPU kernel, B_GLOBAL=8M, K_SYM=1)

| scope | wall (incl. ~5 min compile) |
|---|---|
| 4 pids (smoke)  | ~3 h |
| 8 pids          | ~6 h |
| 10 pids         | ~7 h (close to 9h kernel kill) |

For full-1001 coverage at B=8M K=1, ~100 collaborators × 10 pids each.
That's a lot — most collaborators should pick a *small chunk of the
hardest pids* (the ones where AZ v4 V might find shorter paths than the
current best). If you want broader coverage with the same wall budget,
consider reducing `B_GLOBAL` to 1M (then ~50-60 pids per kernel — see the
table above).

## How to run as a collaborator

1. **Fork** this notebook on Kaggle ("Copy and Edit").
2. The dataset `artgor/megaminx-tpu-artifacts` auto-attaches; it must
   contain `m_az_v4_v_only.pt` (Cell 5 errors clearly if missing).
3. **Edit Cell 1 (CONFIG)** to set `START_PID`, `END_PID`, `K_SYM`.
4. **Switch accelerator to TPU v5e-8** in the right sidebar.
5. Run all cells.
6. Per-iteration partial saves — even if Kaggle hits the 9h kill mid-run,
   every completed pid is recovered from `/kaggle/working/share_jax_partial.json`.
7. After completion, download `/kaggle/working/share_jax_submission.csv` and
   send back to the owner for min-merging.

## Notes for collaborators

- The 9h Kaggle TPU kernel kill is real; pick a scope that fits under it
  (~10 pids at B=8M K=1, per the table above).
- Kaggle's weekly TPU quota is 20h per account.
- If you change `B_GLOBAL` or `K_SYM` mid-run the next pid pays the ~1 min
  compile penalty again (per-shape JIT cache).
- `verify=True` in the per-pid output line is the integrity check (path
  applied to init state actually produces V0). Should always be True for
  found pids; report if you see verify=False.
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
#   START_PID, END_PID  - pid range to solve (half-open). At B=8M V-only,
#                         ~10 pids fits one 9h kernel.
#   K_SYM               - sym-ensemble rotation count: 1 / 2 / 4 / 8.
#                         K>1 runs the beam K times per pid (sequentially)
#                         and takes the per-pid min. Wall scales linearly
#                         with K. Memory does NOT scale.
# ============================================================================

START_PID = 0
END_PID = 4       # 4-pid smoke at B=8M K=1 (~3h). Change for production.
K_SYM = 1         # 1 / 2 / 4 / 8

B_GLOBAL = 8 * 1024 * 1024   # 8M shared beam (verified ceiling on TPU v5e-8)
ALPHA_QSHORT = 2             # used for K_PER_PEER calc; same role as in qshort
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
assert B_GLOBAL % N_DEVICES == 0
B_LOCAL = B_GLOBAL // N_DEVICES
K_PER_PEER = (ALPHA_QSHORT * B_LOCAL) // N_DEVICES
N_PIDS = END_PID - START_PID
print(f"\\n== CONFIG ==")
print(f"pid range: [{START_PID}, {END_PID})  ({N_PIDS} pids)")
print(f"K_SYM: {K_SYM}")
print(f"B_GLOBAL: {B_GLOBAL:,}  B_LOCAL: {B_LOCAL:,}  K_PER_PEER: {K_PER_PEER:,}")
print(f"NUM_STEPS: {NUM_STEPS}  alpha: {ALPHA_QSHORT}")

# V-only is ~6x qshort per-step compute (V on B*N_GEN children vs. qshort's
# student/teacher on B + αB). Scale wall estimates accordingly.
_per_pid = {1*1024*1024: 450, 4*1024*1024: 1300, 8*1024*1024: 2600}.get(B_GLOBAL, 2600)
_n_pairs = N_PIDS * K_SYM
_proj = 0.1 + (_n_pairs * _per_pid) / 3600
print(f"estimated wall: {_proj:.1f}h (compile + {_n_pairs} pairs × ~{_per_pid}s)")
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
# Cell 9: Load V checkpoint, build JAX params dict, set up mesh.
print("loading AZ v4 V (V teacher, V-only) ...")
v_params = load_params_from_pt(v_path, hidden_dims=(2048, 512))
print(f"  V params: {num_params(v_params):,}")

hash_vec_np = make_hash_vec(STATE_SIZE, seed=0)
hash_vec = jnp.asarray(hash_vec_np)

mesh = make_mesh(devices)
print(f"mesh: {mesh}")
'''

CELL_10_SOLVE_LOOP = '''\
# Cell 10: SPMD V-only solve loop — 1 pid at a time, all 8 cores cooperating.
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
        r = beam_solve_v_only_spmd(
            list(s_to_solve), v_params,
            all_moves, V0, hash_vec, mesh,
            B_local=B_LOCAL, K_per_peer=K_PER_PEER,
            n_gen=N_GEN, state_size=STATE_SIZE,
            num_steps=NUM_STEPS, dtype=jnp.bfloat16,
            internal_bs=INTERNAL_BS,
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

    with open("/kaggle/working/share_jax_partial.json", "w") as f:
        json.dump(results_list, f)

with open("/kaggle/working/share_jax_final.json", "w") as f:
    json.dump(results_list, f, indent=2)
print(f"\\n== pipeline done in {time.time() - t_pipeline:.1f}s, {len(results_list)} pairs ==")
'''

CELL_11_AGGREGATE = '''\
# Cell 11: Per-pid min over rotations + submission CSV (send back).
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

out_nb = HERE / "cayleypy-megaminx-beam-az-v4-shareable-jax.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes, "
      f"{len(ALL_PID_STATES)} pid states embedded)")

meta = {
    "id": "artgor/cayleypy-megaminx-beam-az-v4-shareable-jax",
    "title": "Cayleypy Megaminx Beam AZ v4 Shareable JAX",   # plain title -> slug
                                                             # matches `id`, no
                                                             # "tpu-" substring.
    "code_file": "cayleypy-megaminx-beam-az-v4-shareable-jax.ipynb",
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
