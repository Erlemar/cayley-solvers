"""Builds the SHAREABLE TPU beam-search notebook (B=2M, K=2).

Designed for fan-out across multiple collaborators' Kaggle TPU quotas:
  * Cell 1 of the generated notebook is a CONFIG cell where each collaborator
    edits START_PID / END_PID for their chunk. Recommended chunk: ~340 pids
    (estimated 8h kernel wall; full-1001 covered by 3 chunks).
  * All 1001 pid states are embedded; the config cell slices into them.
  * B=2M K=2 is an untried slice of the (K, B) plane vs our v16-v19b sweeps.
    Same-compute alternative to B=4M K=1 with HBM headroom and minimal sym
    diversity preserved.

Output: cayleypy-tpu-beam-b2m-k2.ipynb in the same directory.
Run via: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_b2m_k2/build_notebook.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # megaminx/
HERE = Path(__file__).resolve().parent

# --- Inputs ---------------------------------------------------------------

puzzle_info = json.loads((ROOT / "data" / "puzzle_info.json").read_text())

# Embed ALL 1001 pid states so the config cell can slice at notebook runtime.
rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
ALL_PID_STATES = {}
for pid in range(len(rows)):
    s = [int(x) for x in rows[pid]["initial_state"].split(",")]
    ALL_PID_STATES[pid] = s
N_TOTAL_PIDS = len(ALL_PID_STATES)

# --- Cell sources ---------------------------------------------------------

CELL_0_README = '''\
# Megaminx beam search on Kaggle TPU — B=1M (shareable, K configurable)

Multi-collaborator TPU beam search for the
[CayleyPy Megaminx Kaggle competition](https://www.kaggle.com/competitions/cayley-py-megaminx).
Each collaborator runs a different chunk of pids on their own TPU quota,
and we union the results back via per-pid min.

**Production stack:**
m05 V teacher (6M params) + m23_v2 sym-aware Q-shortlister (12.4M, alpha=2)
+ K-sym-ensemble (configurable, default K=4) + B=1M beam, 120 steps.

## How to run as a collaborator

1. Fork this notebook on Kaggle ("Copy and Edit").
2. The dataset `artgor/megaminx-tpu-artifacts` (public) is auto-attached
   via kernel metadata — no manual step.
3. **Edit Cell 1 (CONFIG) to set your pid range and K_SYM.** Defaults are
   `START_PID=0, END_PID=50, K_SYM=4` (a ~2.5h smoke). Recommended full-run
   chunk: ~375 pids at K=4 for an ~8.3h wall (margin under the 9h kill).
4. **Switch accelerator to TPU v3-8** in the right sidebar (the kernel
   was pushed in CPU mode because owner's TPU quota was exhausted).
5. Run all cells. Per-iteration partial saves mean even if Kaggle hits the
   9h kill mid-run, every completed (pid, rotation) pair is recovered.
6. After completion, download `/kaggle/working/tpu_qshort_sym_results.json`
   and `/kaggle/working/tpu_qshort_sym_submission.csv` and send to the
   owner for merging.

## Why B=1M (K configurable)

Prior sweeps + tried/OOM'd:
- v16: K=4 B=131k ✓
- v17: K=8 B=262k ✓
- v19a-fix / v19b: K=4 B=1M ✓ (proven safe — that's why we're at B=1M here)
- B=2M K=2: OOM at runtime (480 MB shortlist alloc)
- B=1.5M K=any: OOM at XLA program load (5.58 GB workspace > free HBM
  on the chip when 2 ranks share it via xmp.spawn)

B=1M is the only verified-safe value above v17. K is configurable per
collaborator: K=2 covers the most pids per kernel (and is an UNTRIED slice
vs v19b's K=4), K=4 matches v19b exactly (will mostly duplicate existing
wins), K=8 maxes sym diversity (also untried at B=1M).

**To add new wins to our merge, prefer K=2 or K=8.** K=4 is fine for a
sanity check but will mostly reproduce v19b's coverage.

## Key constants (do not change unless you know what you're doing)

| constant | value | reason |
|---|---|---|
| B | 1M (1,048,576) | only verified-safe size for two-ranks-per-chip topology |
| K_SYM | configurable | 1, 2, 4 (default), or 8 rotations per pid |
| INTERNAL_BS | 32768 | B*N_GEN/32768 = 768 chunks/forward — in sweet spot |
| NUM_STEPS | 120 | depth at which paths plateau in our V model |
| ALPHA_QSHORT | 2.0 | student picks 2B-best, teacher reranks to B |

## Wall projection (per-pair cached ~150s; first-pair compile ~30 min)

Wall ≈ compile + (N_pids × K / 8) × 150s. Recommended scope per kernel:

| K | recommended pids | per-rank pairs | wall |
|---|---|---|---|
| 2 | 750 | 188 | 0.5h + 7.8h = 8.3h |
| **4** | **375** | **188** | **0.5h + 7.8h = 8.3h** ← default |
| 8 | 188 | 188 | 0.5h + 7.8h = 8.3h |
'''

PUZZLE_INFO_LITERAL = json.dumps(puzzle_info, separators=(",", ":"))
ALL_PID_STATES_LITERAL = json.dumps(ALL_PID_STATES, separators=(",", ":"))

CELL_1_CONFIG = f'''\
# ============================================================================
# CONFIG — EDIT THIS CELL, then Run All
# ============================================================================
#
# Megaminx test set has 1001 pids (0..1000), ordered by random-walk length
# (pid 0 = easiest scramble, pid 1000 = hardest).
#
# Two knobs:
#   START_PID, END_PID  — pid range to solve (half-open, like Python ranges)
#   K_SYM               — number of sym-ensemble rotations per pid (1, 2, 4, or 8)
#
# K_SYM trades coverage for per-pid quality:
#   K=2: each pid gets 2 shots; covers the most pids per kernel.
#   K=4: each pid gets 4 shots; better per-pid quality, half the coverage.
#   K=8: each pid gets 8 shots; max diversity, quarter the coverage.
#
# Recommended kernel scopes (each ~8.3h wall, under the 9h Kaggle TPU kill):
#   K=2: ~750 pids per kernel  → full-1001 needs 2 collaborators
#   K=4: ~375 pids per kernel  → full-1001 needs 3 collaborators (DEFAULT)
#   K=8: ~188 pids per kernel  → full-1001 needs 6 collaborators
#
# Suggested splits for K=4 (default) full-1001 coverage across 3 collaborators:
#   Collab A: START_PID=0,   END_PID=334   (~7.5h)
#   Collab B: START_PID=334, END_PID=667   (~7.5h)
#   Collab C: START_PID=667, END_PID=1001  (~7.5h)
#
# NOTE: K=4 B=1M matches our v19b config exactly — likely duplicates existing
# wins. Prefer K=2 or K=8 to add new diversity to our merge.
# ============================================================================

START_PID = 0     # inclusive
END_PID = 50      # exclusive — defaults to a 50-pid smoke at K=4 (~2.5h)
K_SYM = 4         # 1, 2, 4, or 8 — number of rotations per pid

# ============================================================================
# Do not edit below this line
# ============================================================================
'''

CELL_2_SETUP = '''\
# Cell 2: Setup, _Tee logging, system info.
# CRITICAL workaround for Kaggle TPU multiprocessing (per the official Kaggle
# PyTorch/XLA notebook): pop TPU_PROCESS_ADDRESSES and CLOUD_TPU_TASK_ID env
# vars BEFORE importing torch_xla submodules / calling xmp.spawn. Kaggle pre-
# sets these for single-process mode; xmp.spawn re-sets them for multi-process
# but only if they were unset to begin with.

import sys, os, time, json, subprocess

for _env_key in ("TPU_PROCESS_ADDRESSES", "CLOUD_TPU_TASK_ID"):
    if _env_key in os.environ:
        os.environ.pop(_env_key)

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

import torch
print("torch", torch.__version__)
try:
    import torch_xla
    print("torch_xla", torch_xla.__version__)
    XLA_AVAILABLE = True
except Exception as e:
    print("XLA import failed:", repr(e))
    XLA_AVAILABLE = False

N_DEVICES = 8
print("XLA_AVAILABLE =", XLA_AVAILABLE, "  N_DEVICES =", N_DEVICES)

# Validate config cell.
assert 0 <= START_PID < END_PID <= ''' + str(N_TOTAL_PIDS) + ''', \\
    f"invalid pid range: START_PID={START_PID}, END_PID={END_PID}, max={''' + str(N_TOTAL_PIDS) + '''}"
assert K_SYM in (1, 2, 4, 8), f"K_SYM must be 1, 2, 4, or 8 (got {K_SYM})"
N_PIDS = END_PID - START_PID
print(f"\\n== CONFIG ==")
print(f"pid range: [{START_PID}, {END_PID})  ({N_PIDS} pids)")
print(f"K_SYM: {K_SYM}")

# Wall projection (150s/pair cached at B=1M, 0.5h compile per rank).
_per_rank_pairs = (N_PIDS * K_SYM + N_DEVICES - 1) // N_DEVICES
_proj_compile_h = 0.5
_proj_compute_h = _per_rank_pairs * 150 / 3600
_proj_total_h = _proj_compile_h + _proj_compute_h
print(f"per-rank pairs: {_per_rank_pairs}")
print(f"estimated wall: {_proj_compile_h:.1f}h compile + {_proj_compute_h:.1f}h compute = {_proj_total_h:.1f}h")
if _proj_total_h > 8.5:
    print(f"WARNING: estimated wall {_proj_total_h:.1f}h exceeds 8.5h target.")
    print(f"  Consider reducing chunk size — Kaggle kills TPU kernels around 9h.")
    print(f"  Per-iteration JSON saves recover partial work, but you'll miss the tail.")
'''

CELL_3_PUZZLE = f'''\
# Cell 3: Puzzle + ALL pid states embedded; slice to TEST_PIDS via config.
PUZZLE_INFO = json.loads({PUZZLE_INFO_LITERAL!r})
GENERATORS = PUZZLE_INFO["generators"]            # name -> [120 ints]
SOLVED = tuple(PUZZLE_INFO["central_state"])      # 120 ints, identity
MOVE_NAMES = list(GENERATORS.keys())
N_GEN = len(MOVE_NAMES)
STATE_SIZE = len(SOLVED)
print(f"N_GEN={{N_GEN}}  STATE_SIZE={{STATE_SIZE}}")

import torch as _torch
all_moves = _torch.tensor(
    [GENERATORS[n] for n in MOVE_NAMES], dtype=_torch.int64
)
V0_cpu = _torch.tensor(SOLVED, dtype=_torch.int8)
print("all_moves shape", tuple(all_moves.shape), "  V0 shape", tuple(V0_cpu.shape))

# All 1001 pid states baked in; the config cell selects a slice.
_ALL_PID_STATES = json.loads({ALL_PID_STATES_LITERAL!r})
ALL_PID_STATES = {{int(k): v for k, v in _ALL_PID_STATES.items()}}
TEST_PIDS = list(range(START_PID, END_PID))
PID_STATES = {{pid: ALL_PID_STATES[pid] for pid in TEST_PIDS}}
print(f"selected {{len(TEST_PIDS)}} pids: [{{TEST_PIDS[0]}}..{{TEST_PIDS[-1]}}]")
'''

CELL_4_MODEL = '''\
# Cell 4: ResMLPDistance — supports output_dim=1 (V) and output_dim=24 (Q).
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x):
        h = F.relu(self.ln1(self.lin1(x)))
        h = self.ln2(self.lin2(h))
        return F.relu(x + h)

class ResMLPDistance(nn.Module):
    def __init__(self, state_size=120, num_classes=120,
                 hidden_dims=(2048, 512), num_res_blocks=2,
                 embed_dim=16, output_dim=1):
        super().__init__()
        self.state_size = state_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(num_classes, embed_dim)
        in_dim = state_size * embed_dim
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
            prev = h
        self.input_stack = nn.Sequential(*layers)
        self.res_blocks = nn.ModuleList([ResBlock(prev) for _ in range(num_res_blocks)])
        self.head = nn.Linear(prev, output_dim)
    def encode(self, x):
        e = self.embedding(x.long())
        return e.flatten(start_dim=-2)
    def forward(self, x):
        target_dtype = self.input_stack[0].weight.dtype
        h = self.encode(x).to(target_dtype)
        h = self.input_stack(h)
        for blk in self.res_blocks:
            h = blk(h)
        out = self.head(h)
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out
'''

CELL_5_LOAD = '''\
# Cell 5: Load m05 (V teacher) + m23_v2 (Q student) + rotations from Kaggle dataset.
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

m05_path = dataset_root / "m05_epoch_0499.pt"
m23v2_path = dataset_root / "m23_v2_epoch_0499.pt"
rotations_path = dataset_root / "rotations.npy"
for p in (m05_path, m23v2_path, rotations_path):
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}")

def _load_state_dict_strip(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

M05_STATE_DICT_CPU = _load_state_dict_strip(m05_path)
M23V2_STATE_DICT_CPU = _load_state_dict_strip(m23v2_path)
print(f"m05 (V): {len(M05_STATE_DICT_CPU)} keys, "
      f"{sum(v.numel() for v in M05_STATE_DICT_CPU.values())/1e6:.2f}M params")
print(f"m23_v2 (Q): {len(M23V2_STATE_DICT_CPU)} keys, "
      f"{sum(v.numel() for v in M23V2_STATE_DICT_CPU.values())/1e6:.2f}M params")

import numpy as np
rotations_np = np.load(rotations_path)
print(f"rotations: {rotations_np.shape} dtype={rotations_np.dtype}")
'''

CELL_6_SYM = '''\
# Cell 6: sym ensemble — K_SYM rotations from rotations.npy.
# K_SYM is set by Cell 1 (CONFIG); we just read it here.
SYM_SEED = 0
ALPHA_QSHORT = 2.0

identity_arr = np.arange(STATE_SIZE, dtype=rotations_np.dtype)
identity_idx = None
for i in range(rotations_np.shape[0]):
    if np.array_equal(rotations_np[i], identity_arr):
        identity_idx = i
        break
assert identity_idx is not None, "rotations.npy missing identity row"

rng_sym = np.random.default_rng(SYM_SEED)
other_idxs = [i for i in range(rotations_np.shape[0]) if i != identity_idx]
chosen_rot_idxs = [identity_idx]
if K_SYM > 1:
    chosen_rot_idxs += list(rng_sym.choice(other_idxs, size=K_SYM - 1, replace=False).tolist())
print(f"sym K={K_SYM}, chose rotation idxs: {chosen_rot_idxs}")

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
            raise ValueError(f"rotation {rot_idx} not a symmetry: R_inv * gen[{nm}] * R is not a generator")
        conj_idx[move_name_to_idx[nm]] = move_name_to_idx[perm_to_name[conj_perm]]
    ROTATIONS.append({"rot_idx": rot_idx, "R": R, "R_inv": R_inv, "conj_idx": conj_idx})

print(f"prepared {len(ROTATIONS)} rotations with conjugation maps")

def apply_rotation_state(state, R, R_inv):
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))

PID_ROT_PAIRS = []
for pid in TEST_PIDS:
    for ri in range(len(ROTATIONS)):
        PID_ROT_PAIRS.append((pid, ri))
print(f"total (pid, rotation) pairs: {len(PID_ROT_PAIRS)}")
'''

CELL_7_BEAM = '''\
# Cell 7: qshort beam search on XLA. Static-shape throughout.

def _make_hash_vec(state_size, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, int(1e15), (state_size,), dtype=torch.int64, generator=g)

def _model_predict_chunked(model, states, chunk_size, xla_available):
    if xla_available:
        import torch_xla.core.xla_model as xm
    n = states.size(0)
    out = None
    with torch.no_grad():
        for i in range(0, n, chunk_size):
            chunk = states[i : i + chunk_size]
            sz = chunk.size(0)
            if sz < chunk_size:
                pad = chunk[-1:].expand(chunk_size - sz, -1)
                padded = torch.cat([chunk, pad], dim=0)
                vals = model(padded.long())
                vals = vals[:sz]
            else:
                vals = model(chunk.long())
            if out is None:
                if vals.dim() == 1:
                    out = torch.empty(n, dtype=torch.bfloat16, device=states.device)
                else:
                    out = torch.empty((n, vals.size(-1)), dtype=torch.bfloat16, device=states.device)
            out[i : i + sz] = vals.to(torch.bfloat16)
            if xla_available:
                xm.mark_step()
    return out


def beam_solve_xla_qshort(initial_state_list, teacher, student,
                          all_moves_dev, V0_dev, hash_vec, V0_hash,
                          device, B, num_steps, internal_bs, alpha,
                          xla_available, state_size, n_gen):
    if xla_available:
        import torch_xla.core.xla_model as xm
    state = torch.tensor(initial_state_list, dtype=torch.int8)
    V0_cpu_local = torch.tensor(SOLVED, dtype=torch.int8)
    if torch.equal(state, V0_cpu_local):
        return {"found": True, "path_len": 0, "path_idx": [], "wall_s": 0.0,
                "timings": {"total": 0.0}, "min_v_trajectory": []}

    aB = int(alpha * B)
    t_start = time.time()

    states0 = state.unsqueeze(0).to(device)
    neighbors0 = states0[:, all_moves_dev].view(-1, state_size)
    values0 = _model_predict_chunked(teacher, neighbors0, internal_bs, xla_available)
    k0 = min(B, n_gen)
    top_v0, top_idx0 = torch.topk(values0, k0, largest=False)
    chosen_states0 = neighbors0[top_idx0]
    if k0 < B:
        pad = chosen_states0[-1:].expand(B - k0, -1)
        states = torch.cat([chosen_states0, pad], dim=0)
    else:
        states = chosen_states0
    if xla_available:
        xm.mark_step()
    t_step0 = time.time() - t_start

    tree_idx = torch.zeros((num_steps, B), dtype=torch.int32, device=device)
    tree_move = torch.zeros((num_steps, B), dtype=torch.int8, device=device)
    min_v_log = torch.full((num_steps,), 1e6, dtype=torch.float32, device=device)
    move0_full = torch.full((B,), -1, dtype=torch.int8, device=device)
    move0_full[:k0] = top_idx0.to(torch.int8)
    tree_move[0] = move0_full

    found_step = torch.tensor(-1, dtype=torch.int32, device=device)
    found_pos = torch.tensor(-1, dtype=torch.int32, device=device)

    eq0 = (states == V0_dev).all(dim=1)
    any0 = eq0.any()
    pos0 = eq0.to(torch.int32).argmax()
    found_step = torch.where(any0, torch.tensor(0, dtype=torch.int32, device=device), found_step)
    found_pos = torch.where(any0, pos0.to(torch.int32), found_pos)

    BIG_VAL = torch.tensor(1e6, dtype=torch.bfloat16, device=device)
    BIG_VAL_INF = torch.tensor(1e6, dtype=torch.float32, device=device)

    t_loop = time.time()
    first_iter_t = None

    for j in range(1, num_steps):
        t_iter = time.time()

        student_q = _model_predict_chunked(student, states, internal_bs, xla_available)

        neighbors_2d = states[:, all_moves_dev]
        neighbors = neighbors_2d.view(-1, state_size)
        h = (neighbors.to(torch.int64) * hash_vec).sum(dim=1)

        sort_h, sort_idx = torch.sort(h)
        is_dup_sorted = torch.cat([
            torch.zeros(1, dtype=torch.bool, device=device),
            (sort_h[1:] == sort_h[:-1]),
        ])
        restore = torch.argsort(sort_idx)
        dup_mask = is_dup_sorted[restore]

        student_q_flat = student_q.view(-1).to(torch.float32)
        student_q_masked = torch.where(dup_mask, BIG_VAL_INF, student_q_flat)
        _, shortlist_indices = torch.topk(student_q_masked, aB, largest=False)

        shortlist_states = neighbors[shortlist_indices]

        teacher_v = _model_predict_chunked(teacher, shortlist_states, internal_bs, xla_available)

        top_v, chosen_local = torch.topk(teacher_v, B, largest=False)
        chosen_states = shortlist_states[chosen_local]
        chosen_global = shortlist_indices[chosen_local]
        chosen_h = h[chosen_global]

        parent = (chosen_global // n_gen).to(torch.int32)
        move = (chosen_global % n_gen).to(torch.int8)

        states = chosen_states
        tree_idx[j] = parent
        tree_move[j] = move
        min_v_log[j] = top_v[0].to(torch.float32)

        eq_v0_h = (chosen_h == V0_hash)
        any_hit = eq_v0_h.any()
        pos_hit = eq_v0_h.to(torch.int32).argmax()
        is_first_hit = (found_step == -1) & any_hit
        j_tensor = torch.tensor(j, dtype=torch.int32, device=device)
        found_step = torch.where(is_first_hit, j_tensor, found_step)
        found_pos = torch.where(is_first_hit, pos_hit.to(torch.int32), found_pos)

        if xla_available:
            xm.mark_step()
        if first_iter_t is None:
            first_iter_t = time.time() - t_iter

    t_loop_end = time.time()

    if xla_available:
        xm.mark_step()
    found_step_h = int(found_step.item())
    found_pos_h = int(found_pos.item())
    min_v_h = min_v_log.detach().cpu().tolist()
    if found_step_h < 0:
        timings = {"step0": t_step0, "loop": t_loop_end - t_loop,
                   "first_iter": first_iter_t or 0.0, "total": time.time() - t_start}
        return {"found": False, "path_len": 0, "path_idx": [], "wall_s": timings["total"],
                "timings": timings, "min_v_trajectory": min_v_h}

    tree_idx_h = tree_idx[: found_step_h + 1].cpu()
    tree_move_h = tree_move[: found_step_h + 1].cpu()
    t_total = time.time() - t_start

    path = []
    pos = found_pos_h
    for j in range(found_step_h, -1, -1):
        m = int(tree_move_h[j, pos].item())
        path.append(m)
        pos = int(tree_idx_h[j, pos].item())
    path.reverse()
    while path and path[0] < 0:
        path.pop(0)

    timings = {"step0": t_step0, "loop": t_loop_end - t_loop,
               "first_iter": first_iter_t or 0.0, "total": t_total}
    return {"found": True, "path_len": len(path), "path_idx": path,
            "wall_s": t_total, "found_step": found_step_h, "timings": timings,
            "min_v_trajectory": min_v_h}
'''

CELL_8_MP_FN = '''\
# Cell 8: Per-rank worker. Multi-process via xmp.spawn (4 chips * 2 cores = 8).

B_TEST = 1_048_576      # 1M. B=2M OOM'd at runtime (alexandervc fork v1).
                        # B=1.5M OOM'd at XLA program load (5.58 GB workspace request,
                        # 3.85 GB free) — two ranks per chip share HBM. B=1M is the
                        # only verified-safe value (matches v19b's working config).
INTERNAL_BS = 32768     # B*N_GEN/32768 = 768 chunks/forward, in sweet spot
NUM_STEPS = 120

def _mp_fn(index, pid_rot_pairs):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    try:
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
    except Exception:
        rank = index
        world_size = 8
    print(f"[rank {rank}/{world_size}] device={device}", flush=True)

    teacher = ResMLPDistance(state_size=120, num_classes=120, hidden_dims=(2048, 512),
                             num_res_blocks=2, embed_dim=16, output_dim=1)
    teacher.load_state_dict(M05_STATE_DICT_CPU, strict=False)
    teacher = teacher.to(torch.bfloat16).to(device).eval()

    student = ResMLPDistance(state_size=120, num_classes=120, hidden_dims=(2048, 1024),
                             num_res_blocks=2, embed_dim=16, output_dim=24)
    missing_s, unexpected_s = student.load_state_dict(M23V2_STATE_DICT_CPU, strict=False)
    if missing_s or unexpected_s:
        print(f"[rank {rank}] student load: missing={len(missing_s)} unexpected={len(unexpected_s)}",
              flush=True)
    student = student.to(torch.bfloat16).to(device).eval()

    all_moves_dev = all_moves.to(device)
    V0_dev = V0_cpu.to(device)
    hash_vec = _make_hash_vec(STATE_SIZE, seed=0).to(device)
    V0_hash = (hash_vec * V0_cpu.to(torch.int64).to(device)).sum().reshape(()).to(torch.int64)

    my_pairs = [pr for i, pr in enumerate(pid_rot_pairs) if i % world_size == rank]
    print(f"[rank {rank}] my_pairs ({len(my_pairs)}): {my_pairs[:5]}{'...' if len(my_pairs) > 5 else ''}", flush=True)

    rank_local = []
    for i, (pid, rot_i) in enumerate(my_pairs):
        rot = ROTATIONS[rot_i]
        s0 = PID_STATES[pid]
        s_to_solve = apply_rotation_state(s0, rot["R"], rot["R_inv"])

        t0 = time.time()
        r = beam_solve_xla_qshort(
            list(s_to_solve), teacher, student,
            all_moves_dev, V0_dev, hash_vec, V0_hash,
            device, B_TEST, NUM_STEPS, INTERNAL_BS, ALPHA_QSHORT,
            XLA_AVAILABLE, STATE_SIZE, N_GEN,
        )
        wall = time.time() - t0

        path_orig = None
        verify_ok = False
        if r["found"]:
            conj_idx = rot["conj_idx"]
            path_orig = [conj_idx[m] for m in r["path_idx"]]
            cur = list(s0)
            for m in path_orig:
                gen = GENERATORS[MOVE_NAMES[m]]
                cur = [cur[g] for g in gen]
            verify_ok = (tuple(cur) == SOLVED)

        result_dict = {
            "pid": pid, "rot_idx": rot_i, "rank": rank,
            "found": r["found"], "verify_ok": verify_ok,
            "path_len": len(path_orig) if path_orig is not None else 0,
            "path_idx_orig": path_orig if verify_ok else [],
            "path_idx_rotated": r.get("path_idx", []),
            "wall_s": wall,
            "timings": r.get("timings", {}),
            "min_v_trajectory": r.get("min_v_trajectory", []),
        }
        rank_local.append(result_dict)

        first_iter = r.get("timings", {}).get("first_iter", 0)
        mvt = r.get("min_v_trajectory", [])
        last_v = mvt[-1] if mvt else None
        print(f"[rank {rank}] {i+1}/{len(my_pairs)} pid={pid:>4} rot={rot_i} "
              f"found={r['found']} verify={verify_ok} path_len={result_dict['path_len']:>3} "
              f"wall={wall:.1f}s first_iter={first_iter:.2f}s last_min_V={last_v}", flush=True)

        # Per-iteration partial save — recovers everything if Kaggle hits 9h kill.
        try:
            with open(f"/kaggle/working/rank_{rank}_partial.json", "w") as f:
                json.dump(rank_local, f)
        except Exception as e:
            print(f"[rank {rank}] WARN partial save failed: {e}", flush=True)

    with open(f"/kaggle/working/rank_{rank}_final.json", "w") as f:
        json.dump(rank_local, f)
    print(f"[rank {rank}] done, processed {len(rank_local)} pairs", flush=True)
'''

CELL_9_SPAWN = '''\
# Cell 9: Launch via xmp.spawn.

print()
print(f"== xmp.spawn qshort+sym (B=2M K=2): K_SYM={K_SYM}, alpha={ALPHA_QSHORT}, "
      f"beam={B_TEST}, internal_bs={INTERNAL_BS}, max_steps={NUM_STEPS} ==")
print(f"== pid range: [{START_PID}, {END_PID})  ({len(TEST_PIDS)} pids, "
      f"{len(PID_ROT_PAIRS)} pid-rot pairs) ==")
print()

import glob
for stale in glob.glob("/kaggle/working/rank_*_partial.json"):
    try: os.remove(stale)
    except: pass
for stale in glob.glob("/kaggle/working/rank_*_final.json"):
    try: os.remove(stale)
    except: pass

import torch_xla.distributed.xla_multiprocessing as xmp

t_launch = time.time()
xmp.spawn(_mp_fn, args=(PID_ROT_PAIRS,), start_method="fork", nprocs=None)
t_launch_total = time.time() - t_launch
print(f"== xmp.spawn complete in {t_launch_total:.1f}s ==")
'''

CELL_10_AGGREGATE = '''\
# Cell 10: Aggregate per-rank JSON files. Reads *_final.json preferentially;
# falls back to *_partial.json if Kaggle killed the kernel mid-run.
import csv as _csv
import glob

def _load_rank_results():
    by_rank = {}
    for path in glob.glob("/kaggle/working/rank_*_final.json"):
        rank = int(path.split("rank_")[1].split("_")[0])
        try:
            with open(path) as f:
                by_rank[rank] = ("final", json.load(f))
        except Exception as e:
            print(f"WARN failed reading {path}: {e}")
    for path in glob.glob("/kaggle/working/rank_*_partial.json"):
        rank = int(path.split("rank_")[1].split("_")[0])
        if rank in by_rank:
            continue
        try:
            with open(path) as f:
                by_rank[rank] = ("partial", json.load(f))
        except Exception as e:
            print(f"WARN failed reading {path}: {e}")
    out = []
    for rank in sorted(by_rank.keys()):
        kind, recs = by_rank[rank]
        print(f"rank {rank}: {kind} file with {len(recs)} records")
        out.extend(recs)
    return out

results_list = _load_rank_results()
print(f"loaded {len(results_list)} total results across {len({r['rank'] for r in results_list})} ranks")

with open("/kaggle/working/tpu_qshort_sym_results.json", "w") as f:
    json.dump(results_list, f, indent=2)
print(f"wrote /kaggle/working/tpu_qshort_sym_results.json")

by_pid = {}
for r in results_list:
    by_pid.setdefault(r["pid"], []).append(r)

print()
print("== per-pid (min over K rotations) ==")
print(f"{'pid':>6}  {'best_rot':>8}  {'best_path':>9}  {'all_found/K':>12}  "
      f"{'wall_max':>9}  {'last_min_V':>10}")
final_per_pid = {}
total_path = 0
n_solved = 0
for pid in sorted(by_pid.keys()):
    rs = by_pid[pid]
    valid = [r for r in rs if r["verify_ok"]]
    n_found = len(valid)
    if valid:
        best = min(valid, key=lambda r: r["path_len"])
        final_per_pid[pid] = {
            "found": True,
            "best_path_len": best["path_len"],
            "best_path_idx": best["path_idx_orig"],
            "best_rot_idx": best["rot_idx"],
        }
        total_path += best["path_len"]
        n_solved += 1
        wall_max = max(r["wall_s"] for r in rs)
        last_v = (best.get("min_v_trajectory") or [None])[-1]
        print(f"{pid:>6}  {best['rot_idx']:>8}  {best['path_len']:>9}  "
              f"{n_found:>5}/{len(rs):<5}  {wall_max:>9.1f}  {last_v}")
    else:
        final_per_pid[pid] = {"found": False}
        wall_max = max(r["wall_s"] for r in rs)
        print(f"{pid:>6}  {'-':>8}  {'-':>9}  {'0':>5}/{len(rs):<5}  {wall_max:>9.1f}")

print()
print(f"== summary ==")
print(f"solved (any rotation): {n_solved}/{len(by_pid)}")
print(f"total_path (best per pid): {total_path}")

csv_path = "/kaggle/working/tpu_qshort_sym_submission.csv"
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

CELLS = [
    ("markdown", CELL_0_README),
    ("code", CELL_1_CONFIG),
    ("code", CELL_2_SETUP),
    ("code", CELL_3_PUZZLE),
    ("code", CELL_4_MODEL),
    ("code", CELL_5_LOAD),
    ("code", CELL_6_SYM),
    ("code", CELL_7_BEAM),
    ("code", CELL_8_MP_FN),
    ("code", CELL_9_SPAWN),
    ("code", CELL_10_AGGREGATE),
]

# --- Notebook assembly -----------------------------------------------------

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
        lines = src.splitlines(keepends=True)
        cell["source"] = lines

out_nb = HERE / "cayleypy-tpu-beam-b2m-k2.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes, {len(ALL_PID_STATES)} pid states embedded)")

meta = {
    # Slug intentionally avoids "tpu-" prefix: Kaggle's kernel-creation
    # validation rejected `cayleypy-tpu-beam-*` slugs with enable_tpu=False
    # (returned a misleading "Notebook not found" error). Kept "megaminx" in
    # the slug since this is megaminx-specific.
    "id": "artgor/cayleypy-megaminx-beam-shareable",
    "title": "cayleypy megaminx beam shareable",
    "code_file": "cayleypy-tpu-beam-b2m-k2.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": False,           # public so collaborators can fork
    "enable_gpu": False,
    "enable_tpu": True,            # intended config; toggle off to push when quota exhausted
    "enable_internet": True,
    "dataset_sources": ["artgor/megaminx-tpu-artifacts"],
    "competition_sources": [],
    "kernel_sources": [],
}
out_meta = HERE / "kernel-metadata.json"
out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {out_meta}")
