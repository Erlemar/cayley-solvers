"""Builds the SHAREABLE TPU beam-search notebook with AZ v4 multi-checkpoint per-pid select.

Companion to `tpu_beam_az_v4_v_only/`: same V-only beam search and TPU multi-process
fan-out, but for each (pid, rotation) we run **N_CKPTS = 4 AZ v4 checkpoints**
(epochs 24, 49, 74, 99) and the aggregator takes per-pid min across the whole
(ckpt x rotation) matrix.

Motivation: in the strat-5 acceptance bench, ep 24 wins globally (874 / 87.4
avg) and later epochs regress (ep 99 = 898, 9/10 only). But the per-pid winner
might not always be ep 24 — different scrambles may suit slightly differently
calibrated V landscapes. This kernel produces the data to answer open question
#4 from the AZ v4 breakthrough notes: "Earlier checkpoints could find an even
better stop point — cheap experiment."

Per (pid, rot, ckpt) cost is identical to the v_only kernel; the matrix
multiplies the work by N_CKPTS, so default END_PID is correspondingly smaller.
Per-rank workers load all four state dicts once, sort their work queue by
ckpt_idx so each ckpt's params stay on-device for a contiguous run (avoids
needless state_dict reloads).

Output: cayleypy-tpu-beam-az-v4-multickpt.ipynb in the same directory.
Run via: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_az_v4_multickpt/build_notebook.py
"""

from __future__ import annotations

import csv
import json
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

# Hardcoded ckpt manifest. Filename basename : which dataset to look in.
# Both datasets are attached via kernel-metadata.json dataset_sources.
CKPT_MANIFEST = [
    {"epoch": 24, "filename": "m_az_v4_v_only.pt",     "dataset": "megaminx-tpu-artifacts"},
    {"epoch": 49, "filename": "m_az_v4_v_only_e49.pt", "dataset": "megaminx-az-v4-multi-ckpt"},
    {"epoch": 74, "filename": "m_az_v4_v_only_e74.pt", "dataset": "megaminx-az-v4-multi-ckpt"},
    {"epoch": 99, "filename": "m_az_v4_v_only_e99.pt", "dataset": "megaminx-az-v4-multi-ckpt"},
]
N_CKPTS = len(CKPT_MANIFEST)
CKPT_MANIFEST_LITERAL = json.dumps(CKPT_MANIFEST, separators=(",", ":"))

# --- Cell sources ---------------------------------------------------------

CELL_0_README = '''\
# Megaminx beam search on Kaggle TPU - AZ v4 multi-checkpoint per-pid select (shareable)

Multi-collaborator TPU beam search for the
[CayleyPy Megaminx Kaggle competition](https://www.kaggle.com/competitions/cayley-py-megaminx).
Each collaborator runs a different chunk of pids on their own TPU quota,
and we union the results back via per-pid min.

**Stack:** AZ v4 V (6M params), **4 checkpoints** (epochs 24, 49, 74, 99),
K-sym-ensemble (configurable, default K=4), B configurable (default B=131,072),
120 steps. Per pid the aggregator takes the **min across all 4 x K cells** of
the (ckpt x rotation) matrix.

## Why multi-checkpoint

The single-checkpoint sibling kernel (`cayleypy-megaminx-beam-az-v4-shareable`)
uses only ep 24, the globally-best 10-pid bench winner (874 / 87.4 avg). Later
epochs regress on the bench mean: ep 49 = 902, ep 74 = 958, ep 99 = 898 (9/10).

But **the per-pid winner may not always be ep 24.** As the trunk specializes
for policy memorization past ep ~25, the V landscape shifts; some pids may
sit in a basin where a later checkpoint's V is sharper, even if globally the
mean regresses. This kernel produces the data to answer that question.

Per pid we expand: 4 checkpoints x K rotations = 4K cells. The merged result
is the shortest path across all cells. The aggregation cell additionally
prints a "best CHECKPOINT per pid" histogram - if >= 2 distinct ckpt epochs
contribute wins, per-pid select is doing useful work; if all wins are ep 24,
the existing single-ckpt shareable is sufficient and this kernel costs 4x for
no gain.

## How to run as a collaborator

1. Fork this notebook on Kaggle ("Copy and Edit").
2. Two datasets are auto-attached via kernel metadata - no manual step:
   - `artgor/megaminx-tpu-artifacts` (for `m_az_v4_v_only.pt` = ep 24 + `rotations.npy`)
   - `artgor/megaminx-az-v4-multi-ckpt` (for ep 49 / 74 / 99 V-only checkpoints)
3. **Edit Cell 1 (CONFIG) to set your pid range, K_SYM, and B_TEST.**
   Defaults are `START_PID=0, END_PID=2, K_SYM=4, B_TEST=131072` - a smoke
   that covers 2 pids x 4 rot x 4 ckpts = 32 cells (~10 min compute + 0.5 h
   compile). Wall projection is printed in Cell 2.
4. **Switch accelerator to TPU v3-8** in the right sidebar.
5. Run all cells. Per-iteration partial saves recover work if Kaggle hits
   the 9h kill mid-run.
6. After completion, download `/kaggle/working/tpu_az_v4_multickpt_results.json`
   and `/kaggle/working/tpu_az_v4_multickpt_submission.csv` and send to the
   owner for merging.

## Trade-off vs the single-ckpt shareable

| Aspect | Single (ep24 only) | Multi (4 ckpts) |
|---|---|---|
| Per-pid cells | K | 4K |
| Per-pair wall (compute) | unchanged | unchanged |
| Total wall for same pid coverage | 1x | 4x |
| Quality | strong (ep24 global best) | >= single (per-pid min dominates) |
| Cost-effectiveness | high if all pids prefer ep24 | high if >= 2 ckpts contribute wins |

Recommendation: run a small coverage (e.g. 8-16 pids) of THIS kernel first to
see if the histogram is non-trivial. If only ep24 contributes wins, switch to
the single-ckpt shareable for production coverage. If multiple ckpts contribute,
this kernel is the right tool.
'''

PUZZLE_INFO_LITERAL = json.dumps(puzzle_info, separators=(",", ":"))
ALL_PID_STATES_LITERAL = json.dumps(ALL_PID_STATES, separators=(",", ":"))

CELL_1_CONFIG = f'''\
# ============================================================================
# CONFIG - EDIT THIS CELL, then Run All
# ============================================================================
#
# Megaminx test set has 1001 pids (0..1000), ordered by random-walk length
# (pid 0 = easiest scramble, pid 1000 = hardest).
#
# Three knobs:
#   START_PID, END_PID  - pid range to solve (half-open, like Python ranges)
#   K_SYM               - number of sym-ensemble rotations per pid (1, 2, 4, or 8)
#   B_TEST              - beam width (65536, 131072, 262144, or 524288)
#
# This is the MULTI-CHECKPOINT variant: per pid expands to 4 checkpoints x
# K_SYM rotations = 4*K cells. Wall scales 4x vs the single-ckpt shareable.
#
# Suggested launches at default K_SYM=4 B_TEST=131072 (~120s/pair):
#   Smoke    : START_PID=0, END_PID=2,   ~10 min   (32 cells)
#   Diagnostic: START_PID=0, END_PID=16, ~5 h      (256 cells, full strat-5 vibe)
#   Half     : START_PID=0, END_PID=64,  ~21 h    (DO NOT SOLO; split into kernels)
#
# The notebook automatically prints a wall projection in Cell 2; adjust the
# config if it warns about the 9h kill.
#
# ============================================================================

START_PID = 0     # inclusive
END_PID = 2       # exclusive - tiny smoke (2 pids x 4 rot x 4 ckpts = 32 cells)
K_SYM = 4         # 1, 2, 4, or 8 - number of rotations per pid
B_TEST = 131072   # 65536, 131072, 262144, or 524288 - beam width

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
assert B_TEST in (65536, 131072, 262144, 524288), \\
    f"B_TEST must be 65536, 131072, 262144, or 524288 (got {B_TEST})"
N_PIDS = END_PID - START_PID
N_CKPTS = ''' + str(N_CKPTS) + '''
print(f"\\n== CONFIG ==")
print(f"pid range: [{START_PID}, {END_PID})  ({N_PIDS} pids)")
print(f"K_SYM: {K_SYM}")
print(f"B_TEST: {B_TEST}")
print(f"N_CKPTS: {N_CKPTS}")

# Wall projection. Anchor: V-only B=131k ~ 120s/pair (one V forward over
# B*N_GEN=3.1M candidates per step). Linear in B*N_GEN. Multi-ckpt: per pid
# we evaluate N_CKPTS * K_SYM cells, so projection multiplies accordingly.
_per_pair_s = 120.0 * (B_TEST / 131072.0)
_total_cells = N_PIDS * K_SYM * N_CKPTS
_per_rank_pairs = (_total_cells + N_DEVICES - 1) // N_DEVICES
_proj_compile_h = 0.5
_proj_compute_h = _per_rank_pairs * _per_pair_s / 3600
_proj_total_h = _proj_compile_h + _proj_compute_h
print(f"total cells (pid x rot x ckpt): {_total_cells}")
print(f"per-rank cells: {_per_rank_pairs}")
print(f"estimated wall: {_proj_compile_h:.1f}h compile + "
      f"{_proj_compute_h:.1f}h compute = {_proj_total_h:.1f}h "
      f"(per-pair anchor: {_per_pair_s:.0f}s)")
if _proj_total_h > 8.5:
    print(f"WARNING: estimated wall {_proj_total_h:.1f}h exceeds 8.5h target.")
    print(f"  Kaggle kills TPU kernels around 9h. Per-iteration JSON saves recover")
    print(f"  partial work, but consider reducing pid range, K_SYM, or B_TEST.")
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
# Cell 4: ResMLPDistance - output_dim=1 (V).
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

CELL_5_LOAD = f'''\
# Cell 5: Load 4 AZ v4 V-only checkpoints + rotations from Kaggle datasets.
from pathlib import Path

CKPT_MANIFEST = json.loads({CKPT_MANIFEST_LITERAL!r})

def _resolve_dataset_root(slug):
    """Probe both standard and /datasets/<owner>/ mount paths."""
    for cand in [
        Path(f"/kaggle/input/{{slug}}"),
        Path(f"/kaggle/input/datasets/artgor/{{slug}}"),
    ]:
        if cand.exists():
            return cand
    return None

# Find every required dataset.
DATASET_ROOTS = {{}}
for ds_slug in set(c["dataset"] for c in CKPT_MANIFEST) | {{"megaminx-tpu-artifacts"}}:
    root = _resolve_dataset_root(ds_slug)
    if root is None:
        listing = list(Path("/kaggle/input").rglob("*.pt"))[:20]
        raise SystemExit(
            f"dataset '{{ds_slug}}' not found under /kaggle/input. "
            f"Found .pt files: {{listing}}. Attach via 'Add Data'."
        )
    DATASET_ROOTS[ds_slug] = root
    print(f"dataset {{ds_slug}}: {{root}}")

def _load_state_dict_strip(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    return {{(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in sd.items()}}

# Load all 4 ckpt state dicts into CPU memory. Each is ~24 MB; total ~96 MB.
V_STATE_DICTS_CPU = []
for ck in CKPT_MANIFEST:
    p = DATASET_ROOTS[ck["dataset"]] / ck["filename"]
    if not p.exists():
        raise SystemExit(f"missing artifact: {{p}}")
    sd = _load_state_dict_strip(p)
    V_STATE_DICTS_CPU.append({{
        "ckpt_idx": len(V_STATE_DICTS_CPU),
        "epoch": ck["epoch"],
        "filename": ck["filename"],
        "state_dict": sd,
    }})
    print(f"  ckpt {{len(V_STATE_DICTS_CPU)-1}} (ep {{ck['epoch']}}): "
          f"{{len(sd)}} keys, {{sum(v.numel() for v in sd.values())/1e6:.2f}}M params, "
          f"{{p.name}}")

import numpy as np
rotations_path = DATASET_ROOTS["megaminx-tpu-artifacts"] / "rotations.npy"
rotations_np = np.load(rotations_path)
print(f"rotations: {{rotations_np.shape}} dtype={{rotations_np.dtype}}")
'''

CELL_6_SYM = '''\
# Cell 6: sym ensemble + (pid, rot_idx, ckpt_idx) triples for the matrix.
# K_SYM is set by Cell 1 (CONFIG); N_CKPTS is len(V_STATE_DICTS_CPU).
SYM_SEED = 0

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

# Build (pid, rot_idx, ckpt_idx) triples. Sort by ckpt_idx primarily so each
# rank processes contiguous blocks of the same ckpt - keeps the on-device
# state_dict swap to a minimum (1 swap per ckpt boundary, vs O(triples) if
# interleaved).
PID_ROT_CKPT_TRIPLES = []
for ck_i in range(len(V_STATE_DICTS_CPU)):
    for pid in TEST_PIDS:
        for ri in range(len(ROTATIONS)):
            PID_ROT_CKPT_TRIPLES.append((pid, ri, ck_i))
print(f"total (pid, rot, ckpt) triples: {len(PID_ROT_CKPT_TRIPLES)} "
      f"({len(TEST_PIDS)} pids x {len(ROTATIONS)} rot x {len(V_STATE_DICTS_CPU)} ckpts)")
'''

CELL_7_BEAM = '''\
# Cell 7: V-only beam search on XLA. Static-shape throughout.
# Per step: one V forward over all B*N_GEN neighbors; pick top B by V.
# This is the identical beam-search code as the single-ckpt shareable - the
# only difference is the calling loop iterates over (pid, rot, ckpt) triples
# instead of (pid, rot) pairs.

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


def beam_solve_xla_v_only(initial_state_list, value_model,
                          all_moves_dev, V0_dev, hash_vec, V0_hash,
                          device, B, num_steps, internal_bs,
                          xla_available, state_size, n_gen):
    """Plain V-only beam search. Static-shape throughout for XLA."""
    if xla_available:
        import torch_xla.core.xla_model as xm
    state = torch.tensor(initial_state_list, dtype=torch.int8)
    V0_cpu_local = torch.tensor(SOLVED, dtype=torch.int8)
    if torch.equal(state, V0_cpu_local):
        return {"found": True, "path_len": 0, "path_idx": [], "wall_s": 0.0,
                "timings": {"total": 0.0}, "min_v_trajectory": []}

    t_start = time.time()

    states0 = state.unsqueeze(0).to(device)
    neighbors0 = states0[:, all_moves_dev].view(-1, state_size)
    values0 = _model_predict_chunked(value_model, neighbors0, internal_bs, xla_available)
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

    BIG_VAL_INF = torch.tensor(1e6, dtype=torch.float32, device=device)

    t_loop = time.time()
    first_iter_t = None

    for j in range(1, num_steps):
        t_iter = time.time()

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

        neighbor_v = _model_predict_chunked(
            value_model, neighbors, internal_bs, xla_available
        )
        neighbor_v_flat = neighbor_v.to(torch.float32)
        neighbor_v_masked = torch.where(dup_mask, BIG_VAL_INF, neighbor_v_flat)

        top_v, chosen_global = torch.topk(neighbor_v_masked, B, largest=False)
        chosen_states = neighbors[chosen_global]
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
# Differs from the single-ckpt shareable in two places:
#   1. Iterates over (pid, rot_idx, ckpt_idx) triples instead of (pid, rot_idx) pairs.
#   2. Swaps the model's state_dict between ckpt blocks. Each rank's work queue
#      is already sorted by ckpt_idx (Cell 6) so swaps happen at block boundaries.

INTERNAL_BS = 4096
NUM_STEPS = 120

def _mp_fn(index, pid_rot_ckpt_triples):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    try:
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
    except Exception:
        rank = index
        world_size = 8
    print(f"[rank {rank}/{world_size}] device={device}", flush=True)

    # One model object; we swap state_dicts between ckpt blocks.
    value_model = ResMLPDistance(state_size=120, num_classes=120,
                                 hidden_dims=(2048, 512), num_res_blocks=2,
                                 embed_dim=16, output_dim=1)
    value_model = value_model.to(torch.bfloat16).to(device).eval()

    all_moves_dev = all_moves.to(device)
    V0_dev = V0_cpu.to(device)
    hash_vec = _make_hash_vec(STATE_SIZE, seed=0).to(device)
    V0_hash = (hash_vec * V0_cpu.to(torch.int64).to(device)).sum().reshape(()).to(torch.int64)

    # Per-rank work slice. Note triples are already sorted by ckpt_idx in Cell 6,
    # so each rank's slice still has ckpt-contiguous blocks (the stride-by-
    # world_size partition preserves the relative ordering within each ckpt).
    my_triples = [tr for i, tr in enumerate(pid_rot_ckpt_triples) if i % world_size == rank]
    print(f"[rank {rank}] my_triples ({len(my_triples)}): "
          f"{my_triples[:5]}{'...' if len(my_triples) > 5 else ''}", flush=True)

    rank_local = []
    current_ckpt_idx = None
    for i, (pid, rot_i, ck_i) in enumerate(my_triples):
        # Swap state_dict when we cross a ckpt boundary.
        if ck_i != current_ckpt_idx:
            t_swap = time.time()
            sd_cpu = V_STATE_DICTS_CPU[ck_i]["state_dict"]
            missing, unexpected = value_model.load_state_dict(sd_cpu, strict=False)
            if missing or unexpected:
                print(f"[rank {rank}] load ckpt {ck_i}: "
                      f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
            # Ensure params live on device + cast to bf16.
            value_model = value_model.to(torch.bfloat16).to(device).eval()
            current_ckpt_idx = ck_i
            ep = V_STATE_DICTS_CPU[ck_i]["epoch"]
            print(f"[rank {rank}] switched to ckpt {ck_i} (ep {ep}) in {time.time()-t_swap:.1f}s",
                  flush=True)

        rot = ROTATIONS[rot_i]
        s0 = PID_STATES[pid]
        s_to_solve = apply_rotation_state(s0, rot["R"], rot["R_inv"])

        t0 = time.time()
        r = beam_solve_xla_v_only(
            list(s_to_solve), value_model,
            all_moves_dev, V0_dev, hash_vec, V0_hash,
            device, B_TEST, NUM_STEPS, INTERNAL_BS,
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
            "pid": pid, "rot_idx": rot_i, "ckpt_idx": ck_i,
            "epoch": V_STATE_DICTS_CPU[ck_i]["epoch"],
            "rank": rank,
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
        print(f"[rank {rank}] {i+1}/{len(my_triples)} "
              f"pid={pid:>4} rot={rot_i} ckpt={ck_i}(ep{result_dict['epoch']}) "
              f"found={r['found']} verify={verify_ok} path_len={result_dict['path_len']:>3} "
              f"wall={wall:.1f}s first_iter={first_iter:.2f}s last_min_V={last_v}", flush=True)

        try:
            with open(f"/kaggle/working/rank_{rank}_partial.json", "w") as f:
                json.dump(rank_local, f)
        except Exception as e:
            print(f"[rank {rank}] WARN partial save failed: {e}", flush=True)

    with open(f"/kaggle/working/rank_{rank}_final.json", "w") as f:
        json.dump(rank_local, f)
    print(f"[rank {rank}] done, processed {len(rank_local)} triples", flush=True)
'''

CELL_9_SPAWN = '''\
# Cell 9: Launch via xmp.spawn.

print()
print(f"== xmp.spawn multi-ckpt (AZ v4): K_SYM={K_SYM}, beam={B_TEST}, "
      f"internal_bs={INTERNAL_BS}, max_steps={NUM_STEPS}, N_CKPTS={N_CKPTS} ==")
print(f"== pid range: [{START_PID}, {END_PID})  ({len(TEST_PIDS)} pids, "
      f"{len(PID_ROT_CKPT_TRIPLES)} triples) ==")
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
xmp.spawn(_mp_fn, args=(PID_ROT_CKPT_TRIPLES,), start_method="fork", nprocs=None)
t_launch_total = time.time() - t_launch
print(f"== xmp.spawn complete in {t_launch_total:.1f}s ==")
'''

CELL_10_AGGREGATE = '''\
# Cell 10: Aggregate per-rank JSON files. Reads *_final.json preferentially;
# falls back to *_partial.json if Kaggle killed the kernel mid-run.
# Per pid, the winner is the shortest valid path across ALL (rot, ckpt) cells.
# Histogram cell at the end shows which (ckpt, rotation) contributed wins.

import csv as _csv
import glob
from collections import Counter

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

with open("/kaggle/working/tpu_az_v4_multickpt_results.json", "w") as f:
    json.dump(results_list, f, indent=2)
print(f"wrote /kaggle/working/tpu_az_v4_multickpt_results.json")

# Per-pid aggregation: shortest valid path across the (rot, ckpt) matrix.
by_pid = {}
for r in results_list:
    by_pid.setdefault(r["pid"], []).append(r)

print()
print("== per-pid (min over rot x ckpt) ==")
print(f"{'pid':>6}  {'best_ck':>7}  {'best_rot':>8}  {'best_path':>9}  "
      f"{'cells_valid/total':>17}  {'wall_max':>9}")
final_per_pid = {}
total_path = 0
n_solved = 0
ckpt_winners = Counter()
rot_winners = Counter()
for pid in sorted(by_pid.keys()):
    rs = by_pid[pid]
    valid = [r for r in rs if r["verify_ok"]]
    if valid:
        best = min(valid, key=lambda r: r["path_len"])
        final_per_pid[pid] = {
            "found": True,
            "best_path_len": best["path_len"],
            "best_path_idx": best["path_idx_orig"],
            "best_ckpt_idx": best["ckpt_idx"],
            "best_epoch": best["epoch"],
            "best_rot_idx": best["rot_idx"],
        }
        total_path += best["path_len"]
        n_solved += 1
        ckpt_winners[best["ckpt_idx"]] += 1
        rot_winners[best["rot_idx"]] += 1
        wall_max = max(r["wall_s"] for r in rs)
        print(f"{pid:>6}  {best['ckpt_idx']:>3}(ep{best['epoch']:>2})  "
              f"{best['rot_idx']:>8}  {best['path_len']:>9}  "
              f"{len(valid):>5}/{len(rs):<5}      {wall_max:>9.1f}")
    else:
        final_per_pid[pid] = {"found": False}
        wall_max = max(r["wall_s"] for r in rs)
        print(f"{pid:>6}  {'-':>7}  {'-':>8}  {'-':>9}  "
              f"{'0':>5}/{len(rs):<5}      {wall_max:>9.1f}")

print()
print(f"== summary ==")
print(f"solved (any cell): {n_solved}/{len(by_pid)}")
print(f"total_path (best per pid): {total_path}")

# The journey-insight payoff: which ckpts and rotations contributed wins?
print()
print("== best CHECKPOINT per pid (which epoch's V landscape won the most) ==")
for ck_i in sorted(ckpt_winners.keys()):
    ep = next(r["epoch"] for r in results_list if r["ckpt_idx"] == ck_i)
    bar = "#" * ckpt_winners[ck_i]
    print(f"  ckpt {ck_i} (ep {ep:>3}):  {ckpt_winners[ck_i]:>3}  {bar}")

print()
print("== best ROTATION per pid ==")
for ri in sorted(rot_winners.keys()):
    bar = "#" * rot_winners[ri]
    print(f"  rot {ri}:  {rot_winners[ri]:>3}  {bar}")

print()
if len(ckpt_winners) > 1:
    print(f"-> {len(ckpt_winners)} distinct checkpoints contributed wins; per-pid select is useful.")
else:
    print("-> All wins came from one checkpoint; the single-ckpt shareable is sufficient.")

csv_path = "/kaggle/working/tpu_az_v4_multickpt_submission.csv"
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

out_nb = HERE / "cayleypy-tpu-beam-az-v4-multickpt.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes, "
      f"{len(ALL_PID_STATES)} pid states embedded, {N_CKPTS} ckpts in manifest)")

# Title slugs to "cayleypy-megaminx-beam-az-v4-multickpt-shareable"; align id.
meta = {
    "id": "artgor/cayleypy-megaminx-beam-az-v4-multickpt-shareable",
    "title": "cayleypy megaminx beam az v4 multickpt shareable",
    "code_file": "cayleypy-tpu-beam-az-v4-multickpt.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,             # flip after dry-run
    "enable_gpu": False,
    "enable_tpu": True,
    "enable_internet": True,
    "dataset_sources": [
        "artgor/megaminx-tpu-artifacts",          # ep24 + rotations.npy
        "artgor/megaminx-az-v4-multi-ckpt",       # ep49 + ep74 + ep99
    ],
    "competition_sources": [],
    "kernel_sources": [],
}
out_meta = HERE / "kernel-metadata.json"
out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {out_meta}")
