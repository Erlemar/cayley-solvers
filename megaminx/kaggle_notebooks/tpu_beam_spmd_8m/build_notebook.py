"""Builds the TPU beam-search notebook with B=8M SPMD shared beam.

This is EXPERIMENTAL: a single B=8M beam shared across all 8 TPU cores via
cross-rank `xm.all_gather` at each topk barrier. Compared to the existing
data-parallel notebooks (8 independent B=1M beams):

  - Each pid sees a B=8M effective beam (vs B=1M per rank in the data-parallel
    setup) — meaningful for hard pids that don't solve at B=1M.
  - All 8 ranks work on ONE pid at a time — throughput is ~1/8 of the
    data-parallel version, so this is ONLY worth running on hard pids that
    a B=1M beam fails to crack.
  - Stack: m_curr_v3 V teacher + m23_v2 Q-shortlister + m_pi_v2 policy term
    (same as the m_curr_v3 fork). Plus optional sym K rotations.

Design (see CELL_7 for the implementation):
  - Sharded state: each rank holds B_LOCAL = B_GLOBAL/8 = 1M states.
  - Each step, ranks compute children + student topk + teacher rerank LOCALLY
    (each rank produces aB_LOCAL = 2M shortlist with teacher_v).
  - Cross-rank `xm.all_gather` of (teacher_v, shortlist_states, parent_local,
    move, child_hash) -> 16M global candidates.
  - Cross-rank dedup over the global hashes, then global topk(B_GLOBAL=8M)
    on deduped teacher_v.
  - Round-robin distribution: each rank takes sorted-top positions [R::8],
    giving exactly B_LOCAL = 1M chosen states per rank.
  - Tree: per-rank tree with (parent_rank, parent_local, move). Path
    reconstruction at end via cross-rank xm.all_gather of trees.

Output: cayleypy-tpu-beam-spmd-8m.ipynb
Run via: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_spmd_8m/build_notebook.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# --- Inputs ---------------------------------------------------------------

puzzle_info = json.loads((ROOT / "data" / "puzzle_info.json").read_text())

rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
ALL_PID_STATES = {}
for pid in range(len(rows)):
    s = [int(x) for x in rows[pid]["initial_state"].split(",")]
    ALL_PID_STATES[pid] = s
N_TOTAL_PIDS = len(ALL_PID_STATES)

# --- Cell sources ---------------------------------------------------------

CELL_0_README = '''\
# Megaminx beam search on Kaggle TPU - B=4M shared beam (SPMD v6, host-side aggregation)

Single B=4M beam sharded across all 8 TPU cores (B_LOCAL = 512K per rank).
Compared to the data-parallel notebooks (8 independent B=1M beams), this gives
a B=4M *effective* beam at the cost of ~1/8 throughput. **Use only on hard pids.**

**Production stack:**
m_curr_v3 V teacher (6M params) + m23_v2 sym-aware Q-shortlister (12.4M, alpha=2)
+ m_pi_v2 policy term (lambda=0.05) + B=4M shared beam, NUM_STEPS=120.

## v6 design (ported from JAX recovery)

v5 used in-body `xm.all_reduce(REDUCE_MIN/SUM)` to converge `found_step` and
walk back the path. That cross-rank carry mis-propagated on TPU v3 (and v5e-8
in the JAX port) and produced false-positive paths.

**v6 fix (host-side aggregation):**
- Each rank tracks ONLY its own local first-hit during the per-step XLA loop.
- After the loop, each rank dumps its (`fs`, `fpl`, `verify_state`, full tree)
  to `/tmp/rank_<r>_partial.npz`.
- `xm.rendezvous` (host-side gRPC barrier — not an XLA collective) syncs ranks.
- Rank 0 reads all 8 partials, picks the global winner host-side (smallest
  `fstep`, tie-break smallest rank), and walks back through cross-rank trees
  in pure numpy.

Trade-off: per-pid I/O of ~3 GB (8 ranks x 360 MB) added to `/tmp`. Cost is
~5-15s per pid on tmpfs.

## When to use this notebook

Run this **only on the ~50-100 hardest pids** that the data-parallel B=1M
notebooks failed to solve. For easy/medium pids, prefer
`cayleypy-megaminx-beam-m-curr-v3` (data-parallel B=1M, 8x throughput).

A natural workflow: take the worst path-length pids from current best CSV,
re-solve them here at B=4M, then merge.

## How to run

1. Fork this notebook on Kaggle ("Copy and Edit").
2. The dataset `artgor/megaminx-tpu-artifacts` (public) is auto-attached.
3. **Edit Cell 1 (CONFIG)** to set your pid range and K_SYM.
   Default: a 5-pid smoke at K=1, ~25min.
4. **Switch accelerator to TPU v3-8** in the right sidebar.
5. Run all cells.
6. After completion, download `/kaggle/working/spmd_results.json` and
   `/kaggle/working/spmd_submission.csv`.

## Memory budget per rank (B_LOCAL = 512K, B_GLOBAL = 4M)

| component | size |
|---|---|
| local states int8 (512K x 120) | 60 MB |
| local children int8 (512K x 24 x 120) | 1.44 GB |
| send_buckets int8 (8 x 128K x 128) | 128 MB |
| recv_buckets int8 (8 x 128K x 128) | 128 MB |
| model (m_curr_v3 + m23_v2 + m_pi_v2) | ~1 GB |
| tree per rank (NUM_STEPS=120, B_LOCAL=512K, ~6 bytes/entry) | ~360 MB |
| **total per rank (peak)** | **~3-4 GB** |

TPU v3-8 has 16 GB HBM per chip with 2 cores per chip. ~3-4 GB per rank fits.

## Caveats / known limits

1. **First-pid compile is expensive** (~30 min) - same as other forks.
2. **`/tmp` usage**: each rank writes ~360 MB per pid; total ~3 GB. On Kaggle
   TPU VMs `/tmp` is typically tmpfs (RAM-backed). If RAM is tight, the files
   are overwritten each pid so peak usage stays at ~3 GB.
3. **Host-side aggregation is per-pid**: rank 0 reads all 8 partials and
   walks back after each pid. Cost is ~5-15s per pid (negligible vs ~150-300s
   compute).
'''

PUZZLE_INFO_LITERAL = json.dumps(puzzle_info, separators=(",", ":"))
ALL_PID_STATES_LITERAL = json.dumps(ALL_PID_STATES, separators=(",", ":"))

CELL_1_CONFIG = f'''\
# ============================================================================
# CONFIG - EDIT THIS CELL, then Run All
# ============================================================================
#
# This is the SPMD B=8M shared-beam notebook. It runs ONE pid at a time
# across all 8 ranks, with the beam state sharded (B_LOCAL = 1M per rank).
#
# Three knobs:
#   START_PID, END_PID  - pid range (half-open). Default: 5-pid smoke.
#   K_SYM               - sym rotations per pid (1 default; 2 if you want
#                         extra diversity at the cost of 2x wall).
#
# Recommended kernel scopes:
#   smoke:   START=0, END=5,  K=1  -> ~25 min (validates pipeline)
#   medium:  START=0, END=30, K=1  -> ~3 h    (short rescue list)
#   full:    START=0, END=120,K=1  -> ~9 h    (top-120 hardest pids)
#
# At K=1, no symmetry - the shared B=8M beam IS the diversity vs the
# data-parallel K=4 setup.
# ============================================================================

START_PID = 0     # inclusive
END_PID = 8       # exclusive - first 8 pids for v6 fix validation
K_SYM = 1         # 1 (default) or 2 - sym rotations per pid

B_GLOBAL = 4 * 1024 * 1024  # 4M shared beam (v1 of this notebook OOM'd at 8M -
                            # XLA workspace exceeded per-rank HBM with 2 ranks/chip).
                            # 4M still gives 4x the data-parallel 1M effective beam.
                            # Future: 8M needs chunked all_gather (split into 2x 4M).

# ============================================================================
# Do not edit below this line
# ============================================================================
'''

CELL_2_SETUP = '''\
# Cell 2: Setup, _Tee logging, system info.
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

assert 0 <= START_PID < END_PID <= ''' + str(N_TOTAL_PIDS) + ''', \\
    f"invalid pid range: START_PID={START_PID}, END_PID={END_PID}"
assert K_SYM in (1, 2), f"K_SYM must be 1 or 2 (got {K_SYM})"
assert B_GLOBAL % N_DEVICES == 0, f"B_GLOBAL ({B_GLOBAL}) must be divisible by {N_DEVICES}"
B_LOCAL = B_GLOBAL // N_DEVICES
N_PIDS = END_PID - START_PID
print(f"\\n== CONFIG ==")
print(f"pid range: [{START_PID}, {END_PID})  ({N_PIDS} pids)")
print(f"K_SYM: {K_SYM}")
print(f"B_GLOBAL: {B_GLOBAL:,}  B_LOCAL: {B_LOCAL:,}")

_proj_compile_h = 0.5
_proj_compute_h = (N_PIDS * K_SYM) * 300 / 3600  # 300s/pair midpoint estimate
_proj_total_h = _proj_compile_h + _proj_compute_h
print(f"estimated wall: {_proj_compile_h:.1f}h compile + {_proj_compute_h:.1f}h compute = {_proj_total_h:.1f}h")
if _proj_total_h > 8.5:
    print(f"WARNING: estimated wall {_proj_total_h:.1f}h exceeds 8.5h target.")
    print(f"  Consider reducing pid range - Kaggle kills TPU kernels around 9h.")
'''

CELL_3_PUZZLE = f'''\
# Cell 3: Puzzle + ALL pid states embedded; slice to TEST_PIDS via config.
PUZZLE_INFO = json.loads({PUZZLE_INFO_LITERAL!r})
GENERATORS = PUZZLE_INFO["generators"]
SOLVED = tuple(PUZZLE_INFO["central_state"])
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

_ALL_PID_STATES = json.loads({ALL_PID_STATES_LITERAL!r})
ALL_PID_STATES = {{int(k): v for k, v in _ALL_PID_STATES.items()}}
TEST_PIDS = list(range(START_PID, END_PID))
PID_STATES = {{pid: ALL_PID_STATES[pid] for pid in TEST_PIDS}}
print(f"selected {{len(TEST_PIDS)}} pids: [{{TEST_PIDS[0]}}..{{TEST_PIDS[-1]}}]")
'''

CELL_4_MODEL = '''\
# Cell 4: ResMLPDistance.
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
# Cell 5: Load m_curr_v3 (V) + m23_v2 (Q) + m_pi_v2 (policy) + rotations.
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

m05_path = dataset_root / "m_curr_v3_epoch_0499.pt"
m23v2_path = dataset_root / "m23_v2_epoch_0499.pt"
m_pi_path = dataset_root / "m_pi_v2_epoch_0199.pt"
rotations_path = dataset_root / "rotations.npy"
for p in (m05_path, m23v2_path, m_pi_path, rotations_path):
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}")

def _load_state_dict_strip(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    return {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}

M05_STATE_DICT_CPU = _load_state_dict_strip(m05_path)
M23V2_STATE_DICT_CPU = _load_state_dict_strip(m23v2_path)
M_PI_STATE_DICT_CPU = _load_state_dict_strip(m_pi_path)
print(f"V teacher (m_curr_v3): {len(M05_STATE_DICT_CPU)} keys")
print(f"Q student (m23_v2):    {len(M23V2_STATE_DICT_CPU)} keys")
print(f"policy (m_pi_v2):      {len(M_PI_STATE_DICT_CPU)} keys")

import numpy as np
rotations_np = np.load(rotations_path)
print(f"rotations: {rotations_np.shape} dtype={rotations_np.dtype}")
'''

CELL_6_SYM = '''\
# Cell 6: sym ensemble - K_SYM rotations per pid.
SYM_SEED = 0
ALPHA_QSHORT = 2.0
LAMBDA_POLICY = 0.05

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
            raise ValueError(f"rotation {rot_idx} not a symmetry")
        conj_idx[move_name_to_idx[nm]] = move_name_to_idx[perm_to_name[conj_perm]]
    ROTATIONS.append({"rot_idx": rot_idx, "R": R, "R_inv": R_inv, "conj_idx": conj_idx})

print(f"prepared {len(ROTATIONS)} rotations")

def apply_rotation_state(state, R, R_inv):
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))

PID_ROT_PAIRS = []
for pid in TEST_PIDS:
    for ri in range(len(ROTATIONS)):
        PID_ROT_PAIRS.append((pid, ri))
print(f"total (pid, rotation) pairs to solve: {len(PID_ROT_PAIRS)}")
'''

CELL_7_BEAM = '''\
# Cell 7: SPMD v6 - host-side cross-rank winner selection (ported from JAX recovery).
#
# v5 used in-body xm.all_reduce(REDUCE_MIN/SUM) to converge found_step/found_pos
# across ranks every iteration. That carry mis-propagated on TPU v3 — observed
# fingerprint: "found_step = i - 1" false-positives across 4 different cross-rank
# architectures (see spmd_dead_end.md).
#
# The JAX port (spmd_jax_recovery.md) confirmed the same failure mode on v5e-8
# and isolated the bug to the in-body cross-rank reduce specifically. Fix that
# worked in JAX, ported here: REMOVE all cross-rank reduces from inside the
# step loop. Each rank tracks ONLY its own local first-hit. After the loop
# finishes, each rank dumps its (found_step, found_pos_local, verify_state,
# tree) to a per-rank temp file; rank 0 reads all 8, picks the global winner
# host-side (smallest fstep, tie-break smallest rank), and walks back.
#
# Per-rank state (sharded by hash, unchanged from v5):
#   states (B_LOCAL, state_size) - this rank's owned states in the global beam
#   tree_parent_local  (NUM_STEPS, B_LOCAL) int32 - parent's local idx in owning rank
#   tree_parent_rank   (NUM_STEPS, B_LOCAL) int8  - rank that owned the parent
#   tree_move          (NUM_STEPS, B_LOCAL) int8
#   verify_state       (state_size,) int8         - captured chosen_state at first
#                                                   local V0 hit (diagnostic)
#
# Per step (UNCHANGED from v5 *except* the convergence reduce is removed):
#   1. Each rank generates 24 children of its B_LOCAL states.
#   2. For each child, hash -> owner_rank.
#   3. Per owner rank S, top K_PER_PEER candidates by student_score.
#   4. Pack (state, parent_local, move) into uint8 records, bucketed by owner.
#   5. xm.all_to_all routes each bucket to its owner rank.
#   6. Owner rank: dedup, teacher rerank, topk(B_LOCAL).
#   7. Update state and tree (parent_rank = sender_rank).
#   8. LOCAL V0 detection. found_step = j on first local hit, otherwise unchanged.
#      NO cross-rank reduce. found_pos_rank dropped (was always = this rank when
#      this rank found V0; host re-derives via argmin over per-rank fstep).
#
# Cross-rank coordination is OUTSIDE the XLA-compiled per-step body, done via
# xm.rendezvous (host-side gRPC barrier — not an XLA collective) + filesystem.

PACK_SIZE = 128  # state(120) + parent_local(4 bytes) + move(1 byte) + pad(3)


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


def beam_solve_spmd_local(initial_state_list, teacher, student, policy,
                          all_moves_dev, V0_dev, hash_vec,
                          device, B_local, world_size, rank,
                          num_steps, internal_bs, alpha, lambda_policy,
                          xla_available, state_size, n_gen):
    """SPMD shared-beam per-rank solver. Returns per-rank scalars + tree on CPU.

    Each rank tracks ONLY its own local first-hit; no in-body cross-rank reduce.
    Aggregation happens host-side in _mp_fn after all ranks finish the per-pid loop.
    """
    if xla_available:
        import torch_xla.core.xla_model as xm

    K_PER_PEER = (int(alpha) * B_local) // world_size
    aB_local = K_PER_PEER * world_size  # = alpha * B_local (assuming divisible)

    state = torch.tensor(initial_state_list, dtype=torch.int8)
    V0_cpu_local = torch.tensor(SOLVED, dtype=torch.int8)
    if torch.equal(state, V0_cpu_local):
        return {"trivial": True, "fs": 0, "fpl": 0,
                "verify_state": np.asarray(SOLVED, dtype=np.int8),
                "tp_local": None, "tp_rank": None, "tmove": None,
                "min_v_trajectory": [], "wall_s": 0.0, "timings": {}}

    BIG_VAL_INF = torch.tensor(1e6, dtype=torch.float32, device=device)
    BIG_F32 = torch.tensor(1e9, dtype=torch.float32, device=device)

    t_start = time.time()

    # ===== Step 0 init: every rank computes the same 24 seeds deterministically.
    # Replicated padding makes the rank-local beam B_LOCAL even though only 24
    # seeds are unique. The first all_to_all at j=1 distributes them by hash.
    states0 = state.unsqueeze(0).to(device)
    seeds = states0[:, all_moves_dev].view(-1, state_size)  # (24, 120)
    if 24 < B_local:
        pad = seeds[-1:].expand(B_local - 24, -1)
        states = torch.cat([seeds, pad], dim=0)
    else:
        states = seeds[:B_local]
    if xla_available:
        xm.mark_step()
    t_step0 = time.time() - t_start

    # Tree storage (per-rank, NUM_STEPS x B_LOCAL).
    tree_parent_local = torch.zeros((num_steps, B_local), dtype=torch.int32, device=device)
    tree_parent_rank = torch.zeros((num_steps, B_local), dtype=torch.int8, device=device)
    tree_move = torch.full((num_steps, B_local), -1, dtype=torch.int8, device=device)

    # Step 0 tree_move: position k stored seed-move = k for k<24, else 23 (last seed).
    move0_arr = torch.arange(24, dtype=torch.int8, device=device)
    if 24 < B_local:
        move0_pad = torch.full((B_local - 24,), 23, dtype=torch.int8, device=device)
        move0_full = torch.cat([move0_arr, move0_pad], dim=0)
    else:
        move0_full = move0_arr[:B_local]
    tree_move[0] = move0_full

    min_v_log = torch.full((num_steps,), 1e6, dtype=torch.float32, device=device)
    # Local-only carries. Each rank only ever sees its OWN first-hit.
    found_step = torch.tensor(-1, dtype=torch.int32, device=device)
    found_pos_local = torch.tensor(-1, dtype=torch.int32, device=device)
    verify_state = torch.zeros((state_size,), dtype=torch.int8, device=device)

    # Initial check: V0 in step-0 states (LOCAL only).
    eq0 = (states == V0_dev).all(dim=1)
    any0 = eq0.any()
    pos0 = eq0.to(torch.int32).argmax()
    found_step = torch.where(any0, torch.tensor(0, dtype=torch.int32, device=device), found_step)
    found_pos_local = torch.where(any0, pos0.to(torch.int32), found_pos_local)
    v0_state_step0 = states[pos0]
    verify_state = torch.where(any0, v0_state_step0, verify_state)

    # Pre-compute static-shape helpers.
    n_total = B_local * n_gen
    flat_idx = torch.arange(n_total, dtype=torch.int64, device=device)
    parent_local_per_child = (flat_idx // n_gen).to(torch.int32)  # (n_total,) int32
    move_per_child = (flat_idx % n_gen).to(torch.int8)  # (n_total,) int8
    arange_aB = torch.arange(aB_local, dtype=torch.int64, device=device)
    sender_rank_per_recv = (arange_aB // K_PER_PEER).to(torch.int8)  # (aB_local,) int8

    t_loop = time.time()
    first_iter_t = None

    for j in range(1, num_steps):
        t_iter = time.time()

        # ===== Local: forward passes =====
        student_q = _model_predict_chunked(student, states, internal_bs, xla_available)
        policy_q = _model_predict_chunked(policy, states, internal_bs, xla_available)
        policy_q_f32 = policy_q.to(torch.float32)
        log_pi = policy_q_f32 - torch.logsumexp(policy_q_f32, dim=-1, keepdim=True)
        neg_log_pi_flat = -log_pi.view(-1)

        # Children
        neighbors = states[:, all_moves_dev].view(-1, state_size)  # (n_total, state_size)

        student_q_flat = student_q.view(-1).to(torch.float32)
        student_score = student_q_flat + lambda_policy * neg_log_pi_flat

        # Hash and owner rank for each child
        h = (neighbors.to(torch.int64) * hash_vec).sum(dim=1)
        owner_rank = (h % world_size).to(torch.int32)

        # ===== Per-rank top-K_PER_PEER bucketing =====
        send_buckets = torch.zeros((world_size, K_PER_PEER, PACK_SIZE), dtype=torch.uint8, device=device)
        zero_state = torch.zeros((K_PER_PEER, state_size), dtype=torch.uint8, device=device)
        zero_int32 = torch.zeros(K_PER_PEER, dtype=torch.int32, device=device)
        zero_int8 = torch.zeros(K_PER_PEER, dtype=torch.int8, device=device)

        for S in range(world_size):
            mask_S = (owner_rank == S)
            score_for_S = torch.where(mask_S, student_score, BIG_F32)
            top_v_S, top_idx_S = torch.topk(score_for_S, K_PER_PEER, largest=False)
            is_pad_S = top_v_S >= (BIG_F32 * 0.5)  # padded entries (no real candidate for owner S)

            sel_states = neighbors[top_idx_S]  # (K_PER_PEER, state_size) int8
            sel_parent_local = parent_local_per_child[top_idx_S]  # (K_PER_PEER,) int32
            sel_move = move_per_child[top_idx_S]  # (K_PER_PEER,) int8

            # Zero out padding entries (so receiver detects via state.sum() == 0).
            sel_states_u8 = torch.where(is_pad_S.unsqueeze(-1), zero_state, sel_states.to(torch.uint8))
            sel_parent_local_z = torch.where(is_pad_S, zero_int32, sel_parent_local)
            sel_move_z = torch.where(is_pad_S, zero_int8, sel_move)

            # Pack: bytes 0..119 = state, 120..123 = parent_local (LE), 124 = move, 125..127 = zero.
            send_buckets[S, :, 0:120] = sel_states_u8
            send_buckets[S, :, 120] = (sel_parent_local_z & 0xFF).to(torch.uint8)
            send_buckets[S, :, 121] = ((sel_parent_local_z >> 8) & 0xFF).to(torch.uint8)
            send_buckets[S, :, 122] = ((sel_parent_local_z >> 16) & 0xFF).to(torch.uint8)
            send_buckets[S, :, 123] = ((sel_parent_local_z >> 24) & 0xFF).to(torch.uint8)
            send_buckets[S, :, 124] = sel_move_z.to(torch.uint8)

        if xla_available:
            xm.mark_step()

        # ===== xm.all_to_all: bucket S goes to rank S =====
        if xla_available and world_size > 1:
            recv_buckets = xm.all_to_all(send_buckets, split_dimension=0,
                                          concat_dimension=0, split_count=world_size)
        else:
            recv_buckets = send_buckets  # single-rank fallback

        if xla_available:
            xm.mark_step()

        # ===== Unpack received candidates: (W, K_PER_PEER, PACK_SIZE) -> (aB_local, ...) =====
        recv_flat = recv_buckets.reshape(-1, PACK_SIZE)  # (aB_local, PACK_SIZE)
        recv_states_u8 = recv_flat[:, 0:120]
        recv_states = recv_states_u8.to(torch.int8)  # values 0..119, int8 == uint8 below 128
        recv_parent_local = (
            recv_flat[:, 120].to(torch.int32) |
            (recv_flat[:, 121].to(torch.int32) << 8) |
            (recv_flat[:, 122].to(torch.int32) << 16) |
            (recv_flat[:, 123].to(torch.int32) << 24)
        )
        recv_move = recv_flat[:, 124].to(torch.int8)
        # sender_rank: after concat_dim=0, recv_buckets[i,:,:] from rank i.
        # After reshape(-1, ...): rows [i*K_PER_PEER : (i+1)*K_PER_PEER] from rank i.
        recv_sender_rank = sender_rank_per_recv

        # Detect padding (all-zero state). Real states have sum = sum(0..119) = 7140.
        recv_state_sum = recv_states.sum(dim=1, dtype=torch.int32)
        is_padding = (recv_state_sum == 0)

        # In-rank dedup
        recv_h = (recv_states.to(torch.int64) * hash_vec).sum(dim=1)
        sort_h, sort_idx = torch.sort(recv_h)
        is_dup_sorted = torch.cat([
            torch.zeros(1, dtype=torch.bool, device=device),
            (sort_h[1:] == sort_h[:-1]),
        ])
        restore = torch.argsort(sort_idx)
        dup_mask = is_dup_sorted[restore]

        # Teacher rerank
        teacher_v = _model_predict_chunked(teacher, recv_states, internal_bs, xla_available)
        teacher_score = teacher_v.to(torch.float32)
        teacher_score_masked = torch.where(dup_mask | is_padding, BIG_VAL_INF, teacher_score)

        # Topk B_local
        top_v_keep, keep_idx = torch.topk(teacher_score_masked, B_local, largest=False)

        chosen_states = recv_states[keep_idx]
        chosen_parent_local = recv_parent_local[keep_idx]
        chosen_parent_rank = recv_sender_rank[keep_idx]
        chosen_move = recv_move[keep_idx]

        # Update state and tree
        states = chosen_states
        tree_parent_local[j] = chosen_parent_local
        tree_parent_rank[j] = chosen_parent_rank
        tree_move[j] = chosen_move
        min_v_log[j] = top_v_keep[0].to(torch.float32)

        # ===== V0 detection — LOCAL ONLY, no cross-rank reduce =====
        # Belt-and-suspenders: gate on non-padding (chosen_state.sum != 0) since
        # padding states are all zeros; real V0 sums to 7140. Without this filter
        # the all-zero padded state could spuriously match V0 if its hash collided.
        chosen_state_sum = chosen_states.sum(dim=1, dtype=torch.int32)
        chosen_is_real = (chosen_state_sum != 0)
        eq_v0 = (chosen_states == V0_dev).all(dim=1) & chosen_is_real
        any_hit = eq_v0.any()
        pos_hit = eq_v0.to(torch.int32).argmax()
        is_first_hit = (found_step == -1) & any_hit
        j_tensor = torch.tensor(j, dtype=torch.int32, device=device)
        found_step = torch.where(is_first_hit, j_tensor, found_step)
        found_pos_local = torch.where(is_first_hit, pos_hit.to(torch.int32), found_pos_local)
        # Capture the chosen state at first local hit, for diagnostic verify.
        v0_state_current = chosen_states[pos_hit]
        verify_state = torch.where(is_first_hit, v0_state_current, verify_state)

        if xla_available:
            xm.mark_step()
        if first_iter_t is None:
            first_iter_t = time.time() - t_iter

    t_loop_end = time.time()

    if xla_available:
        xm.mark_step()

    # ===== Pull per-rank scalars + trees to CPU for host-side aggregation =====
    # The master aggregator (rank 0 in _mp_fn) reads all 8 ranks' partial files,
    # picks the global winner, and walks back.
    fs_h = int(found_step.item())
    fpl_h = int(found_pos_local.item())
    verify_state_h = verify_state.detach().cpu().numpy().astype(np.int8)
    min_v_h = min_v_log.detach().cpu().tolist()
    tp_local_h = tree_parent_local.detach().cpu().numpy().astype(np.int32)
    tp_rank_h = tree_parent_rank.detach().cpu().numpy().astype(np.int8)
    tmove_h = tree_move.detach().cpu().numpy().astype(np.int8)
    t_total = time.time() - t_start
    return {
        "trivial": False,
        "fs": fs_h, "fpl": fpl_h,
        "verify_state": verify_state_h,
        "tp_local": tp_local_h, "tp_rank": tp_rank_h, "tmove": tmove_h,
        "min_v_trajectory": min_v_h, "wall_s": t_total,
        "timings": {"step0": t_step0, "loop": t_loop_end - t_loop,
                    "first_iter": first_iter_t or 0.0, "total": t_total},
    }
'''

CELL_8_MP_FN = '''\
# Cell 8: Per-rank worker. ALL ranks work on each (pid, rotation) pair together.
#
# Flow per (pid, rot) pair:
#   1. All 8 ranks run beam_solve_spmd_local (returns per-rank scalars + tree
#      on CPU). No cross-rank reduce inside this function.
#   2. Each rank writes its (fs, fpl, verify_state, tree) to /tmp/rank_<r>_partial.npz
#      (overwritten per pid; ~360 MB per rank at B_LOCAL=512K).
#   3. xm.rendezvous("partial_done_pid_<i>") syncs all ranks. This is a HOST-SIDE
#      gRPC barrier, NOT an XLA collective — it's safe regardless of the in-body
#      collective bug that motivated this v6 redesign.
#   4. Rank 0 reads all 8 partials, picks global winner (smallest fstep; tie-break
#      smallest rank), walks back through trees host-side using numpy indexing,
#      verifies the path, and appends the result.
#   5. xm.rendezvous("aggregated_pid_<i>") syncs before next iteration so ranks
#      don't overwrite their partials while rank 0 is still reading.

INTERNAL_BS = 32768
NUM_STEPS = 120

PARTIAL_DIR = "/tmp/spmd_partials"


def _host_walkback(per_rank, fs_h, winner_rank, winner_pos):
    """Walk back through per-rank trees to recover the path. All numpy.

    per_rank[r] = {"tp_local": (NUM_STEPS, B_LOCAL) int32,
                   "tp_rank":  (NUM_STEPS, B_LOCAL) int8,
                   "tmove":    (NUM_STEPS, B_LOCAL) int8}

    Returns list of move indices (oldest -> newest).
    """
    path_idx_list = []
    cur_rank = winner_rank
    cur_pos = winner_pos
    for j in range(fs_h, -1, -1):
        m = int(per_rank[cur_rank]["tmove"][j, cur_pos])
        if m >= 0:
            path_idx_list.append(m)
        if j > 0:
            new_rank = int(per_rank[cur_rank]["tp_rank"][j, cur_pos])
            new_pos = int(per_rank[cur_rank]["tp_local"][j, cur_pos])
            cur_rank = new_rank
            cur_pos = new_pos
    return list(reversed(path_idx_list))


def _aggregate_pid(world_size, V0_np, pid, rot_i, conj_idx, s0,
                   partial_paths, t_pid_start, first_iter_log,
                   min_v_traj_rank0):
    """Read all per-rank partial files and pick global winner. Rank 0 only."""
    per_rank = []
    for r in range(world_size):
        d = np.load(partial_paths[r], allow_pickle=False)
        per_rank.append({
            "fs": int(d["fs"]),
            "fpl": int(d["fpl"]),
            "verify_state": d["verify_state"],
            "tp_local": d["tp_local"],
            "tp_rank": d["tp_rank"],
            "tmove": d["tmove"],
        })

    INT_MAX_SENT = 2 ** 30
    fs_per_rank = np.array([r_["fs"] for r_ in per_rank], dtype=np.int64)
    fs_signed = np.where(fs_per_rank >= 0, fs_per_rank, INT_MAX_SENT)
    global_min_step = int(fs_signed.min())

    per_rank_summary = []
    for r in range(world_size):
        vs = per_rank[r]["verify_state"].tolist()
        per_rank_summary.append({
            "rank": r,
            "fstep": per_rank[r]["fs"],
            "fpos": per_rank[r]["fpl"],
            "vstate_head": vs[:12],
            "vstate_is_V0": vs == V0_np.tolist(),
        })

    if global_min_step >= INT_MAX_SENT:
        return {
            "pid": pid, "rot_idx": rot_i, "found": False, "verify_ok": False,
            "path_len": 0, "path_idx_orig": [], "path_idx_rotated": [],
            "wall_s": time.time() - t_pid_start, "first_iter": first_iter_log,
            "per_rank_summary": per_rank_summary,
            "winner_rank": -1, "found_step": -1,
            "min_v_trajectory": min_v_traj_rank0,
        }

    winner_ranks_arr = np.where(fs_signed == global_min_step)[0]
    winner_rank = int(winner_ranks_arr[0])
    fs_winner = int(fs_per_rank[winner_rank])
    fpl_winner = int(per_rank[winner_rank]["fpl"])

    detected_state = per_rank[winner_rank]["verify_state"]
    detected_is_V0 = bool(np.array_equal(detected_state, V0_np))

    # Host-side walkback through cross-rank trees.
    path_rotated = _host_walkback(per_rank, fs_winner, winner_rank, fpl_winner)
    path_orig = [conj_idx[m] for m in path_rotated]

    # Verify the path against the un-rotated initial state.
    cur = list(s0)
    for m in path_orig:
        gen = GENERATORS[MOVE_NAMES[m]]
        cur = [cur[g] for g in gen]
    verify_ok = (tuple(cur) == SOLVED)

    return {
        "pid": pid, "rot_idx": rot_i, "found": True, "verify_ok": verify_ok,
        "detected_is_V0": detected_is_V0,
        "path_len": len(path_orig), "path_idx_orig": path_orig if verify_ok else [],
        "path_idx_rotated": path_rotated,
        "wall_s": time.time() - t_pid_start, "first_iter": first_iter_log,
        "winner_rank": winner_rank, "found_step": fs_winner,
        "per_rank_summary": per_rank_summary,
        "min_v_trajectory": min_v_traj_rank0,
    }


def _mp_fn(index, pid_rot_pairs):
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    try:
        rank = xm.get_ordinal()
        world_size = xm.xrt_world_size()
    except Exception:
        rank = index
        world_size = 8
    print(f"[rank {rank}/{world_size}] device={device}  B_LOCAL={B_LOCAL:,}  B_GLOBAL={B_GLOBAL:,}", flush=True)

    os.makedirs(PARTIAL_DIR, exist_ok=True)

    teacher = ResMLPDistance(state_size=120, num_classes=120, hidden_dims=(2048, 512),
                             num_res_blocks=2, embed_dim=16, output_dim=1)
    teacher.load_state_dict(M05_STATE_DICT_CPU, strict=False)
    teacher = teacher.to(torch.bfloat16).to(device).eval()

    student = ResMLPDistance(state_size=120, num_classes=120, hidden_dims=(2048, 1024),
                             num_res_blocks=2, embed_dim=16, output_dim=24)
    missing_s, unexpected_s = student.load_state_dict(M23V2_STATE_DICT_CPU, strict=False)
    if missing_s or unexpected_s:
        print(f"[rank {rank}] student load: missing={len(missing_s)} unexpected={len(unexpected_s)}", flush=True)
    student = student.to(torch.bfloat16).to(device).eval()

    policy = ResMLPDistance(state_size=120, num_classes=120, hidden_dims=(2048, 512),
                            num_res_blocks=2, embed_dim=16, output_dim=24)
    missing_p, unexpected_p = policy.load_state_dict(M_PI_STATE_DICT_CPU, strict=False)
    if missing_p or unexpected_p:
        print(f"[rank {rank}] policy load: missing={len(missing_p)} unexpected={len(unexpected_p)}", flush=True)
    policy = policy.to(torch.bfloat16).to(device).eval()

    all_moves_dev = all_moves.to(device)
    V0_dev = V0_cpu.to(device)
    V0_np = V0_cpu.numpy().astype(np.int8)
    hash_vec = _make_hash_vec(STATE_SIZE, seed=0).to(device)

    results_list = []
    for i, (pid, rot_i) in enumerate(pid_rot_pairs):
        rot = ROTATIONS[rot_i]
        s0 = PID_STATES[pid]
        s_to_solve = apply_rotation_state(s0, rot["R"], rot["R_inv"])

        t_pid_start = time.time()
        r = beam_solve_spmd_local(
            list(s_to_solve), teacher, student, policy,
            all_moves_dev, V0_dev, hash_vec,
            device, B_LOCAL, world_size, rank,
            NUM_STEPS, INTERNAL_BS, ALPHA_QSHORT, LAMBDA_POLICY,
            XLA_AVAILABLE, STATE_SIZE, N_GEN,
        )

        # Each rank writes its per-rank partial (overwritten next pid).
        partial_path = os.path.join(PARTIAL_DIR, f"rank_{rank}_partial.npz")

        if r["trivial"]:
            # Initial state IS V0. Write a placeholder partial so the aggregator
            # still sees consistent shape; the rank-0 short-circuit will skip the
            # actual aggregation.
            np.savez(partial_path,
                     fs=np.int32(0),
                     fpl=np.int32(0),
                     verify_state=np.asarray(SOLVED, dtype=np.int8),
                     tp_local=np.zeros((1, 1), dtype=np.int32),
                     tp_rank=np.zeros((1, 1), dtype=np.int8),
                     tmove=np.zeros((1, 1), dtype=np.int8))
        else:
            np.savez(partial_path,
                     fs=np.int32(r["fs"]),
                     fpl=np.int32(r["fpl"]),
                     verify_state=r["verify_state"],
                     tp_local=r["tp_local"],
                     tp_rank=r["tp_rank"],
                     tmove=r["tmove"])

        # Host-side gRPC barrier — sync all ranks before rank 0 reads partials.
        xm.rendezvous(f"partial_done_pid_{i}")

        if rank == 0:
            t_agg = time.time()
            if r["trivial"]:
                result = {
                    "pid": pid, "rot_idx": rot_i, "found": True, "verify_ok": True,
                    "path_len": 0, "path_idx_orig": [], "path_idx_rotated": [],
                    "wall_s": time.time() - t_pid_start, "first_iter": 0.0,
                    "winner_rank": 0, "found_step": -1,
                    "per_rank_summary": [],
                    "min_v_trajectory": [],
                }
            else:
                partial_paths = [os.path.join(PARTIAL_DIR, f"rank_{rr}_partial.npz")
                                 for rr in range(world_size)]
                result = _aggregate_pid(
                    world_size, V0_np, pid, rot_i, rot["conj_idx"], s0,
                    partial_paths, t_pid_start,
                    r["timings"].get("first_iter", 0.0),
                    r.get("min_v_trajectory", []),
                )
            agg_wall = time.time() - t_agg

            mvt = result.get("min_v_trajectory", [])
            last_v = mvt[-1] if mvt else None
            print(f"[SPMD] {i+1}/{len(pid_rot_pairs)} pid={pid:>4} rot={rot_i} "
                  f"found={result['found']} verify={result['verify_ok']} "
                  f"path_len={result['path_len']:>3} "
                  f"winner_rank={result.get('winner_rank', -1)} found_step={result.get('found_step', -1)} "
                  f"wall={result['wall_s']:.1f}s agg={agg_wall:.1f}s last_min_V={last_v}", flush=True)

            results_list.append(result)
            try:
                with open("/kaggle/working/spmd_partial.json", "w") as f:
                    json.dump(results_list, f)
            except Exception as e:
                print(f"[rank 0] WARN partial save failed: {e}", flush=True)

        # Sync before next pid (so non-rank-0 ranks don't overwrite their partial
        # before rank 0 has read it).
        xm.rendezvous(f"aggregated_pid_{i}")

    if rank == 0:
        with open("/kaggle/working/spmd_final.json", "w") as f:
            json.dump(results_list, f)
        print(f"[SPMD] done, processed {len(results_list)} pairs", flush=True)
'''

CELL_9_SPAWN = '''\
# Cell 9: Launch via xmp.spawn.

print()
print(f"== xmp.spawn SPMD shared-beam: K_SYM={K_SYM}, alpha={ALPHA_QSHORT}, "
      f"B_GLOBAL={B_GLOBAL:,}, B_LOCAL={B_LOCAL:,}, internal_bs=32768, max_steps=120 ==")
print(f"== pid range: [{START_PID}, {END_PID})  ({len(TEST_PIDS)} pids, "
      f"{len(PID_ROT_PAIRS)} pid-rot pairs) ==")
print()

import glob
for stale in glob.glob("/kaggle/working/spmd_*.json"):
    try: os.remove(stale)
    except: pass

import torch_xla.distributed.xla_multiprocessing as xmp

t_launch = time.time()
xmp.spawn(_mp_fn, args=(PID_ROT_PAIRS,), start_method="fork", nprocs=None)
t_launch_total = time.time() - t_launch
print(f"== xmp.spawn complete in {t_launch_total:.1f}s ==")
'''

CELL_10_AGGREGATE = '''\
# Cell 10: Aggregate. SPMD writes a single shared file (rank 0 only).
import csv as _csv
import glob

results_list = []
for path in ["/kaggle/working/spmd_final.json", "/kaggle/working/spmd_partial.json"]:
    if Path(path).exists():
        with open(path) as f:
            results_list = json.load(f)
        print(f"loaded {len(results_list)} results from {path}")
        break

with open("/kaggle/working/spmd_results.json", "w") as f:
    json.dump(results_list, f, indent=2)

by_pid = {}
for r in results_list:
    by_pid.setdefault(r["pid"], []).append(r)

print()
print("== per-pid (min over K rotations) ==")
print(f"{'pid':>6}  {'best_rot':>8}  {'best_path':>9}  {'all_found/K':>12}  "
      f"{'wall':>9}  {'last_min_V':>10}")
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

csv_path = "/kaggle/working/spmd_submission.csv"
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
        lines = src.splitlines(keepends=True)
        cell["source"] = lines

out_nb = HERE / "cayleypy-tpu-beam-spmd-8m.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes, {len(ALL_PID_STATES)} pid states embedded)")

meta = {
    "id": "artgor/cayleypy-megaminx-beam-spmd-b-8m",
    "title": "cayleypy megaminx beam SPMD B 8M",  # avoid '=' which Kaggle slugifies to 'b-8m'
    "code_file": "cayleypy-tpu-beam-spmd-8m.ipynb",
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
