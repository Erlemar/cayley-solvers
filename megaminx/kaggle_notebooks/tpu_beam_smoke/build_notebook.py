"""Builds the TPU multi-core qshort+sym-ensemble notebook (.ipynb).

v10 — adds qshort + sym-ensemble to the multi-core threading port.
  * qshort: m23_v2 (Q-head) shortlists alpha*B candidates per step; m05 (V) reranks.
    Static-shape via +inf masking on duplicates.
  * sym-ensemble: K=4 rotations from rotations.npy. Each (pid, rotation) is an
    independent beam search; paths translated back to original frame via
    precomputed conjugation maps; per-pid take min over rotations.
  * Smoke: 5 pids x K=4 = 20 (pid, rotation) pairs across 8 TPU cores.

Output: cayleypy-tpu-beam-smoke.ipynb in the same directory.
Run via the project venv: .venv/Scripts/python.exe megaminx/kaggle_notebooks/tpu_beam_smoke/build_notebook.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # megaminx/
HERE = Path(__file__).resolve().parent

# --- Inputs ---------------------------------------------------------------

puzzle_info = json.loads((ROOT / "data" / "puzzle_info.json").read_text())

# v19b: second half of B=1M K=4 split. pids 500-1000.
# Per-pair ~150s confirmed in v19a-fix. 501 pids × 4 / 8 ranks ≈ 251 per rank.
# Per-rank wall: 30min compile + 251×150s = ~10.7h. Will hit 9h kill at ~80%.
# Recover via follow-up kernel for missing pids.
TEST_PIDS = list(range(500, 1001))

rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
pid_states = {}
for pid in TEST_PIDS:
    s = [int(x) for x in rows[pid]["initial_state"].split(",")]
    pid_states[pid] = s

# --- Cell sources ---------------------------------------------------------

CELL_0_README = '''\
# Megaminx beam search on Kaggle TPU — qshort + sym-ensemble (v10)

Multi-core port of the production GPU stack:
  m05 V teacher + m23_v2 Q-shortlister + K=4 sym-ensemble + beam=131k.

This is what produced our 82,481 best on GPU (with --qshort, --sym-ensemble,
--qshort-student m23_v2). v10 ports those features to TPU v3-8 (8 cores) via
threading.

## v10 vs prior versions

| version | feature | result |
|---|---|---|
| v3-v5 | single-core, no qshort, no sym | works, ~118h projected for full-1001 |
| v8 | multi-core threading, no qshort, no sym | works, ~17h projected |
| **v10** | **multi-core + qshort + sym-K=4** | **target ~9h projected, prod-quality paths** |

## Multi-core via threading (NOT xmp.spawn)

Kaggle TPU v3-8 is single-host — all 8 cores accessible from one process via
xla:0..xla:7. xmp.spawn fails with "Expected 8 worker addresses, got 1"
(it expects multi-host TPU pod topology). We use Python threading:

  for rank in range(N_DEVICES):
      threading.Thread(target=_worker, args=(rank, ..., device=xla:rank))

Each thread pins to its xla:r device. Static graph compiles once per shape
signature per core.

## qshort step (XLA-static-shape)

Original CUDA implementation (beam_lab/beam_search_qshort.py) uses
torch.unique to dedup, producing a variable-size tensor. XLA-hostile.
Our static-shape rewrite:

  1. student_q = student(states)                           # (B, N_GEN) bf16
  2. neighbors = states[:, all_moves].view(-1, ...)        # (B*N_GEN, S)
  3. h = hash(neighbors)                                    # (B*N_GEN,)
  4. dup_mask = sort+adjacent-equal+restore                # (B*N_GEN,) bool
  5. student_q_masked = where(dup_mask, +inf, student_q.flatten())
  6. shortlist = topk(student_q_masked, alpha*B, smallest) # (alpha*B,)
  7. shortlist_states = neighbors[shortlist]               # (alpha*B, S)
  8. teacher_v = teacher(shortlist_states)                 # (alpha*B,) bf16
  9. chosen = topk(teacher_v, B, smallest)                 # (B,)

All shapes are static. Same algorithm semantics: student picks
alpha*B-best (excluding dups), teacher reranks to B. alpha=2 is production.

## Sym-ensemble

  * rotations.npy: 360 elements of A_5 x C_6 group (megaminx symmetries).
  * For each rotation R: precompute R_inv = argsort(R) and conj_idx[g] =
    name_of(R_inv * gen[g] * R) - i.e., the gen-index in original frame
    that, when applied, has the same effect as gen[g] on the rotated state.
  * For each pid: rotated_state = R . state . R_inv  (vector form: rotated[i] = R[state[R_inv[i]]])
  * Run qshort beam on rotated_state, get path_rotated.
  * Translate: path_orig = [conj_idx[m] for m in path_rotated].
  * Verify path_orig on original state (sanity check).
  * Take min path length across all K rotations.

Distribution: (pid, rotation_idx) pairs are round-robin across N_DEVICES cores.
For 5 pids x K=4 = 20 pairs / 8 cores: most cores get 2-3 pairs.

## Compile cost

XLA compiles two graphs (student step + teacher step + dedup; full beam loop
fused). One-time per shape signature, cached on each core. First (pid, rot)
pair pays ~25 min compile; subsequent ones run cached at ~30-80 s each (5-10x
faster than v9's no-qshort due to shrinking model forwards from 768 chunks
of size internal_bs to ~96 chunks).
'''

PUZZLE_INFO_LITERAL = json.dumps(puzzle_info, separators=(",", ":"))

CELL_1_SETUP = '''\
# Cell 1: Setup, _Tee logging, system info.
# CRITICAL workaround for Kaggle TPU multiprocessing (per the official Kaggle
# PyTorch/XLA notebook): pop TPU_PROCESS_ADDRESSES and CLOUD_TPU_TASK_ID env
# vars BEFORE importing torch_xla submodules / calling xmp.spawn. Kaggle pre-
# sets these for single-process mode; xmp.spawn re-sets them for multi-process
# but only if they were unset to begin with. Without this pop, you get
# "Expected 8 worker addresses, got 1" or "Runtime is already initialized".

import sys, os, time, json, subprocess

# Env var pop must happen BEFORE any torch_xla submodule import.
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
# v3-8 TPU: 4 chips * 2 cores = 8 logical devices. xmp.spawn creates 4 processes
# (one per chip), each with 2 threads (one per TensorCore). N_DEVICES is the
# total xm.xrt_world_size() seen across all replicas.
N_DEVICES = 8
print("XLA_AVAILABLE =", XLA_AVAILABLE, "  N_DEVICES =", N_DEVICES)
'''

CELL_2_PUZZLE = f'''\
# Cell 2: Puzzle + test pid embedding.
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

TEST_PIDS = {json.dumps(TEST_PIDS)}
PID_STATES = {json.dumps(pid_states)}
PID_STATES = {{int(k): v for k, v in PID_STATES.items()}}
print("test pids:", TEST_PIDS)
'''

CELL_3_MODEL = '''\
# Cell 3: ResMLPDistance — supports output_dim=1 (V) and output_dim=24 (Q).
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

CELL_4_LOAD = '''\
# Cell 4: Load m05 (V teacher) + m23_v2 (Q student) + rotations from Kaggle dataset.
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

CELL_4B_SYM = '''\
# Cell 4b: Pick K sym rotations + precompute conjugation maps.
# v19a: K=4 (production sweet spot, half v17's K=8). Trades sym diversity for beam.
K_SYM = 4
SYM_SEED = 0
ALPHA_QSHORT = 2.0   # alpha*B candidates from student shortlist; production value

# Find identity row in rotations.
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

# Precompute (R, R_inv, conj_idx) for each chosen rotation.
# R is a permutation of state positions (length STATE_SIZE).
# R_inv = argsort(R).
# conj_idx[g] = index of generator g_orig such that g_orig == R_inv * g_rot * R (in puzzle terms).
ROTATIONS = []   # list of dicts: {"R", "R_inv", "conj_idx"}
move_name_to_idx = {n: i for i, n in enumerate(MOVE_NAMES)}
perm_to_name = {tuple(v): n for n, v in GENERATORS.items()}

for rot_idx in chosen_rot_idxs:
    R = tuple(int(x) for x in rotations_np[rot_idx])
    R_inv = tuple(int(x) for x in np.argsort(rotations_np[rot_idx]))
    # Conjugation map: for each gen name in rotated frame, find equivalent in original frame.
    # conj[name_rot] = name_orig such that gen[name_orig] == R_inv * gen[name_rot] * R
    # Vector form: (R_inv * g * R)[i] = R_inv[g[R[i]]]
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
    """rotated[i] = R[state[R_inv[i]]] — applied to a length-STATE_SIZE int sequence."""
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))

# Build (pid, rotation_idx) pairs for round-robin distribution.
PID_ROT_PAIRS = []
for pid in TEST_PIDS:
    for ri in range(len(ROTATIONS)):
        PID_ROT_PAIRS.append((pid, ri))
print(f"total (pid, rotation) pairs: {len(PID_ROT_PAIRS)}")
'''

CELL_5_BEAM = '''\
# Cell 5: qshort beam search on XLA. Static-shape throughout.

def _make_hash_vec(state_size, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randint(0, int(1e15), (state_size,), dtype=torch.int64, generator=g)

def _model_predict_chunked(model, states, chunk_size, xla_available):
    """Run model on states (returns 1D values for V model, 2D logits for Q model).

    Imports torch_xla locally to avoid pre-initializing the runtime in the
    parent process (would block torch_xla.launch from forking).
    """
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
    """Beam search with student shortlist + teacher rerank. Static-shape.

    Step 0 uses V-only (24 candidates, no qshort). Steps 1+ use qshort.
    """
    state = torch.tensor(initial_state_list, dtype=torch.int8)
    V0_cpu_local = torch.tensor(SOLVED, dtype=torch.int8)
    if torch.equal(state, V0_cpu_local):
        return {"found": True, "path_len": 0, "path_idx": [], "wall_s": 0.0,
                "timings": {"total": 0.0}, "min_v_trajectory": []}

    aB = int(alpha * B)
    t_start = time.time()

    # ---- Step 0: V-only (no qshort, only 24 candidates) ----
    states0 = state.unsqueeze(0).to(device)                 # (1, S)
    neighbors0 = states0[:, all_moves_dev].view(-1, state_size)  # (24, S)
    values0 = _model_predict_chunked(teacher, neighbors0, internal_bs, xla_available)  # (24,)
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

    # Tree backpointers on TPU.
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

    # ---- Steps 1..num_steps-1: qshort ----
    for j in range(1, num_steps):
        t_iter = time.time()

        # 1. Student forward on B parents -> (B, n_gen)
        student_q = _model_predict_chunked(student, states, internal_bs, xla_available)  # (B, n_gen) bf16

        # 2. Compute neighbors + hash for dedup
        neighbors_2d = states[:, all_moves_dev]                        # (B, n_gen, S)
        neighbors = neighbors_2d.view(-1, state_size)                  # (B*n_gen, S)
        h = (neighbors.to(torch.int64) * hash_vec).sum(dim=1)          # (B*n_gen,)

        # 3. Dedup mask (sort+adjacent-equal+restore)
        sort_h, sort_idx = torch.sort(h)
        is_dup_sorted = torch.cat([
            torch.zeros(1, dtype=torch.bool, device=device),
            (sort_h[1:] == sort_h[:-1]),
        ])
        restore = torch.argsort(sort_idx)
        dup_mask = is_dup_sorted[restore]                              # (B*n_gen,) bool

        # 4. Student shortlist: mask dups with +inf, topk alpha*B
        student_q_flat = student_q.view(-1).to(torch.float32)          # (B*n_gen,)
        student_q_masked = torch.where(dup_mask, BIG_VAL_INF, student_q_flat)
        _, shortlist_indices = torch.topk(student_q_masked, aB, largest=False)  # (alpha*B,)

        # 5. Materialize shortlist states (alpha*B candidates only)
        shortlist_states = neighbors[shortlist_indices]                # (alpha*B, S)

        # 6. Teacher forward on shortlist -> (alpha*B,)
        teacher_v = _model_predict_chunked(teacher, shortlist_states, internal_bs, xla_available)

        # 7. Top B by teacher V
        top_v, chosen_local = torch.topk(teacher_v, B, largest=False)  # (B,)
        chosen_states = shortlist_states[chosen_local]                 # (B, S)
        chosen_global = shortlist_indices[chosen_local]                # (B,) into B*n_gen
        chosen_h = h[chosen_global]                                    # (B,)

        parent = (chosen_global // n_gen).to(torch.int32)
        move = (chosen_global % n_gen).to(torch.int8)

        states = chosen_states
        tree_idx[j] = parent
        tree_move[j] = move
        min_v_log[j] = top_v[0].to(torch.float32)

        # V0 hit
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

CELL_6_MP_FN = '''\
# Cell 6: Per-rank worker. Threading-based (v8 pattern). torch_xla.launch and
# xmp.spawn both fail on Kaggle TPU v3-8 single-host with "Expected 8 worker
# addresses, got 1" — they assume multi-host topology. Threading works (single
# process, multiple xla:N device targets) but suffers from GIL contention on
# the dispatch path. Still useful because qshort's 5-8x model-forward speedup
# dominates the wall budget regardless.

B_TEST = 1048576        # v19a-fix: B=1M, 8x v16's beam.
# CRITICAL: at B=1M with internal_bs=4096, per-chunk dispatch overhead
# became the bottleneck (per-step time 42s vs 0.31s at v17). Bumping to
# 32768 cuts chunks/forward from 5859 to 732, should bring per-step
# back into the linear regime.
INTERNAL_BS = 32768
NUM_STEPS = 120

def _mp_fn(index, pid_rot_pairs):
    """xmp.spawn worker. Called once per of the 8 logical devices; each is one
    of: 4 processes x 2 threads on TPU v3-8. Get rank/world from xm runtime."""
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

    # m23_v2 is bigger than m05: hidden_dims=(2048, 1024) per its state_dict shapes.
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
    print(f"[rank {rank}] my_pairs ({len(my_pairs)}): {my_pairs}", flush=True)

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

        # Per-rank JSON for cross-process aggregation. Written every iteration so
        # we don't lose work if Kaggle kills at 12h.
        try:
            with open(f"/kaggle/working/rank_{rank}_partial.json", "w") as f:
                json.dump(rank_local, f)
        except Exception as e:
            print(f"[rank {rank}] WARN partial save failed: {e}", flush=True)

    # Final per-rank result file (used by aggregate cell after launch returns).
    with open(f"/kaggle/working/rank_{rank}_final.json", "w") as f:
        json.dump(rank_local, f)
    print(f"[rank {rank}] done, processed {len(rank_local)} pairs", flush=True)
'''

CELL_7_SPAWN = '''\
# Cell 7: Launch via xmp.spawn — official Kaggle PyTorch/XLA pattern. Requires
# the env var pop in Cell 1 (TPU_PROCESS_ADDRESSES, CLOUD_TPU_TASK_ID) to have
# happened first. start_method='fork' is REQUIRED in interactive notebooks.
# nprocs=None: auto-detects 4 processes (one per TPU v3-8 chip), each with 2
# threads (one per TensorCore) -> 8 invocations of _mp_fn total.
#
# 4-process parallelism reduces GIL contention 4x vs pure-threading, and
# avoids the threading deadlock we saw in v13 (8 threads loading 2 models
# each into XLA simultaneously).

print()
print(f"== xmp.spawn qshort+sym: K_SYM={K_SYM}, alpha={ALPHA_QSHORT}, "
      f"beam={B_TEST}, internal_bs={INTERNAL_BS}, max_steps={NUM_STEPS} ==")
print(f"== test pids: {TEST_PIDS}, total (pid,rot) pairs: {len(PID_ROT_PAIRS)} ==")
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

CELL_8_AGGREGATE = '''\
# Cell 8: Aggregate per-rank JSON files.
# Reads *_final.json preferentially; falls back to *_partial.json if Kaggle
# killed the kernel mid-run (so we always have *some* useful result).
import csv as _csv
import glob

def _load_rank_results():
    """Pick the most complete per-rank file available."""
    by_rank = {}
    for path in glob.glob("/kaggle/working/rank_*_final.json"):
        rank = int(path.split("rank_")[1].split("_")[0])
        try:
            with open(path) as f:
                by_rank[rank] = ("final", json.load(f))
        except Exception as e:
            print(f"WARN failed reading {path}: {e}")
    # Top up with partials for ranks missing a final.
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

# Persist combined results for downstream tooling.
with open("/kaggle/working/tpu_qshort_sym_results.json", "w") as f:
    json.dump(results_list, f, indent=2)
print(f"wrote /kaggle/working/tpu_qshort_sym_results.json")

# Group by pid.
by_pid = {}
for r in results_list:
    by_pid.setdefault(r["pid"], []).append(r)

print()
print("== per-pid (min over K rotations) ==")
print(f"{'pid':>6}  {'best_rot':>8}  {'best_path':>9}  {'all_found/K':>12}  "
      f"{'wall_max':>9}  {'last_min_V':>10}")
final_per_pid = {}  # pid -> {found, best_path_len, best_path_idx, ranks_used}
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

# Write submission CSV (only for pids attempted; empty for unsolved).
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
    ("code", CELL_1_SETUP),
    ("code", CELL_2_PUZZLE),
    ("code", CELL_3_MODEL),
    ("code", CELL_4_LOAD),
    ("code", CELL_4B_SYM),
    ("code", CELL_5_BEAM),
    ("code", CELL_6_MP_FN),
    ("code", CELL_7_SPAWN),
    ("code", CELL_8_AGGREGATE),
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

out_nb = HERE / "cayleypy-tpu-beam-smoke.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb}  ({out_nb.stat().st_size} bytes)")

meta = {
    "id": "artgor/cayleypy-tpu-beam-smoke",
    "title": "cayleypy tpu beam smoke",
    "code_file": "cayleypy-tpu-beam-smoke.ipynb",
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
