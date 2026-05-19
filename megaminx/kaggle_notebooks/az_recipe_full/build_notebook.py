"""Builds the public AZ-recipe end-to-end notebook for cayleypy-megaminx.

One kernel, full recipe from zero:
  Stage A — 1-epoch walk-depth cold-start (gives Bellman a warm-start it requires)
  Stage B — 10-epoch m_dd_v0 Bellman with PDB lambda=5 (faithful to the production recipe)
  Stage C — Build AZ (state, action) policy dataset from the community-merged
            76,304-move CSV `merge_v9_with_77152.csv`
  Stage D — 10-epoch AZ dual-head (policy + value) training, checkpoint every 2 epochs
            -> 5 checkpoints driving Stage E
  Stage E — Per-pid selection matrix: 5 checkpoints x 4 sym-ensemble rotations x
            51 strat-5 pids = 1,020 beam runs at B=16k. Per pid, take the shortest
            valid path across the 5x4 matrix (the "both stacked" select).
  Stage F — Fill from BFS-d6 fallback, verify, write submission CSV.

EPOCHS_* defaults are 10 for the recipe demo; user scales to 50/100 in the
production cell at the end.

Output:
  cayleypy-megaminx-az-recipe.ipynb in this directory.

Run via:
  .venv/Scripts/python.exe megaminx/kaggle_notebooks/az_recipe_full/build_notebook.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # megaminx/
HERE = Path(__file__).resolve().parent

# --- Inputs embedded into the notebook ------------------------------------

puzzle_info = json.loads((ROOT / "data" / "puzzle_info.json").read_text())
PUZZLE_INFO_LITERAL = json.dumps(puzzle_info, separators=(",", ":"))

# All 1001 pid scrambles, embedded so the notebook is decoupled from
# /kaggle/input/cayleypy-megaminx/test.csv (which would also work, but using
# the embedded copy keeps the kernel self-contained for forks).
rows = list(csv.DictReader(open(ROOT / "data" / "test.csv")))
ALL_PID_STATES = {pid: [int(x) for x in rows[pid]["initial_state"].split(",")]
                  for pid in range(len(rows))}
N_TOTAL_PIDS = len(ALL_PID_STATES)
ALL_PID_STATES_LITERAL = json.dumps(ALL_PID_STATES, separators=(",", ":"))

# --- Cells -----------------------------------------------------------------

CELL_INTRO = '''\
# CayleyPy Megaminx — AlphaZero recipe end-to-end

Public reference notebook for the
[CayleyPy Megaminx Kaggle competition](https://www.kaggle.com/competitions/cayley-py-megaminx).
Trains a 6M-parameter dual-head AlphaZero model from zero, then uses it to solve
the strat-5 acceptance bench, demonstrating a **per-pid checkpoint x rotation
selection** that exploits an open question from prior work: *which AZ checkpoint
is best depends on the pid*.

## Why this notebook exists

The breakthrough on this project (`m_az_v4/epoch_0024.pt`) was the first 6M-param
model to break sub-89 strat-5 standalone (51/51 / mean 87.5 vs the m05 baseline's
50/51 / mean 89.4). The recipe was scattered across ~20 scripts and 10 days of
experimentation; this notebook consolidates it into one runnable artifact so
collaborators can reproduce, audit, and extend.

## The recipe at a glance

1. **Stage A** - Cold-start a tiny ResMLP V model with 1 epoch of walk-depth
   labels. Bellman refinement requires a warm-start; this is the cheapest path
   to one when starting from zero.
2. **Stage B** - `m_dd_v0` Bellman recipe: random walks + BFS-d6 exact-distance
   anchor mixin + V0 / d=1 anchors + 25% beam-frontier mixin + admissibility-aware
   loss using a max-of-4 disjoint K=5 corner-PDB lower bound (lambda=5).
3. **Stage C** - Replay the community-merged 76,304-move submission CSV through
   the puzzle generators to extract (state, action) pairs for policy training.
4. **Stage D** - AlphaZero-style dual-head: shared ResMLP trunk (warm-started
   from Stage B), policy head (24 logits, CE loss on Stage C pairs), value head
   (Bellman targets, identical recipe to Stage B). Joint loss
   `alpha * CE + beta * MSE`. Checkpoint every 2 epochs to seed Stage E.
5. **Stage E** - Per-pid selection matrix: for every (checkpoint, rotation) pair,
   solve every strat-5 pid; take the per-pid min across the matrix. Surfaces the
   pid-dependent best-checkpoint signal that motivated this notebook.
6. **Stage F** - Fill missing pids from the BFS-d6 fallback, verify, write the
   submission CSV.

## The early-stop story (why ep24 is famous)

In the production run, the AZ dual-head trains for 100 epochs but the *best
strat-5 result* is at epoch 24 — earlier than naive smoothed-loss early-stop
would pick. Past ep ~25 the trunk specializes for policy memorization (top-1
acc 4.5 -> 28.5 -> 48 -> 74 -> 91 percent at ep 0, 24, 49, 74, 99) at the cost
of value-head calibration (v_loss 0.096 -> 0.110 -> 0.140 -> 0.250 -> 0.481).
The "8% dual-head trunk-sharing tax" we used to cite was a misdiagnosis —
the real cause was over-training on a fast-converging policy distribution.

This notebook's per-pid selection matrix exploits a related observation: even
within the early window, *different pids prefer different checkpoints*. By
running the full matrix you get a small but real lift over picking any single
global-best checkpoint.

## Credits

The 76,304-move policy training source `merge_v9_with_77152.csv` is a min-merge
of three submissions: ours (78,029) + two from collaborators in the cayleypy
community (79,911 + 77,152). This notebook would not exist without their
contributions — please credit them when sharing forks.
'''


CELL_RECIPE_DIAGRAM = '''\
## Recipe diagram

```
                (1 epoch walk-depth pretrain)
                          |
                          v
      Stage A: cold_start.pt  (provides warmstart for Bellman)
                          |
                          v
      Stage B: m_dd_v0 Bellman + PDB (10 epochs)
              [RW] + [BFS-d6 mixin 10%] + [V0 + d=1 anchors] + [frontier 25%]
              + admissibility penalty lambda * relu(h_PDB - V_pred)^2
                          |
                          v
            m_dd_v0/epoch_0009.pt    (V model, 6M params)
                          |
                          | warm-start trunk + value head
                          v
      Stage C: build az_dataset_76304.pt
              merge_v9_with_77152.csv ---> (state, action) replay tuples
                          |
                          v
      Stage D: AZ dual-head training (10 epochs, ckpt every 2)
              shared trunk + policy head (24 logits) + value head (1)
              joint loss = CE(pi, action) + MSE(V, bellman_target)
                          |
                          v
            m_az/epoch_{1,3,5,7,9}.pt    (5 checkpoints)
                          |
                          v
      Stage E: per-pid selection matrix
              5 ckpts x 4 rotations x 51 strat-5 pids = 1,020 beam runs
              per pid: shortest valid path across the 5x4 matrix
                          |
                          v
      Stage F: verify + write az_recipe_demo_submission.csv
```
'''


CELL_SETUP = f'''\
# ============================================================================
# CONFIG - safe to scale up after the demo runs cleanly
# ============================================================================
#
# Defaults (this cell) produce the recipe demo:
#   Stage A: 1 epoch walk-depth pretrain               (~1 min on T4)
#   Stage B: 10 epochs Bellman + PDB                   (~12 min on T4)
#   Stage C: replay 76,304-move CSV                    (~30 s)
#   Stage D: 10 epochs AZ dual-head, ckpt every 2 ep   (~6 min on T4)
#   Stage E: 5 ckpts x 4 rotations x 51 pids @ B=16k  (~30-50 min on T4)
#   Stage F: verify + write CSV                        (<1 min)
#                                                      ----------
#                                                  Total: ~55-75 min
#
# To scale toward the production AZ v4 result (51/51 strat-5, mean 87.5),
# bump EPOCHS_PRETRAIN to 50 and EPOCHS_AZ to 100, then in Stage E either
# (a) widen BEAM to 65536 at the same K_SYM=4, or (b) run on full-1001 pids.
# See the Production cell at the bottom for the canonical settings.
#
# IMPORTANT: leave TARGET_UPDATE_EVERY=10 unchanged when scaling. With
# EPOCHS=10 here, the Bellman target net refreshes exactly once at end-of-
# training (matches the m_dd_v0 recipe). At EPOCHS=100 it refreshes 10 times,
# which is the documented cadence.
# ============================================================================

import os

EPOCHS_PRETRAIN     = 10        # Stage B Bellman epochs
EPOCHS_AZ           = 10        # Stage D AZ dual-head epochs
CHECKPOINT_EVERY    = 2         # Stage D ckpt cadence -> 5 ckpts at ep 1,3,5,7,9
TARGET_UPDATE_EVERY = 10        # Bellman target-net refresh cadence (epochs)

K_SYM        = 4                # number of sym-ensemble rotations per pid
BEAM         = 16384            # KhoruzhiiSearchConfig.beam_width for Stage E
MAX_STEPS    = 120              # KhoruzhiiSearchConfig.num_steps
INTERNAL_BS  = 4096             # solver internal_batch_size

STRAT_PER_BUCKET = 5            # strat-5 -> 5 pids per (pid // 100) bucket
STRAT_SEED       = 0            # canonical strat-5 seed used since m05

SAMPLES_PER_EPOCH = 500_000     # random-walk sample budget per epoch
RW_BATCH_SIZE     = 8192        # Bellman & AZ value-side batch
POLICY_BATCH_SIZE = 1024        # AZ policy-side batch
K_MAX             = 80          # random-walk depth (k_max)

# Local-test override: KAGGLE_TEST_PIDS=11 in env truncates Stage E to 11 pids
# for a quick smoke before publishing. Unset in the published kernel.
LIMIT_FOR_LOCAL_TEST = int(os.environ.get("KAGGLE_TEST_PIDS", "0"))

# Resolve DATA (read-only inputs) + WORKDIR (writable scratch).
from pathlib import Path
import sys

_DATA_CANDIDATES = [
    Path("/kaggle/input/megaminx-az-recipe-artifacts"),
    Path("/kaggle/input/cayleypy-megaminx-az-recipe-artifacts"),
    # Some Kaggle kernels mount under /kaggle/input/datasets/<owner>/<slug>/
    # instead of /kaggle/input/<slug>/. Both paths can show up depending on
    # how the dataset was attached.
    Path("/kaggle/input/datasets/artgor/megaminx-az-recipe-artifacts"),
]
DATA = next((c for c in _DATA_CANDIDATES if c.exists()), None)

if DATA is None:
    # Local-test fallback: search for the repo from cwd upward.
    _cur = Path.cwd().resolve()
    for _p in [_cur, *_cur.parents]:
        if (_p / "megaminx" / "data" / "puzzle_info.json").exists():
            DATA = _p / "megaminx" / "data"
            print(f"[setup] not on Kaggle; using local repo data at {{DATA}}")
            break

if DATA is None:
    _listing = list(Path("/kaggle/input").rglob("puzzle_info.json"))[:5] if Path("/kaggle/input").exists() else []
    raise SystemExit(
        f"Dataset not found. Probed Kaggle paths: {{_DATA_CANDIDATES}}. "
        f"Found puzzle_info.json at: {{_listing}}. Attach the dataset via "
        f"'Add Data' on Kaggle, or run from a checkout that has megaminx/data/."
    )
print(f"[setup] DATA = {{DATA}}")

# True iff we're actually on Kaggle (not just a Windows box where /kaggle/foo resolves to C:/kaggle/foo).
ON_KAGGLE = str(DATA).startswith("/kaggle/input/")
WORKDIR = "/kaggle/working" if ON_KAGGLE else str(Path.cwd() / "az_recipe_workdir")
Path(WORKDIR).mkdir(parents=True, exist_ok=True)
print(f"[setup] WORKDIR = {{WORKDIR}}  (on_kaggle={{ON_KAGGLE}})")

# Add bundled cayley + megaminx source to sys.path. On Kaggle this lives under
# DATA/cayley_src/. On local, fall back to the checkout's src/ + megaminx/src/.
_SRC = DATA / "cayley_src"
if (_SRC / "src" / "cayley").exists() and (_SRC / "megaminx" / "src" / "megaminx").exists():
    sys.path.insert(0, str(_SRC / "src"))
    sys.path.insert(0, str(_SRC / "megaminx" / "src"))
    print(f"[setup] sys.path += cayley_src/src + cayley_src/megaminx/src")
else:
    # Local fallback: use the checkout's src/ + megaminx/src/.
    _repo_root = DATA.parent.parent
    _local_cayley = _repo_root / "src"
    _local_megaminx = _repo_root / "megaminx" / "src"
    if (_local_cayley / "cayley").exists() and (_local_megaminx / "megaminx").exists():
        sys.path.insert(0, str(_local_cayley))
        sys.path.insert(0, str(_local_megaminx))
        print(f"[setup] sys.path += {{_local_cayley}} + {{_local_megaminx}} (local checkout)")
    else:
        raise SystemExit(
            f"cayley + megaminx source not found. Expected one of:\\n"
            f"  Kaggle:  {{DATA}}/cayley_src/src/cayley/  AND  /megaminx/src/megaminx/\\n"
            f"  Local:   {{_local_cayley}}/cayley/  AND  {{_local_megaminx}}/megaminx/\\n"
            f"Re-upload the dataset per RUNBOOK.md."
        )

# Banner.
print()
print("=" * 72)
print(f"  EPOCHS_PRETRAIN = {{EPOCHS_PRETRAIN}}   EPOCHS_AZ = {{EPOCHS_AZ}}")
print(f"  CHECKPOINT_EVERY = {{CHECKPOINT_EVERY}}   K_SYM = {{K_SYM}}   BEAM = {{BEAM}}")
print(f"  STRAT-{{STRAT_PER_BUCKET}} acceptance bench, seed = {{STRAT_SEED}}")
if LIMIT_FOR_LOCAL_TEST:
    print(f"  LOCAL TEST: Stage E truncated to {{LIMIT_FOR_LOCAL_TEST}} pids")
print()
print("  10 epochs is recipe demonstration. Production = 50 / 100. The Bellman")
print(f"  target-net refreshes once at epoch {{EPOCHS_PRETRAIN}} with this setting -")
print("  matches the m_dd_v0 recipe. If you scale EPOCHS up, leave")
print("  TARGET_UPDATE_EVERY=10 unchanged (the standard cadence).")
print("=" * 72)
'''


CELL_IMPORTS = '''\
import time
import json
import csv
import copy
import random
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# cayley source (mounted from the dataset)
from cayley.bellman import BellmanConfig, train_bellman
from cayley.training import TrainConfig, train as train_walkdepth
from cayley.model import ResMLPDistance
from cayley.gflow_model import ResMLPGFlowNet
from cayley.search import load_model_checkpoint
from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.verify import verify_path, verify_submission, load_test_states
from cayley.data import GeneratorTable, generate_walks_torch

# megaminx source
from megaminx.puzzle import Megaminx
from megaminx.pdb_heuristic import CornerPDBHeuristic
from megaminx.post_process import full_post_process

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[imports] torch={torch.__version__}  device={DEVICE}")
if DEVICE == "cuda":
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"[imports] GPU={name}  compute_capability={cap}")
    # bf16 needs Ampere+ (sm_80) for native HW. T4 (sm_75) and P100 (sm_60) do
    # not. Pick the right autocast dtype.
    USE_BF16 = (cap[0] >= 8)
    AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
    print(f"[imports] autocast dtype = {AUTOCAST_DTYPE} (bf16 supported: {USE_BF16})")
else:
    USE_BF16 = False
    AUTOCAST_DTYPE = torch.float32

# Initialize the puzzle once. Megaminx.load duck-types into the cayley
# trainer (cayley.bellman type-hints PictureCube, accepts any compatible).
PUZZLE = Megaminx.load(DATA / "puzzle_info.json")
STATE_SIZE = len(PUZZLE.solved_state)
N_GEN = len(PUZZLE.move_names)
print(f"[imports] puzzle: state_size={STATE_SIZE}  n_gen={N_GEN}  "
      f"move_names[:6]={PUZZLE.move_names[:6]}")
'''


CELL_STAGE_A = '''\
# ============================================================================
# Stage A - 1-epoch walk-depth cold-start
# ============================================================================
#
# `train_bellman` requires a non-empty `warmstart_path` (cayley/bellman.py:359
# raises ValueError otherwise). Bellman from random init rarely converges -
# the walk-depth signal is needed first to learn rough state representations
# the Bellman bootstrap can refine.
#
# Cheapest from-zero path: one epoch of standard walk-depth training (matches
# megaminx/scripts/02_train.py). Saves cold_start.pt; Stage B warm-starts from it.
# ============================================================================

print("=" * 72)
print("Stage A - cold-start walk-depth pretrain (1 epoch)")
print("=" * 72)

cold_model = ResMLPDistance(
    state_size=STATE_SIZE, num_classes=STATE_SIZE,
    hidden_dims=(2048, 512), num_res_blocks=2,
    encoding="embedding", embed_dim=16,
)
print(f"[stage A] model params: {cold_model.num_parameters():,}")

cold_tc = TrainConfig(
    n_epochs=1,
    samples_per_epoch=SAMPLES_PER_EPOCH,
    batch_size=RW_BATCH_SIZE,
    k_max=K_MAX,
    lr=5e-4,
    weight_decay=0.0,
    device=DEVICE,
    seed=34,
    n_back=1,
    loss="mse",
    amp=USE_BF16,
    compile_model=False,        # Kaggle's torch may be old enough to break compile
    fused_optimizer=(DEVICE == "cuda"),
    checkpoint_every_epochs=1,  # save the single epoch
)

_cold_dir = Path(WORKDIR) / "cold_start"
_t = time.time()
train_walkdepth(
    cold_model, PUZZLE, cold_tc,
    checkpoint_dir=_cold_dir,
    on_epoch_end=lambda s: print(
        f"  [cold] ep {s.epoch:3d} | loss {s.loss:.4f} | lr {s.lr:.2e} | {s.elapsed_s:.1f}s"
    ),
)
COLD_START_PATH = _cold_dir / "epoch_0000.pt"
print(f"[stage A] saved {COLD_START_PATH}  ({time.time()-_t:.1f}s)")
del cold_model
torch.cuda.empty_cache() if DEVICE == "cuda" else None
'''


CELL_STAGE_B0_FRONTIER_MD = '''\
## Stage B0 - where do the frontier states come from?

The Bellman pre-training recipe mixes **25% of every batch** with states drawn
from past beam-search frontiers (states actually visited by the solver during
real solves), with the remaining 75% from random walks. The hypothesis: the
random-walk distribution and the beam-frontier distribution don't fully overlap
- random walks tend to underweight the kind of states the solver actually
encounters deep into a search. Mixing real frontier states sharpens V calibration
where it matters for inference.

These states are produced by `megaminx/scripts/42_log_frontier_states.py`: it
runs vanilla beam search with a previous V model (e.g. m05) on N pids, hooks
`KhoruzhiiSolver._do_greedy_step` to capture the surviving states at every beam
step, deduplicates across all pids by state-bytes, and saves
`data/frontier_states.pt` (states only, no labels — the Bellman loop computes
`1 + min_a V_target(apply(s, a))` on the fly, identical to the random-walk path).
Default args: `--n-pids 100 --beams 16384 --max-steps 80 --bucket-from 4
--bucket-to 7`, ~36 MB output.

This notebook ships the pre-built `frontier_states.pt` in the Kaggle dataset
because generating it from scratch needs an existing V model (chicken-and-egg
from a true cold start). A "purist" zero-input variant would set
`frontier_fraction=0.0` in Stage B and lose the 25% mixin — recipe drift but
still functional.

The next code cell shows the regeneration command (commented out — only run
this if you want to rebuild `frontier_states.pt` from a different V checkpoint).
'''


CELL_STAGE_B0_REGEN = '''\
# Optional: regenerate frontier_states.pt from your own V checkpoint.
# Uncomment + adjust paths to use. Skipped by default.
#
# import subprocess
# subprocess.run([
#     "python", str(_SRC / "megaminx" / "scripts" / "42_log_frontier_states.py"),
#     "--checkpoint", str(_cold_dir / "epoch_0000.pt"),  # or your trained V
#     "--out", str(Path(WORKDIR) / "frontier_states.pt"),
#     "--n-pids", "100",
#     "--beams", "16384",
#     "--max-steps", "80",
#     "--bucket-from", "4",
#     "--bucket-to", "7",
# ], check=True)
print("[stage B0] using shipped frontier_states.pt (skip regeneration)")
'''


CELL_STAGE_B = '''\
# ============================================================================
# Stage B - m_dd_v0 Bellman with PDB lambda=5
# ============================================================================
#
# `train_bellman` orchestrates EVERYTHING internally:
#   - per batch composition: random walks + BFS-d6 mixin (10%) + V0 anchor x32
#     + 24 d=1 anchors x4 + 25% beam-frontier states (with Bellman targets)
#   - target-net forward to compute `1 + min_a V_target(apply(s, a))`
#   - target-net refresh every TARGET_UPDATE_EVERY epochs
#   - PDB admissibility penalty lambda * mean(relu(h_PDB(s) - V_pred(s))^2)
#
# We just build the configs and call it.
# ============================================================================

print("=" * 72)
print(f"Stage B - m_dd_v0 Bellman (epochs={EPOCHS_PRETRAIN}, lambda_pdb=5.0)")
print("=" * 72)

dd_model = ResMLPDistance(
    state_size=STATE_SIZE, num_classes=STATE_SIZE,
    hidden_dims=(2048, 512), num_res_blocks=2,
    encoding="embedding", embed_dim=16,
)

dd_tc = TrainConfig(
    n_epochs=EPOCHS_PRETRAIN,
    samples_per_epoch=SAMPLES_PER_EPOCH,
    batch_size=RW_BATCH_SIZE,
    k_max=K_MAX,
    lr=5e-4,
    weight_decay=0.0,
    device=DEVICE,
    seed=34,
    n_back=1,
    loss="mse",
    amp=USE_BF16,
    compile_model=False,
    fused_optimizer=(DEVICE == "cuda"),
    checkpoint_every_epochs=EPOCHS_PRETRAIN,  # only save final
)

dd_bc = BellmanConfig(
    warmstart_path=str(COLD_START_PATH),
    target_update_every_epochs=TARGET_UPDATE_EVERY,
    target_net_chunk=4096,
    clip_upper=True,
    clip_lower=True,
    frontier_path=str(DATA / "frontier_states.pt"),
    frontier_fraction=0.25,
    frontier_walk_depth_cap=200.0,
    bfs_d6_path=str(DATA / "bfs_d6_train.pt"),
    bfs_d6_fraction=0.10,
    lambda_pdb=5.0,
    n_anchor_v0=32,
    n_anchor_d1=4,
)

# Build the PDB heuristic (max-of-4 disjoint K=5 corner PDBs covering all 20 corners).
pdb_paths = [DATA / f"pdb_corner_K5{suffix}.pkl" for suffix in ["", "_p1", "_p2", "_p3"]]
for p in pdb_paths:
    if not p.exists():
        raise SystemExit(f"missing PDB file: {p} (re-upload the dataset)")
pdb = CornerPDBHeuristic(
    pdb_paths=pdb_paths,
    corner_tables_path=DATA / "corner_tables.pkl",
    device=DEVICE,
)

def pdb_lookup_fn(states):
    return pdb.lookup(states)

_dd_dir = Path(WORKDIR) / "m_dd_v0"
_t = time.time()
train_bellman(
    dd_model, PUZZLE, dd_tc, dd_bc,
    checkpoint_dir=_dd_dir,
    on_epoch_end=lambda s: print(
        f"  [m_dd_v0] ep {s.epoch:3d} | loss {s.loss:.4f} | lr {s.lr:.2e} | {s.elapsed_s:.1f}s"
    ),
    pdb_lookup_fn=pdb_lookup_fn,
)
DD_PATH = _dd_dir / f"epoch_{EPOCHS_PRETRAIN - 1:04d}.pt"
print(f"[stage B] saved {DD_PATH}  ({time.time()-_t:.1f}s)")
del dd_model, pdb
torch.cuda.empty_cache() if DEVICE == "cuda" else None
'''


CELL_STAGE_C = '''\
# ============================================================================
# Stage C - Build AZ (state, action) policy dataset
# ============================================================================
#
# Replays the community-merged 76,304-move CSV through puzzle.generators to
# produce (state, action) pairs. Each pid contributes path_len tuples:
# at step i, state = scramble after first i moves, action = move taken at step i.
# Output dict matches what 71_train_az_v3.py expects.
# ============================================================================

print("=" * 72)
print("Stage C - build AZ policy dataset from merge_v9_with_77152.csv")
print("=" * 72)

# Locate the community-merged CSV. On Kaggle it's at DATA/. Locally it's at
# megaminx/submissions/ (sibling of megaminx/data/).
_csv_candidates = [
    DATA / "merge_v9_with_77152.csv",
    DATA.parent / "submissions" / "merge_v9_with_77152.csv",
]
_csv_path = next((p for p in _csv_candidates if p.exists()), None)
if _csv_path is None:
    raise SystemExit(
        f"merge_v9_with_77152.csv not found. Probed: {[str(p) for p in _csv_candidates]}. "
        f"On Kaggle the dataset must include it at the top level."
    )
_test_rows = list(csv.DictReader(open(DATA / "test.csv")))
_sub_rows = list(csv.DictReader(open(_csv_path)))
print(f"[stage C] submission: {len(_sub_rows)} rows; test: {len(_test_rows)} pids")

_move_to_idx = {name: i for i, name in enumerate(PUZZLE.move_names)}
_all_states, _all_actions, _all_values = [], [], []
_n_skipped = 0

for _row in _sub_rows:
    _pid = int(_row["initial_state_id"])
    _path_str = _row["path"].strip()
    if not _path_str:
        _n_skipped += 1
        continue
    _moves = _path_str.split(".")
    _n = len(_moves)
    _s0 = tuple(int(x) for x in _test_rows[_pid]["initial_state"].split(","))
    _cur = list(_s0)
    for _i, _name in enumerate(_moves):
        if _name not in _move_to_idx:
            print(f"  pid={_pid}: unknown move {_name!r} at step {_i}, skipping rest")
            break
        _all_states.append(np.array(_cur, dtype=np.int8))
        _all_actions.append(_move_to_idx[_name])
        _all_values.append(_n - _i)
        _gen = PUZZLE.generators[_name]
        _cur = [_cur[_g] for _g in _gen]
    if tuple(_cur) != PUZZLE.solved_state:
        print(f"  pid={_pid}: path doesn't reach solved (kept tuples anyway)")

AZ_DATASET = {
    "states":  torch.tensor(np.stack(_all_states), dtype=torch.int8),
    "actions": torch.tensor(_all_actions, dtype=torch.int64),
    "values":  torch.tensor(_all_values, dtype=torch.float32),
    "source":  str(_csv_path),
}
AZ_DATASET_PATH = Path(WORKDIR) / "az_dataset.pt"
torch.save(AZ_DATASET, AZ_DATASET_PATH)
print(f"[stage C] saved {AZ_DATASET_PATH}: "
      f"{AZ_DATASET['states'].shape[0]:,} (state, action) pairs from "
      f"{len(_sub_rows) - _n_skipped} solved pids ({_n_skipped} skipped); "
      f"value range [{AZ_DATASET['values'].min():.0f}, {AZ_DATASET['values'].max():.0f}]")
'''


CELL_STAGE_D_HELPERS = '''\
# Stage D - helper functions ported inline from 71_train_az_v3.py.

def warmstart_az_from_v(az_model, v_ckpt_path):
    """Copy V-model trunk + head into the AZ model.

    Loads embedding + input_stack + res_blocks (the trunk), and copies the
    V model's `head` linear layer into the AZ model's `value_head`. Initializing
    value_head from a calibrated distance head accelerates convergence and
    keeps the value scale in the (0, ~max_dist) range from epoch 0.
    """
    ckpt = torch.load(v_ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
          for k, v in sd.items()}
    trunk_sd = {k: v for k, v in sd.items()
                if k.startswith(("embedding.", "input_stack.", "res_blocks."))}
    az_model.load_state_dict(trunk_sd, strict=False)
    print(f"  [warm-start] loaded {len(trunk_sd)} trunk tensors from {v_ckpt_path.name}")
    if "head.weight" in sd:
        with torch.no_grad():
            az_model.value_head.weight.copy_(sd["head.weight"])
            az_model.value_head.bias.copy_(sd["head.bias"])
        print(f"  [warm-start] initialized value_head from V-model head")


def az_value_only_forward(az_model, states, chunk_size=4096):
    """Run only the value head of an AZ model on `states`. For target-net forward."""
    outs = []
    for i in range(0, states.size(0), chunk_size):
        h = az_model.trunk(states[i : i + chunk_size])
        outs.append(az_model.value_head(h).squeeze(-1))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def az_bellman_targets(target_az, states, walk_depths, generators, solved_state,
                       chunk_size=4096):
    """Bellman target = clip(1 + min_a target(apply(s, a)), 0, walk_depth).

    Uses the value head only of the target network. Mirrors cayley.bellman
    `_bellman_targets` for the Stage D mixed batch.
    """
    B, S = states.shape
    n_gen = generators.shape[0]
    children = states.unsqueeze(1).expand(B, n_gen, S).clone()
    children = torch.gather(children, 2, generators.unsqueeze(0).expand(B, n_gen, S))
    children_flat = children.reshape(B * n_gen, S)
    is_solved = (children_flat == solved_state).all(dim=1)
    target_az.eval()
    child_v = az_value_only_forward(target_az, children_flat, chunk_size).float()
    child_v = torch.where(is_solved, torch.zeros_like(child_v), child_v)
    child_v = child_v.view(B, n_gen)
    target = 1.0 + child_v.min(dim=1).values
    target = torch.minimum(target, walk_depths)
    return torch.clamp(target, min=0.0)
'''


CELL_STAGE_D_TRAIN = '''\
# ============================================================================
# Stage D - AZ dual-head training (10 epochs, ckpt every 2)
# ============================================================================
#
# Faithful inline port of megaminx/scripts/71_train_az_v3.py main loop.
# Per training step:
#   1. Generate random-walk states + Bellman targets (value side).
#   2. Concatenate V0 anchors (32) + 24 D=1 anchors x4 + BFS-d6 mixin (10%).
#   3. Sample a separate batch of (state, action) pairs from AZ_DATASET (policy side).
#   4. Forward both batches through the shared trunk; compute joint loss
#      `alpha * CE(policy_logits, action) + beta * MSE(value_pred, target)`.
#
# Knob deltas vs the script defaults:
#   * `target_update_every_epochs = TARGET_UPDATE_EVERY` (10) - default fires once at end of training.
#   * `checkpoint_every = CHECKPOINT_EVERY` (2) - need every checkpoint for Stage E.
#   * `warmstart_trunk = DD_PATH`
#   * fp16 autocast on T4 (no native bf16); bf16 on Ampere+ via AUTOCAST_DTYPE.
# ============================================================================

print("=" * 72)
print(f"Stage D - AZ dual-head (epochs={EPOCHS_AZ}, ckpt every {CHECKPOINT_EVERY} ep)")
print("=" * 72)

ALPHA_POLICY = 1.0
BETA_VALUE   = 1.0
N_ANCHOR_V0  = 32
N_ANCHOR_D1  = 4
BFS_D6_FRAC  = 0.10

az_model = ResMLPGFlowNet(
    state_size=STATE_SIZE, num_classes=STATE_SIZE,
    hidden_dims=(2048, 512), num_res_blocks=2,
    encoding="embedding", embed_dim=16, n_actions=N_GEN,
).to(DEVICE)
print(f"[stage D] model params: {az_model.num_parameters():,}")
warmstart_az_from_v(az_model, DD_PATH)
az_model.train()

# Frozen target network for Bellman bootstrap (value side).
az_target = copy.deepcopy(az_model).eval()
for _p in az_target.parameters():
    _p.requires_grad = False

# Generators + solved state on device.
_gens_table = GeneratorTable.from_puzzle(PUZZLE)
GENERATORS_DEV = torch.from_numpy(_gens_table.perms).to(DEVICE)
SOLVED_DEV = torch.tensor(PUZZLE.solved_state, dtype=torch.int64, device=DEVICE)
ANCHOR_V0 = SOLVED_DEV.unsqueeze(0)                                    # (1, S)
ANCHOR_D1 = torch.gather(ANCHOR_V0.expand(N_GEN, STATE_SIZE), 1, GENERATORS_DEV)  # (24, S)

# Policy dataset (already in memory as AZ_DATASET; move to device).
states_p_dev  = AZ_DATASET["states"].to(DEVICE).long()
actions_p_dev = AZ_DATASET["actions"].to(DEVICE).long()
NP = states_p_dev.size(0)

# BFS-d6 mixin pre-load.
print(f"[stage D] loading BFS-d6 from {DATA / 'bfs_d6_train.pt'}")
_bfs6 = torch.load(DATA / "bfs_d6_train.pt", map_location="cpu", weights_only=False)
BFS6_STATES = _bfs6["states"]
BFS6_DISTS  = _bfs6["distances"]
BFS6_PER_BATCH = max(1, int(round(RW_BATCH_SIZE * BFS_D6_FRAC)))
print(f"[stage D] BFS-d6 mixin: {BFS6_PER_BATCH}/{RW_BATCH_SIZE} per batch")

az_optim = torch.optim.AdamW(
    az_model.parameters(), lr=5e-4, weight_decay=0.0,
    fused=(DEVICE == "cuda"),
)
az_sched = torch.optim.lr_scheduler.CosineAnnealingLR(az_optim, T_max=EPOCHS_AZ)
az_autocast = (torch.amp.autocast("cuda", dtype=AUTOCAST_DTYPE)
               if DEVICE == "cuda" else None)

_rng_p = torch.Generator(device=DEVICE).manual_seed(71)
_rng_bfs = torch.Generator(device="cpu").manual_seed(72)

_az_dir = Path(WORKDIR) / "m_az"
_az_dir.mkdir(parents=True, exist_ok=True)
AZ_CKPTS = []

for _epoch in range(EPOCHS_AZ):
    _t = time.time()
    _n_walks = max(1, SAMPLES_PER_EPOCH // K_MAX)
    _rw_states, _rw_depths = generate_walks_torch(
        PUZZLE, n_walks=_n_walks, k_max=K_MAX,
        seed=71 + _epoch * 1000, device=DEVICE, n_back=1,
    )
    _rw_depths_f = _rw_depths.to(torch.float32)
    _N_RW = _rw_states.size(0)

    _rw_per_batch = RW_BATCH_SIZE - BFS6_PER_BATCH - N_ANCHOR_V0 - 24 * N_ANCHOR_D1
    _n_per_epoch = BFS6_PER_BATCH * (_N_RW // _rw_per_batch + 1)
    _n_per_epoch = min(_n_per_epoch, BFS6_STATES.size(0))
    _ep_perm = torch.randperm(BFS6_STATES.size(0), generator=_rng_bfs)[:_n_per_epoch]
    _ep_bfs_states = BFS6_STATES[_ep_perm].to(DEVICE).long()
    _ep_bfs_dists  = BFS6_DISTS[_ep_perm].to(DEVICE).float()

    _n_batches = max(1, _N_RW // _rw_per_batch)
    _bfs_cursor = 0
    _tot_p = _tot_v = _tot_acc = 0.0

    for _b in range(_n_batches):
        _start = _b * _rw_per_batch
        _end = min(_start + _rw_per_batch, _N_RW)
        _bs_rw = _rw_states[_start:_end]
        _bd_rw = _rw_depths_f[_start:_end]
        with torch.no_grad():
            _target_rw = az_bellman_targets(
                az_target, _bs_rw, _bd_rw, GENERATORS_DEV, SOLVED_DEV,
            )
        _bs_parts = [
            _bs_rw,
            ANCHOR_V0.expand(N_ANCHOR_V0, -1),
            ANCHOR_D1.repeat(N_ANCHOR_D1, 1),
        ]
        _target_parts = [
            _target_rw,
            torch.zeros(N_ANCHOR_V0, dtype=torch.float32, device=DEVICE),
            torch.ones(24 * N_ANCHOR_D1, dtype=torch.float32, device=DEVICE),
        ]
        if _bfs_cursor + BFS6_PER_BATCH <= _ep_bfs_states.size(0):
            _bs_parts.append(_ep_bfs_states[_bfs_cursor : _bfs_cursor + BFS6_PER_BATCH])
            _target_parts.append(_ep_bfs_dists[_bfs_cursor : _bfs_cursor + BFS6_PER_BATCH])
            _bfs_cursor += BFS6_PER_BATCH
        _bs_v = torch.cat(_bs_parts, dim=0)
        _target_v = torch.cat(_target_parts, dim=0)

        # Policy batch (random sample).
        _idx_p = torch.randint(0, NP, (POLICY_BATCH_SIZE,),
                               generator=_rng_p, device=DEVICE)
        _bs_p = states_p_dev[_idx_p]
        _ba_p = actions_p_dev[_idx_p]

        if az_autocast is not None:
            with az_autocast:
                _h_v = az_model.trunk(_bs_v)
                _pred_v = az_model.value_head(_h_v).squeeze(-1)
                _v_loss = F.mse_loss(_pred_v.float(), _target_v)
                _h_p = az_model.trunk(_bs_p)
                _logits_p = az_model.policy_head(_h_p)
                _p_loss = F.cross_entropy(_logits_p, _ba_p)
                _loss = ALPHA_POLICY * _p_loss + BETA_VALUE * _v_loss
        else:
            _h_v = az_model.trunk(_bs_v)
            _pred_v = az_model.value_head(_h_v).squeeze(-1)
            _v_loss = F.mse_loss(_pred_v.float(), _target_v)
            _h_p = az_model.trunk(_bs_p)
            _logits_p = az_model.policy_head(_h_p)
            _p_loss = F.cross_entropy(_logits_p, _ba_p)
            _loss = ALPHA_POLICY * _p_loss + BETA_VALUE * _v_loss

        az_optim.zero_grad(set_to_none=True)
        _loss.backward()
        az_optim.step()
        _tot_p += float(_p_loss.item())
        _tot_v += float(_v_loss.item())
        with torch.no_grad():
            _tot_acc += float((_logits_p.argmax(dim=1) == _ba_p).float().mean().item())
    az_sched.step()
    _avg_p = _tot_p / _n_batches
    _avg_v = _tot_v / _n_batches
    _avg_acc = _tot_acc / _n_batches
    _wall = time.time() - _t
    print(f"  [AZ] ep {_epoch:3d} | p_loss {_avg_p:.4f} | v_loss {_avg_v:.4f} | "
          f"top-1 {_avg_acc:.4f} | lr {az_sched.get_last_lr()[0]:.2e} | {_wall:.1f}s")

    if (_epoch + 1) % TARGET_UPDATE_EVERY == 0:
        az_target.load_state_dict(az_model.state_dict())
        print(f"  [AZ] refreshed target net at ep {_epoch}")

    if (_epoch + 1) % CHECKPOINT_EVERY == 0 or _epoch == EPOCHS_AZ - 1:
        _ckpt_path = _az_dir / f"epoch_{_epoch:04d}.pt"
        torch.save({
            "epoch": _epoch,
            "state_dict": az_model.state_dict(),
            "model_config": {
                "state_size": STATE_SIZE, "num_classes": STATE_SIZE,
                "hidden_dims": [2048, 512], "num_res_blocks": 2,
                "encoding": "embedding", "embed_dim": 16, "n_actions": N_GEN,
                "model_class": "ResMLPGFlowNet",
            },
            "p_loss": _avg_p, "v_loss": _avg_v, "top1_acc": _avg_acc,
        }, _ckpt_path)
        AZ_CKPTS.append(_ckpt_path)
        print(f"  [AZ] saved {_ckpt_path}")

print(f"[stage D] {len(AZ_CKPTS)} checkpoints: {[p.name for p in AZ_CKPTS]}")
del az_target
torch.cuda.empty_cache() if DEVICE == "cuda" else None
'''


CELL_STAGE_E1_PIDS = '''\
# ============================================================================
# Stage E1 - strat-5 pid selection (canonical 51-pid acceptance bench)
# ============================================================================
#
# Reproduces 03_solve.py:261-279 stratification: bucket pids by (pid // 100),
# pick STRAT_PER_BUCKET=5 from each bucket with random.Random(seed=0). The
# resulting 51 pids have been the project's strat-5 acceptance gate since m05.
# ============================================================================

print("=" * 72)
print(f"Stage E1 - strat-{STRAT_PER_BUCKET} pid selection (seed={STRAT_SEED})")
print("=" * 72)

_test_states = load_test_states(DATA / "test.csv")
_all_ids = sorted(_test_states.keys())
_buckets = {}
for _pid in _all_ids:
    _buckets.setdefault(_pid // 100, []).append(_pid)

_rng_strat = random.Random(STRAT_SEED)
SELECT_PIDS = []
for _b in sorted(_buckets):
    SELECT_PIDS.extend(sorted(_rng_strat.sample(_buckets[_b], min(STRAT_PER_BUCKET, len(_buckets[_b])))))

if LIMIT_FOR_LOCAL_TEST:
    SELECT_PIDS = SELECT_PIDS[:LIMIT_FOR_LOCAL_TEST]

print(f"[stage E1] {len(SELECT_PIDS)} pids: {SELECT_PIDS[:8]}{'...' if len(SELECT_PIDS) > 8 else ''}")
'''


CELL_STAGE_E2_HELPERS = '''\
# ============================================================================
# Stage E2 - sym-ensemble setup (rotations + GFlowValueAdapter + helpers)
# ============================================================================
#
# Conjugation: applying R then solving R*s_0*R_inv yields a path of generator
# names IN THE ROTATED FRAME. To use it on the original state we translate
# each move name through the conjugation map:
#     R_inv * g_n * R = g_{conj(n)}
# The conjugated path solves the ORIGINAL state.
#
# K_SYM=4 means: identity + 3 random non-identity rotations from rotations.npy.
# ============================================================================

print("=" * 72)
print(f"Stage E2 - sym-ensemble setup (K_SYM={K_SYM})")
print("=" * 72)


class GFlowValueAdapter(nn.Module):
    """Wraps ResMLPGFlowNet so __call__ returns just the distance scalar.

    sign=+1 for AZ regime (value head outputs predicted distance directly).
    """

    def __init__(self, gflow_model, sign=1.0, inference_chunk_size=4096):
        super().__init__()
        self.gflow = gflow_model
        self.sign = sign
        self.state_size = gflow_model.state_size
        self.num_classes = gflow_model.num_classes
        self.inference_chunk_size = inference_chunk_size

    def forward(self, x):
        if x.shape[0] <= self.inference_chunk_size:
            _logits, value = self.gflow(x)
            return self.sign * value
        outs = []
        for i in range(0, x.shape[0], self.inference_chunk_size):
            _logits, value = self.gflow(x[i : i + self.inference_chunk_size])
            outs.append(self.sign * value)
        return torch.cat(outs, dim=0)


def load_az_value_adapter(ckpt_path, device, dtype):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}
    mc = ckpt["model_config"]
    g = ResMLPGFlowNet(
        state_size=mc["state_size"], num_classes=mc["num_classes"],
        hidden_dims=tuple(mc["hidden_dims"]), num_res_blocks=mc["num_res_blocks"],
        encoding=mc.get("encoding", "embedding"),
        embed_dim=mc.get("embed_dim", 16),
        n_actions=mc.get("n_actions", 24),
    )
    g.load_state_dict(sd, strict=False)
    g = g.to(device=device, dtype=dtype).eval()
    return GFlowValueAdapter(g, sign=+1.0).to(device).eval()


def apply_rotation_to_state(state, R, R_inv):
    """R · state · R_inv. Convention: out[i] = R[state[R_inv[i]]]."""
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))


def compute_conjugation_map(R, R_inv, generators_dict, gen_names):
    """For each gen name n, find m such that R_inv * g_n * R = g_m."""
    n = len(R)
    perm_to_name = {tuple(g): nm for nm, g in generators_dict.items()}
    out = {}
    for nm in gen_names:
        g = generators_dict[nm]
        conj = tuple(R_inv[g[R[i]]] for i in range(n))
        if conj not in perm_to_name:
            raise ValueError(f"rotation is not a symmetry: R_inv * g_{nm} * R is not a generator")
        out[nm] = perm_to_name[conj]
    return out


# Load rotations and pick K_SYM of them (identity + K_SYM-1 random non-identity).
ROTATIONS_NP = np.load(DATA / "rotations.npy")
print(f"[stage E2] rotations.npy: shape={ROTATIONS_NP.shape}")

_identity = np.arange(STATE_SIZE, dtype=ROTATIONS_NP.dtype)
_id_idx = None
for _i in range(ROTATIONS_NP.shape[0]):
    if np.array_equal(ROTATIONS_NP[_i], _identity):
        _id_idx = _i
        break
assert _id_idx is not None, "identity row missing from rotations.npy"

_rng_sym = np.random.default_rng(0)
_other = [_i for _i in range(ROTATIONS_NP.shape[0]) if _i != _id_idx]
CHOSEN_ROT_IDXS = [_id_idx]
if K_SYM > 1:
    CHOSEN_ROT_IDXS += list(_rng_sym.choice(_other, size=K_SYM - 1, replace=False).tolist())

# Pre-compute conjugation maps once per rotation (each is a {move_name: move_name} dict).
ROTATIONS_DATA = []
for _r_idx in CHOSEN_ROT_IDXS:
    _R = tuple(int(x) for x in ROTATIONS_NP[_r_idx])
    _R_inv = tuple(int(x) for x in np.argsort(ROTATIONS_NP[_r_idx]))
    _conj_map = compute_conjugation_map(_R, _R_inv, PUZZLE.generators, PUZZLE.move_names)
    ROTATIONS_DATA.append({"rot_idx": _r_idx, "R": _R, "R_inv": _R_inv, "conj_map": _conj_map})
print(f"[stage E2] prepared {len(ROTATIONS_DATA)} rotations: rot_idxs={CHOSEN_ROT_IDXS}")
'''


CELL_STAGE_E3_MATRIX = '''\
# ============================================================================
# Stage E3 - Per-pid selection matrix (5 ckpts x 4 rotations x 51 pids)
# ============================================================================
#
# For each (ckpt, rotation) pair, build a fresh KhoruzhiiSolver and solve
# every selected pid. Per pid, the winner is the shortest VALID path across
# the matrix. This is the "both stacked" select - dimensions are
# (checkpoint diversity) x (rotation diversity).
#
# Wall budget on T4: 5 * 4 * 51 = 1020 beam runs at B=16384, ~3s each = ~50 min.
# ============================================================================

print("=" * 72)
print(f"Stage E3 - per-pid selection matrix "
      f"({len(AZ_CKPTS)} ckpts x {len(ROTATIONS_DATA)} rotations x {len(SELECT_PIDS)} pids)")
print("=" * 72)

INFERENCE_DTYPE = torch.bfloat16 if (DEVICE == "cuda" and USE_BF16) else torch.float32

# results[pid][(ckpt_idx, rot_idx)] = {"path_len": int, "path": list[str], "wall_s": float}
RESULTS = {pid: {} for pid in SELECT_PIDS}

_solver_cfg = KhoruzhiiSearchConfig(beam_width=BEAM, num_steps=MAX_STEPS,
                                    internal_batch_size=INTERNAL_BS)

_t_outer = time.time()
for _ck_i, _ck_path in enumerate(AZ_CKPTS):
    _adapter = load_az_value_adapter(_ck_path, DEVICE, INFERENCE_DTYPE)
    for _ri, _rot in enumerate(ROTATIONS_DATA):
        _solver = KhoruzhiiSolver(
            puzzle=PUZZLE, model=_adapter, device=DEVICE,
            internal_batch_size=INTERNAL_BS, random_seed=0,
            state_dtype=torch.int8,
        )
        _t_pair = time.time()
        _n_solved = 0
        _total_moves = 0
        for _pid in SELECT_PIDS:
            _s0 = _test_states[_pid]  # already a tuple of ints (load_test_states)
            _s_to_solve = apply_rotation_to_state(_s0, _rot["R"], _rot["R_inv"])
            _t_pid = time.time()
            _found, _plen, _path_rotated = _solver.solve(_s_to_solve, _solver_cfg)
            _wall = time.time() - _t_pid
            if not _found:
                continue
            # Translate path back through the conjugation map.
            _path_orig = [_rot["conj_map"][_nm] for _nm in _path_rotated]
            _vp = verify_path(PUZZLE, _s0, _path_orig)
            if not _vp.ok:
                # Should not happen if rotation conjugation is correct; skip silently.
                continue
            # Optional post-process (same-face reduction + adjacent-inverse cancellation).
            _path_pp = full_post_process(_path_orig, puzzle=PUZZLE, bfs_table=None)
            if not verify_path(PUZZLE, _s0, _path_pp).ok:
                _path_pp = _path_orig  # fall back if post-process broke verification
            _entry = {"path_len": len(_path_pp), "path": _path_pp, "wall_s": _wall}
            RESULTS[_pid][(_ck_i, _ri)] = _entry
            _n_solved += 1
            _total_moves += len(_path_pp)
        _pair_wall = time.time() - _t_pair
        print(f"  [E3] ckpt={_ck_path.name} rot_idx={_rot['rot_idx']:>2}: "
              f"solved {_n_solved}/{len(SELECT_PIDS)} | "
              f"total {_total_moves:>5} | wall {_pair_wall:>6.1f}s")
    del _adapter
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
print(f"[stage E3] matrix complete in {(time.time() - _t_outer) / 60:.1f} min")

# Per-pid winner.
PER_PID_WIN = {}
for _pid, _cells in RESULTS.items():
    if not _cells:
        PER_PID_WIN[_pid] = None
        continue
    (_best_key, _best_entry) = min(_cells.items(), key=lambda kv: kv[1]["path_len"])
    PER_PID_WIN[_pid] = {
        "ckpt_idx": _best_key[0],
        "rot_idx": _best_key[1],
        "path_len": _best_entry["path_len"],
        "path": _best_entry["path"],
        "n_valid_in_matrix": len(_cells),
    }

_n_won = sum(1 for v in PER_PID_WIN.values() if v is not None)
_total_moves = sum(v["path_len"] for v in PER_PID_WIN.values() if v is not None)
print(f"[stage E3] per-pid winners: {_n_won}/{len(SELECT_PIDS)} pids; "
      f"total moves = {_total_moves}; mean = {_total_moves / max(_n_won, 1):.2f}")
'''


CELL_STAGE_F = '''\
# ============================================================================
# Stage F - Verify + write submission CSV
# ============================================================================
#
# Per-pid winners cover the strat-5 subset (51 pids). The remaining pids in
# the 1001-pid test set get filled from the BFS-d6 fallback (`pp_bfs6_fallback.csv`).
# Then verify_submission asserts every path is valid.
# ============================================================================

print("=" * 72)
print("Stage F - verify + write submission CSV")
print("=" * 72)

# Load fallback paths for missing pids.
_fallback = {}
with open(DATA / "pp_bfs6_fallback.csv") as f:
    for _row in csv.DictReader(f):
        _pid = int(_row["initial_state_id"])
        _path_str = _row["path"].strip()
        _fallback[_pid] = _path_str.split(".") if _path_str else []

OUT_CSV = Path(WORKDIR) / "az_recipe_demo_submission.csv"
with open(OUT_CSV, "w", newline="") as _f:
    _w = csv.writer(_f)
    _w.writerow(["initial_state_id", "path"])
    for _pid in sorted(_test_states.keys()):
        _winner = PER_PID_WIN.get(_pid)
        if _winner is not None:
            _w.writerow([_pid, ".".join(_winner["path"])])
        elif _pid in _fallback and _fallback[_pid]:
            _w.writerow([_pid, ".".join(_fallback[_pid])])
        else:
            _w.writerow([_pid, ""])
print(f"[stage F] wrote {OUT_CSV}")

# Verify the full submission. With LIMIT_FOR_LOCAL_TEST set, most pids come
# from fallback, but every path must still be valid.
_report = verify_submission(PUZZLE, DATA / "test.csv", OUT_CSV)
print(f"[stage F] verify: {_report.n_valid}/{_report.n_total} valid; "
      f"total_moves = {_report.total_moves}")
if not _report.all_valid:
    _bad = _report.failures[:5]
    raise SystemExit(f"submission has {len(_report.failures)} invalid paths; first 5: {_bad}")
print(f"[stage F] all paths valid")
'''


CELL_INTERPRET = '''\
# ============================================================================
# Interpret results
# ============================================================================

print("=" * 72)
print("Result interpretation")
print("=" * 72)

# Strat-only stats (the bench we control).
_strat_winners = [PER_PID_WIN[p] for p in SELECT_PIDS if PER_PID_WIN[p] is not None]
_strat_solved = len(_strat_winners)
_strat_total = sum(w["path_len"] for w in _strat_winners)
print(f"strat-{STRAT_PER_BUCKET}: solved {_strat_solved}/{len(SELECT_PIDS)} | "
      f"total {_strat_total} | mean {_strat_total / max(_strat_solved, 1):.2f} | "
      f"max {max((w['path_len'] for w in _strat_winners), default=0)}")

# Best checkpoint per pid - histogram. The journey-insight payoff: if all pids
# pick the same global-best checkpoint, per-pid selection adds no signal. If
# multiple checkpoints win, the matrix is doing useful work.
from collections import Counter
_ckpt_winners = Counter(w["ckpt_idx"] for w in _strat_winners)
_rot_winners  = Counter(w["rot_idx"]  for w in _strat_winners)
print()
print("best CHECKPOINT per pid (Stage D ckpt index, 0..N-1):")
for _ck_i in sorted(_ckpt_winners.keys()):
    _bar = "#" * _ckpt_winners[_ck_i]
    print(f"  ckpt {_ck_i} ({AZ_CKPTS[_ck_i].name}):  {_ckpt_winners[_ck_i]:>3}  {_bar}")

print()
print("best ROTATION per pid (sym-ensemble rot index, 0..K_SYM-1):")
for _ri in sorted(_rot_winners.keys()):
    _bar = "#" * _rot_winners[_ri]
    print(f"  rot {_ri} (rotations.npy idx {ROTATIONS_DATA[_ri]['rot_idx']}):  "
          f"{_rot_winners[_ri]:>3}  {_bar}")

print()
if len(_ckpt_winners) > 1:
    print(f"{len(_ckpt_winners)} distinct checkpoints contributed wins -> per-pid selection added signal.")
else:
    print("All pids picked the same checkpoint -> per-pid selection didn't help here.")
    print("(Expected at very low EPOCHS_AZ; scale up to see the spread.)")
'''


CELL_PRODUCTION_MD = '''\
## Scaling to production

The defaults in this notebook (10 / 10 epochs, B=16k, K=4, strat-5) are tuned
for a single ~1-hour Kaggle GPU session - enough to demonstrate the full recipe
and produce a small submission, but not enough to compete on the leaderboard.

To scale toward the project's best result (`m_az_v4/epoch_0024.pt`,
51/51 strat-5 / mean 87.5 / first sub-89 standalone for a 6M model):

| knob | demo | production |
|---|---|---|
| `EPOCHS_PRETRAIN` | 10 | 50 |
| `EPOCHS_AZ` | 10 | 100 |
| `CHECKPOINT_EVERY` | 2 | 5 |
| `BEAM` | 16384 | 65536 |
| `K_SYM` | 4 | 8 |
| pids | strat-5 (51) | full-1001 |
| beam pass schedule | single B=16k | `[16384, 65536]` two-pass with `--niss` |

A full-1001 production solve uses a multi-pass + NISS recipe (CLAUDE.md rule 11
in the source repo) that can't fit a single 9h Kaggle GPU kernel - 5 ckpts x 8
rotations x 1001 pids at B=65k is ~17h. Either (a) run on local GPU, or
(b) split into two chained kernels with partial-result handoff.

The next code cell prints the production knob block (commented out so it
doesn't override your demo run).
'''


CELL_PRODUCTION_CODE = '''\
# # ===== PRODUCTION KNOB BLOCK - uncomment to use, requires multiple kernel sessions =====
# EPOCHS_PRETRAIN     = 50
# EPOCHS_AZ           = 100
# CHECKPOINT_EVERY    = 5             # 20 ckpts; pick a sliding window for Stage E
# TARGET_UPDATE_EVERY = 10
# K_SYM               = 8
# BEAM                = 65536
# # Stage E pids: full 1001 (drops STRAT_PER_BUCKET / STRAT_SEED entirely)
# SELECT_PIDS = sorted(_test_states.keys())
# # Production multi-pass + NISS lives in megaminx/scripts/03_solve.py - see
# # the ARGS-style usage in CLAUDE.md rule 11. The simplest portable form:
# #   --beams 16384,65536 --max-steps 60,150 --niss --bf16 --resume
# # (The current Stage E loop above runs single-pass; for multi-pass, wrap
# # the per-pid solve in a beam-escalation loop and try NISS = solve the
# # inverted scramble too, inverting the path back.)
print("[production cell] currently commented out; uncomment + re-run from Stage A.")
'''


CELL_CREDITS = '''\
## Credits and acknowledgments

The community-merged training source `merge_v9_with_77152.csv` (76,304 moves)
that drives Stage C is a min-merge of three submissions:

1. **78,029 moves** - this project's `merge_v7_curr_v3_pi_v2_rescue.csv`,
   produced by the m05 + qshort + sym-ensemble + rescue stack.
2. **79,911 moves** - colleague A's submission. *(Author: TODO_FILL_IN)*
3. **77,152 moves** - colleague B's submission. *(Author: TODO_FILL_IN)*

Per-pid min-merge means each pid contributes whichever of the three found the
shortest valid path. Higher path consistency across the merge (vs single-source)
makes the policy CE converge faster - empirically the AZ v4 recipe trained on
this merged dataset hit top-1 90%+ by ep 75 vs the single-source v3 dataset
plateauing at top-1 50% by ep 100.

If you fork this notebook and produce a substantially shorter merge (e.g. by
adding your own submissions to the per-pid min), please share back so the
community baseline keeps improving.

### Source repository

[Source repo on GitHub](TODO_FILL_IN_REPO_URL) - this notebook is generated
from `megaminx/kaggle_notebooks/az_recipe_full/build_notebook.py`.

### Related Kaggle artifacts

- Dataset: `artgor/megaminx-az-recipe-artifacts` (this notebook's data
  attachment).
- Sibling notebook: `artgor/cayleypy-megaminx-beam-az-v4-shareable` -
  TPU-friendly inference using the production-trained AZ v4 V model.
'''


CELL_FINAL = '''\
print()
print("=" * 72)
print("Notebook complete. Artifacts in /kaggle/working:")
print("=" * 72)
for _name in sorted(os.listdir(WORKDIR)):
    _full = Path(WORKDIR) / _name
    if _full.is_dir():
        _size = sum(p.stat().st_size for p in _full.rglob("*") if p.is_file())
        print(f"  {_name}/   {_size / 1e6:>7.1f} MB  ({sum(1 for _ in _full.rglob('*'))} entries)")
    else:
        print(f"  {_name}     {_full.stat().st_size / 1e6:>7.1f} MB")
print()
print(f"Submission: {OUT_CSV}")
print(f"Verify report: {_report.n_valid}/{_report.n_total} valid, "
      f"total_moves = {_report.total_moves}")
'''


# --- Cell ordering ---------------------------------------------------------

CELLS = [
    ("markdown", CELL_INTRO),
    ("markdown", CELL_RECIPE_DIAGRAM),
    ("code",     CELL_SETUP),
    ("code",     CELL_IMPORTS),
    ("code",     CELL_STAGE_A),
    ("markdown", CELL_STAGE_B0_FRONTIER_MD),
    ("code",     CELL_STAGE_B0_REGEN),
    ("code",     CELL_STAGE_B),
    ("code",     CELL_STAGE_C),
    ("code",     CELL_STAGE_D_HELPERS),
    ("code",     CELL_STAGE_D_TRAIN),
    ("code",     CELL_STAGE_E1_PIDS),
    ("code",     CELL_STAGE_E2_HELPERS),
    ("code",     CELL_STAGE_E3_MATRIX),
    ("code",     CELL_STAGE_F),
    ("code",     CELL_INTERPRET),
    ("markdown", CELL_PRODUCTION_MD),
    ("code",     CELL_PRODUCTION_CODE),
    ("markdown", CELL_CREDITS),
    ("code",     CELL_FINAL),
]


# --- Notebook assembly -----------------------------------------------------

def _build():
    nb = {
        "cells": [
            {
                "cell_type": ctype,
                "metadata": {},
                "source": src.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if ctype == "code" else {}),
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

    out_nb = HERE / "cayleypy-megaminx-az-recipe.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
    print(f"wrote {out_nb}  ({out_nb.stat().st_size:,} bytes, {len(CELLS)} cells)")

    meta = {
        # Slug must match what Kaggle derived from the title on first push.
        # The CLI warning "kernel title does not resolve to the specified id"
        # means subsequent pushes either (a) create a NEW kernel under the
        # title-slug, or (b) version the title-slug kernel — both surprising
        # if id differs. Keep id and title-slug aligned to avoid the trap.
        "id": "artgor/cayleypy-megaminx-az-recipe-end-to-end",
        "title": "cayleypy megaminx az recipe end to end",
        "code_file": "cayleypy-megaminx-az-recipe.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,           # flip to False after the dry-run completes
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": ["artgor/megaminx-az-recipe-artifacts"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    out_meta = HERE / "kernel-metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_meta}")


if __name__ == "__main__":
    _build()
