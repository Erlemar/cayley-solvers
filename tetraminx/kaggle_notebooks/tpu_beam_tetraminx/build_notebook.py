"""Build the Kaggle TPU notebook for the Professor Tetraminx beam.

Design difference from the IHES/megaminx builders: the JAX kernel is NOT inlined
into the notebook by source-munging.  `jax_model.py` and `jax_beam_spmd_v_only.py`
ship INSIDE the asset dataset and the notebook just puts the dataset on sys.path.
Updating the kernel is then a dataset version bump, and what runs on TPU is
byte-identical to what `cpu_smoke.py` validates locally.

Asset dataset `artgor/tetraminx-tpu-artifacts` must contain:
    puzzle_info.json
    tetra_symmetries.npy  tetra_symmetries_inv.npy  tetra_move_relabel.npy
    jax_model.py  jax_beam_spmd_v_only.py
    taz_v1_v_only.pt        <- AZ value head (sha256 e42e0027ba33), NOT tv0_bellman
    bfs_endgame.npz         <- exact d<=6 table, ~250 MB

    .venv/Scripts/python.exe tetraminx/kaggle_notebooks/tpu_beam_tetraminx/build_notebook.py

CONFIG SYNCED 2026-08-01 to the settings that produced -143 moves on GCP v6e-8.
The previous version of this notebook was several findings out of date -- it used
the pure ResMLP V, a 1M beam, two frames, no history dedup and no endgame table.
Every one of those is a measured loss:

  * MODEL. `taz_v1_v_only.pt` is the AZ dual-head's VALUE head with the 24-wide
    policy head stripped. It beat `tv0_bellman.pt` in an A/B at B=1M. The two have
    IDENTICAL param counts (4,996,481) and are distinguishable only by digest --
    e42e0027ba33 (AZ) vs 4447572792cc (ResMLP V) -- so the loader prints the sha.
  * FRAMES. Measured over 171 pids: the k0 INVERSE frame alone returns 4.54
    moves/TPU-hour, against 1.84 for all four and 1.99 for k0 forward alone. The
    inverse antisymmetry carries the win, not the spatial rotation, and frames 3-4
    add only 0.75 moves/hour. Prefer breadth (more pids) over depth (more frames).
  * HISTORY_DEPTH 1. Excludes candidates seen in the previous layer.
  * ENDGAME. The beam's goal test becomes "inside the d<=6 table"; the tail is then
    the table's optimal descent, so the last moves of every solve are provably
    optimal and the beam stops ~6 steps early where it is narrowest.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "cayleypy-tetraminx-tpu-beam"

MD_INTRO = r"""# Professor Tetraminx beam search on Kaggle TPU (JAX SPMD, V-only + sym ensemble)

Solves [CayleyPy Professor Tetraminx](https://www.kaggle.com/competitions/cayley-py-professor-tetraminx-solve-optimally)
with one **shared beam sharded across all 8 TPU cores**: every core owns a slice
of the same global beam, children are hash-routed to their owner core with
`all_to_all`, and dedup + top-k happen per owner.  A `B_GLOBAL` of 32M is one
beam of 32M states, not 8 independent beams of 4M.

**The puzzle.** 88 facelets, all distinct (a permutation puzzle, not a colour
puzzle).  24 generators = 4 vertex axes x 3 layers x 2 directions, every one of
order 3.  |G| = 1.54e32 and the non-backtracking branching factor is ~15.8, so a
uniformly random state sits ~27 moves from solved -- the whole leaderboard lives
within a few percent of optimal and single moves matter.

**Sym ensemble.** The facelet-automorphism group has order 15552 = 24 x 648,
where the 648 is a centralizer that acts trivially on reachable states.  The
useful quotient is the 24-element tetrahedral group (12 rotations + 12 mirrors,
mirrors flipping move direction).  Adding inverse antisymmetry (solve `s^-1`,
then reverse and invert the path) gives **48 independent frames per puzzle**.
Each frame is a separate beam; keep the shortest.  `FRAMES` selects which.

**Backpointers.** Per-step parent pointers are packed into a uint32
(24 parent-local bits + 3 rank + 5 move) and streamed to a host memmap, so beam
width is limited by HBM for the states, not by tree storage.

**Exact endgame.** The goal test is "inside the BFS d<=6 table", not "solved". The
beam stops ~6 steps early -- exactly where it is narrowest and least reliable --
and the tail is replaced by the table's optimal descent, so the last moves of every
solve are provably optimal. Every emitted path is replayed against the ORIGINAL
test state before it is recorded, which also catches any hash false-positive.

**Which frames, and why only one.** Measured over 171 pids on v6e-8: the k0
**inverse** frame alone returns **4.54 moves per TPU-hour**, against 1.84 for all
four frames and 1.99 for the k0 forward frame alone. The inverse antisymmetry is
what finds the shorter path, not the spatial rotation, and frames 3-4 together add
only 0.75 moves/hour. With a fixed TPU budget, more pids at one frame beats fewer
pids at four. `FRAMES` is a list -- add more if you want depth on a small pid set.

**Scale.** Defaults are the GCP configuration: AZ value head, 8M beam,
`history_depth=1`, one inverse frame. On a smaller/slower TPU runtime, drop
`B_GLOBAL` to 1M first and confirm a solve before scaling up; the tree memmap is
`NUM_STEPS * B_GLOBAL * 4` bytes per frame (2 GB at 8M) in /kaggle/working.

**Any pid.** `PIDS = None` runs all 1000; otherwise pass any list or range.
`RESUME` skips pids already complete in the output JSON, so a run that hits the
session limit can be continued by simply re-running the notebook.

Outputs `tetraminx_tpu_results.json`: one record per (pid, frame) with the move
path already mapped back to ORIGINAL puzzle coordinates and replay-verified, plus
`tetraminx_tpu_submission.csv` holding the best path per pid.
"""

SETUP = r'''
import os
# Must precede the jax import: the kernel hashes 88-facelet states into int64.
os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import json, sys, time, csv
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
devices = jax.devices()
print(f"jax {jax.__version__}  x64={jax.config.jax_enable_x64}")
print(f"{len(devices)} devices: {devices[0].device_kind}")
assert len(devices) >= 2, "expected a multi-core TPU runtime"
'''

CONFIG = r'''
# ---------------- configuration ----------------
# The AZ dual-head's VALUE head, policy head stripped. NOT tv0_bellman.pt: both are
# 4,996,481 params and identical in shape, so the sha printed at load is the only
# thing that tells them apart. e42e0027ba33 = AZ (this one), 4447572792cc = ResMLP V.
V_CHECKPOINT = "taz_v1_v_only.pt"
HIDDEN_DIMS  = (2048, 512)         # must match the checkpoint
NUM_RES_BLOCKS = 2

B_GLOBAL   = 48 * 1024 * 1024      # 8M is the width every GCP result came from;
                                   # 48M/64M work since the 2026-08-02 streaming
                                   # port (see PARENT_CHUNK). Drop to 1M to
                                   # smoke-test unfamiliar hardware.
NUM_STEPS  = 60
INTERNAL_BS = 16384                # V forward chunk; must be <= B_LOCAL

# Children are (B_LOCAL, n_gen, S) = B_LOCAL * 24 * 88 bytes if built in one go:
# 13.3 GB per rank at 48M, which does not fit a 16 GB TPU core. PARENT_CHUNK streams
# them in slices instead (277 MB per chunk at 131072), which is what makes >~16M
# possible at all. None is fine at <=8M, so it is derived rather than left to the
# reader.
#
# The streaming step body reached feature parity with the non-streaming one on
# 2026-08-02 (history_depth / endgame / no-backtrack / q_mode). Before that it took
# 7 inner args and returned 11 where the wrapper expected 9 / 13, so any
# parent_chunk run raised `TypeError: ... unexpected keyword argument 'q_mode'`.
# Verified by running cpu_smoke.py both ways on the same puzzle: identical paths,
# with history + no-backtrack + endgame all active. NOTE: that is CPU validation --
# it proves shapes and logic, NOT the TPU all_to_all collectives.
PARENT_CHUNK = None if B_GLOBAL <= 8 * 1024 * 1024 else 131072

# Phase D: pack the send-side bf16 score into the bucket and skip the receive-side
# V re-run. Saves ~8% of V compute. OFF by default here, unlike the megaminx 48M
# notebook: bf16 tie-breaks differently from fp32 and the kernel documents
# "path-length drift of 0-2 moves per pid" -- on this puzzle we are fighting for
# ~80 moves across 1000 pids, so an 8% speedup is not worth unquantified drift.
# Turn it on only if you have measured the drift on your pid set.
PACK_V_SCORE = False
PROGRESS_EVERY = 5                 # per-step timing; 0 = silent. A 48M call is hours.
# K_per_peer = ALPHA * B_LOCAL // world. ALPHA=1 makes each owner receive exactly
# B_LOCAL candidates, so its top-k is a NO-OP and all selection degenerates to a
# per-(sender,owner) bucket top-k -- a far worse approximation of the global
# top-k, and the reason the first 1M run found nothing on any of 8 calls.
# The validated cube kernel uses 2 ("receive-side 2x oversampling for quality").
ALPHA      = 2

HISTORY_DEPTH = 1                  # drop candidates seen in the previous layer
HISTORY_EXACT = True               # binary search; the bitmask variant false-positives
ENDGAME_NPZ   = "bfs_endgame.npz"  # exact d<=6 table; set None to solve to the goal
NO_BACKTRACK  = False              # measured neutral-to-negative here; kept as a knob

# Frames: (symmetry index 0..23, use_inverse). The k0 INVERSE frame alone is the
# efficient choice -- 4.54 moves/TPU-hour vs 1.84 for all four. Add (0, False) or
# (1, True) if you want depth on a small pid set instead of breadth.
FRAMES = [(0, True)]

# WHICH PUZZLES. None = every pid in test.csv. Otherwise any list/range, e.g.
#   PIDS = [990, 991, 992]
#   PIDS = range(0, 100)
#   PIDS = None            # all 1000
PIDS       = [990, 991, 992, 993]
PID_LIMIT  = None                  # cap after expansion, e.g. 60 to fit a session
RESUME     = True                  # skip pids already complete in OUT_JSON

# Floor for submission.csv. None = the competition sample_submission.csv (very long
# paths). Name a stronger CSV placed in the asset dataset to merge on top of real
# work instead -- the output is a per-pid min, so it can never be worse than this.
BASELINE_CSV = None

OUT_JSON = "/kaggle/working/tetraminx_tpu_results.json"

N_DEVICES = len(devices)
B_LOCAL = B_GLOBAL // N_DEVICES
K_PER_PEER = (ALPHA * B_LOCAL) // N_DEVICES
assert B_LOCAL * N_DEVICES == B_GLOBAL, "B_GLOBAL must divide by device count"
assert INTERNAL_BS <= B_LOCAL, "INTERNAL_BS must be <= B_LOCAL"
if PARENT_CHUNK:
    assert B_LOCAL % PARENT_CHUNK == 0, "PARENT_CHUNK must divide B_LOCAL"
    # The streaming body scores each chunk with internal_bs, so the chunk's child
    # count must be divisible by it (and in q_mode, the chunk's parent count).
    assert (PARENT_CHUNK * 24) % INTERNAL_BS == 0, (
        f"PARENT_CHUNK*24={PARENT_CHUNK * 24} must be divisible by "
        f"INTERNAL_BS={INTERNAL_BS}")

# The packed backpointer is 24 parent-local bits + 3 rank + 5 move, so B_LOCAL must
# fit in 2^24 and the device count in 2^3. Violating either corrupts the walkback
# SILENTLY -- paths come back wrong rather than erroring -- so assert it.
assert B_LOCAL <= (1 << 24), (
    f"B_LOCAL {B_LOCAL:,} exceeds the 24-bit parent-local field (max 16,777,216); "
    f"B_GLOBAL can reach {(1 << 24) * N_DEVICES:,} on {N_DEVICES} devices")
assert N_DEVICES <= (1 << 3), f"{N_DEVICES} devices exceeds the 3-bit rank field"

_NG, _S = 24, 88                   # n_gen, state_size (asserted in the next cell)
_child_gb = B_LOCAL * _NG * _S / 1e9
_chunk_gb = (PARENT_CHUNK or B_LOCAL) * _NG * _S / 1e9
_tree_gb = NUM_STEPS * B_GLOBAL * 4 / 1e9
print(f"B_GLOBAL {B_GLOBAL:,}  B_LOCAL {B_LOCAL:,}  K_PER_PEER {K_PER_PEER:,}")
print(f"frames {FRAMES}  history_depth {HISTORY_DEPTH}  pack_v_score {PACK_V_SCORE}")
print(f"children per rank: {_child_gb:.1f} GB unchunked -> {_chunk_gb:.2f} GB per chunk"
      + (f" (PARENT_CHUNK {PARENT_CHUNK:,}, {B_LOCAL // PARENT_CHUNK} chunks/step)"
         if PARENT_CHUNK else "  [NO CHUNKING]"))
if PARENT_CHUNK is None and _child_gb > 8:
    print(f"  *** {_child_gb:.1f} GB of children per rank will not fit a TPU core. "
          f"Set PARENT_CHUNK (e.g. 131072). ***")
print(f"tree memmap: {_tree_gb:.1f} GB per frame in /kaggle/working "
      f"(~20 GB writable; unlinked after each frame)")
if _tree_gb > 17:
    print(f"  *** {_tree_gb:.1f} GB is close to the /kaggle/working limit -- lower "
          f"NUM_STEPS or B_GLOBAL ***")
'''

LOAD = r'''
# ---------------- mounts ----------------
dataset_root = None
for cand in [Path("/kaggle/input/tetraminx-tpu-artifacts"),
             Path("/kaggle/input/datasets/artgor/tetraminx-tpu-artifacts")]:
    if cand.exists():
        dataset_root = cand
        break
if dataset_root is None:
    raise SystemExit("attach the artgor/tetraminx-tpu-artifacts dataset")

comp_root = None
for cand in [Path("/kaggle/input/cayley-py-professor-tetraminx-solve-optimally"),
             Path("/kaggle/input/competitions/cayley-py-professor-tetraminx-solve-optimally")]:
    if cand.exists():
        comp_root = cand
        break
if comp_root is None:
    raise SystemExit("attach the competition data")
print("assets:", dataset_root, "\ncomp:  ", comp_root)

sys.path.insert(0, str(dataset_root))
from jax_model import load_params_from_pt, num_params
from jax_beam_spmd_v_only import beam_solve_v_only_spmd_packed, make_mesh

PUZZLE_INFO = json.loads((dataset_root / "puzzle_info.json").read_text(encoding="utf-8"))
GENERATORS = PUZZLE_INFO["generators"]
MOVE_NAMES = list(GENERATORS.keys())
SOLVED = np.array(PUZZLE_INFO["central_state"], dtype=np.int64)
N_GEN, STATE_SIZE = len(MOVE_NAMES), len(SOLVED)
print(f"N_GEN={N_GEN}  STATE_SIZE={STATE_SIZE}")
assert (N_GEN, STATE_SIZE) == (24, 88)

all_moves_np = np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32)
all_moves = jnp.asarray(all_moves_np)
V0 = jnp.asarray(SOLVED.astype(np.int8))
INV_IDX = np.array([MOVE_NAMES.index(n[1:] if n.startswith("-") else "-" + n)
                    for n in MOVE_NAMES])

SYM     = np.load(dataset_root / "tetra_symmetries.npy").astype(np.int64)
SYM_INV = np.load(dataset_root / "tetra_symmetries_inv.npy").astype(np.int64)
RELABEL = np.load(dataset_root / "tetra_move_relabel.npy").astype(np.int64)
print(f"symmetries: {SYM.shape[0]} spatial ({2 * SYM.shape[0]} frames with antisymmetry)")

STATES = {}
with open(comp_root / "test.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        STATES[int(r["initial_state_id"])] = np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
print(f"{len(STATES)} test states")

v_params = load_params_from_pt(dataset_root / V_CHECKPOINT,
                              hidden_dims=HIDDEN_DIMS, num_res_blocks=NUM_RES_BLOCKS)
# Identify the CHECKPOINT, not just its shape: the AZ value head and the pure
# ResMLP V are both 4,996,481 params with head width 1, so a param count cannot
# tell you which model produced a result. Expect e42e0027ba33 (AZ value head).
import hashlib
_ck = dataset_root / V_CHECKPOINT
_sha = hashlib.sha256(_ck.read_bytes()).hexdigest()[:12]
print(f"V checkpoint: {V_CHECKPOINT} ({_ck.stat().st_size:,} B, sha256:{_sha})")
print(f"V params: {num_params(v_params):,}"
      + ("   [AZ value head]" if _sha == "e42e0027ba33" else
         "   [ResMLP V]" if _sha == "4447572792cc" else "   [unrecognised]"))

# ---------------- exact endgame table ----------------
EG = eg_hashes = eg_ztab = None
if ENDGAME_NPZ:
    _p = dataset_root / ENDGAME_NPZ
    if _p.exists():
        _eg = np.load(_p, allow_pickle=True)
        EG = {"hashes": _eg["hashes"], "depths": _eg["depths"],
              "ztab": _eg["ztab"], "max_depth": int(_eg["max_depth"])}
        eg_hashes = jnp.asarray(EG["hashes"])
        eg_ztab = jnp.asarray(EG["ztab"])
        print(f"endgame table: {EG['hashes'].size:,} states <= d{EG['max_depth']}")
    else:
        print(f"[warn] {ENDGAME_NPZ} not in the dataset -- solving to the exact goal, "
              f"which is slower and gives up the provably-optimal tail")

def eg_lookup(states):
    st = np.atleast_2d(states)
    h = np.zeros(st.shape[0], dtype=np.int64)
    for i in range(st.shape[1]):
        h ^= EG["ztab"][i][st[:, i]]
    pos = np.clip(np.searchsorted(EG["hashes"], h), 0, EG["hashes"].size - 1)
    out = np.full(st.shape[0], -1, dtype=np.int64)
    hit = EG["hashes"][pos] == h
    out[hit] = EG["depths"][pos[hit]]
    return out

def eg_descend(state):
    """Optimal move list from a table state to solved (frame coordinates)."""
    d = int(eg_lookup(state[None, :])[0])
    assert d >= 0, "descend called on a state outside the table"
    out, cur = [], state
    while d > 0:
        children = cur[all_moves_np]
        nxt = int(np.nonzero(eg_lookup(children) == d - 1)[0][0])
        out.append(nxt)
        cur = children[nxt]
        d -= 1
    return out

INV_MOVE_TBL = jnp.asarray(INV_IDX.astype(np.int32)) if NO_BACKTRACK else None

rng = np.random.default_rng(0)
hash_vec = jnp.asarray(rng.integers(0, int(1e15), size=STATE_SIZE, dtype=np.int64))
owner_hash_vec = jnp.asarray(np.random.default_rng(12345).integers(
    0, np.iinfo(np.uint32).max, size=STATE_SIZE, dtype=np.uint32))
mesh = make_mesh(devices)
'''

SOLVE = r'''
# ---------------- frame transforms (mirrors tetraminx/scripts/30_solve.py) ----
def to_frame(s0, k, inverted):
    """s -> (optional group-inverse) -> conjugate by symmetry k."""
    t = np.argsort(s0) if inverted else s0
    return SYM_INV[k][t[SYM[k]]]

def from_frame(path_idx, k, inverted):
    """Frame move indices -> original move indices.

    A path solving conj(s) maps through sigma to a path solving s; the inverse
    frame is then undone by reversing the path and inverting each move.
    """
    q = [int(RELABEL[k][m]) for m in path_idx]
    if inverted:
        q = [int(INV_IDX[m]) for m in reversed(q)]
    return q

def replay(s0, path_idx):
    cur = s0
    for m in path_idx:
        cur = cur[all_moves_np[m]]
    return cur

# ---------------- which pids ----------------
pid_list = sorted(STATES) if PIDS is None else [int(p) for p in PIDS]
missing = [p for p in pid_list if p not in STATES]
if missing:
    raise SystemExit(f"pids not in test.csv: {missing[:10]}")

results = []
done = set()
if RESUME and Path(OUT_JSON).exists():
    try:
        results = json.load(open(OUT_JSON, encoding="utf-8"))
        from collections import Counter
        cnt = Counter(r["pid"] for r in results)
        done = {p for p, n in cnt.items() if n >= len(FRAMES)}
        print(f"resuming: {len(done)} pids already complete in {OUT_JSON}")
    except Exception as e:
        print(f"[warn] could not read {OUT_JSON} ({e}); starting fresh")
        results = []

pid_list = [p for p in pid_list if p not in done]
if PID_LIMIT:
    pid_list = pid_list[:PID_LIMIT]
print(f"solving {len(pid_list)} pids x {len(FRAMES)} frame(s)")

for pid in pid_list:
    s0 = STATES[pid]
    for (k, inverted) in FRAMES:
        u = to_frame(s0, k, inverted)
        tree = f"/kaggle/working/tree_pid{pid}_k{k}_inv{int(inverted)}.u32"
        t0 = time.time()
        try:
            r = beam_solve_v_only_spmd_packed(
                list(u), v_params, all_moves, V0, hash_vec, mesh,
                B_local=B_LOCAL, K_per_peer=K_PER_PEER,
                n_gen=N_GEN, state_size=STATE_SIZE,
                num_steps=NUM_STEPS, dtype=jnp.bfloat16,
                internal_bs=INTERNAL_BS, tree_path=tree,
                parent_chunk=PARENT_CHUNK, owner_hash_vec=owner_hash_vec,
                history_depth=HISTORY_DEPTH, history_exact=HISTORY_EXACT,
                eg_hashes=eg_hashes, eg_ztab=eg_ztab,
                inv_move_tbl=INV_MOVE_TBL,
                pack_v_score=PACK_V_SCORE, progress_every=PROGRESS_EVERY,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"pid": pid, "sym": k, "inverted": inverted,
                            "found": False, "error": str(e),
                            "wall_s": time.time() - t0})
            continue

        rec = {"pid": pid, "sym": k, "inverted": inverted,
               "found": bool(r["found"]), "wall_s": r["wall_s"]}
        if r["found"]:
            frame_path = list(r["path_idx"])
            if EG is not None:
                # With an endgame table the beam stops on a TABLE node, not the
                # goal; splice the table's optimal tail. The replay below is what
                # catches a hash false-positive, so this is safe by construction.
                reached = replay(u, frame_path)
                if not np.array_equal(reached, SOLVED):
                    frame_path = frame_path + eg_descend(reached)
            orig = from_frame(frame_path, k, inverted)
            ok = bool(np.array_equal(replay(s0, orig), SOLVED))
            rec.update({"path_len": len(orig), "verify_ok": ok,
                        "path": ".".join(MOVE_NAMES[m] for m in orig)})
            print(f"pid {pid} k={k} inv={int(inverted)}: {len(orig)} moves  "
                  f"verify={ok}  {r['wall_s']:.0f}s")
        else:
            # min-V trajectory is the diagnostic that separates "beam is making
            # progress but ran out of steps" from "beam is not descending at all".
            traj = r.get("min_v_trajectory_rank0") or []
            rec["min_v_trajectory"] = traj
            print(f"pid {pid} k={k} inv={int(inverted)}: NOT FOUND  {r['wall_s']:.0f}s")
            if traj:
                print("   min V per step: "
                      + " ".join(f"{v:.1f}" for v in traj[:NUM_STEPS]))
        results.append(rec)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)

best = {}
for r in results:
    if r.get("found") and r.get("verify_ok") and r.get("path"):
        if r["pid"] not in best or r["path_len"] < len(best[r["pid"]].split(".")):
            best[r["pid"]] = r["path"]
print("\\nsolved pids:", len(best))
print("total over solved pids:", sum(len(p.split(".")) for p in best.values()))

# ---------------- submission.csv (ALL pids) ----------------
# A beam run touches a subset of pids, so a file holding only those is not a
# submission. Prefill every row from a baseline and overwrite only where this run is
# STRICTLY shorter -- a per-pid min, so the output can never score worse than the
# baseline it started from (CLAUDE.md Rule 26).
#
# BASELINE_CSV picks the floor: None uses the competition sample_submission.csv,
# whose paths are very long. Put a stronger CSV in the asset dataset and name it
# here to merge on top of real work instead.
baseline = {}
_bl_src = None
if BASELINE_CSV:
    _cand = dataset_root / BASELINE_CSV
    if _cand.exists():
        _bl_src = _cand
    else:
        print(f"[warn] BASELINE_CSV {BASELINE_CSV} not in the dataset")
if _bl_src is None:
    for _c in [comp_root / "sample_submission.csv", comp_root / "sample_submission.csv.zip"]:
        if _c.exists():
            _bl_src = _c
            break
if _bl_src is None:
    print("[warn] no baseline CSV found -- writing only the pids this run solved")
else:
    with open(_bl_src, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("path"):
                baseline[int(r["initial_state_id"])] = r["path"]
    print(f"baseline: {_bl_src.name} ({len(baseline)} rows, "
          f"{sum(len(p.split('.')) for p in baseline.values()):,} moves)")

rows = dict(baseline)
n_better = n_worse = 0
for pid, path in best.items():
    if pid in rows:
        if len(path.split(".")) < len(rows[pid].split(".")):
            rows[pid] = path; n_better += 1
        else:
            n_worse += 1
    else:
        rows[pid] = path; n_better += 1

# Re-verify EVERY emitted row against the original test state, not just ours -- this
# is what catches a merge that pairs a path with the wrong pid.
bad = []
for pid, path in rows.items():
    if pid in STATES and not np.array_equal(
            replay(STATES[pid], [MOVE_NAMES.index(m) for m in path.split(".")]), SOLVED):
        bad.append(pid)

SUB_CSV = "/kaggle/working/submission.csv"
with open(SUB_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["initial_state_id", "path"])
    for pid in sorted(rows):
        w.writerow([pid, rows[pid]])
total = sum(len(p.split(".")) for p in rows.values())
print(f"\\nwrote {SUB_CSV}: {len(rows)} rows, {total:,} moves")
print(f"  improved by this run : {n_better} pids")
print(f"  kept baseline        : {len(rows) - n_better} pids "
      f"({n_worse} where this run was not shorter)")
print(f"  invalid rows         : {len(bad)}" + (f"  {bad[:10]}" if bad else "  (all verified)"))
if len(rows) < len(STATES):
    print(f"  WARNING: {len(STATES) - len(rows)} pids missing -- not a complete submission")
'''


REPORT = r'''
# ---------------- results table ----------------
# Re-runnable on its own: reads `results` (or OUT_JSON) rather than recomputing.
import json as _json
from pathlib import Path as _Path

_recs = results if "results" in dir() else _json.load(open(OUT_JSON, encoding="utf-8"))

# Best verified record per pid, and the frame + wall that produced it.
_best = {}
_wall = {}
for _r in _recs:
    _p = _r["pid"]
    _wall[_p] = _wall.get(_p, 0.0) + float(_r.get("wall_s", 0.0))
    if _r.get("found") and _r.get("verify_ok") and _r.get("path"):
        _L = len(_r["path"].split("."))
        if _p not in _best or _L < _best[_p]["len"]:
            _best[_p] = {"len": _L, "sym": _r["sym"], "inv": int(_r["inverted"]),
                         "wall": float(_r.get("wall_s", 0.0))}

_base = {}
try:
    _base = {p: len(q.split(".")) for p, q in baseline.items()}
except Exception:
    pass

_solved = sorted(_best)
_unsolved = sorted(set(_wall) - set(_best))

print(f"{'pid':>5} {'len':>4} {'base':>5} {'delta':>6}  {'frame':>8} {'wall_s':>8}")
print("-" * 45)
for _p in _solved:
    _b = _best[_p]
    _bl = _base.get(_p)
    _d = (_b["len"] - _bl) if _bl is not None else None
    print(f"{_p:>5} {_b['len']:>4} "
          f"{(_bl if _bl is not None else '-'):>5} "
          f"{(f'{_d:+d}' if _d is not None else '-'):>6}  "
          f"{('k%d,inv%d' % (_b['sym'], _b['inv'])):>8} {_b['wall']:>8.0f}")
for _p in _unsolved:
    print(f"{_p:>5} {'--':>4} {(_base.get(_p, '-')):>5} {'-':>6}  {'-':>8} {_wall[_p]:>8.0f}")

_lens = [_best[_p]["len"] for _p in _solved]
print("-" * 45)
if _lens:
    _mean = sum(_lens) / len(_lens)
    print(f"solved      : {len(_solved)} / {len(_wall)} attempted"
          + (f"   ({len(_unsolved)} not found)" if _unsolved else ""))
    print(f"total moves : {sum(_lens):,}")
    print(f"AVERAGE     : {_mean:.2f} moves per solved pid")
    print(f"min / max   : {min(_lens)} / {max(_lens)}")
    _h = {}
    for _L in _lens:
        _h[_L] = _h.get(_L, 0) + 1
    print(f"histogram   : {dict(sorted(_h.items()))}")
    _cmp = [(_best[_p]["len"] - _base[_p]) for _p in _solved if _p in _base]
    if _cmp:
        _w = [d for d in _cmp if d < 0]
        print(f"vs baseline : {len(_w)} improved, {sum(_w)} moves saved "
              f"(compared on {len(_cmp)} pids)")
    _tw = sum(_wall.values())
    print(f"wall        : {_tw / 3600:.2f} h total, {_tw / max(1, len(_wall)) / 60:.1f} min/pid")
else:
    print("no verified solves in this run")
'''


def cell(src: str, kind: str = "code") -> dict:
    lines = src.strip("\n").splitlines(keepends=True)
    if kind == "code":
        return {"cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": lines}
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def main() -> int:
    nb = {
        "cells": [cell(MD_INTRO, "markdown"), cell(SETUP), cell(CONFIG),
                  cell(LOAD), cell(SOLVE), cell(REPORT)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = HERE / f"{SLUG}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = {
        "id": f"artgor/{SLUG}",
        "title": "CayleyPy Tetraminx TPU beam",
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": False,
        "dataset_sources": ["artgor/tetraminx-tpu-artifacts"],
        "competition_sources": ["cayley-py-professor-tetraminx-solve-optimally"],
        "kernel_sources": [],
    }
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {HERE / 'kernel-metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
