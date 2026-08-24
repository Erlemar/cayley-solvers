"""Build the cube555 Kaggle TPU notebook.

    .venv/Scripts/python.exe cube555/tpu/build_notebook.py

Writes cayleypy-cube555-tpu-beam-q.ipynb + its kernel-metadata JSON next to this
file. The notebook is generated rather than hand-edited so the cells stay diffable
and a fix lands in one place.

Adapted from tetraminx/kaggle_notebooks/tpu_beam_tetraminx/build_notebook{,_q}.py.
The tetraminx pair is a base + a string-patch; cube555 gets ONE file because there
is no V-only variant to share with -- every cube555 checkpoint is a 30-wide Q head.

WHAT DIFFERS FROM THE TETRAMINX KERNEL, and the cube555 measurement behind each:

  MODEL     ResMLPQ 24.76M, 30-wide Q head. NO transformer -- cube555 never trained
            one, so the PieceTransformer branch is absent from jax_model.py rather
            than present-and-unused.

  WIDTH     B_GLOBAL defaults to 2^21, NOT the 16M the tetraminx kernel uses.
            On cube555 width is measured NON-MONOTONIC: 2^20 solved 2/6, 2^21 6/6,
            2^22 3/6, 2^23 0/1 (RESULTS.md s4). This inverts the usual TPU story --
            the win here is THROUGHPUT (more pids and frames per session), not width.
            Caveat carried into the notebook: that sweep ran on the 20k checkpoint,
            which the same document calls the worst of six, and the stated mechanism
            ("a wider beam admits candidates the scorer cannot rank") is about scorer
            quality -- so the curve is UNVERIFIED for q555_2k_BEST and re-measuring it
            is one of the cheapest things this kernel can do.

  QV        QV_CONSISTENCY defaults to 0.0, not tetraminx's 0.3. On cube555 it is
            rejected twice independently: it trades solve rate for length (3/3 -> 2/3
            at lam 0.3), and on 5 matched pids it was +8.0 mean with 1 win in 5. A
            lost pid on this puzzle falls back to a ~900-move baseline.

  FRAMES    The PyTorch solver takes the FIRST frame that succeeds; this notebook
            runs every frame in FRAMES and keeps the per-pid MINIMUM. That is
            best-of-frames, which RESULTS.md s7 lists as an open lead ("strictly
            better and has never been measured") -- the kernel gets it for free.

  HISTORY   4, not 1. Measured -14% path length on cube555, saturating at 4 because
            all 30 generators are ODD, so the graph is parity-bipartite and revisits
            occur only at even lags.

  STATE     uint8 end to end. See the header of jax_beam_spmd_v_only.py -- this is
            the port's one dangerous change and it has a negative control.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "cayleypy-cube555-tpu-beam-q"

MD_INTRO = r"""# CayleyPy 5x5x5 beam search on Kaggle TPU -- 30-wide Q head, JAX SPMD

Solves [CayleyPy 555 cube](https://www.kaggle.com/competitions/cayley-py-555-cube) with one
**shared beam sharded across all 8 TPU cores**, scored by an **all-neighbours Q head**
instead of a per-child value function.

**Why a Q head.** A V model must expand every child and score it: `B x 30` forwards per
beam step. A Q head with `output_dim = 30` scores all 30 children from **one forward on
the parent**. That 30x is the entire inference economics of this puzzle -- it is why a
two-million-wide beam runs on one device at all.

**The objective behind the head.** Walk `k ~ U[2,80]` steps from solved, pick a pivot `p`,
and label exactly two of the 30 actions: `Q(s, undo) = p-1`, `Q(s, next) = p+1`. The two
labels always differ by exactly 2 with zero conditional variance, so the loss cannot be
reduced by flattening the gap between a good and a bad child -- which is what an
MSE-on-walk-depth V loss does at depth. Exact BFS anchors (`d<=4`, all 30 columns) are in
every batch; without them the Bellman bootstrap settles at `V(solved) ~ 2`.

---

## This is NOT a "width is the lever" puzzle

Every sibling kernel in this family buys quality with beam width. On cube555 that is
measured **false**. Same pids, same frame, only width varying:

| width | solved |
|---|---|
| 2^20 | 2/6 |
| **2^21** | **6/6** |
| 2^22 | 3/6 |
| 2^23 | 0/1 |

No OOM anywhere -- 2^23 ran fine and simply found less. A wider beam admits candidates
the scorer cannot rank, and they crowd out the good ones in a global top-B. So the
default here is `B_GLOBAL = 2^21`, and **the TPU's contribution is throughput, not
width**: eight cores at the measured-best width means more pids and more symmetry frames
per session, which is where the remaining score actually is.

*Caveat, stated because it changes what you should do with this kernel:* that width sweep
ran on `q555_20k_deployed.pt`, which the source project measures as **the worst of six
checkpoints**. The mechanism it proposes is about scorer quality, so the curve may well
move with a better scorer. Re-running it on `q555_2k_BEST.pt` is a two-hour experiment
and one of the highest-value things this notebook can do -- see `WIDTH_SWEEP` in the
config cell.

## Checkpoints: the deployed one is not the good one

Six Bellman refinement lengths, 24 disjoint pids, `--frames 0`, 2^21, only the checkpoint
varying:

| Bellman steps | solved | mean length |
|---|---|---|
| **2,000** | **18/24** | **130.1** |
| 4,000 | 9/24 | 141.4 |
| 6,000 | 12/24 | 140.9 |
| 8,000 | 9/24 | 143.9 |
| 20,000 (shipped the 187,780 submission) | 10/24 | 174.5 |

`q555_2k_BEST.pt` is the default here. An independent check on 2026-08-23 at **half** the
production width reproduced this: pid 1034 came back at **122 moves against the shipped
submission's 176**, pid 1020 at 134 against 138.

The older checkpoints are still worth running -- a min-merge over all six arms solved
24/24 where the best single arm solved 18/24, so they are complementary. Run them as
separate arms and merge; `BLEND_CHECKPOINTS` (averaging scores inside one step) is a
*different* and untested thing, and defaults off.

## Best-of-frames, for free

Each of the 48 conjugation frames is close to an independent ~33-50% draw, so `k` frames
solve `1-(1-q)^k`. The PyTorch solver stops at the **first** frame that succeeds; this
notebook runs every frame in `FRAMES` and keeps the per-pid **minimum**, which the source
project lists as an untested improvement. `--invert` is legal on this puzzle (a state is
a group element), giving 48 x 2 = 96 distinct trajectories.

## Exact endgame

The goal test is "inside the exact BFS `d<=5` ball", not "solved" -- widening the target
from 1 state to **10,739,017** -- and the tail is spliced from the table's optimal
descent. Membership on device is a 64-bit Zobrist hash test, so a false positive is
possible (~0.06 expected over ~1e11 lookups); the host-side splice re-derives the descent
and **fails soft**, dropping that one frame instead of the run.

Every emitted path is replayed against the ORIGINAL test state before being recorded, and
the submission is a per-pid min against a baseline, so it can never score worse than the
floor it started from.
"""

SETUP = r'''
import os
# Must precede the jax import: the kernel hashes 150-sticker states into int64.
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
# ResMLPQ, 24,757,807 params, 30-wide Q head + auxiliary V head.
#   q555_2k_BEST.pt        2,000 Bellman steps  -- 18/24 solved, mean 130.1  <-- default
#   q555_6k.pt             6,000                -- 12/24, 140.9
#   q555_20k_deployed.pt  20,000                -- 10/24, 174.5 (shipped the 187,780)
#   q555_pretrained.pt         0                -- the parent all three warm-started from
# Keep the losers for MERGING (min over arms was 24/24 vs 18/24 for the best single
# one), not for blending.
Q_CHECKPOINT = "q555_2k_BEST.pt"

# Score-space blend of several checkpoints inside ONE beam step. Legitimate in
# principle -- same architecture, same objective, same pretrained parent, so they
# share a distance scale -- but UNMEASURED on cube555. The measured multi-checkpoint
# result is a per-pid MIN over separate runs, which is not this. None = off.
BLEND_CHECKPOINTS = None           # e.g. ["q555_6k.pt"]
BLEND_WEIGHTS     = None           # e.g. [0.7, 0.3]; first entry is Q_CHECKPOINT

# score = Q + lam*|Q - (V(s)-1)|, both heads from ONE trunk pass, so it is free.
# REJECTED TWICE on cube555 -- 3/3 -> 2/3 solved at lam 0.30, and +8.0 mean on 5
# matched pids with 1 win in 5. On this puzzle a lost pid falls back to a ~900-move
# baseline, so a length lever that costs coverage is a large net loss. 0.0 = off.
QV_CONSISTENCY = 0.0

# 2^21. WIDER IS WORSE HERE: 2^20 2/6, 2^21 6/6, 2^22 3/6, 2^23 0/1 -- no OOM, the
# scorer simply cannot rank what a wider beam admits. See the intro for why this
# number should be re-measured on q555_2k_BEST rather than trusted.
B_GLOBAL   = 2 * 1024 * 1024
NUM_STEPS  = 300                   # a beam finds a length-L solution AT step L, so
                                   # this only truncates. Real solutions run 118-250.
INTERNAL_BS = 16384                # model forward chunk; must divide B_LOCAL

# Children are (B_LOCAL, 30, 150) bytes if built in one go. At B_GLOBAL=2M that is
# 1.2 GB/rank and fits a 16 GB core unchunked. cube555 states are 150 bytes against
# tetraminx's 88 and there are 30 moves against 24, so a child block here costs 2.1x
# the tetraminx one -- the chunking threshold arrives at a LOWER width than in the
# sibling kernel. PARENT_CHUNK streams them in slices when it does not fit.
PARENT_CHUNK = None if B_GLOBAL <= 4 * 1024 * 1024 else 131072

PACK_V_SCORE = True                # REQUIRED by q_mode: a Q head returns 30 action
                                   # scores, not a state value, so the receive side
                                   # cannot rescore a bare state.
PROGRESS_EVERY = 10                # per-step timing; 0 = silent
ALPHA      = 2                     # receive-side oversampling. 1 makes the owner's
                                   # top-k a no-op and selection degenerates.

HISTORY_DEPTH = 4                  # -14% path length on cube555; saturates at 4
HISTORY_EXACT = True
ENDGAME_NPZ   = "bfs_endgame.npz"  # exact d<=5 ball; None solves to the goal alone
NO_BACKTRACK  = False              # neutral: history_depth>=1 already drops these

# Frames: (symmetry index 0..47, use_inverse). EVERY frame listed is run and the
# per-pid MINIMUM is kept -- this is best-of-frames, not first-of-frames.
# 0,7,19,33,41,47 are the six the PyTorch campaign used; inverse frames are a second
# axis that is legal here and never swept.
FRAMES = [(0, False), (7, True)]

# WHICH PUZZLES. None = every pid in test.csv. The score lives in the DEEP pids:
# pids 35..1034 are random walks whose baseline is length pid-34, so solving a deep
# one saves hundreds of moves while the 35 Santa pids (0..34) save ZERO -- our ~177
# loses to their 93.6 baseline. Run deepest-first.
PIDS       = list(range(1034, 1000, -1))
PID_LIMIT  = None
RESUME     = True

# Floor for submission.csv. Put a strong CSV in the asset dataset and name it here;
# the output is a per-pid min so it can never be worse. None = the competition
# sample_submission.csv, which is very long.
BASELINE_CSV = None

OUT_JSON = "/kaggle/working/cube555_tpu_results.json"

N_DEVICES = len(devices)
B_LOCAL = B_GLOBAL // N_DEVICES
K_PER_PEER = (ALPHA * B_LOCAL) // N_DEVICES
assert B_LOCAL * N_DEVICES == B_GLOBAL, "B_GLOBAL must divide by device count"
assert INTERNAL_BS <= B_LOCAL, "INTERNAL_BS must be <= B_LOCAL"
assert B_LOCAL % INTERNAL_BS == 0, "q_mode scans over PARENTS: INTERNAL_BS must divide B_LOCAL"
if PARENT_CHUNK:
    assert B_LOCAL % PARENT_CHUNK == 0, "PARENT_CHUNK must divide B_LOCAL"
    assert PARENT_CHUNK % INTERNAL_BS == 0, "q_mode: INTERNAL_BS must divide PARENT_CHUNK"
    assert (PARENT_CHUNK * 30) % INTERNAL_BS == 0

# The packed backpointer is 24 parent-local bits + 3 rank + 5 move. 30 generators fit
# the 5-bit move field with 2 to spare. Violating either budget corrupts the walkback
# SILENTLY -- paths come back wrong rather than erroring -- so assert it.
assert B_LOCAL <= (1 << 24), f"B_LOCAL {B_LOCAL:,} exceeds the 24-bit parent-local field"
assert N_DEVICES <= (1 << 3), f"{N_DEVICES} devices exceeds the 3-bit rank field"

_NG, _S = 30, 150
_child_gb = B_LOCAL * _NG * _S / 1e9
_chunk_gb = (PARENT_CHUNK or B_LOCAL) * _NG * _S / 1e9
_tree_gb = NUM_STEPS * B_GLOBAL * 4 / 1e9
print(f"B_GLOBAL {B_GLOBAL:,}  B_LOCAL {B_LOCAL:,}  K_PER_PEER {K_PER_PEER:,}")
print(f"frames {FRAMES} (best-of, not first-of)  history_depth {HISTORY_DEPTH}  "
      f"qv {QV_CONSISTENCY}")
print(f"children per rank: {_child_gb:.2f} GB unchunked -> {_chunk_gb:.2f} GB per chunk"
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
for cand in [Path("/kaggle/input/cube555-tpu-artifacts"),
             Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts")]:
    if cand.exists():
        dataset_root = cand
        break
if dataset_root is None:
    raise SystemExit("attach the artgor/cube555-tpu-artifacts dataset")

comp_root = None
for cand in [Path("/kaggle/input/cayley-py-555-cube"),
             Path("/kaggle/input/competitions/cayley-py-555-cube")]:
    if cand.exists():
        comp_root = cand
        break
if comp_root is None:
    raise SystemExit("attach the cayley-py-555-cube competition data")
print("assets:", dataset_root, "\ncomp:  ", comp_root)

sys.path.insert(0, str(dataset_root))
from jax_model import (load_params_from_pt, make_blend, has_value_head, num_params)
from jax_beam_spmd_v_only import beam_solve_v_only_spmd_packed, make_mesh

PUZZLE_INFO = json.loads((dataset_root / "puzzle_info.json").read_text(encoding="utf-8"))
GENERATORS = PUZZLE_INFO["generators"]
MOVE_NAMES = list(GENERATORS.keys())
SOLVED = np.array(PUZZLE_INFO["central_state"], dtype=np.int64)
N_GEN, STATE_SIZE = len(MOVE_NAMES), len(SOLVED)
print(f"N_GEN={N_GEN}  STATE_SIZE={STATE_SIZE}")
assert (N_GEN, STATE_SIZE) == (30, 150)
# Picture cube: the solved state is the identity permutation, so a state IS a group
# element. This is what makes the inverse frames below legal.
assert np.array_equal(SOLVED, np.arange(STATE_SIZE)), "expected the identity central state"

all_moves_np = np.array([GENERATORS[n] for n in MOVE_NAMES], dtype=np.int32)
all_moves = jnp.asarray(all_moves_np)
# uint8, NOT int8 -- classes 128..149 wrap negative under int8 and the failure is
# version-dependent (silent on some numpy/jax builds). See jax_beam_spmd_v_only.py.
V0 = jnp.asarray(SOLVED.astype(np.uint8))
INV_IDX = np.array([MOVE_NAMES.index(n[1:] if n.startswith("-") else "-" + n)
                    for n in MOVE_NAMES])

SYM     = np.load(dataset_root / "cube555_sym.npy").astype(np.int64)
SYM_INV = np.load(dataset_root / "cube555_sym_inv.npy").astype(np.int64)
# FRAME move -> ORIGINAL move. The sibling table (sym_move_relabel_48) goes the other
# way; using it would give paths of the right length that do not solve the original.
RELABEL = np.load(dataset_root / "cube555_move_relabel_inv.npy").astype(np.int64)
print(f"symmetries: {SYM.shape[0]} conjugation frames "
      f"({2 * SYM.shape[0]} with the inverse axis)")

STATES = {}
with open(comp_root / "test.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        STATES[int(r["initial_state_id"])] = np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
print(f"{len(STATES)} test states")

import hashlib
v_params = load_params_from_pt(dataset_root / Q_CHECKPOINT)
_ck = dataset_root / Q_CHECKPOINT
_sha = hashlib.sha256(_ck.read_bytes()).hexdigest()[:12]
print(f"Q checkpoint: {Q_CHECKPOINT} ({_ck.stat().st_size:,} B, sha256:{_sha})")
if BLEND_CHECKPOINTS:
    _members = [v_params] + [load_params_from_pt(dataset_root / b)
                             for b in BLEND_CHECKPOINTS]
    v_params = make_blend(_members, BLEND_WEIGHTS)
    print(f"blend: {len(_members)} members, weights {v_params['weights']}")
_head = v_params["members"][0] if "members" in v_params else v_params
Q_MODE = int(_head["head_w"].shape[-1]) == N_GEN
print(f"params: {num_params(v_params):,}   q_mode {Q_MODE}   qv {QV_CONSISTENCY}")
assert Q_MODE, "this kernel is the Q-head recipe; head width must be 30"
if QV_CONSISTENCY and not has_value_head(v_params):
    raise SystemExit("QV_CONSISTENCY needs a checkpoint with v_head.* (az_head=True)")

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
    st = np.atleast_2d(np.asarray(states, dtype=np.int64))
    h = np.zeros(st.shape[0], dtype=np.int64)
    for i in range(st.shape[1]):
        h ^= EG["ztab"][i][st[:, i]]
    pos = np.clip(np.searchsorted(EG["hashes"], h), 0, EG["hashes"].size - 1)
    out = np.full(st.shape[0], -1, dtype=np.int64)
    hit = EG["hashes"][pos] == h
    out[hit] = EG["depths"][pos[hit]]
    return out

def eg_descend(state):
    """Optimal move list from a table state to solved, in frame coordinates.

    Returns None instead of raising when the state is NOT really in the table. The
    device-side membership test compares a 64-bit Zobrist hash only, so at 10.7M
    entries and ~1e11 lookups a false positive is expected roughly 0.06 times per
    long campaign -- and one killed a 29-hour run on the PyTorch side. A false
    positive must cost one frame, never the session.
    """
    d = int(eg_lookup(state[None, :])[0])
    if d < 0:
        return None
    out, cur = [], state
    while d > 0:
        children = cur[all_moves_np]
        hit = np.nonzero(eg_lookup(children) == d - 1)[0]
        if hit.size == 0:
            return None
        out.append(int(hit[0]))
        cur = children[int(hit[0])]
        d -= 1
    return out if np.array_equal(cur, SOLVED) else None

INV_MOVE_TBL = jnp.asarray(INV_IDX.astype(np.int32)) if NO_BACKTRACK else None

rng = np.random.default_rng(0)
hash_vec = jnp.asarray(rng.integers(0, int(1e15), size=STATE_SIZE, dtype=np.int64))
owner_hash_vec = jnp.asarray(np.random.default_rng(12345).integers(
    0, np.iinfo(np.uint32).max, size=STATE_SIZE, dtype=np.uint32))
mesh = make_mesh(devices)
'''

SOLVE = r'''
# ---------------- frame transforms (mirror cube555/scripts/30_solve.py) ----------
def to_frame(s0, k, inverted):
    """s -> (optional group-inverse) -> conjugate by symmetry k.

    invert_state on a picture cube is just argsort: out[v] = i. Legal here and
    forbidden on the 4x4x4 sibling, which is a COLOUR cube where a state is not a
    group element.
    """
    t = np.argsort(s0) if inverted else s0
    return SYM_INV[k][t[SYM[k]]] if k else t

def from_frame(path_idx, k, inverted):
    """Frame move indices -> original move indices."""
    q = [int(RELABEL[k][m]) for m in path_idx] if k else list(path_idx)
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
                q_mode=Q_MODE, qv_consistency=QV_CONSISTENCY,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({"pid": pid, "sym": k, "inverted": inverted,
                            "found": False, "error": str(e),
                            "wall_s": time.time() - t0})
            continue

        rec = {"pid": pid, "sym": k, "inverted": inverted,
               "found": bool(r["found"]), "wall_s": r["wall_s"],
               "checkpoint": Q_CHECKPOINT, "b_global": B_GLOBAL}
        if r["found"]:
            frame_path = list(r["path_idx"])
            ok = True
            if EG is not None:
                # The beam stopped on a TABLE node, not the goal; splice the exact
                # tail. eg_descend returns None on a hash false positive rather than
                # raising -- one bad frame must not cost the session.
                reached = replay(u, frame_path)
                if not np.array_equal(reached, SOLVED):
                    tail = eg_descend(reached)
                    if tail is None:
                        print(f"pid {pid} k={k} inv={int(inverted)}: endgame hash "
                              f"FALSE POSITIVE -- frame dropped")
                        rec.update({"found": False, "endgame_false_positive": True})
                        results.append(rec)
                        continue
                    frame_path = frame_path + tail
            orig = from_frame(frame_path, k, inverted)
            ok = bool(np.array_equal(replay(s0, orig), SOLVED))
            rec.update({"path_len": len(orig), "verify_ok": ok,
                        "path": ".".join(MOVE_NAMES[m] for m in orig)})
            if not ok:
                # A wrong RELABEL direction looks exactly like this and nothing else
                # does: right length, solves the frame, does not solve the original.
                print(f"pid {pid} k={k}: TRANSLATED PATH DOES NOT SOLVE -- discarded")
            print(f"pid {pid} k={k} inv={int(inverted)}: {len(orig)} moves  "
                  f"verify={ok}  {r['wall_s']:.0f}s")
        else:
            traj = r.get("min_v_trajectory_rank0") or []
            rec["min_v_trajectory"] = traj
            print(f"pid {pid} k={k} inv={int(inverted)}: NOT FOUND  {r['wall_s']:.0f}s")
            if traj:
                print("   min Q per step: "
                      + " ".join(f"{v:.1f}" for v in traj[:NUM_STEPS]))
        results.append(rec)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)

# BEST of frames, not first of frames.
best = {}
for r in results:
    if r.get("found") and r.get("verify_ok") and r.get("path"):
        if r["pid"] not in best or r["path_len"] < len(best[r["pid"]].split(".")):
            best[r["pid"]] = r["path"]
print("\nsolved pids:", len(best))
print("total over solved pids:", sum(len(p.split(".")) for p in best.values()))

# ---------------- submission.csv (ALL pids) ----------------
# A beam run touches a subset of pids, so a file holding only those is not a
# submission. Prefill every row from a baseline and overwrite only where this run is
# STRICTLY shorter -- a per-pid min, so the output can never score worse than its
# floor.
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
print(f"\nwrote {SUB_CSV}: {len(rows)} rows, {total:,} moves")
print(f"  improved by this run : {n_better} pids")
print(f"  kept baseline        : {len(rows) - n_better} pids "
      f"({n_worse} where this run was not shorter)")
print(f"  invalid rows         : {len(bad)}" + (f"  {bad[:10]}" if bad else "  (all verified)"))
if len(rows) < len(STATES):
    print(f"  WARNING: {len(STATES) - len(rows)} pids missing -- not a complete submission")
'''

REPORT = r'''
# ---------------- results table ----------------
# Re-runnable on its own: reads OUT_JSON rather than recomputing anything.
import json
from pathlib import Path
import numpy as np

recs = json.load(open(OUT_JSON, encoding="utf-8")) if Path(OUT_JSON).exists() else []
if not recs:
    print("no results yet")
else:
    by_pid = {}
    for r in recs:
        by_pid.setdefault(r["pid"], []).append(r)

    print(f"{'pid':>5} {'base':>6} {'best':>6} {'saved':>7}  {'frames':>7}  per-frame")
    print("-" * 74)
    tot_saved = n_solved = 0
    for pid in sorted(by_pid):
        rs = by_pid[pid]
        okr = [r for r in rs if r.get("found") and r.get("verify_ok")]
        # pids 35..1034 are random walks whose shipped baseline is length pid-34.
        base = (pid - 34) if pid >= 35 else None
        if okr:
            b = min(r["path_len"] for r in okr)
            n_solved += 1
            saved = (base - b) if base is not None else 0
            tot_saved += max(0, saved)
            per = " ".join(
                f"k{r['sym']}{'i' if r['inverted'] else 'f'}:"
                + (str(r["path_len"]) if r.get("verify_ok") else "x")
                for r in rs)
            print(f"{pid:>5} {str(base or '-'):>6} {b:>6} {saved:>7}  "
                  f"{len(okr)}/{len(rs):<5}  {per}")
        else:
            why = "fp" if any(r.get("endgame_false_positive") for r in rs) else "none"
            print(f"{pid:>5} {str(base or '-'):>6} {'-':>6} {0:>7}  {0}/{len(rs):<5}  {why}")
    print("-" * 74)
    print(f"solved {n_solved}/{len(by_pid)} pids; moves saved vs baseline: {tot_saved:,}")

    walls = [r["wall_s"] for r in recs if "wall_s" in r]
    if walls:
        print(f"wall: {sum(walls) / 3600:.2f} h total, {np.mean(walls):.0f}s mean per "
              f"frame ({len(walls)} frame-runs)")
    lens = [min(r["path_len"] for r in v if r.get("verify_ok"))
            for v in by_pid.values() if any(r.get("verify_ok") for r in v)]
    if lens:
        print(f"path length: mean {np.mean(lens):.1f}  min {min(lens)}  max {max(lens)}")
        print("  reference: q555_2k_BEST measured mean 130.1 at 2^21 on 24 disjoint "
              "pids; the shipped 187,780 submission averages 167.8 over its 599 solves")

    # Frame contribution: which frames actually earned their wall time. This is the
    # measurement the notebook exists to make cheap -- frames are the main lever and
    # their marginal value decays as the residue enriches in hard pids.
    from collections import defaultdict
    won = defaultdict(int)
    for pid, rs in by_pid.items():
        okr = [r for r in rs if r.get("verify_ok")]
        if okr:
            b = min(r["path_len"] for r in okr)
            for r in okr:
                if r["path_len"] == b:
                    won[(r["sym"], r["inverted"])] += 1
    if won:
        print("\nframes that produced a pid's best path (ties counted for each):")
        for (k, inv), n in sorted(won.items(), key=lambda kv: -kv[1]):
            print(f"  k{k}{'i' if inv else 'f'}: {n}")
'''


def cell(src: str, kind: str = "code") -> dict:
    return ({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
            if kind == "markdown" else
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": src.splitlines(True)})


DEFAULT_PIDS_LINE = "PIDS       = list(range(1034, 1000, -1))"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", type=str, default=None,
                    help="override the PIDS line, e.g. --pids 1034,1020 for a TPU "
                         "smoke. Comma-separated; the default is the deepest-34 queue.")
    ap.add_argument("--b-global", type=int, default=None,
                    help="override B_GLOBAL (e.g. for a width sweep arm)")
    args = ap.parse_args()

    config = CONFIG
    if args.pids:
        pids = [int(p) for p in args.pids.split(",")]
        assert DEFAULT_PIDS_LINE in config, "PIDS line moved"
        config = config.replace(
            DEFAULT_PIDS_LINE,
            f"PIDS       = {pids}   # OVERRIDE via build_notebook.py --pids")
        print(f"PIDS overridden: {pids}")
    if args.b_global:
        old = "B_GLOBAL   = 2 * 1024 * 1024"
        assert old in config, "B_GLOBAL line moved"
        config = config.replace(
            old, f"B_GLOBAL   = {args.b_global}   # OVERRIDE via --b-global")
        print(f"B_GLOBAL overridden: {args.b_global:,}")

    nb = {
        "cells": [cell(MD_INTRO, "markdown"), cell(SETUP), cell(config),
                  cell(LOAD), cell(SOLVE), cell(REPORT)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = HERE / f"{SLUG}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = {
        "id": f"artgor/{SLUG}",
        # Kaggle requires the title to slugify to `id` -- keep them in step.
        "title": "CayleyPy cube555 TPU beam q",
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": True,
        "enable_internet": False,
        "dataset_sources": ["artgor/cube555-tpu-artifacts"],
        "competition_sources": ["cayley-py-555-cube"],
        "kernel_sources": [],
    }
    (HERE / f"kernel-metadata-{SLUG}.json").write_text(json.dumps(meta, indent=2),
                                                       encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {HERE / f'kernel-metadata-{SLUG}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
