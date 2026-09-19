"""Build the cube555 notebook that runs the PATCHED MultiGPUBeamSearch engine on 2xT4.

    .venv/Scripts/python.exe cube555/gpu/build_engine_notebook.py

This is the engine path that `README.md` said was blocked. Two things unblocked it:

  1. cube555's ResMLPQ has a ONE-level input stack; the engine's resmlp-layernorm
     exporter reads `input_stack.3` unconditionally. `make_engine_model.py` rewrites
     the checkpoint into the two-level form EXACTLY (+/- duplication with W3=k[I,-I]),
     verified 512/512 argmin and 512/512 full ordering against the original.
  2. The runner capped `state_len` at 120 while its own CMake already sizes the payload
     per puzzle. Fixed in Erlemar/MultiGPUBeamSearch@widen-state-payload, which is what
     this notebook clones -- upstream PR TryDotAtwo/MultiGPUBeamSearch#3.

One patch is still applied in-notebook rather than in that PR: the exporter's
`if embed_dim != 16` guard. The fold it protects, `(embedding @ block.t()).t()`, is
embed_dim-agnostic and `state_len = in_dim // embed_dim` still derives 150 at 24, so
it is a one-line generalisation -- kept separate because it is a different concern
from the state payload.

Verified locally before this notebook existed: the engine's REAL
`tools.cayleypy_public.model.export_checkpoint()` accepts the rewritten checkpoint and
emits a complete manifest -- format=resmlp-layernorm, backend=mlp, state_len=150,
num_classes=150, hd1=2048, hd2=1024, nrd=10, output_dim=30, fp16. So the model and
export path are known-good; what this run tests is the CUDA build and solve at
state_len 150, which is exactly the evidence PR #3 is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "cayleypy-cube555-2xt4-engine"

MD_INTRO = r"""# CayleyPy 5x5x5 on the MultiGPUBeamSearch engine (2xT4)

Runs [TryDotAtwo/MultiGPUBeamSearch](https://github.com/TryDotAtwo/MultiGPUBeamSearch)'s
CUDA beam against the CayleyPy 5x5x5 cube, using this project's `ResMLPQ` 30-wide Q head.

Two things had to be fixed to get here, and both are worth stating because they are the
reason this notebook is not simply a copy of the tetraminx one.

## 1. The model shape

cube555's `ResMLPQ` has a **one-level** input stack (built from a single `d_model`),
while the engine's `resmlp-layernorm` exporter reads `input_stack.3` unconditionally --
a two-level stack, with the C++ noting *"hidden1 must be >= hidden2 because Stream1
reuses hidden1 scratch for residual output"*.

Training a new two-level front end onto the teacher's activation **failed**: MSE fell
12x while argmin agreement against the teacher stayed at chance (1/30). The within-parent
action spread is ~0.6 against a Q level of ~44, so the front end has to be right to well
under a percent before the ranking -- the only thing a beam consumes -- survives.

The rewrite that works is exact algebra, no training. With `v = W0 e + b0` and
`h = ReLU(LN(v))`, set `W0' = [W0; -W0]`, `b0' = [b0; -b0]`, `LN1 = (1, 0)`,
`W3 = k[I, -I]`, and `LN2` = the teacher's affine. Then `u = [v; -v]` has mean 0 exactly,
so `LN1(u) = [v; -v]/s`, `ReLU` splits it into `[v+; v-]/s`, `W3` recombines it to `v/s`,
and LayerNorm's scale-invariance makes `LN2(v/s) = LN2(v)`. The intermediate ReLU is
exactly what the +/- pair absorbs. Residual blocks and head are copied verbatim.

Measured against the original on 512 real test states: **512/512 argmin agreement and
512/512 identical full 30-action ordering**, max abs dQ 1.755e-4 (the fp32 round-off
floor of the extra matmul -- identical at `k=32` and `k=128`).

## 2. The state length

The runner rejected `state_len > 120` for "the State128 logical payload", but its own
CMake already infers `BEAM_STATE_LOGICAL_BYTES` from `puzzle_info.json` and rounds the
physical size to the alignment -- pointed at cube555 an unmodified checkout reports
`150 / 160 / 30`. Raising that bound alone would have been unsafe: Zobrist rows are
indexed by state **value** (`zobrist[p * STATE_VALUE_PAD + value]`) with the pad pinned
at 128, so a 150-class alphabet would read into the next position's row silently.

This notebook clones the branch that fixes both together
([PR #3](https://github.com/TryDotAtwo/MultiGPUBeamSearch/pull/3)) -- the pad is inferred
alongside the other sizes, with a `static_assert` keeping the two bounds in step.

**This run is the end-to-end evidence that PR is missing** -- it could not be produced on
the authoring machine because the CUDA target needs NCCL, which NVIDIA does not ship for
Windows.

## Where the score is

Pids 35..1034 are random walks whose shipped baseline is length `pid-34`, so a deep pid
saves hundreds of moves while the 35 Santa pids save zero. Run deepest-first. Every
returned path is replayed against the ORIGINAL test state before it is recorded.
"""

SETUP = r'''
import json, os, subprocess, sys, time, csv, shutil, random
from pathlib import Path
import numpy as np

# ---------------- USER CONFIG ----------------
# Fork + branch carrying the state-payload fix (upstream PR #3).
SOLVER_REPO   = "https://github.com/Erlemar/MultiGPUBeamSearch.git"
SOLVER_BRANCH = "cube555-engine"   # our branch: payload sizing + the two stream1 fixes

PUZZLE_IDS = [1034]          # deepest-first; the baseline for pid p is length p-34
SANITIZE = False             # set True to re-run under compute-sanitizer memcheck
BEAM_WIDTH_POWER = 14 if SANITIZE else 21   # 2^21 is cube555's measured optimum
MAX_DEPTH = 6 if SANITIZE else 250          # real solutions run 118-250
FRAMES = [(0, False)]        # (symmetry 0..47, use_inverse); each frame is a full run
TOUCH_BFS_RADIUS = 4
SOLUTION_MODE = "first"

BEAM_WIDTH = 2 ** BEAM_WIDTH_POWER
WORK = Path("/kaggle/working")
REPO = Path("/tmp/MultiGPUBeamSearch")
BUNDLE = Path("/tmp/bundle")
BUNDLE.mkdir(parents=True, exist_ok=True)

CODE = next((p for p in [Path("/kaggle/input/cube555-gpu-code"),
                         Path("/kaggle/input/datasets/artgor/cube555-gpu-code")]
             if p.exists()), None)
ASSETS = next((p for p in [Path("/kaggle/input/cube555-tpu-artifacts"),
                           Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts")]
               if p.exists()), None)
COMP = next((p for p in [Path("/kaggle/input/cayley-py-555-cube"),
                         Path("/kaggle/input/competitions/cayley-py-555-cube")]
             if p.exists()), None)
for name, p in [("cube555-gpu-code", CODE), ("cube555-tpu-artifacts", ASSETS),
                ("cayley-py-555-cube", COMP)]:
    if p is None:
        raise SystemExit(f"attach {name}")
print("code:", CODE, "\nassets:", ASSETS, "\ncomp:", COMP)

gpus = [l.strip() for l in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines() if l.strip()]
print(f"{len(gpus)} GPU(s): {gpus}")
# The engine's public runner requires exactly two T4s and fails closed otherwise, so
# check here rather than after a multi-minute CUDA build.
if len(gpus) != 2 or any(g not in {"Tesla T4", "NVIDIA T4"} for g in gpus):
    raise SystemExit(f"this engine requires exactly two T4 GPUs; observed {gpus}. "
                     f"Set the accelerator to 'GPU T4 x2' (machine_shape=NvidiaTeslaT4).")
'''

CLONE = r'''
# ---------------- clone the patched engine ----------------
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(["git", "clone", "--depth", "1", "--branch", SOLVER_BRANCH,
                SOLVER_REPO, str(REPO)], check=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
print(f"engine @ {SOLVER_BRANCH} = {head}")

# ---- the one patch NOT in PR #3: generalise the exporter's embed_dim guard --------
# `if embed_dim != 16: raise` guards a fold, `(embedding @ block.t()).t()`, that is
# embed_dim-agnostic; `state_len = input_stack.0.weight.shape[1] // embed_dim` still
# derives 150 at embed_dim 24, and embed_dim never reaches the CUDA because the runtime
# consumes the FOLDED (state_len x num_classes x hidden1) table. Kept out of the PR
# because it is a separate concern from the state payload.
_exp = REPO / "tools" / "export_stream1_mlp.py"
_t = _exp.read_text(encoding="utf-8")
_needle = "    if embed_dim != 16:"
assert _needle in _t, "export_stream1_mlp.py embed_dim guard moved -- re-read before patching"
_exp.write_text(_t.replace(_needle, "    if embed_dim not in (16, 24):", 1), encoding="utf-8")
print("patched export_stream1_mlp.py: embed_dim guard -> (16, 24)")

# ---- confirm the payload really does size itself to 150 --------------------------
_sz = subprocess.run(["cmake", "-P", "cmake/CheckBeamStateSizing.cmake"],
                     cwd=REPO, capture_output=True, text=True)
print(_sz.stdout.strip() or _sz.stderr.strip())
'''

BUNDLE_CELL = r'''
# ---------------- bundle: checkpoint + generators ----------------
CHECKPOINT = BUNDLE / "q555_engine.pt"
shutil.copy2(CODE / "q555_2k_BEST_engine.pt", CHECKPOINT)

# The C++ runner text-scans ONE file for both move names and the generator
# permutations, so puzzle_info.json is re-emitted with an explicit move_names list in
# the same order -- that order is what the Q head's 30 outputs are indexed by, and a
# mismatch would silently misalign every action.
_info = json.loads((ASSETS / "puzzle_info.json").read_text(encoding="utf-8"))
MOVE_NAMES = list(_info["generators"].keys())
GENERATOR_JSON = BUNDLE / "cube555_generators.json"
GENERATOR_JSON.write_text(json.dumps({**_info, "move_names": MOVE_NAMES}), encoding="utf-8")
PUZZLE_INFO_JSON = BUNDLE / "puzzle_info.json"
shutil.copy2(ASSETS / "puzzle_info.json", PUZZLE_INFO_JSON)

CENTRAL = np.array(_info["central_state"], dtype=np.int64)
ALL_MOVES = np.array([_info["generators"][m] for m in MOVE_NAMES], dtype=np.int64)
N_GEN, STATE_LEN = len(MOVE_NAMES), len(CENTRAL)
print(f"N_GEN={N_GEN} STATE_LEN={STATE_LEN} num_classes={int(CENTRAL.max())+1}")
assert (N_GEN, STATE_LEN) == (30, 150)

INV_IDX = np.array([MOVE_NAMES.index(m[1:] if m.startswith("-") else "-" + m)
                    for m in MOVE_NAMES])
SYM     = np.load(ASSETS / "cube555_sym.npy").astype(np.int64)
SYM_INV = np.load(ASSETS / "cube555_sym_inv.npy").astype(np.int64)
RELABEL = np.load(ASSETS / "cube555_move_relabel_inv.npy").astype(np.int64)

STATES = {}
with open(COMP / "test.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        STATES[int(r["initial_state_id"])] = np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
print(f"{len(STATES)} test states")

# ---- run the engine's OWN export against our checkpoint, before any CUDA ---------
# This is the cheap gate: it is pure Python, and it is what fails if the algebraic
# rewrite or the embed_dim patch is wrong.
sys.path.insert(0, str(REPO))
import importlib
_model_mod = importlib.import_module("tools.cayleypy_public.model")
importlib.reload(_model_mod)
_check = Path("/tmp/_export_selfcheck")
if _check.exists():
    shutil.rmtree(_check)
_res = _model_mod.export_checkpoint(
    CHECKPOINT, _check, num_classes=150, state_len=150, move_count=30,
    metadata_json=None, generator_json=GENERATOR_JSON, source_root=None)
print(f"export self-check: format={_res.format} backend={_res.backend} "
      f"state_len={_res.manifest['state_len']} hd1={_res.manifest['hd1']} "
      f"hd2={_res.manifest['hd2']} nrd={_res.manifest['nrd']} "
      f"output_dim={_res.manifest['output_dim']}")
assert _res.format == "resmlp-layernorm" and _res.manifest["output_dim"] == 30
assert _res.manifest["state_len"] == 150
shutil.rmtree(_check)
'''

SOLVE = r'''
# ---------------- frame transforms + solve loop ----------------
def to_frame(s0, k, inverted):
    t = np.argsort(s0) if inverted else s0
    return SYM_INV[k][t[SYM[k]]] if k else t

def from_frame(path_idx, k, inverted):
    q = [int(RELABEL[k][m]) for m in path_idx] if k else list(path_idx)
    return [int(INV_IDX[m]) for m in reversed(q)] if inverted else q

def replay(s0, path_idx):
    cur = s0
    for m in path_idx:
        cur = cur[ALL_MOVES[m]]
    return cur

OUT_JSON = WORK / "cube555_engine_results.json"
results, BEST = [], {}
for (k, inverted) in FRAMES:
    # the engine reads the puzzle instances from a CSV, so the frame transform is
    # applied by writing a framed test.csv rather than by any engine-side flag
    rows = []
    with open(COMP / "test.csv", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f); fields = rdr.fieldnames
        for r in rdr:
            pid = int(r["initial_state_id"])
            row = dict(r)
            if pid in STATES and pid in PUZZLE_IDS:
                row["initial_state"] = ",".join(
                    str(int(x)) for x in to_frame(STATES[pid], k, inverted))
            rows.append(row)
    framed = BUNDLE / f"test_k{k}_i{int(inverted)}.csv"
    with open(framed, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    for pid in PUZZLE_IDS:
        outdir = WORK / f"out_k{k}_i{int(inverted)}_p{pid}"
        cfg = {
            "author_name": "artgor-cube555-engine",
            "checkpoint_path": str(CHECKPOINT),
            "checkpoint_metadata_json": None,
            "checkpoint_generator_json": str(GENERATOR_JSON),
            "checkpoint_source_root": None,
            "puzzle_info_json": str(PUZZLE_INFO_JSON),
            "test_csv": str(framed),
            "sample_submission_csv": str(COMP / "sample_submission.csv"),
            "puzzle_id_start": pid, "puzzle_id_end": pid,
            "beam_width": BEAM_WIDTH, "max_depth": MAX_DEPTH,
            "reflect_mode": "off", "reflect_source_csv": None,
            "solution_mode": SOLUTION_MODE, "collect_until_depth": MAX_DEPTH,
            "max_collected_solutions": 100, "touch_bfs_radius": TOUCH_BFS_RADIUS,
            "enable_debug": False, "enable_depth_logs": True, "enable_debug_logs": False,
            "debug_stream_timing": False, "debug_inference_trace": False,
            "debug_path_trace": False, "debug_final_validate": False,
            "debug_final_exchange_trace": False, "debug_final_histogram_trace": False,
            "debug_stream4_histogram_trace": False, "debug_depth_flow_trace": False,
            "depth_log_every": 1, "puzzle_log_every": 1,
            "publish_results": False,          # nothing leaves this session
            "results_ingest_url": "", "competition": "cayley-py-555-cube",
            "kaggle_owner": "artgor", "kaggle_slug": "cayleypy-cube555-2xt4-engine",
            "kaggle_version": 1, "kaggle_username": "artgor",
            "solver_commit": head,
            # optional=True, but "" fails nonempty_string(); publish_results is off
            # so no provenance hash is required.
            "kaggle_notebook_sha256": None,
        }
        cfgp = BUNDLE / f"cfg_k{k}_i{int(inverted)}_p{pid}.json"
        cfgp.write_text(json.dumps(cfg, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n=== pid {pid} frame k={k} inv={int(inverted)} "
              f"beam 2^{BEAM_WIDTH_POWER} depth {MAX_DEPTH} ===", flush=True)
        t0 = time.time()
        rc = subprocess.run([sys.executable, "-m", "tools.run_cayleypy_public",
                             "--config-json", str(cfgp), "--output-dir", str(outdir)],
                            cwd=REPO, check=False).returncode
        wall = time.time() - t0
        rec = {"pid": pid, "sym": k, "inverted": inverted, "rc": rc,
               "wall_s": round(wall, 1)}
        sol = outdir / "solutions" / "solutions.csv"
        # solutions.csv exists with a header even when nothing solved, so a size check
        # is not enough -- pandas raises EmptyDataError on a header-only file.
        _rows = []
        if rc == 0 and sol.is_file() and sol.stat().st_size > 0:
            import pandas as pd
            try:
                _rows = list(pd.read_csv(sol).iterrows())
            except pd.errors.EmptyDataError:
                _rows = []
        if _rows:
            for _, row in _rows:
                if int(row["puzzle_id"]) != pid or not isinstance(row.get("path"), str):
                    continue
                fpath = [MOVE_NAMES.index(m) for m in row["path"].split(".")]
                orig = from_frame(fpath, k, inverted)
                ok = bool(np.array_equal(replay(STATES[pid], orig), CENTRAL))
                rec.update({"path_len": len(orig), "verify_ok": ok,
                            "path": ".".join(MOVE_NAMES[m] for m in orig)})
                print(f"  -> {len(orig)} moves, verify={ok}, {wall:.0f}s")
                if ok and (pid not in BEST or len(orig) < len(BEST[pid].split("."))):
                    BEST[pid] = rec["path"]
        else:
            print(f"  -> no solution (rc={rc}, {wall:.0f}s)")
        results.append(rec)
        OUT_JSON.write_text(json.dumps(results, indent=1), encoding="utf-8")

print(f"\nsolved {len(BEST)}/{len(PUZZLE_IDS)} pid(s)")
for pid, path in BEST.items():
    base = pid - 34 if pid >= 35 else None
    n = len(path.split("."))
    print(f"  pid {pid}: {n} moves" + (f"  (baseline {base}, saved {base-n})" if base else ""))
'''

REPORT = r'''
# ---------------- submission ----------------
# Prefill every row from the competition sample and overwrite only where this run is
# STRICTLY shorter, so the output can never score worse than its floor.
baseline = {}
with open(COMP / "sample_submission.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("path"):
            baseline[int(r["initial_state_id"])] = r["path"]
rows = dict(baseline)
n_better = 0
for pid, path in BEST.items():
    if pid not in rows or len(path.split(".")) < len(rows[pid].split(".")):
        rows[pid] = path; n_better += 1

bad = []
for pid, path in rows.items():
    if pid in STATES and not np.array_equal(
            replay(STATES[pid], [MOVE_NAMES.index(m) for m in path.split(".")]), CENTRAL):
        bad.append(pid)

SUB = WORK / "submission.csv"
with open(SUB, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["initial_state_id", "path"])
    for pid in sorted(rows):
        w.writerow([pid, rows[pid]])
total = sum(len(p.split(".")) for p in rows.values())
print(f"wrote {SUB}: {len(rows)} rows, {total:,} moves "
      f"({n_better} improved, {len(bad)} invalid)")

# ---- what PR #3 needs to hear back -----------------------------------------------
print("\n=== engine build/solve summary (for TryDotAtwo/MultiGPUBeamSearch#3) ===")
for r in results:
    print(f"  pid {r['pid']} k{r['sym']}{'i' if r['inverted'] else 'f'}: "
          f"rc={r['rc']} len={r.get('path_len','-')} verify={r.get('verify_ok','-')} "
          f"wall={r['wall_s']}s")
_lg = sorted((WORK).glob("out_*/logs/cmake-configure.log"))
if _lg:
    for line in _lg[0].read_text(encoding="utf-8", errors="replace").splitlines():
        if "Beam state" in line or "Beam move" in line:
            print("  " + line.strip())
'''

SANITIZE_CELL = r'''
# ---------------- compute-sanitizer memcheck ----------------
# The runner builds `production_runner` itself and launches it through torchrun, so the
# way to get a sanitizer stack without touching engine code is to swap the built binary
# for a wrapper that execs compute-sanitizer on the real one. The first run above has
# already produced the binary (it builds before it crashes).
if SANITIZE:
    import glob
    cands = sorted(glob.glob("/tmp/cayleypy_public_build_*/production_runner"))
    print("built runners:", cands)
    if not cands:
        print("no production_runner found -- the build did not get that far")
    else:
        real = Path(cands[-1])
        shadow = real.with_suffix(".real")
        if not shadow.exists():
            real.rename(shadow)
            wrapper = ("#!/bin/sh\n"
                       "exec compute-sanitizer --tool memcheck --launch-timeout 120 "
                       "--print-limit 20 --show-backtrace yes " + str(shadow) + " \"$@\"\n")
            real.write_text(wrapper, encoding="utf-8")
            real.chmod(0o755)
            print(f"wrapped {real.name} -> compute-sanitizer memcheck {shadow.name}")
        print(subprocess.run(["compute-sanitizer", "--version"], capture_output=True,
                             text=True).stdout.strip()[:200])

        env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
        outdir = WORK / "sanitize_out"
        cfgp = sorted(BUNDLE.glob("cfg_*.json"))[-1]
        print(f"\nre-running under memcheck: beam 2^{BEAM_WIDTH_POWER} depth {MAX_DEPTH}\n")
        pr = subprocess.run([sys.executable, "-m", "tools.run_cayleypy_public",
                             "--config-json", str(cfgp), "--output-dir", str(outdir)],
                            cwd=REPO, env=env, capture_output=True, text=True)
        blob = (pr.stdout or "") + "\n" + (pr.stderr or "")
        (WORK / "sanitizer_raw.txt").write_text(blob, encoding="utf-8")
        # surface the FIRST violation with its stack -- that is what pins the kernel
        lines = blob.splitlines()
        first = next((i for i, l in enumerate(lines)
                      if "Invalid" in l or "Misaligned" in l or "========= ERROR" in l), None)
        print("=== first sanitizer violation ===")
        if first is None:
            tail = [l for l in lines if l.startswith("=========")][:40]
            print("\n".join(tail) if tail else blob[-3000:])
        else:
            print("\n".join(lines[max(0, first - 3): first + 45]))
'''


def cell(src, kind="code"):
    return ({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
            if kind == "markdown" else
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": src.splitlines(True)})


def main() -> int:
    nb = {"cells": [cell(MD_INTRO, "markdown"), cell(SETUP), cell(CLONE),
                    cell(BUNDLE_CELL), cell(SOLVE), cell(SANITIZE_CELL), cell(REPORT)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"{SLUG}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"artgor/{SLUG}",
        "title": "CayleyPy cube555 2xT4 engine",
        "code_file": f"{SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,          # clones the engine fork + CUTLASS
        "dataset_sources": ["artgor/cube555-gpu-code", "artgor/cube555-tpu-artifacts"],
        "competition_sources": ["cayley-py-555-cube"],
        "kernel_sources": [],
    }
    push = HERE / "push_engine"
    push.mkdir(exist_ok=True)
    (push / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (push / f"{SLUG}.ipynb").write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {out}\nwrote {push}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
