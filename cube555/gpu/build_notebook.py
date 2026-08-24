"""Build the cube555 2xT4 GPU beam notebook.

    .venv/Scripts/python.exe cube555/gpu/build_notebook.py
    .venv/Scripts/python.exe cube555/gpu/build_notebook.py --pids 1034,1033

This is the answer to "a large beam on GPU for cube555" after the third-party
MultiGPUBeamSearch engine turned out to cap state_len at 120 (cube555 is 150) -- see
README.md. It runs OUR solver (`KhoruzhiiSolver`, the same one `30_solve.py` drives)
on both T4s at once.

DESIGN, and why it is not a sharded beam. Width on cube555 is measured non-monotonic
(2^20 2/6, 2^21 6/6, 2^22 3/6, 2^23 0/1, no OOM anywhere), so the optimum is 2^21 and a
single 16 GB T4 holds it. A second card is therefore worth more as a second WORKER than
as half of a bigger beam: it buys frames and pids, which is where the remaining score
is. Two processes, one per device, zero cross-device communication, and no failure mode
where one card stalls the other.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "cayleypy-cube555-2xt4-beam"

MD_INTRO = r"""# CayleyPy 5x5x5 -- 2xT4 beam search, 30-wide Q head

Solves [CayleyPy 555 cube](https://www.kaggle.com/competitions/cayley-py-555-cube) by
running **one beam per GPU**, both cards working at once, scored by the project's
`ResMLPQ` 30-wide Q head.

## Why two beams and not one big one

Every sibling kernel in this family shards a single beam across devices to buy width.
On cube555 that would buy nothing. Same pids, same frame, only width varying:

| width | solved |
|---|---|
| 2^20 | 2/6 |
| **2^21** | **6/6** |
| 2^22 | 3/6 |
| 2^23 | 0/1 |

No OOM anywhere -- 2^23 ran fine and simply found less. A wider beam admits candidates
the scorer cannot rank and they crowd out the good ones in a global top-B. The optimum
is 2^21, and one 2^21 beam fits a single 16 GB T4.

So the second card is spent on **throughput**, which is where the remaining score
actually is:

- **frames** -- each conjugation frame is close to an independent ~33-50% draw, so `k`
  frames solve `1-(1-q)^k`. This notebook keeps the per-pid **minimum** over every frame
  it runs. The PyTorch solver stops at the *first* frame that succeeds; taking the
  shortest is listed in the project's own results as untested and "strictly better".
- **pids** -- deeper coverage of the queue per session.

Two independent worker processes, one per device, no cross-device communication.

## fp16, not bf16

T4 is Turing (sm75) with no native bf16. Measured on 512 real test states against fp32:

| dtype | max abs dQ | argmin agreement |
|---|---|---|
| **fp16** | **0.119** | **501/512** |
| bf16 | 0.876 | 427/512 |

7x tighter. Anything in this family that defaults to bf16 is assuming an A100 or a TPU.

## Where the score is

Pids 35..1034 are random walks whose shipped baseline is length `pid-34`, so solving a
deep pid saves hundreds of moves while the 35 Santa pids (0..34) save **zero** -- ours
is ~177 against their 93.6 baseline. Run deepest-first; zero Santa pids belong in a
merge.

## Exact endgame

The goal test is "inside the exact BFS `d<=5` ball" -- 10,739,017 states rather than 1 --
with the table's optimal descent spliced on arrival. The ball is not shipped (1.9 GB); it
rebuilds in about 70 s in the cell below. Membership compares the **state**, not just a
64-bit hash: at 10.7M entries a hash-only test false-positives often enough that one
killed a 29-hour run, and here the splice additionally fails soft -- a bad frame is
dropped, not the session.

Every path is replayed against the ORIGINAL test state before it is recorded, and the
submission is a per-pid min against a baseline, so it can never score worse than its floor.
"""

SETUP = r'''
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

# ---------------- mounts ----------------
CODE = None
for c in [Path("/kaggle/input/cube555-gpu-code"),
          Path("/kaggle/input/datasets/artgor/cube555-gpu-code")]:
    if c.exists():
        CODE = c
        break
ASSETS = None
for c in [Path("/kaggle/input/cube555-tpu-artifacts"),
          Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts")]:
    if c.exists():
        ASSETS = c
        break
COMP = None
for c in [Path("/kaggle/input/cayley-py-555-cube"),
          Path("/kaggle/input/competitions/cayley-py-555-cube")]:
    if c.exists():
        COMP = c
        break
if CODE is None:
    raise SystemExit("attach artgor/cube555-gpu-code")
if ASSETS is None:
    raise SystemExit("attach artgor/cube555-tpu-artifacts (checkpoints + sym tables)")
if COMP is None:
    raise SystemExit("attach the cayley-py-555-cube competition data")
print("code  :", CODE, "\nassets:", ASSETS, "\ncomp  :", COMP)

gpus = [l.strip() for l in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).splitlines()
    if l.strip()]
print(f"{len(gpus)} GPU(s): {gpus}")
DEVICES = ",".join(f"cuda:{i}" for i in range(len(gpus)))

# PREFLIGHT. Check the architecture, not the name. A P100 (sm_60) against a torch
# built for sm_70+ raises cudaErrorNoKernelImageForDevice on the FIRST cuda call,
# which surfaced here as "anchor build failed (rc=1)" after 70 s of BFS -- a message
# that points nowhere near the cause. Two seconds of checking is worth it.
import torch
_arch = [int(a.split("_")[1]) for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
_min_sm = min(_arch) if _arch else 70
for _i in range(len(gpus)):
    _cap = torch.cuda.get_device_capability(_i)
    _sm = _cap[0] * 10 + _cap[1]
    print(f"  cuda:{_i} {gpus[_i]}  sm_{_sm}")
    if _sm < _min_sm:
        raise SystemExit(
            f"cuda:{_i} is {gpus[_i]} (sm_{_sm}) but this image's torch supports "
            f"{torch.cuda.get_arch_list()}. Set the accelerator to 'GPU T4 x2' "
            f"(kernel-metadata machine_shape=NvidiaTeslaT4); a bare enable_gpu can "
            f"be scheduled onto a P100.")
if len(gpus) < 2:
    print("  [warn] written for the 2xT4 accelerator; runs on one card at half throughput")

# ---------------- stage a working tree ----------------
# build_anchors.py and gpu_beam.py both resolve paths relative to a cube555/ project
# root (PROJECT = parents[1], then PROJECT/src and PROJECT.parent/src). /kaggle/input is
# read-only and flat, so mirror the expected layout into /kaggle/working once.
# /tmp, NOT /kaggle/working: the d<=5 anchors file is 1.9 GB and /kaggle/working is
# captured as kernel output, which made pulling results take minutes.
ROOT = Path("/tmp/cube555_tree")
PROJ = ROOT / "cube555"
if not (PROJ / "data" / "puzzle_info.json").exists():
    for d in [ROOT / "src", PROJ / "src", PROJ / "scripts", PROJ / "data"]:
        d.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CODE / "cayley", ROOT / "src" / "cayley", dirs_exist_ok=True)
    shutil.copytree(CODE / "cube555", PROJ / "src" / "cube555", dirs_exist_ok=True)
    shutil.copy2(CODE / "build_anchors.py", PROJ / "scripts" / "build_anchors.py")
    shutil.copy2(ASSETS / "puzzle_info.json", PROJ / "data" / "puzzle_info.json")
    shutil.copy2(COMP / "test.csv", PROJ / "data" / "test.csv")
print("staged", PROJ)
'''

CONFIG = r'''
# ---------------- configuration ----------------
# q555_2k_BEST.pt   2,000 Bellman steps -- 18/24 solved, mean 130.1   <-- default
# q555_6k.pt        6,000               -- 12/24, 140.9
# q555_20k_deployed 20,000              -- 10/24, 174.5 (shipped the 187,780 submission)
# q555_pretrained         0             -- the parent all three warm-started from
# The losers are worth running as SEPARATE arms and merging (a min over six arms was
# 24/24 vs 18/24 for the best single one), not blending.
CHECKPOINT = "q555_2k_BEST.pt"

BEAMS       = "2097152"     # 2^21, the measured optimum. Wider is WORSE here.
MAX_STEPS   = "300"         # a beam finds a length-L solution AT step L; this truncates
INTERNAL_BS = 65536         # model forward / expansion chunk. Lower it if a T4 OOMs.
ENDGAME_DEPTH = 5           # 10,739,017-state ball; 4 is 446,403 and much cheaper
DTYPE       = "fp16"        # T4 has no native bf16 -- see the intro table
HISTORY_DEPTH = 4           # -14% path length; saturates at 4 (all 30 generators odd)
NO_BACKTRACK  = False       # neutral: history_depth>=1 already drops these

# (frame, direction): 'f' forward, 'i' inverse. EVERY frame listed is run and the per-pid
# MINIMUM kept. 0,7,19,33,41,47 are the six the PyTorch campaign used; the inverse axis
# is legal here (a state is a group element) and doubles the pool to 96 trajectories.
FRAMES = "0f,7i"

# Deepest-first: saving per pid is base_p - L with base_p = pid-34, so ordering
# front-loads the gain and any stopping point is near-optimal.
# deepest:4 = 8 tasks over 2 workers. At ~30-60 min/task on a T4 that finishes
# well inside the session wall; deepest:8 risked truncation, and a truncated run
# loses the merge step even though the per-worker JSONs survive.
PIDS = "deepest:4"          # or an explicit list like "1034,1033,1032"

# Floor for submission.csv. The competition sample is very long; point this at a
# stronger CSV in the asset dataset to merge on top of real work instead.
# Merge against our BEST file, not the competition sample. With "" the emitted
# submission.csv is 503,776 + whatever this run solved, which is worse than what
# we already have and therefore not submittable as-is.
BASELINE_CSV = "baseline_187780.csv"   # "" falls back to the competition sample

OUT = "/kaggle/working/submission.csv"
WORK = "/kaggle/working/run"
print(f"checkpoint {CHECKPOINT}  beams {BEAMS}  frames {FRAMES}  pids {PIDS}")
print(f"devices {DEVICES}  dtype {DTYPE}  endgame d<={ENDGAME_DEPTH}  history {HISTORY_DEPTH}")
'''

ANCHORS = r'''
# ---------------- exact endgame ball ----------------
# Deliberately not shipped: the d<=5 table is 1.9 GB. It rebuilds from the generators on
# CPU in about 6 minutes (see below for why not on the GPU), and the builder cross-checks its own level counts against the BFS pass and
# prints the collision headroom -- a silent hash collision would write a WRONG exact
# label, which is the one thing the endgame splice trusts as truth.
_anch = PROJ / "data" / f"anchors_d{ENDGAME_DEPTH}.pt"
if _anch.exists():
    print(f"{_anch.name} already built ({_anch.stat().st_size/1e6:.0f} MB)")
else:
    t0 = time.time()
    # --device cpu, deliberately. The GPU build peaks just over a T4: `expand()`
    # allocates its whole (N*30, 150) output in one go and the run OOM'd asking for
    # 4.69 GiB with 3.87 GiB free on a 14.56 GiB card. It fits a 16 GB card, which is
    # why it passed locally and failed here. The CPU has ~30 GB and takes 347 s
    # against 72 s -- a fine trade once per 9-hour session, and it leaves the whole
    # of both GPUs to the beam.
    _p = subprocess.run([sys.executable, str(PROJ / "scripts" / "build_anchors.py"),
                         "--depth", str(ENDGAME_DEPTH), "--device", "cpu"], check=False)
    if _p.returncode != 0 or not _anch.exists():
        raise SystemExit(f"anchor build failed (rc={_p.returncode})")
    print(f"built {_anch.name} in {time.time()-t0:.0f}s "
          f"({_anch.stat().st_size/1e6:.0f} MB)")
'''

RUN = r'''
# ---------------- solve ----------------
# One worker process per device. run_multi_gpu.py round-robins (pid, frame) tasks over
# the devices, waits, then merges best-of-frames per pid and replay-verifies every row.
_cmd = [sys.executable, str(CODE / "run_multi_gpu.py"),
        "--handoff", str(PROJ), "--assets", str(ASSETS),
        "--checkpoint", str(ASSETS / CHECKPOINT),
        "--devices", DEVICES, "--pids", PIDS, "--frames", FRAMES,
        "--beams", BEAMS, "--max-steps", MAX_STEPS,
        "--endgame-depth", str(ENDGAME_DEPTH),
        "--internal-batch-size", str(INTERNAL_BS), "--dtype", DTYPE,
        "--history-depth", str(HISTORY_DEPTH),
        "--test-csv", str(PROJ / "data" / "test.csv"),
        "--work", WORK, "--out", OUT]
if NO_BACKTRACK:
    _cmd.append("--no-backtrack")
if BASELINE_CSV:
    # Search BOTH mounts. baseline_187780.csv lives in cube555-gpu-code while the
    # checkpoints live in cube555-tpu-artifacts, and resolving it against the wrong one
    # silently produced a 3-row submission.csv instead of 1035 -- the run looked fine
    # and its output was unusable.
    _bl = next((d / BASELINE_CSV for d in (CODE, ASSETS)
                if (d / BASELINE_CSV).exists()), None)
    if _bl is None:
        raise SystemExit(
            f"BASELINE_CSV {BASELINE_CSV!r} not found in {CODE} or {ASSETS}. "
            f"Without it the submission would contain only the pids this run solved.")
    print(f"baseline: {_bl}")
    _cmd += ["--baseline-csv", str(_bl)]
else:
    _cmd += ["--baseline-csv", str(COMP / "sample_submission.csv")]
print(" ".join(_cmd), "\n")
_rc = subprocess.run(_cmd, check=False).returncode
print({"return_code": _rc})

# Fail loudly. The first 2xT4 run reported COMPLETE while every worker had died and the
# submission was the untouched baseline -- "an empty CSV passes verification vacuously".
_res_all = Path(WORK) / "results_all.json"
_recs = json.loads(_res_all.read_text(encoding="utf-8")) if _res_all.exists() else []
_solved = sum(1 for r in _recs if r.get("path_len") is not None)
print(f"workers returned {_rc}; {_solved} verified path(s) across {len(_recs)} task(s)")
if _rc != 0:
    raise SystemExit(f"beam workers returned {_rc} -- see the traceback above")
if _recs and _solved == 0:
    raise SystemExit(f"{len(_recs)} tasks ran and NONE produced a verified path -- "
                     f"failing rather than shipping a baseline-only submission")
'''

REPORT = r'''
# ---------------- report ----------------
# Reads the merged worker output rather than recomputing anything, so it is re-runnable.
import csv
from collections import defaultdict

_res = Path(WORK) / "results_all.json"
recs = json.loads(_res.read_text(encoding="utf-8")) if _res.exists() else []
if not recs:
    print("no results")
else:
    by_pid = defaultdict(list)
    for r in recs:
        by_pid[r["pid"]].append(r)
    print(f"{'pid':>6} {'base':>6} {'best':>6} {'saved':>7}  {'wall_s':>7}  per-frame")
    print("-" * 76)
    tot = solved = 0
    for pid in sorted(by_pid, reverse=True):
        rs = by_pid[pid]
        ok = [r for r in rs if "path_len" in r]
        base = pid - 34 if pid >= 35 else None
        wall = sum(r.get("wall_s", 0) for r in rs)
        per = " ".join(f"k{r['sym']}{'i' if r['inverted'] else 'f'}:"
                       + str(r.get("path_len", r.get("note", "-")[:4])) for r in rs)
        if ok:
            b = min(r["path_len"] for r in ok)
            solved += 1
            sv = (base - b) if base else 0
            tot += max(0, sv)
            print(f"{pid:>6} {str(base or '-'):>6} {b:>6} {sv:>7}  {wall:>7.0f}  {per}")
        else:
            print(f"{pid:>6} {str(base or '-'):>6} {'-':>6} {0:>7}  {wall:>7.0f}  {per}")
    print("-" * 76)
    print(f"solved {solved}/{len(by_pid)} pids; moves saved vs baseline {tot:,}")

    lens = [min(r["path_len"] for r in v if "path_len" in r)
            for v in by_pid.values() if any("path_len" in r for r in v)]
    if lens:
        print(f"path length: mean {np.mean(lens):.1f} min {min(lens)} max {max(lens)}")
        print("  reference: q555_2k_BEST measured mean 130.1 at 2^21 on 24 disjoint "
              "pids; the shipped 187,780 submission averages 167.8 over its 599 solves")

    # Which frames earned their wall time. Frames are the main lever here and their
    # marginal value decays as the residue enriches in genuinely hard pids, so this is
    # the measurement that should drive the next FRAMES setting.
    won, ran = defaultdict(int), defaultdict(int)
    for pid, rs in by_pid.items():
        ok = [r for r in rs if "path_len" in r]
        for r in rs:
            ran[(r["sym"], r["inverted"])] += 1
        if ok:
            b = min(r["path_len"] for r in ok)
            for r in ok:
                if r["path_len"] == b:
                    won[(r["sym"], r["inverted"])] += 1
    print("\nframe contribution (produced a pid's best path / times run):")
    for key in sorted(ran, key=lambda k: -won[k]):
        k, inv = key
        print(f"  k{k}{'i' if inv else 'f'}: {won[key]}/{ran[key]}")

    _sub = Path(OUT)
    if _sub.is_file():
        rows = list(csv.DictReader(open(_sub, encoding="utf-8")))
        print(f"\n{OUT}: {len(rows)} rows, "
              f"{sum(len(r['path'].split('.')) for r in rows):,} moves")
'''


def cell(src: str, kind: str = "code") -> dict:
    return ({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
            if kind == "markdown" else
            {"cell_type": "code", "execution_count": None, "metadata": {},
             "outputs": [], "source": src.splitlines(True)})


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pids", default=None, help='override PIDS, e.g. "1034,1033"')
    ap.add_argument("--frames", default=None, help='override FRAMES, e.g. "0f,7i,19f"')
    args = ap.parse_args()

    cfg = CONFIG
    if args.pids:
        old = 'PIDS = "deepest:8"          # or an explicit list like "1034,1033,1032"'
        assert old in cfg, "PIDS line moved"
        cfg = cfg.replace(old, f'PIDS = "{args.pids}"   # OVERRIDE via build_notebook.py')
        print(f"PIDS overridden: {args.pids}")
    if args.frames:
        old = 'FRAMES = "0f,7i"'
        assert old in cfg, "FRAMES line moved"
        cfg = cfg.replace(old, f'FRAMES = "{args.frames}"   # OVERRIDE')
        print(f"FRAMES overridden: {args.frames}")

    nb = {
        "cells": [cell(MD_INTRO, "markdown"), cell(SETUP), cell(cfg),
                  cell(ANCHORS), cell(RUN), cell(REPORT)],
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
        "title": "CayleyPy cube555 2xT4 beam",
        "code_file": f"{SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # REQUIRED. Without machine_shape Kaggle handed this kernel a single
        # Tesla P100 (sm_60), and the image's torch ships sm_70..sm_120 only, so
        # every CUDA op died with cudaErrorNoKernelImageForDevice. The failure is
        # not a memory or code problem and reads nothing like one.
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": False,
        "dataset_sources": ["artgor/cube555-gpu-code", "artgor/cube555-tpu-artifacts"],
        "competition_sources": ["cayley-py-555-cube"],
        "kernel_sources": [],
    }
    push = HERE / "push"
    push.mkdir(exist_ok=True)
    (HERE / f"kernel-metadata-{SLUG}.json").write_text(json.dumps(meta, indent=2),
                                                       encoding="utf-8")
    (push / "kernel-metadata.json").write_text(json.dumps(meta, indent=2),
                                               encoding="utf-8")
    (push / f"{SLUG}.ipynb").write_text(out.read_text(encoding="utf-8"),
                                        encoding="utf-8")
    print(f"wrote {out}\nwrote {push}/ (ready for `kaggle kernels push -p push`)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
