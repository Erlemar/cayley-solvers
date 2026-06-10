"""Build the NISS ablation kernel: AZ v4 V + sym4 on strat-51, with vs without
NISS, production multi-pass recipe. Answers whether --niss earns its 2x cost.

Two arms run by 03_solve.py (production solver) on the same 51 stratified pids:
  A) --sym-ensemble 4                (no NISS)
  B) --sym-ensemble 4 --niss

Source/model/data come from artgor/megaminx-bridge-gpu-artifacts (re-versioned
to include 03_solve.py + pp_bfs6_fallback.csv). Comparison restricts to the 51
strat pids (the full CSV is fallback-filled to 1001, so the whole-file total is
meaningless).

Run: .venv/Scripts/python.exe megaminx/kaggle_notebooks/niss_ablation_strat51/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELL_MD = """\
# NISS ablation - AZ v4 V + sym4, strat-51, production multi-pass

Does `--niss` earn its 2x wall in the production full-1001 recipe? Run the
production solver (`03_solve.py`) on the 51 stratified pids (incl. hard pids
901/907/911/970/992/1000) at `--sym-ensemble 4 --beams 16384,65536
--max-steps 60,150 --bf16`, once **without** NISS and once **with**. Compare
solve count + total moves (model-or-fallback) over the 51 pids.
"""

CELL_RUN = r'''
import os, sys, glob, subprocess, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

hits = glob.glob("/kaggle/input/**/megaminx/scripts/03_solve.py", recursive=True)
assert hits, f"03_solve.py not found; {sorted(glob.glob('/kaggle/input/**', recursive=True))[:30]}"
BUNDLE = hits[0].split("/megaminx/scripts/")[0]
print("mount:", BUNDLE, flush=True)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), flush=True)

script = f"{BUNDLE}/megaminx/scripts/03_solve.py"
ckpt = f"{BUNDLE}/m_az_v4_v_only.pt"
common = [sys.executable, script, "--checkpoint", ckpt,
          "--stratified", "5", "--strat-seed", "0",
          "--sym-ensemble", "4",
          "--beams", "16384,65536", "--max-steps", "60,150", "--bf16"]

def run(extra, out, tag):
    cmd = common + extra + ["--out", out]
    print(f"\n===== ARM {tag}: {' '.join(cmd)} =====", flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    p.wait()
    print(f"ARM {tag} exit {p.returncode}  wall {time.time() - t0:.0f}s", flush=True)

run([], "/kaggle/working/strat51_sym4_noniss.csv", "A_noniss")
run(["--niss"], "/kaggle/working/strat51_sym4_niss.csv", "B_niss")
'''

CELL_COMPARE = r'''
import random, csv
all_ids = list(range(1001)); rng = random.Random(0); buckets = {}
for pid in all_ids:
    buckets.setdefault(pid // 100, []).append(pid)
strat = []
for b in sorted(buckets):
    strat.extend(sorted(rng.sample(buckets[b], min(5, len(buckets[b])))))
strat = set(strat)
print(f"strat-51 pids: {len(strat)}")

def load(p):
    d = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            pid = int(row["initial_state_id"]); path = row["path"]
            d[pid] = 0 if not path else len(path.split("."))
    return d

A = load("/kaggle/working/strat51_sym4_noniss.csv")
B = load("/kaggle/working/strat51_sym4_niss.csv")
TH = 200  # model paths ~50-160; pp_bfs6 fallback ~400 -> clean split

def summ(d, name):
    solved = sorted(p for p in strat if d[p] < TH)
    total = sum(d[p] for p in strat)
    mtotal = sum(d[p] for p in solved)
    print(f"{name}: model-solved {len(solved)}/51 | strat-51 total(model-or-fb) {total} | model-only total {mtotal}")
    return set(solved), total

print()
sA, tA = summ(A, "A no-niss")
sB, tB = summ(B, "B niss   ")
print(f"\nNISS delta on strat-51 total: {tB - tA:+d}  (negative = niss better)")
print(f"solved only WITH niss:    {sorted(sB - sA)}")
print(f"solved only WITHOUT niss: {sorted(sA - sB)}")
common = sA & sB
shaved = sum(A[p] - B[p] for p in common)
print(f"on {len(common)} commonly-solved pids, niss shaved {shaved:+d} moves total "
      f"(positive = niss shorter)")
worse = [p for p in common if B[p] > A[p]]
print(f"  pids where niss was LONGER: {sorted(worse)}")
'''

cells = [("markdown", CELL_MD), ("code", CELL_RUN), ("code", CELL_COMPARE)]
nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
         if t == "markdown" else
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
          "source": s.splitlines(keepends=True)})
        for t, s in cells
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out_nb = HERE / "cayleypy-megaminx-niss-ablation-strat51.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb} ({out_nb.stat().st_size} bytes)")

meta = {
    "id": "artgor/cayleypy-megaminx-niss-ablation-strat51",
    "title": "cayleypy megaminx niss ablation strat51",
    "code_file": "cayleypy-megaminx-niss-ablation-strat51.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["artgor/megaminx-bridge-gpu-artifacts"],
    "competition_sources": [], "kernel_sources": [],
}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {HERE / 'kernel-metadata.json'}")
