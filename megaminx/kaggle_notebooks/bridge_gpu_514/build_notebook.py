"""Build the Kaggle GPU bridge-compression notebook for a single pid.

Runs 81_bridge_compression.py (iterative, sym-ensemble + NISS) on pid 514
against the 75,200 base, on a Kaggle GPU kernel instead of the local 4090
(which is contended by other work). Source + model + data come from the
`artgor/megaminx-bridge-gpu-artifacts` dataset (one bundle.tar.gz, extracted
at runtime to /kaggle/working/bundle).

Run: .venv/Scripts/python.exe megaminx/kaggle_notebooks/bridge_gpu_514/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELL_MD = """\
# Megaminx bridge compression on Kaggle GPU - pid 514

Runs the iterative neural bridge-compression driver (`81_bridge_compression.py`)
on **pid 514** of the 75,200 base submission, with the strong config:
`--beam 65536 --sym-ensemble 4 --niss --max-iterations 5`, windows 8..40.

The local probe (plain beam 65k, no sym/niss) shortened pid 514 from **80 -> 77**.
This kernel measures where the **sym4 + NISS + iterate-to-convergence** config lands.

Source/model/data are bundled in `artgor/megaminx-bridge-gpu-artifacts`
(`bundle.tar.gz`), extracted at runtime. fp32 (no bf16) for P100/T4 safety.
A wall guard (`--max-wall-seconds 37800` = 10.5h) makes the run exit cleanly
and write output before Kaggle's 12h kernel kill.
"""

CELL_RUN = r'''
import os, sys, glob, tarfile, subprocess, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Kaggle auto-extracts the uploaded tar.gz, so the source tree is normally
# present directly under the dataset mount. Fall back to extracting a
# bundle.tar.gz if a future upload isn't auto-extracted.
# Recursive glob: Kaggle may mount at /kaggle/input/<slug>/ OR the nested
# /kaggle/input/datasets/<owner>/<slug>/ path, so search all depths.
hits = glob.glob("/kaggle/input/**/megaminx/scripts/81_bridge_compression.py", recursive=True)
if hits:
    BUNDLE = hits[0].split("/megaminx/scripts/")[0]
    print("source tree mount:", BUNDLE, flush=True)
else:
    cands = glob.glob("/kaggle/input/**/bundle.tar.gz", recursive=True)
    assert cands, f"no source under /kaggle/input; sample: {sorted(glob.glob('/kaggle/input/**', recursive=True))[:30]}"
    BUNDLE = "/kaggle/working/bundle"
    os.makedirs(BUNDLE, exist_ok=True)
    with tarfile.open(cands[0]) as t:
        t.extractall(BUNDLE)
    print("extracted bundle to:", BUNDLE, flush=True)

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""), flush=True)

script = f"{BUNDLE}/megaminx/scripts/81_bridge_compression.py"
cmd = [sys.executable, script,
       "--checkpoint", f"{BUNDLE}/m_az_v4_v_only.pt",
       "--submission", f"{BUNDLE}/base_514.csv",
       "--pids", "514",
       "--beam", "65536",
       "--sym-ensemble", "4",
       "--niss",
       "--max-iterations", "5",
       "--window-sizes", "8,15,20,25,30,40",
       "--internal-batch-size", "8192",
       "--max-wall-seconds", "37800",
       "--out", "/kaggle/working/bridge_514_out.csv",
       "--log-json", "/kaggle/working/bridge_514_log.json"]
print("running:", " ".join(cmd), flush=True)
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in p.stdout:
    print(line, end="", flush=True)
p.wait()
print(f"\nexit code: {p.returncode}  wall: {time.time() - t0:.0f}s", flush=True)
'''

CELL_RESULT = r'''
import json, os
out = "/kaggle/working/bridge_514_out.csv"
log = "/kaggle/working/bridge_514_log.json"
if os.path.exists(out):
    print("=== bridge_514_out.csv ===")
    with open(out) as f:
        txt = f.read()
    print(txt[:500])
    for line in txt.splitlines():
        if line.startswith("514,"):
            path = line.split(",", 1)[1]
            n = 0 if not path else len(path.split("."))
            print(f"\npid 514 bridged length: {n}  (base 80, plain-65k probe 77)")
if os.path.exists(log):
    print("\n=== summary ===")
    with open(log) as f:
        print(json.dumps(json.load(f).get("summary", {}), indent=2))
'''

cells = [("markdown", CELL_MD), ("code", CELL_RUN), ("code", CELL_RESULT)]
nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
         if ctype == "markdown" else
         {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
          "source": src.splitlines(keepends=True)})
        for ctype, src in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out_nb = HERE / "cayleypy-megaminx-bridge-gpu-514.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb} ({out_nb.stat().st_size} bytes)")
