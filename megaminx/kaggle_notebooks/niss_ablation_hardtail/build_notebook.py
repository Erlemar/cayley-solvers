"""NISS ablation v2 - hard tail (10 pids), fp32, batch mode, run.log capture.

Retry after the strat-51 attempt timed out (bf16-on-T4 ~40x slowdown suspected;
the working 514 run used fp32). Tests NISS where it's actually justified: the
10 hardest pids (991-1000), via --pid-from/--pid-to so 03_solve.py runs in
batch mode (no 950-pid fallback fill, and unsolved pids emit no row -> clean
solved/unsolved signal). Two arms: sym4 no-niss vs sym4 niss.

Run: .venv/Scripts/python.exe megaminx/kaggle_notebooks/niss_ablation_hardtail/build_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELL_MD = """\
# NISS ablation v2 - hard tail (pids 991-1000), fp32

Does `--niss` rescue the hardest pids where forward beam stalls? Run the
production solver on pids 991-1000 (batch mode) at `--sym-ensemble 4
--beams 16384,65536 --max-steps 60,150`, once **without** and once **with**
`--niss`. fp32 (bf16 suspected as the strat-51 slowdown). Per-pid output
streamed to `/kaggle/working/run.log` so timing survives a timeout.
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
common = [sys.executable, "-u", script, "--checkpoint", ckpt,
          "--pid-from", "991", "--pid-to", "1001",
          "--sym-ensemble", "4",
          "--beams", "16384,65536", "--max-steps", "60,150"]

LOG = open("/kaggle/working/run.log", "w", buffering=1)

def run(extra, out, tag):
    cmd = common + extra + ["--out", out]
    hdr = f"\n===== ARM {tag}: {' '.join(cmd)} =====\n"
    print(hdr, flush=True); LOG.write(hdr)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True); LOG.write(line)
    p.wait()
    msg = f"ARM {tag} exit {p.returncode}  wall {time.time() - t0:.0f}s\n"
    print(msg, flush=True); LOG.write(msg)

run([], "/kaggle/working/hardtail_noniss.csv", "A_noniss")
run(["--niss"], "/kaggle/working/hardtail_niss.csv", "B_niss")
LOG.close()
'''

CELL_COMPARE = r'''
import csv, os
hard = list(range(991, 1001))

def load(p):
    d = {}
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                pid = int(row["initial_state_id"]); path = row["path"]
                if path:
                    d[pid] = len(path.split("."))
    return d

A = load("/kaggle/working/hardtail_noniss.csv")
B = load("/kaggle/working/hardtail_niss.csv")
sA = [p for p in hard if p in A]
sB = [p for p in hard if p in B]
print(f"no-niss solved {len(sA)}/10: {sA}  total {sum(A[p] for p in sA)}")
print(f"niss    solved {len(sB)}/10: {sB}  total {sum(B[p] for p in sB)}")
print(f"rescued by NISS (solved by B, not A): {sorted(set(sB) - set(sA))}")
print(f"lost (solved by A, not B):            {sorted(set(sA) - set(sB))}")
common = sorted(set(sA) & set(sB))
shaved = sum(A[p] - B[p] for p in common)
print(f"\non {len(common)} commonly-solved pids, NISS shaved {shaved:+d} moves total")
for p in common:
    print(f"  pid {p}: no-niss {A[p]} -> niss {B[p]}  ({A[p] - B[p]:+d})")
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
out_nb = HERE / "cayleypy-megaminx-niss-ablation-hardtail.ipynb"
out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=True), encoding="utf-8")
print(f"wrote {out_nb} ({out_nb.stat().st_size} bytes)")

meta = {
    "id": "artgor/cayleypy-megaminx-niss-ablation-hardtail",
    "title": "cayleypy megaminx niss ablation hardtail",
    "code_file": "cayleypy-megaminx-niss-ablation-hardtail.ipynb",
    "language": "python", "kernel_type": "notebook",
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "dataset_sources": ["artgor/megaminx-bridge-gpu-artifacts"],
    "competition_sources": [], "kernel_sources": [],
}
(HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"wrote {HERE / 'kernel-metadata.json'}")
