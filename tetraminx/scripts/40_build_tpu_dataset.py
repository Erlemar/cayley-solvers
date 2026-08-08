"""Assemble the artgor/tetraminx-tpu-artifacts Kaggle dataset.

The TPU notebook puts this dataset on sys.path and imports the JAX kernel from
it, so what runs on TPU is byte-identical to what `cpu_smoke.py` validates
locally -- no notebook source-munging, and a kernel change is a dataset version
bump.

Contents:
    puzzle_info.json
    tetra_symmetries.npy  tetra_symmetries_inv.npy  tetra_move_relabel.npy
    jax_model.py  jax_beam_spmd_v_only.py
    <checkpoint copied to the name the notebook expects>

    .venv/Scripts/python.exe tetraminx/scripts/40_build_tpu_dataset.py \
        --checkpoint tetraminx/models/tv1_bellman/best.pt

Then (PowerShell -- the Kaggle token must be re-exported per session):
    $env:KAGGLE_API_TOKEN="..."; $env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"
    .venv/Scripts/kaggle.exe datasets create  -p tetraminx/kaggle_datasets/tetraminx-tpu-artifacts   # first time
    .venv/Scripts/kaggle.exe datasets version -p tetraminx/kaggle_datasets/tetraminx-tpu-artifacts -m "msg"
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DEST = PROJECT / "tetraminx" / "kaggle_datasets" / "tetraminx-tpu-artifacts"
KERNEL_DIR = PROJECT / "tetraminx" / "kaggle_notebooks" / "tpu_beam_tetraminx"
DATA = PROJECT / "tetraminx" / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--checkpoint-name", default="tv0_bellman.pt",
                    help="must match V_CHECKPOINT in the notebook config cell")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    files = [
        DATA / "puzzle_info.json",
        DATA / "tetra_symmetries.npy",
        DATA / "tetra_symmetries_inv.npy",
        DATA / "tetra_move_relabel.npy",
        KERNEL_DIR / "jax_model.py",
        KERNEL_DIR / "jax_beam_spmd_v_only.py",
    ]
    missing = [f for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")

    for f in files:
        shutil.copy2(f, DEST / f.name)
        print(f"  {f.name}  ({(DEST / f.name).stat().st_size / 1e6:.1f} MB)")
    shutil.copy2(args.checkpoint, DEST / args.checkpoint_name)
    print(f"  {args.checkpoint_name}  "
          f"({(DEST / args.checkpoint_name).stat().st_size / 1e6:.1f} MB)  "
          f"<- {args.checkpoint}")

    total = sum(f.stat().st_size for f in DEST.iterdir() if f.is_file())
    print(f"\n{DEST}  ({total / 1e6:.0f} MB total)")
    print("next: kaggle datasets create/version -p "
          "tetraminx/kaggle_datasets/tetraminx-tpu-artifacts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
