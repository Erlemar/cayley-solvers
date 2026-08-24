"""Local smoke test: exec every code cell (except the pip-install cell) in one
namespace with quick_test settings, against local stand-ins for the Kaggle mounts.

Usage:
    .venv/Scripts/python.exe megaminx/kaggle_notebooks/az4_train_shareable/smoke_run.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

os.environ.setdefault('AZ4_QUICK_TEST', '1')
os.environ.setdefault('AZ4_STAGES', 'pretrain,curriculum,bellman,bellman_dd,az')
os.environ.setdefault('AZ4_SOLVE', '1')
os.environ.setdefault('AZ4_COMP_DIR', 'C:/Users/and-l/cayley/megaminx/data')
os.environ.setdefault(
    'AZ4_ASSETS_DIR',
    'C:/Users/and-l/cayley/megaminx/kaggle_datasets/az4_training_assets')
os.environ.setdefault(
    'AZ4_WORK_DIR',
    'C:/Users/and-l/AppData/Local/Temp/claude/C--Users-and-l-cayley/'
    '50eb8ed6-008e-4af1-9cdb-444ced88a947/scratchpad/az4_work')

SKIP = {'c04_install.py'}


def main() -> int:
    ns: dict = {}
    t_all = time.time()
    for path in sorted((HERE / 'cells').iterdir()):
        if path.suffix != '.py' or path.name in SKIP:
            continue
        print(f'\n######## exec {path.name} ########', flush=True)
        t0 = time.time()
        code = compile(path.read_text(encoding='utf-8'), str(path), 'exec')
        exec(code, ns)  # shared namespace = notebook semantics
        print(f'######## {path.name} ok ({time.time() - t0:.1f}s) ########', flush=True)
    print(f'\nSMOKE OK in {(time.time() - t_all) / 60:.1f} min')
    return 0


if __name__ == '__main__':
    sys.exit(main())
