"""Sanity-check that beam_lab is correctly installed.

Run from the beam_lab directory:
    python verify_install.py

Prints what's working and what isn't. Does not run a real benchmark — for that
use run_benchmark.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

print("== beam_lab install check ==")

# 1. Imports
try:
    import torch
    print(f"[ok] torch {torch.__version__}")
except ImportError as e:
    print(f"[FAIL] torch: {e}")
    sys.exit(1)
try:
    import numpy as np
    print(f"[ok] numpy {np.__version__}")
except ImportError as e:
    print(f"[FAIL] numpy: {e}")
    sys.exit(1)

# 2. CUDA?
if torch.cuda.is_available():
    print(f"[ok] cuda available: {torch.cuda.get_device_name(0)}")
else:
    print("[warn] cuda not available — beam search will run on CPU at ~50× slower")

# 3. Local module imports
try:
    import puzzle  # noqa: F401
    print("[ok] puzzle.py")
    import model  # noqa: F401
    print("[ok] model.py")
    import beam_search  # noqa: F401
    print("[ok] beam_search.py")
    import beam_search_batch  # noqa: F401
    print("[ok] beam_search_batch.py")
    import beam_search_mitm  # noqa: F401
    print("[ok] beam_search_mitm.py (MITM available; needs --bfs-table to actually run)")
except ImportError as e:
    print(f"[FAIL] local module: {e}")
    sys.exit(1)

# 4. Data
puzzle_info = HERE / "data" / "puzzle_info.json"
samples_csv = HERE / "data" / "sample_puzzles.csv"
if puzzle_info.exists():
    print(f"[ok] {puzzle_info.name} ({puzzle_info.stat().st_size / 1024:.1f} KB)")
else:
    print(f"[FAIL] missing {puzzle_info}")
    sys.exit(1)
if samples_csv.exists():
    n = sum(1 for _ in open(samples_csv)) - 1
    print(f"[ok] {samples_csv.name} ({n} sample puzzles)")
else:
    print(f"[FAIL] missing {samples_csv}")
    sys.exit(1)

# 5. Puzzle loads
from puzzle import Megaminx
p = Megaminx.load(puzzle_info)
print(f"[ok] puzzle loaded: {len(p.move_names)} generators, state_size {len(p.solved_state)}")

# 6. Checkpoint
ckpt = HERE / "models" / "m05_bellman_warm" / "epoch_0499.pt"
if not ckpt.exists():
    print(f"[FAIL] missing checkpoint: {ckpt}")
    print("       Either copy m05's epoch_0499.pt here, or pass a different --checkpoint to run_benchmark.py.")
    sys.exit(1)

from model import load_checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
m = load_checkpoint(str(ckpt), device=device)
n_params = sum(x.numel() for x in m.parameters())
print(f"[ok] checkpoint loaded: {ckpt.name}, {n_params:,} params on {device}")

# 7. Tiny end-to-end run on the easiest sample puzzle
import csv as _csv
with open(samples_csv) as f:
    rdr = _csv.DictReader(f)
    rows = list(rdr)
# Pid 0 is trivially solved (it's the identity); pid 50 is the first non-trivial.
pid_to_test = 50
state = None
for row in rows:
    if int(row["initial_state_id"]) == pid_to_test:
        state = tuple(int(x) for x in row["initial_state"].split(","))
        break
if state is None:
    print(f"[warn] pid {pid_to_test} not in samples; skipping end-to-end test")
    sys.exit(0)

from beam_search import KhoruzhiiSolver, KhoruzhiiSearchConfig, setup_model_for_inference
m_bf = m.to(torch.bfloat16) if device == "cuda" else m
setup_model_for_inference(m_bf)
solver = KhoruzhiiSolver(p, m_bf, device=device, internal_batch_size=8192,
                          random_seed=0, state_dtype=torch.int8)
cfg = KhoruzhiiSearchConfig(beam_width=4096, num_steps=60, num_attempts=1,
                             internal_batch_size=8192)
print(f"[..] solving sample pid {pid_to_test} at beam=4096 num_steps=60 (small budget)...")
ok, plen, names, prof = solver.solve(state, cfg)
if ok:
    print(f"[ok] solved in {plen} moves, {prof.total_s:.2f}s wall, {prof.n_steps} beam steps")
    # Verify
    cur = tuple(state)
    for mv in names:
        cur = p.apply_move(cur, mv)
    if p.is_solved(cur):
        print("[ok] path verified — applies to identity")
    else:
        print("[FAIL] path does NOT solve! Check beam_search.py recently edited?")
        sys.exit(1)
else:
    print(f"[warn] beam=4096 didn't crack pid {pid_to_test} in 60 steps — that's OK, this is a small budget. "
          "Try `python run_benchmark.py --beam 32768 --max-steps 80 --pid 50 --bf16` for a real test.")

print()
print("== install check complete ==")
print("Next: run a benchmark sweep, e.g.")
print("  python run_benchmark.py --checkpoint models/m05_bellman_warm/epoch_0499.pt \\")
print("      --beam 16384,65536,131072 --max-steps 120 --bf16 \\")
print("      --out results/baseline_sweep.csv")
