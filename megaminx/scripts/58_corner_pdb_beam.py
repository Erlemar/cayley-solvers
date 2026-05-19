"""Beam search with K=5 corner-PDB max-combine heuristic.

For each candidate state, value = max(V_neural, max_over_4_PDBs(pdb_dist)).
Admissible since each PDB is a lower bound and max preserves admissibility.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/58_corner_pdb_beam.py \\
        --v-checkpoint megaminx/models/m_curr_v3/epoch_0499.pt \\
        --pids 0,100,200,500,800 \\
        --beam 65536 --max-steps 120 \\
        --out megaminx/submissions/corner_pdb_smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from megaminx.pdb_heuristic import CornerPDBHeuristic
from megaminx.puzzle import Megaminx


def load_v_model(path, device, output_dim=1):
    """Polymorphic loader — handles ResMLP and GraphTransformer checkpoints."""
    return load_model_checkpoint(path, device=device, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-checkpoint", required=True, type=Path)
    ap.add_argument("--pdb-paths", type=str, default=None,
                    help="comma-separated PDB paths. Default: all 4 K5 PDBs.")
    ap.add_argument("--pids", type=str, default="0,100,200,500,800")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no-pdb", action="store_true",
                    help="Disable PDB (baseline run).")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))
    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    mode = "OFF (baseline)" if args.no_pdb else "ON (max-of-4 K=5 corner PDBs)"
    print(f"Corner-PDB beam: {len(pids)} pids, beam={args.beam:,}, PDB={mode}", flush=True)

    v_model = load_v_model(args.v_checkpoint, args.device)
    if args.bf16 and args.device == "cuda":
        v_model = v_model.to(torch.bfloat16)

    pdb_lookup = None
    if not args.no_pdb:
        if args.pdb_paths:
            pdb_paths = [Path(p) for p in args.pdb_paths.split(",")]
        else:
            pdb_paths = [
                PROJECT / "data" / f"pdb_corner_K5{suffix}.pkl"
                for suffix in ["", "_p1", "_p2", "_p3"]
            ]
        for p in pdb_paths:
            assert p.exists(), p
        h = CornerPDBHeuristic(pdb_paths, PROJECT / "data" / "corner_tables.pkl",
                                device=args.device)
        # Wrap as callable returning (B,) float16
        def pdb_lookup_fn(states):
            return h.lookup(states).to(torch.float16)
        pdb_lookup = pdb_lookup_fn

    solver = KhoruzhiiSolver(
        puzzle=puzzle, model=v_model, device=args.device,
        internal_batch_size=2**14,
        pdb_lookup=pdb_lookup,
        pdb_combine_mode="max",
    )
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    f_csv = open(args.out, "w", newline="")
    writer = csv.writer(f_csv)
    writer.writerow(["initial_state_id", "path"])

    print(f"\n{'pid':>4} | {'path_len':>8} | {'wall':>8} | verify", flush=True)
    print("-" * 60, flush=True)

    n_solved = 0
    total_moves = 0
    for pid in pids:
        s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
        t0 = time.time()
        found, path_len, path = solver.solve(s0, cfg)
        wall = time.time() - t0
        if not found:
            print(f"{pid:>4} | {'N/F':>8} | {wall:>8.1f}s | N/F", flush=True)
            writer.writerow([pid, ""])
            continue
        cur = list(s0)
        for name in path:
            gen = puzzle.generators[name]
            cur = [cur[g] for g in gen]
        verify_ok = (tuple(cur) == puzzle.solved_state)
        print(f"{pid:>4} | {path_len:>8} | {wall:>8.1f}s | {'OK' if verify_ok else 'FAIL'}",
              flush=True)
        if verify_ok:
            writer.writerow([pid, ".".join(path)])
            n_solved += 1
            total_moves += path_len
        else:
            writer.writerow([pid, ""])
        f_csv.flush()
    f_csv.close()

    print(f"\nsolved {n_solved}/{len(pids)} | total {total_moves} | "
          f"avg {total_moves/max(1,n_solved):.1f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
