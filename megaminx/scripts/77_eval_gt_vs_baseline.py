"""Quick Phase 4 eval driver: GT-V vs AZ v4 baseline on a small pid set.

Runs `61_eval_v_at_solved.py`-style calibration + `58_corner_pdb_beam.py`-style
short bench for ONE pid set (default: 5 pids). Pure-Python: uses the
polymorphic load_model_checkpoint and runs solves head-to-head.

This is a quick "is GT in the ballpark?" check. Strat-5 (51 pid) is the real
gate but takes ~30-60 min per model.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/77_eval_gt_vs_baseline.py \\
        --models megaminx/models/m_gt_v0_bellman/epoch_0079.pt:gt_v0 \\
                  megaminx/models/m_az_v4_v_only_e99.pt:az_v4 \\
        --pids 0,100,200,500,800 \\
        --beam 65536 --max-steps 120
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


def parse_model_arg(s: str) -> tuple[Path, str]:
    if ":" in s:
        path, label = s.rsplit(":", 1)
    else:
        path, label = s, Path(s).stem
    return Path(path), label


def calibration(model, device: str, n_samples: int = 2000):
    """V(V0), mean V@d=1, V@d=6 on a sample of the BFS-d6 table."""
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    state_size = len(puzzle.solved_state)
    bfs = BfsBytesTable.load(PROJECT / "data" / "bfs_bytes_d6.pkl")
    items = list(bfs.table.items())[:n_samples]
    states = np.empty((len(items), state_size), dtype=np.int64)
    depths = np.empty(len(items), dtype=np.int32)
    for i, (k, p) in enumerate(items):
        states[i] = np.frombuffer(k, dtype=np.int8).astype(np.int64)
        depths[i] = len(p)
    V0 = torch.tensor([puzzle.solved_state], dtype=torch.long, device=device)
    states_t = torch.from_numpy(states).to(device)
    with torch.no_grad():
        v0 = float(model(V0).flatten().cpu().item())
        preds = model(states_t).flatten().float().cpu().numpy()
    out = {"V(V0)": v0}
    for d in (1, 4, 6):
        m = depths == d
        out[f"V@d={d}"] = float(preds[m].mean()) if m.any() else float("nan")
    out["n_undershoot"] = int(np.sum(preds < depths.astype(np.float32) - 0.001))
    out["n_total"] = len(items)
    return out


def solve_pids(model, pids, beam: int, max_steps: int, device: str, bf16: bool):
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    test_rows = list(csv.DictReader(open(PROJECT / "data" / "test.csv")))
    if bf16 and device == "cuda":
        model = model.to(torch.bfloat16)
    solver = KhoruzhiiSolver(
        puzzle=puzzle, model=model, device=device,
        internal_batch_size=2**14,
    )
    cfg = KhoruzhiiSearchConfig(beam_width=beam, num_steps=max_steps)
    out = []
    for pid in pids:
        s0 = tuple(int(x) for x in test_rows[pid]["initial_state"].split(","))
        t0 = time.time()
        try:
            found, path_len, path = solver.solve(s0, cfg)
        except torch.AcceleratorError:
            found, path_len, path = False, 0, []
            torch.cuda.empty_cache()
        wall = time.time() - t0
        verify_ok = False
        if found and path:
            cur = list(s0)
            for name in path:
                gen = puzzle.generators[name]
                cur = [cur[g] for g in gen]
            verify_ok = (tuple(cur) == puzzle.solved_state)
        out.append({
            "pid": pid,
            "found": bool(found),
            "verify": bool(verify_ok),
            "path_len": int(path_len) if found and verify_ok else None,
            "wall_s": wall,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="model_path[:label] pairs (e.g., model.pt:gt_v0)")
    ap.add_argument("--pids", default="0,100,200,500,800")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-calibration", action="store_true")
    args = ap.parse_args()
    pids = [int(x) for x in args.pids.split(",")]

    print(f"\n=== Phase 4 eval: GT vs baselines (pids={pids}) ===\n", flush=True)
    results = {}
    for spec in args.models:
        path, label = parse_model_arg(spec)
        if not path.exists():
            print(f"[SKIP] {label}: {path} not found", flush=True)
            continue
        print(f"--- {label} ({path.name}) ---", flush=True)
        model = load_model_checkpoint(path, device=args.device, dtype=torch.float32)
        if not args.skip_calibration:
            cal = calibration(model, args.device)
            print(f"  calibration: V(V0)={cal['V(V0)']:.3f}  "
                  f"V@d=1={cal['V@d=1']:.3f}  V@d=4={cal['V@d=4']:.3f}  V@d=6={cal['V@d=6']:.3f}  "
                  f"undershoot={cal['n_undershoot']}/{cal['n_total']}", flush=True)
            results.setdefault(label, {})["calibration"] = cal
        rows = solve_pids(model, pids, args.beam, args.max_steps, args.device, args.bf16)
        results.setdefault(label, {})["bench"] = rows
        n_solved = sum(1 for r in rows if r["verify"])
        total = sum((r["path_len"] or 0) for r in rows if r["verify"])
        avg = total / max(1, n_solved)
        print(f"  bench: {n_solved}/{len(rows)} solved | total {total} | avg {avg:.1f}", flush=True)
        for r in rows:
            mark = "OK" if r["verify"] else ("FAIL" if r["found"] else "N/F")
            print(f"    pid {r['pid']:4d}: {r['path_len'] if r['verify'] else '-':>5} moves  "
                  f"({mark}, {r['wall_s']:.1f}s)", flush=True)
        del model
        torch.cuda.empty_cache()
        print(flush=True)

    print("=== Summary ===")
    for label, r in results.items():
        if "bench" in r:
            n = sum(1 for x in r["bench"] if x["verify"])
            t = sum((x["path_len"] or 0) for x in r["bench"] if x["verify"])
            print(f"  {label}: {n}/{len(r['bench'])} solved, total={t}, avg={t/max(1,n):.1f}")


if __name__ == "__main__":
    sys.exit(main())
