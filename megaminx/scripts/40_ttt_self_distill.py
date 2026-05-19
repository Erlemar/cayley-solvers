"""T1.3 v2 — TTT with self-distilled labels (DAGGER-style).

v1 (37_ttt_solve.py) used Bellman-self-consistency on m05 itself for local
targets. m05 is already approximately self-consistent → tiny gradient signal
→ no real shift in beam behavior. Net result: NULL within noise.

v2 fixes the label-source weakness:

  1. Snapshot V weights.
  2. Run a CHEAP first-pass beam (beam=8k, max_steps=60) on the test puzzle.
  3. If it solves: every state along the found path has a TIGHT remaining-
     path-length label. Use these (state, true_remaining_d) pairs as fine-tune
     labels. Targets are EXACT (path-derived), not bootstrap.
  4. Fine-tune V for ~50 SGD steps on these per-puzzle exact labels.
  5. Run FULL beam (beam=65k) with locally-tuned V → typically shorter path.
  6. Take min(first_pass_path, full_beam_path) and verify.
  7. Restore weights for next puzzle.

If first pass fails (rare): fall back to v1 (Bellman bootstrap on local walks).

Distinct from m37/m43 (solver-trace): per-puzzle labels from the CURRENT
puzzle's beam path, not aggregated across past submissions. No OOD risk
because the labels come from this puzzle's own neighborhood.

Usage:
  PYTHONUTF8=1 .venv/Scripts/python.exe -u megaminx/scripts/40_ttt_self_distill.py \\
      --checkpoint megaminx/models/m05_bellman_warm/epoch_0499.pt \\
      --out megaminx/submissions/m05_ttt_v2_strat5.csv \\
      --beams 65536 --max-steps 150 \\
      --stratified 5 --strat-seed 0 --strat-buckets 7,8,9,10
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from cayley.verify import load_test_states, verify_path
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


def ttt_self_distill(
    model: torch.nn.Module,
    puzzle: Megaminx,
    solver: KhoruzhiiSolver,
    test_state,
    cfg_quick: KhoruzhiiSearchConfig,
    cfg_full: KhoruzhiiSearchConfig,
    device: str,
    k_steps: int,
    lr: float,
    batch_size: int,
) -> tuple[list[str] | None, str]:
    """Run quick beam → use path as labels → fine-tune → full beam.
    Returns (best_path_or_None, mode_used)."""
    # Step 1: cheap first pass
    found, _, raw = solver.solve(test_state, cfg_quick)
    if not found:
        return None, "first_pass_failed"
    path_quick = full_post_process(raw)
    if not verify_path(puzzle, test_state, path_quick).ok:
        return None, "first_pass_invalid"
    L = len(path_quick)

    # Step 2: build (state, remaining_distance) labels from the path
    labels = []
    cur_state = list(test_state)
    labels.append((tuple(cur_state), L))
    for i, move_name in enumerate(path_quick):
        gen_idx = puzzle.move_names.index(move_name)
        gen_perm = puzzle.generators[move_name]
        cur_state = [cur_state[gen_perm[j]] for j in range(len(cur_state))]
        labels.append((tuple(cur_state), L - i - 1))
    # remove the solved state (label 0) — model is anchored there already
    labels = [(s, d) for s, d in labels if d > 0]
    if not labels:
        return path_quick, "first_pass_only_no_labels"

    states_t = torch.tensor([list(s) for s, _ in labels], dtype=torch.long, device=device)
    targets_t = torch.tensor([d for _, d in labels], dtype=torch.float32, device=device)

    # Step 3: fine-tune V on per-puzzle exact labels (MSE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = states_t.size(0)
    for step in range(k_steps):
        if n > batch_size:
            idx = torch.randperm(n, device=device)[:batch_size]
            bs = states_t[idx]
            bt = targets_t[idx]
        else:
            bs = states_t
            bt = targets_t
        pred = model(bs).flatten().float()
        loss = F.mse_loss(pred, bt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()

    # Step 4: full beam with locally-tuned V
    found, _, raw = solver.solve(test_state, cfg_full)
    path_full = None
    if found:
        cand = full_post_process(raw)
        if verify_path(puzzle, test_state, cand).ok:
            path_full = cand

    # Step 5: take min(quick, full)
    if path_full is not None and len(path_full) < len(path_quick):
        return path_full, "self_distilled"
    return path_quick, "first_pass_only"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beams", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--quick-beam", type=int, default=8192,
                    help="beam width for the cheap first pass")
    ap.add_argument("--quick-max-steps", type=int, default=60)
    ap.add_argument("--stratified", type=int, default=5)
    ap.add_argument("--strat-seed", type=int, default=0)
    ap.add_argument("--strat-buckets", type=str, default=None)
    ap.add_argument("--pids", type=str, default=None)
    ap.add_argument("--ttt-k-steps", type=int, default=50)
    ap.add_argument("--ttt-lr", type=float, default=1e-3)
    ap.add_argument("--ttt-batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--internal-batch-size", type=int, default=2**14)
    ap.add_argument("--state-dtype", default="int8",
                    choices=["int8", "int16", "int32", "int64"])
    args = ap.parse_args()

    state_dtype = {"int8": torch.int8, "int16": torch.int16,
                   "int32": torch.int32, "int64": torch.int64}[args.state_dtype]

    print(f"loading puzzle + states + model ...")
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states_dict = load_test_states(PROJECT / "data" / "test.csv")
    all_ids = sorted(states_dict)

    if args.pids:
        solve_ids = sorted(int(p) for p in args.pids.split(",") if p.strip())
    else:
        rng = random.Random(args.strat_seed)
        buckets: dict[int, list[int]] = {}
        for pid in all_ids:
            buckets.setdefault(pid // 100, []).append(pid)
        wanted = (set(int(b) for b in args.strat_buckets.split(",") if b.strip())
                  if args.strat_buckets else set(buckets))
        picked = []
        for b in sorted(buckets):
            if b not in wanted:
                continue
            picked.extend(sorted(rng.sample(buckets[b], min(args.stratified, len(buckets[b])))))
        solve_ids = picked

    print(f"solve_ids: {len(solve_ids)} pids")

    # fp32 for fine-tuning gradients
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=torch.float32)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} params")

    solver = KhoruzhiiSolver(
        puzzle, model, device=args.device,
        internal_batch_size=args.internal_batch_size,
        state_dtype=state_dtype,
    )

    cfg_quick = KhoruzhiiSearchConfig(beam_width=args.quick_beam,
                                       num_steps=args.quick_max_steps,
                                       num_attempts=1)
    cfg_full = KhoruzhiiSearchConfig(beam_width=args.beams,
                                      num_steps=args.max_steps,
                                      num_attempts=1)

    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}

    out_paths: dict[int, list[str]] = {}
    n_solved = 0
    n_self_distilled = 0
    t0 = time.time()
    for i, pid in enumerate(solve_ids):
        state = states_dict[pid]
        t_pid = time.time()
        path, mode = ttt_self_distill(
            model, puzzle, solver, state,
            cfg_quick, cfg_full, args.device,
            k_steps=args.ttt_k_steps, lr=args.ttt_lr,
            batch_size=args.ttt_batch_size,
        )
        model.load_state_dict(snapshot)
        bucket = pid // 100
        if path is not None:
            out_paths[pid] = path
            n_solved += 1
            if mode == "self_distilled":
                n_self_distilled += 1
            print(f"  pid={pid:4d} bucket={bucket} mlen={len(path):3d} mode={mode:20s} "
                  f"({time.time()-t_pid:.1f}s)", flush=True)
        else:
            print(f"  pid={pid:4d} bucket={bucket} UNSOLVED mode={mode} "
                  f"({time.time()-t_pid:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== TTT v2 summary ===")
    print(f"  solved: {n_solved}/{len(solve_ids)}")
    print(f"  self-distilled wins (full < first_pass): {n_self_distilled}")
    print(f"  total wall: {elapsed/60:.1f} min")
    if n_solved > 0:
        avg = sum(len(p) for p in out_paths.values()) / n_solved
        print(f"  mean path length (solved): {avg:.2f}")

    by_bucket: dict[int, list[int]] = {}
    for pid in solve_ids:
        by_bucket.setdefault(pid // 100, []).append(pid)
    print(f"\n  per-bucket:")
    for b in sorted(by_bucket):
        pids_b = by_bucket[b]
        solved_b = [out_paths[p] for p in pids_b if p in out_paths]
        avg = sum(len(p) for p in solved_b) / max(len(solved_b), 1) if solved_b else -1
        print(f"  {b*100:4d}-{(b+1)*100-1:<4d} {len(pids_b):3d} "
              f"{len(solved_b):>2d}/{len(pids_b):<2d} {avg:7.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths):
            w.writerow([pid, ".".join(out_paths[pid])])
    print(f"\nwrote {args.out} ({len(out_paths)} solved pids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
