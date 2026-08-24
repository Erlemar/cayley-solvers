"""Harvest certified near-solved calibration anchors (direction 1B,
own_research_directions_2026-06-12.md).

Failure mode being attacked (quantified by Fedor in the CayleyPy chat, and the
documented cause of beam stalls): near the end of stalled solves the V model
predicts ~2 for states whose true distance is 6-8. Those errors are CHEAPLY
CERTIFIABLE with our exact BFS-d6 table (`data/bfs_bytes_d6.pkl`):

    lookup(s) = word  -> exact distance d(s) = len(word)   (d <= 6)
    lookup(s) = None  -> certified lower bound d(s) >= 7

This script runs plain beam solves on a pid list, harvests every frontier
state the model claims is near solved (V < threshold), certifies each against
the d6 table, prints the calibration table (our version of Fedor's plot), and
saves an anchor dataset for a follow-up Bellman refine:

    exact anchors: target d, MSE term
    lb anchors:    hinge max(0, 7 - V)^2 (only pushes V UP past the bound)

Usage (local 4090):
  .venv/Scripts/python.exe megaminx/scripts/89_harvest_stagnation_anchors.py \
      --checkpoint megaminx/models/m_az_v4_v_only.pt \
      --pids 905,920,935,950,965,980,991,995 \
      --beam 8192 --out megaminx/data/stagnation_anchors_v0.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "beam_lab"))

from beam_search import (
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    Profile,
    _state_hash,
    setup_model_for_inference,
)
from cayley.search import load_model_checkpoint
from cayley.verify import load_test_states
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.puzzle import Megaminx


def harvest_pid(solver: KhoruzhiiSolver, state, beam: int, max_steps: int,
                v_thr: float, per_step_cap: int, stall_minv: float, stall_len: int):
    """Run one plain beam solve, harvesting low-V frontier states each step.

    Returns (harvested, solved, n_steps) where harvested is a list of
    (state_tensor_cpu, v_pred, step, in_stall).
    """
    from collections import deque

    cfg = KhoruzhiiSearchConfig(beam_width=beam, num_steps=max_steps)
    prof = Profile()
    device = solver.device
    st = torch.tensor(list(state), dtype=solver.state_dtype, device=device)
    states = st.unsqueeze(0)
    states_hashed = _state_hash(states, solver.hash_vec, solver.internal_batch_size)
    states_bad_hashed = torch.empty(0, dtype=torch.int64, device=device)
    states_hash_log: deque[torch.Tensor] = deque(maxlen=4)

    harvested: list[tuple[torch.Tensor, float, int, bool]] = []
    minv_hist: list[float] = []
    solved = False

    for j in range(max_steps):
        states, values, _moves, _parents, chosen_hashes = solver._do_greedy_step(
            states, states_hashed, states_bad_hashed, cfg.beam_width, prof,
            cfg.use_incremental_hash, cfg.reuse_full_neighbors,
        )
        if states.numel() == 0:
            break
        states_hashed = chosen_hashes
        states_hash_log.append(chosen_hashes)

        vmin = float(values.min().item())
        minv_hist.append(vmin)
        # Stall = the model has been claiming "almost solved" for stall_len
        # consecutive steps without the beam actually solving.
        in_stall = (
            len(minv_hist) >= stall_len
            and all(v < stall_minv for v in minv_hist[-stall_len:])
        )

        mask = values < v_thr
        n_low = int(mask.sum().item())
        if n_low > 0:
            low_idx = torch.nonzero(mask, as_tuple=True)[0]
            if n_low > per_step_cap:
                _, order = torch.topk(values[low_idx].to(torch.float32),
                                      per_step_cap, largest=False, sorted=False)
                low_idx = low_idx[order]
            low_states = states[low_idx].cpu()
            low_vals = values[low_idx].float().cpu()
            for i in range(low_states.size(0)):
                harvested.append((low_states[i], float(low_vals[i]), j, in_stall))

        if solver._find_solved_pos(states, chosen_hashes) is not None:
            solved = True
            break
        if j > 3 and solver._check_stagnation(states_hash_log):
            break

    return harvested, solved, len(minv_hist)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT / "models" / "m_az_v4_v_only.pt")
    ap.add_argument("--pids", type=str, required=True)
    ap.add_argument("--beam", type=int, default=8192)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--v-threshold", type=float, default=6.5,
                    help="harvest frontier states with predicted V below this")
    ap.add_argument("--per-step-cap", type=int, default=512)
    ap.add_argument("--stall-minv", type=float, default=3.5,
                    help="frontier-min V below this counts toward a stall window")
    ap.add_argument("--stall-len", type=int, default=8,
                    help="consecutive steps below stall-minv to call it a stall")
    ap.add_argument("--bfs-table", type=Path, default=PROJECT / "data" / "bfs_bytes_d6.pkl")
    ap.add_argument("--out", type=Path, default=PROJECT / "data" / "stagnation_anchors_v0.pt")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pids = [int(x) for x in args.pids.split(",") if x.strip()]
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    setup_model_for_inference(model)
    solver = KhoruzhiiSolver(puzzle, model, device=args.device,
                             random_seed=args.seed, profile=False)

    all_rows: list[tuple[torch.Tensor, float, int, bool, int]] = []
    for pid in pids:
        t0 = time.time()
        rows, solved, n_steps = harvest_pid(
            solver, states[pid], args.beam, args.max_steps,
            args.v_threshold, args.per_step_cap, args.stall_minv, args.stall_len,
        )
        all_rows.extend((s, v, st, stall, pid) for (s, v, st, stall) in rows)
        print(f"pid={pid:4d} steps={n_steps:3d} solved={solved} "
              f"harvested={len(rows):5d} ({time.time() - t0:.0f}s)", flush=True)

    if not all_rows:
        print("nothing harvested - thresholds too tight or pids too easy")
        return 1

    # Dedup by state bytes (CPU; harvest sizes are modest).
    uniq: dict[bytes, tuple[torch.Tensor, float, int, bool, int]] = {}
    for row in all_rows:
        key = row[0].numpy().astype(np.uint8).tobytes()
        # Keep the LOWEST predicted V per state (the most-wrong claim).
        if key not in uniq or row[1] < uniq[key][1]:
            uniq[key] = row
    rows = list(uniq.values())
    print(f"harvested {len(all_rows)} rows -> {len(rows)} unique states; "
          f"loading d6 table (slow, ~2 GB) ...", flush=True)

    table = BfsBytesTable.load(args.bfs_table)
    print(f"d6 table loaded: {len(table.table):,} states, max_depth={table.max_depth}")

    n = len(rows)
    states_t = torch.stack([r[0] for r in rows])
    v_pred = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    step_t = torch.tensor([r[2] for r in rows], dtype=torch.int16)
    stall_t = torch.tensor([r[3] for r in rows], dtype=torch.bool)
    pid_t = torch.tensor([r[4] for r in rows], dtype=torch.int32)
    label_type = torch.empty(n, dtype=torch.uint8)   # 0 = exact, 1 = lower bound
    label_value = torch.empty(n, dtype=torch.uint8)  # exact d, or the LB (max_depth+1)

    lb = table.max_depth + 1
    for i, r in enumerate(rows):
        word = table.lookup(tuple(int(x) for x in r[0]))
        if word is None:
            label_type[i] = 1
            label_value[i] = lb
        else:
            label_type[i] = 0
            label_value[i] = len(word)

    # Calibration report — our version of Fedor's plot, with PROOFS.
    print(f"\ncalibration of '{args.checkpoint.name}' on harvested states "
          f"(V < {args.v_threshold}):")
    print(f"  {'V bucket':>9s} {'n':>7s} {'cert d>=7':>10s} {'pct':>5s} "
          f"{'mean true d (exact)':>20s} {'in-stall pct':>13s}")
    for lo in range(0, int(args.v_threshold) + 1):
        m = (v_pred >= lo) & (v_pred < lo + 1)
        nb = int(m.sum())
        if nb == 0:
            continue
        n_lb = int((label_type[m] == 1).sum())
        exact_mask = m & (label_type == 0)
        mean_d = float(label_value[exact_mask].float().mean()) if int(exact_mask.sum()) else float("nan")
        stall_pct = 100.0 * float(stall_t[m].float().mean())
        print(f"  [{lo},{lo + 1}) {nb:>7d} {n_lb:>10d} {100.0 * n_lb / nb:>4.0f} "
              f"{mean_d:>20.2f} {stall_pct:>12.0f}")

    n_wrong = int(((label_type == 1) & (v_pred < lb - 0.5)).sum())
    gap = (lb - v_pred[label_type == 1])
    print(f"\n{n_wrong}/{n} states are PROVABLY under-predicted "
          f"(V < {lb - 0.5} but certified d >= {lb}); "
          f"mean certified gap {float(gap.mean()) if gap.numel() else 0:.2f} moves")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "states": states_t,            # int8 (N, 120)
        "v_pred": v_pred,              # float32 - model's claim at harvest time
        "label_type": label_type,      # 0 exact (MSE target), 1 lower-bound (hinge)
        "label_value": label_value,    # exact d, or LB = max_depth + 1
        "step": step_t,
        "in_stall": stall_t,
        "pid": pid_t,
        "meta": {
            "checkpoint": str(args.checkpoint),
            "beam": args.beam,
            "v_threshold": args.v_threshold,
            "bfs_max_depth": table.max_depth,
            "pids": pids,
        },
    }, args.out)
    print(f"wrote {args.out} ({n} anchors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
