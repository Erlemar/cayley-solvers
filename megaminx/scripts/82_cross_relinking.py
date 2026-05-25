"""Cross-solution neural relinking.

For each pid with multiple valid paths {A, B, C, ...}, attempt to recombine
prefix(A) with suffix(B) via a learned bridge:

    candidate = A[:i] + bridge(prefix_A[i] -> prefix_B[j]) + B[j:]

Generalizes whole-path min-merge by exploiting the fact that the same pid
often has multiple paths that diverge and reconverge in useful ways.

V-trajectory scoring (cheap): for each candidate (path_A, i, path_B, j) the
predicted save is

    save_pred = len(best_path) - i - V(residual(A_i, B_j)) - (len(B) - j)

We rank candidates by `save_pred` and solve only the top-K per pid. Solves
that find shorter bridges than the V prediction still count.

Acceptance: every accepted splice must verify from initial state to solved
AND strictly improve on the current best path for that pid.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/82_cross_relinking.py \\
        --checkpoint megaminx/models/m_az_v4_v_only.pt \\
        --base megaminx/submissions/merge_v12_az_v4_plus_our.csv \\
        --candidates megaminx/submissions/m_az_v4_prod_1001.csv,\\
                     megaminx/submissions/merge_v10_our_plus_m_dd_v0.csv,\\
                     megaminx/submissions/m_dd_v0_50ep_prod_1001.csv \\
        --out megaminx/submissions/relink_v0.csv \\
        --top-n-pids 50 --solves-per-pid 8 --beam 32768 --bf16
"""

from __future__ import annotations

import argparse
import csv
import json
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
from cayley.verify import load_submission, load_test_states, verify_path
from megaminx.bridge import compute_prefix_states, make_residual
from megaminx.puzzle import Megaminx


@torch.no_grad()
def eval_v(model, states_np, device, batch_size: int = 4096) -> np.ndarray:
    n = states_np.shape[0]
    x = torch.from_numpy(states_np.astype(np.int64)).to(device)
    out = torch.empty(n, dtype=torch.float32, device=device)
    for i in range(0, n, batch_size):
        chunk = x[i : i + batch_size]
        pred = model(chunk)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        out[i : i + batch_size] = pred.float()
    return out.cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path,
                    help="Baseline CSV. Output preserves any pid not improved by relinking.")
    ap.add_argument("--candidates", required=True, type=str,
                    help="Comma-separated candidate CSVs to draw paths from. May include "
                         "the base; duplicates dedupe automatically.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--test-csv", type=Path,
                    default=PROJECT / "data" / "test.csv")
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--top-n-pids", type=int, default=50,
                    help="Process the N pids with longest best path")
    ap.add_argument("--pids", type=str, default=None,
                    help="Explicit pid list, overrides --top-n-pids")
    ap.add_argument("--positions-frac", type=str, default="0.1,0.25,0.4,0.55,0.7,0.85",
                    help="Comma-separated path-fraction positions to try for both i and j")
    ap.add_argument("--solves-per-pid", type=int, default=8,
                    help="Max actual bridge solves per pid (top-K by V-score)")
    ap.add_argument("--beam", type=int, default=32768)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--log-json", type=Path, default=None)
    args = ap.parse_args()

    pos_fracs = [float(x) for x in args.positions_frac.split(",")]

    puzzle = Megaminx.load(args.puzzle_info)
    solved = tuple(puzzle.solved_state)
    states = load_test_states(args.test_csv)
    base_paths = load_submission(args.base)
    print(f"loaded base: {len(base_paths)} pids, total {sum(len(p) for p in base_paths.values())} moves",
          flush=True)

    # Pool candidate paths per pid from all CSVs (including base, plus extras).
    cand_csvs = [Path(p.strip()) for p in args.candidates.split(",") if p.strip()]
    # Always include the base so per-pid pool covers at least one valid path.
    if args.base not in cand_csvs:
        cand_csvs = [args.base] + cand_csvs
    print(f"candidate CSVs: {[c.name for c in cand_csvs]}", flush=True)

    paths_by_pid: dict[int, list[list[str]]] = {pid: [] for pid in base_paths}
    for c in cand_csvs:
        if not c.exists():
            print(f"  WARNING missing: {c}", flush=True)
            continue
        sub = load_submission(c)
        for pid, p in sub.items():
            paths_by_pid.setdefault(pid, []).append(p)
    # Dedupe per pid (same move sequence appears in multiple CSVs).
    dedupe_count = []
    for pid, plist in paths_by_pid.items():
        seen = set()
        uniq = []
        for p in plist:
            key = ".".join(p)
            if key not in seen:
                seen.add(key)
                uniq.append(p)
        paths_by_pid[pid] = uniq
        dedupe_count.append(len(uniq))
    print(f"per-pid path counts: min={min(dedupe_count)} max={max(dedupe_count)} "
          f"mean={np.mean(dedupe_count):.1f}", flush=True)

    if args.pids:
        selected_pids = [int(p) for p in args.pids.split(",")]
    else:
        # Top-N by current best path length, but only pids with >=2 paths.
        eligible = [pid for pid in base_paths if len(paths_by_pid[pid]) >= 2]
        eligible.sort(key=lambda p: len(base_paths[p]), reverse=True)
        selected_pids = eligible[: args.top_n_pids]
    print(f"selected {len(selected_pids)} pids; longest best = "
          f"{len(base_paths[selected_pids[0]])}", flush=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"loading V model from {args.checkpoint.name} (dtype={dtype})", flush=True)
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    solver = KhoruzhiiSolver(
        puzzle, model, device=args.device,
        internal_batch_size=args.internal_batch_size,
        random_seed=0,
        state_dtype=torch.int8,
    )

    out_paths: dict[int, list[str]] = dict(base_paths)
    attempt_log: list[dict] = []
    n_v_scored = 0
    n_solved_attempts = 0
    n_wins = 0
    total_saved = 0
    t_session = time.time()

    for pid_idx, pid in enumerate(selected_pids):
        if pid not in states:
            continue
        initial_state = states[pid]
        # Verify all candidate paths actually solve this pid (defensive).
        valid_paths = [p for p in paths_by_pid[pid]
                       if verify_path(puzzle, initial_state, p).ok]
        if len(valid_paths) < 2:
            print(f"[{pid_idx+1}/{len(selected_pids)}] pid {pid:4d} | only "
                  f"{len(valid_paths)} valid paths, skipping", flush=True)
            continue
        # Sort by length so shortest is best (greedy floor).
        valid_paths.sort(key=len)
        best_len_initial = len(valid_paths[0])
        if base_paths[pid] not in valid_paths:
            # Base already had something; treat base as the initial best.
            best_len_initial = min(best_len_initial, len(base_paths[pid]))
        best_path = list(base_paths[pid])
        if len(valid_paths[0]) < len(best_path):
            best_path = list(valid_paths[0])

        # Precompute prefixes for each valid path.
        prefix_by_pi = [compute_prefix_states(initial_state, p, puzzle) for p in valid_paths]
        pid_t0 = time.time()

        # Generate candidate (a, i, b, j) tuples.
        candidates = []
        for a_idx, A in enumerate(valid_paths):
            for b_idx, B in enumerate(valid_paths):
                if a_idx == b_idx:
                    continue
                len_A, len_B = len(A), len(B)
                # i positions in A, j positions in B
                i_set = sorted({max(1, min(len_A - 1, int(round(f * len_A)))) for f in pos_fracs})
                j_set = sorted({max(1, min(len_B - 1, int(round(f * len_B)))) for f in pos_fracs})
                for i in i_set:
                    for j in j_set:
                        # Allowed-bridge cap so the candidate has a chance of beating best.
                        suffix_b_len = len_B - j
                        # candidate len = i + bridge_len + suffix_b_len
                        max_bridge = len(best_path) - i - suffix_b_len - 1
                        if max_bridge < 1:
                            continue
                        candidates.append((a_idx, i, b_idx, j, max_bridge))

        if not candidates:
            print(f"[{pid_idx+1}/{len(selected_pids)}] pid {pid:4d} | best={len(best_path)} | "
                  f"no candidates (paths too long), skipping", flush=True)
            continue

        # V-score every candidate (single batched forward).
        residuals = np.empty((len(candidates), len(solved)), dtype=np.int64)
        for k, (ai, i, bi, j, _) in enumerate(candidates):
            s_a = prefix_by_pi[ai][i]
            s_b = prefix_by_pi[bi][j]
            residuals[k] = make_residual(s_a, s_b)
        v_pred = eval_v(model, residuals, args.device)
        n_v_scored += len(candidates)

        # Predicted save = len(best) - i - V(residual) - suffix_b_len
        scores = np.empty(len(candidates), dtype=np.float32)
        for k, (ai, i, bi, j, _) in enumerate(candidates):
            suffix_b_len = len(valid_paths[bi]) - j
            scores[k] = len(best_path) - i - v_pred[k] - suffix_b_len

        # Take top-K by predicted save.
        order = np.argsort(-scores)[: args.solves_per_pid]
        # Filter to scores > 0 (predicted improvement) and skip if V predicted bridge
        # already longer than max_bridge cap.
        wins_here = 0
        saved_here = 0
        attempts_here = 0
        for k in order:
            if scores[k] <= 0:
                break  # remaining candidates predicted to not improve
            ai, i, bi, j, max_bridge = candidates[k]
            cfg = KhoruzhiiSearchConfig(
                beam_width=args.beam,
                num_steps=int(max(1, max_bridge)),
                num_attempts=args.num_attempts,
                internal_batch_size=args.internal_batch_size,
            )
            res = residuals[k].tolist()
            t0 = time.time()
            found, bridge_len, bridge = solver.solve(res, cfg)
            t1 = time.time()
            n_solved_attempts += 1
            attempts_here += 1
            if not found:
                continue
            # Splice and verify.
            A = valid_paths[ai]
            B = valid_paths[bi]
            candidate_path = A[:i] + bridge + B[j:]
            vr = verify_path(puzzle, initial_state, candidate_path)
            if not vr.ok:
                print(f"[pid {pid}] RELINK FAILED VERIFY: {vr.reason}", flush=True)
                continue
            if len(candidate_path) >= len(best_path):
                continue
            saved = len(best_path) - len(candidate_path)
            best_path = candidate_path
            wins_here += 1
            saved_here += saved
            n_wins += 1
            total_saved += saved
            attempt_log.append({"pid": pid, "a_idx": ai, "i": i, "b_idx": bi, "j": j,
                                "len_A": len(A), "len_B": len(B),
                                "bridge_len": bridge_len, "saved": saved,
                                "v_pred_save": round(float(scores[k]), 2),
                                "solve_s": round(t1 - t0, 2)})

        if wins_here > 0:
            out_paths[pid] = best_path
        pid_wall = time.time() - pid_t0
        progress = f"[{pid_idx+1}/{len(selected_pids)}]"
        print(f"{progress} pid {pid:4d} | best {best_len_initial:4d} -> {len(best_path):4d} "
              f"(saved {best_len_initial - len(best_path):+4d}) | wins {wins_here} of {attempts_here} "
              f"attempts | {pid_wall:.1f}s", flush=True)

    t_total = time.time() - t_session
    print(f"\n=== summary ===\n"
          f"  pids processed: {len(selected_pids)}\n"
          f"  V-scored candidates: {n_v_scored}\n"
          f"  actual solves: {n_solved_attempts}\n"
          f"  verified wins: {n_wins}\n"
          f"  total moves saved: {total_saved}\n"
          f"  win rate (per-solve): {n_wins / max(n_solved_attempts, 1):.1%}\n"
          f"  total wall: {t_total:.1f}s", flush=True)

    # Write CSV.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths.keys()):
            w.writerow([pid, ".".join(out_paths[pid])])
    new_total = sum(len(p) for p in out_paths.values())
    base_total = sum(len(p) for p in base_paths.values())
    print(f"wrote merged CSV: {args.out}", flush=True)
    print(f"submission total: {base_total} -> {new_total} ({base_total - new_total:+d})", flush=True)

    if args.log_json is not None:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as f:
            json.dump({"attempts": attempt_log,
                       "summary": {"n_pids": len(selected_pids),
                                   "n_v_scored": n_v_scored,
                                   "n_solved_attempts": n_solved_attempts,
                                   "n_wins": n_wins,
                                   "total_saved": total_saved,
                                   "total_wall_s": round(t_total, 2)}},
                      f, indent=2)
        print(f"wrote attempt log: {args.log_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
