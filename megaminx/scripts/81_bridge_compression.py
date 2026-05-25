"""Neural bridge compression: replace arbitrary middle windows of a valid
path with shorter learned bridges.

For each pid in the selected set:
  1. Parse the current best path.
  2. Compute prefix states along the path.
  3. For each (window_size, position) pair:
     - Compute the residual state X = inv_S_j[S_i].
     - Try to solve X with the V model at a capped depth (window_size - 1).
     - If solved and the bridge is strictly shorter than the original window,
       splice it in. Verify the spliced path solves the puzzle from initial.
  4. Track all verified improvements; keep the shortest path found.

Outputs a merged CSV (original best ∪ bridge wins, per-pid min) plus a
per-pid log of attempts/wins.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/81_bridge_compression.py \\
        --checkpoint megaminx/models/m_az_v4_v_only.pt \\
        --submission megaminx/submissions/merge_v14_plus_min_count_v4.csv \\
        --out megaminx/submissions/bridge_v0.csv \\
        --top-n-pids 10 --window-sizes 20,30,40 \\
        --beam 16384 --bf16
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


def select_positions(path_len: int, window_len: int, n_positions: int) -> list[int]:
    """Stride-sample non-overlapping start positions for a fixed window size.

    Returns indices i in [0, path_len - window_len]. If the path is too short,
    returns empty list.
    """
    max_i = path_len - window_len
    if max_i <= 0:
        return []
    if n_positions >= max_i + 1:
        return list(range(max_i + 1))
    # Even stride across [0, max_i].
    stride = max_i / (n_positions - 1) if n_positions > 1 else 0
    return [int(round(k * stride)) for k in range(n_positions)]


@torch.no_grad()
def _eval_v_on_states(model, states: np.ndarray, device: str,
                     dtype: torch.dtype, batch_size: int = 4096) -> np.ndarray:
    """Run model on a (N, state_size) int numpy array. Returns (N,) float numpy."""
    n = states.shape[0]
    x = torch.from_numpy(states.astype(np.int64)).to(device)
    out = torch.empty(n, dtype=torch.float32, device=device)
    for i in range(0, n, batch_size):
        chunk = x[i : i + batch_size]
        pred = model(chunk)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        out[i : i + batch_size] = pred.float()
    return out.cpu().numpy()


def _hamming_pairwise(s_i: np.ndarray, s_j: np.ndarray) -> int:
    """Hamming distance between two state arrays. Fast numpy op."""
    return int((s_i != s_j).sum())


def select_positions_v_trajectory(
    prefix_states: list[tuple],
    window_len: int,
    n_positions: int,
    model,
    device: str,
    dtype: torch.dtype,
    use_hamming: bool = True,
) -> list[int]:
    """Select start positions where the residual's predicted V is well below
    the window length. Score = window_len - max(V(R), Hamming(S_i, S_j) // 12).

    The Hamming term protects against V saturation: V predicts ~30 for any
    state more than 40 moves from solved, so V-only ranking yields false
    positives on deep residuals. Each megaminx generator moves ~12 stickers,
    so Hamming/12 is a rough lower bound on bridge length.

    Returns up to n_positions starting indices, ranked by predicted savings
    (highest first).
    """
    path_len = len(prefix_states) - 1
    max_i = path_len - window_len
    if max_i <= 0:
        return []
    n_candidates = max_i + 1
    prefix_np = np.asarray(prefix_states, dtype=np.int64)
    residuals = np.empty((n_candidates, prefix_np.shape[1]), dtype=np.int64)
    hamming = np.empty(n_candidates, dtype=np.int32)
    for i in range(n_candidates):
        s_i = prefix_np[i]
        s_j = prefix_np[i + window_len]
        # Inline make_residual for speed in numpy
        inv_s_j = np.empty(prefix_np.shape[1], dtype=np.int64)
        inv_s_j[s_j] = np.arange(prefix_np.shape[1], dtype=np.int64)
        residuals[i] = inv_s_j[s_i]
        hamming[i] = int((s_i != s_j).sum())
    v_pred = _eval_v_on_states(model, residuals, device, dtype)
    if use_hamming:
        # Conservative lower bound on bridge length: max(V_pred, Hamming/12).
        # 12 ≈ stickers moved per single generator (range 10-25 depending on face).
        h_lower = np.maximum(1.0, hamming.astype(np.float32) / 12.0)
        depth_estimate = np.maximum(v_pred, h_lower)
    else:
        depth_estimate = v_pred
    scores = window_len - depth_estimate
    order = np.argsort(-scores)
    return [int(i) for i in order[:n_positions]]


def select_hamming_similar_windows(
    prefix_states: list[tuple],
    n_positions: int,
    min_window: int = 10,
    max_window: int = 80,
) -> list[tuple[int, int]]:
    """Find (i, j) pairs where Hamming(prefix[i], prefix[j]) is small relative
    to j-i — natural "the path effectively returned to a previously-visited
    region" detour candidates. No V eval needed; pure state-space signal.

    Returns up to n_positions (i, j) tuples ranked by (hamming - sqrt(j-i)),
    smallest first (i.e., most-anomalous low-Hamming pairs).
    """
    n = len(prefix_states)
    prefix_np = np.asarray(prefix_states, dtype=np.int64)
    candidates: list[tuple[float, int, int]] = []
    # All (i, j) with min_window <= j-i <= max_window, j < n.
    # n is typically <= 90 so this is O(n^2/2) which is ~4000 pairs — fine.
    for i in range(n - min_window):
        for j in range(i + min_window, min(i + max_window + 1, n)):
            h = int((prefix_np[i] != prefix_np[j]).sum())
            # "Anomaly": small h despite long j-i is the signal.
            # Score = h - sqrt(j-i) → small means anomalous.
            anomaly = h - np.sqrt(j - i)
            candidates.append((anomaly, i, j))
    candidates.sort(key=lambda t: t[0])
    return [(i, j) for _, i, j in candidates[:n_positions]]


def _load_macro_cache(path: Path | None):
    """Load a JSONL macro cache file from prior bridge runs. Each line is
    {"residual": [int,...], "bridge_moves": [str,...]}. Returns dict mapping
    residual tuple to bridge moves list. Empty if file missing.
    """
    cache: dict[tuple, list[str]] = {}
    if path is None or not path.exists():
        return cache
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = tuple(entry["residual"])
                moves = list(entry["bridge_moves"])
                # Keep the shortest bridge if duplicates.
                if key not in cache or len(moves) < len(cache[key]):
                    cache[key] = moves
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def _write_submission(out_path: Path, out_paths: dict) -> None:
    """Write the merged submission CSV. Used for both incremental and final saves."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths.keys()):
            w.writerow([pid, ".".join(out_paths[pid])])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="V model checkpoint (e.g. m_az_v4_v_only.pt)")
    ap.add_argument("--submission", required=True, type=Path,
                    help="Base submission CSV to compress")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output CSV with bridge-improved paths")
    ap.add_argument("--test-csv", type=Path,
                    default=PROJECT / "data" / "test.csv")
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--top-n-pids", type=int, default=10,
                    help="Process the N longest pids (default 10)")
    ap.add_argument("--pids", type=str, default=None,
                    help="Optional comma-separated explicit pid list, overrides --top-n-pids")
    ap.add_argument("--window-sizes", type=str, default="8,15,20,25,30,40,50,60",
                    help="Comma-separated window sizes (default 8,15,20,25,30,40,50,60 — mixed grain)")
    ap.add_argument("--positions-per-size", type=int, default=5,
                    help="Number of start positions to try per window size (default 5)")
    ap.add_argument("--select-mode", default="v_trajectory",
                    choices=("stride", "v_trajectory"),
                    help="Window position strategy: stride (uniform) or v_trajectory "
                         "(rank by V(residual), default). v_trajectory needs ~1 extra "
                         "V eval per candidate position; cost is tiny vs solving.")
    ap.add_argument("--no-hamming-prefilter", action="store_true",
                    help="Disable B5: stop using Hamming(S_i, S_j) as a depth co-predictor "
                         "with V. Pure V-score ranking (legacy).")
    ap.add_argument("--hamming-similar-positions", type=int, default=8,
                    help="F15: extra (i, j) candidates from low-Hamming pairs across the "
                         "whole path (variable window size). 0 disables. Default 8.")
    ap.add_argument("--max-iterations", type=int, default=3,
                    help="C8: rerun the full window-size pass on the same pid until no new "
                         "wins or max iterations. Default 3.")
    ap.add_argument("--macro-cache", type=Path, default=None,
                    help="F16: load a JSONL macro cache from prior bridge runs. For each "
                         "residual, check exact-match before running beam-solve. Cheap win.")
    ap.add_argument("--harvest-bridges", type=Path, default=None,
                    help="F17: append accepted bridges (residual + bridge_moves) to this "
                         "JSONL for future bridge-Bellman training. One line per win.")
    ap.add_argument("--beam", type=int, default=16384)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true", help="Cast model to bf16 for speed")
    ap.add_argument("--log-json", type=Path, default=None,
                    help="Optional path to write per-pid attempt log as JSON")
    # Phase 2: production-stack residual solver
    ap.add_argument("--sym-ensemble", type=int, default=0,
                    help="A1: K rotations applied to each residual; take shortest "
                         "verified bridge. 0=disabled. Always includes identity.")
    ap.add_argument("--sym-rotations", type=Path,
                    default=PROJECT / "data" / "rotations.npy",
                    help="Path to rotations array (default: data/rotations.npy)")
    ap.add_argument("--sym-seed", type=int, default=0,
                    help="RNG seed for sym-ensemble rotation choice")
    ap.add_argument("--niss", action="store_true",
                    help="A3: also solve inverse(residual), keep shorter bridge")
    ap.add_argument("--qshort-student", type=Path, default=None,
                    help="A2: qshort student checkpoint (e.g. m23_v3_az_v4_sym/epoch_0199.pt). "
                         "Uses QShortlisterSolver with --checkpoint as teacher V.")
    ap.add_argument("--qshort-alpha", type=int, default=2,
                    help="qshort top-k multiplier (default 2)")
    ap.add_argument("--max-wall-seconds", type=int, default=0,
                    help="Stop cleanly and write output once this many wall seconds "
                         "elapse (0 = unlimited). Guards against Kaggle's 12h kernel kill.")
    args = ap.parse_args()

    window_sizes = [int(w) for w in args.window_sizes.split(",")]

    puzzle = Megaminx.load(args.puzzle_info)
    states = load_test_states(args.test_csv)
    paths = load_submission(args.submission)
    n_pids = len(states)
    print(f"loaded puzzle, {n_pids} test states, {len(paths)} submitted paths", flush=True)

    # Select pids: top-N longest by current path length (or explicit list).
    if args.pids:
        selected_pids = [int(p) for p in args.pids.split(",")]
    else:
        sorted_pids = sorted(paths.keys(), key=lambda p: len(paths[p]), reverse=True)
        selected_pids = sorted_pids[: args.top_n_pids]
    print(f"selected {len(selected_pids)} pids: {selected_pids[:10]}...", flush=True)
    print(f"  longest selected: pid {selected_pids[0]}, len {len(paths[selected_pids[0]])}", flush=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"loading V model from {args.checkpoint.name} (dtype={dtype})", flush=True)
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)

    # Phase 2: build production-stack solver if any of A1/A2/A3 set, else simple.
    use_production = (args.sym_ensemble > 0 or args.niss or args.qshort_student is not None)
    qshort_model = None
    sym_rotations_list = []
    if use_production:
        from megaminx.bridge_solver import ProductionResidualSolver, load_sym_rotations
        if args.qshort_student is not None:
            print(f"loading qshort student: {args.qshort_student.name} (dtype={dtype})",
                  flush=True)
            qshort_model = load_model_checkpoint(args.qshort_student, device=args.device,
                                                  dtype=dtype)
        if args.sym_ensemble > 0:
            sym_rotations_list = load_sym_rotations(
                args.sym_rotations, args.sym_ensemble, puzzle, args.sym_seed,
            )
            print(f"sym-ensemble: {len(sym_rotations_list)} rotations loaded "
                  f"from {args.sym_rotations.name}", flush=True)
        prod_solver = ProductionResidualSolver(
            puzzle, model, device=args.device,
            internal_batch_size=args.internal_batch_size,
            state_dtype=torch.int8,
            sym_rotations=sym_rotations_list,
            use_niss=args.niss,
            qshort_student=qshort_model,
            qshort_alpha=args.qshort_alpha,
        )
        print(f"production solver: sym_ensemble={args.sym_ensemble} niss={args.niss} "
              f"qshort={'yes' if qshort_model is not None else 'no'}", flush=True)
        solver = None  # not used in this branch
    else:
        prod_solver = None
        solver = KhoruzhiiSolver(
            puzzle, model, device=args.device,
            internal_batch_size=args.internal_batch_size,
            random_seed=0,
            state_dtype=torch.int8,
        )

    out_paths: dict[int, list[str]] = dict(paths)
    attempt_log: list[dict] = []
    total_solver_wall = 0.0
    n_attempts = 0
    n_wins = 0
    n_macro_cache_hits = 0
    total_saved = 0
    t_session = time.time()

    # F16: load macro cache from prior runs
    macro_cache = _load_macro_cache(args.macro_cache)
    if args.macro_cache is not None:
        print(f"loaded macro cache: {len(macro_cache)} entries from {args.macro_cache}",
              flush=True)

    # F17: open bridge-harvest file for append
    harvest_fp = None
    if args.harvest_bridges is not None:
        args.harvest_bridges.parent.mkdir(parents=True, exist_ok=True)
        harvest_fp = open(args.harvest_bridges, "a", buffering=1)
        print(f"harvesting accepted bridges to: {args.harvest_bridges}", flush=True)

    def attempt_bridge(s_i, s_j, max_bridge_len: int):
        """Run the residual solver (with macro-cache short-circuit). Returns
        (found, bridge_len, bridge_moves, source) where source ∈ {'cache','beam'}."""
        nonlocal n_macro_cache_hits, total_solver_wall, n_attempts
        residual_np = make_residual(s_i, s_j)
        # F16: cache hit?
        key = tuple(int(x) for x in residual_np.tolist())
        cached = macro_cache.get(key)
        if cached is not None and len(cached) <= max_bridge_len:
            res_state = tuple(residual_np.tolist())
            try:
                end = puzzle.apply_path(res_state, cached)
                if tuple(end) == tuple(puzzle.solved_state):
                    n_macro_cache_hits += 1
                    return True, len(cached), list(cached), "cache"
            except Exception:
                pass
        # Beam-solve via production or simple stack.
        t0 = time.time()
        if prod_solver is not None:
            found, bridge_len, bridge = prod_solver.solve_residual(
                residual_np.tolist(),
                beam=args.beam,
                max_steps=max(max_bridge_len, 1),
                num_attempts=args.num_attempts,
            )
        else:
            cfg = KhoruzhiiSearchConfig(
                beam_width=args.beam,
                num_steps=max(max_bridge_len, 1),
                num_attempts=args.num_attempts,
                internal_batch_size=args.internal_batch_size,
            )
            found, bridge_len, bridge = solver.solve(residual_np.tolist(), cfg)
        total_solver_wall += (time.time() - t0)
        n_attempts += 1
        return found, bridge_len, bridge, "beam"

    for pid_idx, pid in enumerate(selected_pids):
        orig_path = list(out_paths[pid])
        if pid not in states:
            print(f"[pid {pid}] WARNING not in test states, skipping", flush=True)
            continue
        initial_state = states[pid]
        orig_len = len(orig_path)
        vr = verify_path(puzzle, initial_state, orig_path)
        if not vr.ok:
            print(f"[pid {pid}] WARNING original path does not verify: {vr.reason}", flush=True)
            continue

        best_path = orig_path
        pid_wins = 0
        pid_saved_total = 0
        pid_attempts_count = 0
        pid_cache_hits = 0
        pid_t0 = time.time()

        # C8: iterate full pass until no new wins (or max-iterations).
        for iter_idx in range(args.max_iterations):
            wins_this_iter = 0
            if args.max_wall_seconds and (time.time() - t_session) > args.max_wall_seconds:
                print(f"[pid {pid}] wall budget {args.max_wall_seconds}s hit; stopping", flush=True)
                break
            prefix = compute_prefix_states(initial_state, best_path, puzzle)
            prefix_np = np.asarray(prefix, dtype=np.int64)

            # ---- Fixed-window-size candidates ----
            for window_len in window_sizes:
                if args.max_wall_seconds and (time.time() - t_session) > args.max_wall_seconds:
                    break
                if window_len >= len(best_path):
                    continue
                if args.select_mode == "v_trajectory":
                    positions = select_positions_v_trajectory(
                        prefix, window_len, args.positions_per_size,
                        model, args.device, dtype,
                        use_hamming=not args.no_hamming_prefilter,
                    )
                else:
                    positions = select_positions(len(best_path), window_len, args.positions_per_size)
                for i in positions:
                    j = i + window_len
                    if j > len(best_path):
                        continue
                    s_i = tuple(prefix_np[i].tolist())
                    s_j = tuple(prefix_np[j].tolist())
                    if s_i == s_j:
                        # Closed loop — remove the segment entirely.
                        candidate = best_path[:i] + best_path[j:]
                        if (verify_path(puzzle, initial_state, candidate).ok
                                and len(candidate) < len(best_path)):
                            saved = len(best_path) - len(candidate)
                            best_path = candidate
                            pid_wins += 1
                            wins_this_iter += 1
                            pid_saved_total += saved
                            attempt_log.append({"pid": pid, "iter": iter_idx, "i": i, "j": j,
                                                "window_len": window_len, "bridge_len": 0,
                                                "saved": saved, "kind": "empty_loop"})
                            # Recompute prefix after splice.
                            prefix = compute_prefix_states(initial_state, best_path, puzzle)
                            prefix_np = np.asarray(prefix, dtype=np.int64)
                            continue
                    found, bridge_len, bridge, source = attempt_bridge(s_i, s_j, window_len - 1)
                    pid_attempts_count += 1
                    if not found or bridge_len >= window_len:
                        continue
                    candidate = best_path[:i] + bridge + best_path[j:]
                    vr = verify_path(puzzle, initial_state, candidate)
                    if not vr.ok:
                        print(f"[pid {pid}] BRIDGE VERIFY FAIL i={i} j={j} wlen={window_len}: {vr.reason}",
                              flush=True)
                        continue
                    if len(candidate) >= len(best_path):
                        continue
                    saved = len(best_path) - len(candidate)
                    best_path = candidate
                    pid_wins += 1
                    wins_this_iter += 1
                    pid_saved_total += saved
                    if source == "cache":
                        pid_cache_hits += 1
                    attempt_log.append({"pid": pid, "iter": iter_idx, "i": i, "j": j,
                                        "window_len": window_len, "bridge_len": bridge_len,
                                        "saved": saved, "kind": f"bridge_{source}"})
                    # F17: harvest the (residual, bridge) pair.
                    if harvest_fp is not None and source == "beam":
                        residual_int = make_residual(s_i, s_j).tolist()
                        harvest_fp.write(json.dumps({
                            "pid": pid, "i": i, "j": j, "window_len": window_len,
                            "bridge_len": bridge_len, "saved": saved,
                            "residual": residual_int, "bridge_moves": bridge,
                        }) + "\n")
                    # Splice changes prefix — recompute for further attempts.
                    prefix = compute_prefix_states(initial_state, best_path, puzzle)
                    prefix_np = np.asarray(prefix, dtype=np.int64)

            # ---- F15: Hamming-similar (variable-window) candidates ----
            if args.hamming_similar_positions > 0:
                ham_candidates = select_hamming_similar_windows(
                    prefix, args.hamming_similar_positions,
                    min_window=10, max_window=80,
                )
                for i, j in ham_candidates:
                    if j > len(best_path):
                        continue
                    window_len = j - i
                    s_i = tuple(prefix_np[i].tolist())
                    s_j = tuple(prefix_np[j].tolist())
                    if s_i == s_j:
                        candidate = best_path[:i] + best_path[j:]
                        if (verify_path(puzzle, initial_state, candidate).ok
                                and len(candidate) < len(best_path)):
                            saved = len(best_path) - len(candidate)
                            best_path = candidate
                            pid_wins += 1
                            wins_this_iter += 1
                            pid_saved_total += saved
                            attempt_log.append({"pid": pid, "iter": iter_idx, "i": i, "j": j,
                                                "window_len": window_len, "bridge_len": 0,
                                                "saved": saved, "kind": "empty_loop_ham"})
                            prefix = compute_prefix_states(initial_state, best_path, puzzle)
                            prefix_np = np.asarray(prefix, dtype=np.int64)
                            continue
                    found, bridge_len, bridge, source = attempt_bridge(s_i, s_j, window_len - 1)
                    pid_attempts_count += 1
                    if not found or bridge_len >= window_len:
                        continue
                    candidate = best_path[:i] + bridge + best_path[j:]
                    vr = verify_path(puzzle, initial_state, candidate)
                    if not vr.ok or len(candidate) >= len(best_path):
                        continue
                    saved = len(best_path) - len(candidate)
                    best_path = candidate
                    pid_wins += 1
                    wins_this_iter += 1
                    pid_saved_total += saved
                    if source == "cache":
                        pid_cache_hits += 1
                    attempt_log.append({"pid": pid, "iter": iter_idx, "i": i, "j": j,
                                        "window_len": window_len, "bridge_len": bridge_len,
                                        "saved": saved, "kind": f"bridge_ham_{source}"})
                    if harvest_fp is not None and source == "beam":
                        residual_int = make_residual(s_i, s_j).tolist()
                        harvest_fp.write(json.dumps({
                            "pid": pid, "i": i, "j": j, "window_len": window_len,
                            "bridge_len": bridge_len, "saved": saved,
                            "residual": residual_int, "bridge_moves": bridge,
                            "candidate_source": "hamming",
                        }) + "\n")
                    prefix = compute_prefix_states(initial_state, best_path, puzzle)
                    prefix_np = np.asarray(prefix, dtype=np.int64)

            # Incremental save: persist progress each iteration so a kill
            # (e.g., Kaggle 12h limit) doesn't lose completed work.
            if pid_wins > 0:
                out_paths[pid] = best_path
                _write_submission(args.out, out_paths)
            if wins_this_iter == 0:
                break  # converged; no point doing more iterations

        pid_wall = time.time() - pid_t0
        if pid_wins > 0:
            out_paths[pid] = best_path
            n_wins += pid_wins
            total_saved += pid_saved_total
        new_len = len(best_path)
        progress = f"[{pid_idx+1}/{len(selected_pids)}]"
        cache_note = f" (cache:{pid_cache_hits})" if pid_cache_hits else ""
        print(f"{progress} pid {pid:4d} | orig {orig_len:4d} -> {new_len:4d} "
              f"(saved {orig_len - new_len:+4d}) | wins {pid_wins:2d} of {pid_attempts_count:2d} "
              f"attempts{cache_note} | {pid_wall:.1f}s", flush=True)

    if harvest_fp is not None:
        harvest_fp.close()
    t_total = time.time() - t_session
    print(f"\n=== summary ===\n"
          f"  pids processed: {len(selected_pids)}\n"
          f"  total bridge attempts: {n_attempts}\n"
          f"  total verified wins: {n_wins}\n"
          f"  macro-cache hits: {n_macro_cache_hits}\n"
          f"  total moves saved: {total_saved}\n"
          f"  win rate: {n_wins / max(n_attempts, 1):.1%}\n"
          f"  solver wall: {total_solver_wall:.1f}s | total wall: {t_total:.1f}s",
          flush=True)

    # Write out merged CSV.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths.keys()):
            w.writerow([pid, ".".join(out_paths[pid])])
    print(f"wrote merged CSV: {args.out}", flush=True)

    # Quick post-write verify: sum of moves over all pids.
    new_total = sum(len(p) for p in out_paths.values())
    orig_total = sum(len(paths[p]) for p in out_paths.keys())
    print(f"submission total: {orig_total} -> {new_total} ({orig_total - new_total:+d})", flush=True)

    if args.log_json is not None:
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as f:
            json.dump({"attempts": attempt_log,
                       "summary": {"n_pids": len(selected_pids),
                                   "n_attempts": n_attempts,
                                   "n_wins": n_wins,
                                   "total_saved": total_saved,
                                   "solver_wall_s": round(total_solver_wall, 2),
                                   "total_wall_s": round(t_total, 2)}},
                      f, indent=2)
        print(f"wrote attempt log: {args.log_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
