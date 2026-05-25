"""Generate a residuals.csv for the TPU bridge kernel.

Pipeline:
  1. Load base submission, V model, test states.
  2. For each selected pid: compute prefix states.
  3. Generate candidate (window_size, position) tuples via Phase-1 logic:
     - Mixed window grain
     - V-trajectory scoring with Hamming co-predictor (B5)
     - Hamming-similar candidates (F15)
  4. For each candidate: build residual = inv(prefix[j])[prefix[i]].
  5. Macro-cache short-circuit (F16): cached residuals get marked
     "presolved" — they skip TPU and feed directly into the splice.
  6. Rank remaining residuals by predicted_save = window_len − max(V(R), H/12).
  7. Take top-K total residuals (across pids and window sizes).
  8. Write residuals.csv (test.csv-compatible) + metadata JSON for splicing.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/82_generate_residuals_for_tpu.py \\
        --checkpoint megaminx/models/m_az_v4_v_only.pt \\
        --base megaminx/submissions/merge_v14_plus_min_count_v4.csv \\
        --pids 990,991,992,993,994,995,996,997,998,999,1000 \\
        --window-sizes 8,15,20,25,30,40,50,60 \\
        --positions-per-size 5 \\
        --hamming-similar 8 \\
        --top-k 200 \\
        --out-residuals megaminx/tpu_bridge_data/residuals.csv \\
        --out-metadata megaminx/tpu_bridge_data/metadata.json \\
        --macro-cache megaminx/data/bridge_harvest.jsonl \\
        --bf16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path
from megaminx.bridge import compute_prefix_states, make_residual
from megaminx.puzzle import Megaminx


@torch.no_grad()
def eval_v(model, states_np, device, batch_size: int = 4096):
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
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="V model checkpoint (used for V-trajectory scoring)")
    ap.add_argument("--base", required=True, type=Path,
                    help="Base submission CSV (paths to compress)")
    ap.add_argument("--out-residuals", required=True, type=Path,
                    help="Output residuals CSV (test.csv-compatible)")
    ap.add_argument("--out-metadata", required=True, type=Path,
                    help="Output metadata JSON (residual_id -> {pid, i, j, ...})")
    ap.add_argument("--test-csv", type=Path,
                    default=PROJECT / "data" / "test.csv")
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--pids", type=str, default=None,
                    help="Comma-separated pid list")
    ap.add_argument("--top-n-pids", type=int, default=0,
                    help="If --pids not set: take this many longest pids")
    ap.add_argument("--window-sizes", type=str, default="8,15,20,25,30,40,50,60",
                    help="Comma-separated window sizes")
    ap.add_argument("--positions-per-size", type=int, default=5,
                    help="Candidates per window size via V-trajectory ranking")
    ap.add_argument("--hamming-similar", type=int, default=8,
                    help="Additional candidates from low-Hamming pairs")
    ap.add_argument("--top-k", type=int, default=200,
                    help="Global top-K residuals to send to TPU (by predicted_save)")
    ap.add_argument("--macro-cache", type=Path, default=None,
                    help="Optional macro cache JSONL — cached residuals get marked "
                         "presolved and skip TPU.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args()

    window_sizes = [int(w) for w in args.window_sizes.split(",")]

    puzzle = Megaminx.load(args.puzzle_info)
    test_states = load_test_states(args.test_csv)
    base_paths = load_submission(args.base)

    if args.pids:
        selected_pids = [int(p) for p in args.pids.split(",")]
    elif args.top_n_pids > 0:
        sorted_pids = sorted(base_paths.keys(), key=lambda p: len(base_paths[p]), reverse=True)
        selected_pids = sorted_pids[: args.top_n_pids]
    else:
        ap.error("must provide --pids or --top-n-pids")

    print(f"loaded base: {len(base_paths)} pids; selected: {len(selected_pids)} "
          f"(first 10: {selected_pids[:10]})", flush=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"loading V model from {args.checkpoint.name} (dtype={dtype})", flush=True)
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)

    # Load macro cache
    macro_cache: dict[tuple, list[str]] = {}
    if args.macro_cache and args.macro_cache.exists():
        with open(args.macro_cache) as f:
            for line in f:
                try:
                    e = json.loads(line.strip())
                    key = tuple(e["residual"])
                    moves = list(e["bridge_moves"])
                    if key not in macro_cache or len(moves) < len(macro_cache[key]):
                        macro_cache[key] = moves
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"loaded macro cache: {len(macro_cache)} entries", flush=True)

    # Generate candidates per pid
    all_candidates: list[dict] = []  # each: pid, i, j, window_len, s_i, s_j, residual, predicted_save
    n_state = len(puzzle.solved_state)
    for pid in selected_pids:
        if pid not in test_states or pid not in base_paths:
            print(f"  pid {pid}: not in inputs, skipping", flush=True)
            continue
        initial_state = test_states[pid]
        path = base_paths[pid]
        if not verify_path(puzzle, initial_state, path).ok:
            print(f"  pid {pid}: base path doesn't verify, skipping", flush=True)
            continue
        prefix = compute_prefix_states(initial_state, path, puzzle)
        prefix_np = np.asarray(prefix, dtype=np.int64)
        path_len = len(path)
        seen_pairs: set[tuple[int, int]] = set()

        def add_candidate(i: int, j: int):
            """Compute residual + V-score + add to candidate list."""
            key = (i, j)
            if key in seen_pairs or i >= j or j > path_len:
                return
            seen_pairs.add(key)
            s_i = prefix_np[i]
            s_j = prefix_np[j]
            if np.array_equal(s_i, s_j):
                return  # closed loop; handled directly in splice as empty bridge
            inv_s_j = np.empty(n_state, dtype=np.int64)
            inv_s_j[s_j] = np.arange(n_state, dtype=np.int64)
            residual = inv_s_j[s_i]
            window_len = j - i
            hamming = int((s_i != s_j).sum())
            all_candidates.append({
                "pid": int(pid),
                "i": int(i),
                "j": int(j),
                "window_len": int(window_len),
                "s_i": s_i.tolist(),
                "s_j": s_j.tolist(),
                "residual": residual.tolist(),
                "hamming": hamming,
                "original_window_path": path[i:j],
            })

        # Mixed window grain via V-trajectory selection
        for window_len in window_sizes:
            if window_len >= path_len:
                continue
            max_i = path_len - window_len
            if max_i <= 0:
                continue
            residuals_batch = np.empty((max_i + 1, n_state), dtype=np.int64)
            hammings_batch = np.empty(max_i + 1, dtype=np.int32)
            for i in range(max_i + 1):
                s_i = prefix_np[i]
                s_j = prefix_np[i + window_len]
                inv_s_j = np.empty(n_state, dtype=np.int64)
                inv_s_j[s_j] = np.arange(n_state, dtype=np.int64)
                residuals_batch[i] = inv_s_j[s_i]
                hammings_batch[i] = int((s_i != s_j).sum())
            v_pred = eval_v(model, residuals_batch, args.device)
            depth_estimate = np.maximum(v_pred, np.maximum(1.0, hammings_batch / 12.0))
            scores = window_len - depth_estimate
            order = np.argsort(-scores)
            for k in order[: args.positions_per_size]:
                i = int(k)
                add_candidate(i, i + window_len)

        # Hamming-similar variable-window candidates
        if args.hamming_similar > 0:
            ham_cands: list[tuple[float, int, int]] = []
            for i in range(path_len - 10):
                for j in range(i + 10, min(i + 81, path_len + 1)):
                    h = int((prefix_np[i] != prefix_np[j]).sum())
                    anomaly = h - np.sqrt(j - i)
                    ham_cands.append((anomaly, i, j))
            ham_cands.sort(key=lambda t: t[0])
            for _, i, j in ham_cands[: args.hamming_similar]:
                add_candidate(i, j)

    # Score all candidates globally + pick top-K
    print(f"generated {len(all_candidates)} raw candidates", flush=True)
    if not all_candidates:
        ap.error("no candidates generated")

    # Per-candidate V re-eval (single batched forward over all residuals)
    res_arr = np.array([c["residual"] for c in all_candidates], dtype=np.int64)
    v_pred_all = eval_v(model, res_arr, args.device)
    for c, v in zip(all_candidates, v_pred_all):
        c["v_pred"] = float(v)
        h_lower = max(1.0, c["hamming"] / 12.0)
        depth_est = max(c["v_pred"], h_lower)
        c["predicted_save"] = c["window_len"] - depth_est

    # Split into cache-hits and to-solve
    cache_hits: list[dict] = []
    to_solve: list[dict] = []
    for c in all_candidates:
        key = tuple(c["residual"])
        cached = macro_cache.get(key)
        if cached is not None and len(cached) < c["window_len"]:
            # Verify cached bridge solves residual.
            try:
                end = puzzle.apply_path(tuple(c["residual"]), cached)
                if tuple(end) == tuple(puzzle.solved_state):
                    c["bridge_moves"] = list(cached)
                    c["bridge_len"] = len(cached)
                    c["source"] = "cache"
                    cache_hits.append(c)
                    continue
            except Exception:
                pass
        to_solve.append(c)

    # Sort to_solve by predicted_save desc, take top-K
    to_solve.sort(key=lambda c: -c["predicted_save"])
    if args.top_k > 0 and len(to_solve) > args.top_k:
        to_solve = to_solve[: args.top_k]
    print(f"cache hits: {len(cache_hits)}  to-solve on TPU: {len(to_solve)}", flush=True)

    # Write residuals.csv (only the to-solve set, with sequential residual_ids)
    args.out_residuals.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_residuals, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "initial_state"])
        for rid, c in enumerate(to_solve):
            w.writerow([rid, ",".join(str(x) for x in c["residual"])])
            c["residual_id"] = rid
    print(f"wrote {args.out_residuals} ({len(to_solve)} residuals for TPU)", flush=True)

    # Write metadata
    metadata = {
        "base_csv": str(args.base),
        "test_csv": str(args.test_csv),
        "v_checkpoint": str(args.checkpoint),
        "window_sizes": window_sizes,
        "positions_per_size": args.positions_per_size,
        "hamming_similar": args.hamming_similar,
        "top_k_tpu": args.top_k,
        "selected_pids": selected_pids,
        "cache_hits": [{
            "pid": c["pid"], "i": c["i"], "j": c["j"],
            "window_len": c["window_len"],
            "bridge_len": c["bridge_len"],
            "bridge_moves": c["bridge_moves"],
            "saved": c["window_len"] - c["bridge_len"],
        } for c in cache_hits],
        "tpu_residuals": [{
            "residual_id": c["residual_id"],
            "pid": c["pid"], "i": c["i"], "j": c["j"],
            "window_len": c["window_len"],
            "hamming": c["hamming"],
            "v_pred": c["v_pred"],
            "predicted_save": c["predicted_save"],
        } for c in to_solve],
    }
    args.out_metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"wrote {args.out_metadata}", flush=True)

    # Summary
    by_pid = {}
    for c in to_solve:
        by_pid.setdefault(c["pid"], []).append(c)
    print(f"\nper-pid TPU residual counts:")
    for pid in sorted(by_pid):
        print(f"  pid {pid:4d}: {len(by_pid[pid])} residuals, "
              f"max predicted_save={max(c['predicted_save'] for c in by_pid[pid]):.1f}", flush=True)
    if cache_hits:
        print(f"\ncache-hit savings (will go directly to splice without TPU):")
        for c in sorted(cache_hits, key=lambda c: -(c['window_len'] - c['bridge_len']))[:10]:
            print(f"  pid {c['pid']:4d} i={c['i']:3d} j={c['j']:3d}: "
                  f"{c['window_len']} -> {c['bridge_len']} (save {c['window_len'] - c['bridge_len']})",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
