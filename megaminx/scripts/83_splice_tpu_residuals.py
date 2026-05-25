"""Splice TPU-solved residual bridges back into the base submission.

Inputs:
  - Base submission CSV (the paths we're improving)
  - Metadata JSON from 82_generate_residuals_for_tpu.py (residual_id -> source pid/i/j)
  - TPU solutions CSV (initial_state_id == residual_id, path = bridge)
  - Optional: harvest JSONL for future bridge-Bellman training data

For each pid that has any wins:
  1. Collect all wins (cache-hits from metadata + TPU solutions).
  2. Sort by saved-moves descending; greedily pick non-overlapping wins.
  3. Splice from rightmost to leftmost so indices stay valid.
  4. Verify the final spliced path solves the original initial state.
  5. Replace the pid's path in the output CSV.

Output:
  - Spliced submission CSV (per-pid greedy min over base / cache / TPU)
  - Optional harvest update (appends accepted TPU bridges for future cache hits)

Usage:
    .venv/Scripts/python.exe megaminx/scripts/83_splice_tpu_residuals.py \\
        --base megaminx/submissions/merge_v14_plus_min_count_v4.csv \\
        --metadata megaminx/tpu_bridge_data/metadata.json \\
        --tpu-solutions megaminx/tpu_bridge_data/share_jax_residuals_b1m.csv \\
        --out megaminx/submissions/bridge_tpu_b1m.csv \\
        --harvest megaminx/data/bridge_harvest.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.verify import load_submission, load_test_states, verify_path
from megaminx.puzzle import Megaminx


def select_non_overlapping(wins: list[dict]) -> list[dict]:
    """Greedy: sort by saved descending; keep wins whose (i, j) interval
    doesn't overlap any already-selected interval. Returns list sorted by i."""
    wins_sorted = sorted(wins, key=lambda w: (-w["saved"], w["i"]))
    intervals: list[tuple[int, int]] = []
    selected: list[dict] = []
    for w in wins_sorted:
        i, j = w["i"], w["j"]
        # Overlap test: two intervals [a, b) and [c, d) overlap iff a < d and c < b.
        if any(i < x_j and x_i < j for x_i, x_j in intervals):
            continue
        intervals.append((i, j))
        selected.append(w)
    selected.sort(key=lambda w: w["i"])
    return selected


def splice_path(base_path: list[str], wins_sorted_by_i: list[dict]) -> list[str]:
    """Apply non-overlapping wins (sorted by i ascending) right-to-left
    so earlier-window indices stay valid as later-window splices shorten
    the trailing segment."""
    new_path = list(base_path)
    for w in reversed(wins_sorted_by_i):
        new_path = new_path[: w["i"]] + list(w["bridge_moves"]) + new_path[w["j"] :]
    return new_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="Base submission CSV (the paths we're improving)")
    ap.add_argument("--metadata", required=True, type=Path,
                    help="Metadata JSON from 82_generate_residuals_for_tpu.py")
    ap.add_argument("--tpu-solutions", type=Path, default=None,
                    help="TPU kernel output CSV. May be partial (some residuals "
                         "may not have been solved in the time budget).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output spliced submission CSV")
    ap.add_argument("--test-csv", type=Path,
                    default=PROJECT / "data" / "test.csv")
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--harvest", type=Path, default=None,
                    help="Optional: append accepted TPU bridges to this JSONL")
    args = ap.parse_args()

    puzzle = Megaminx.load(args.puzzle_info)
    test_states = load_test_states(args.test_csv)
    base_paths = load_submission(args.base)
    base_total = sum(len(p) for p in base_paths.values())
    print(f"base: {len(base_paths)} pids, total {base_total} moves", flush=True)

    with open(args.metadata) as f:
        metadata = json.load(f)
    cache_hits = metadata.get("cache_hits", [])
    tpu_meta = {int(r["residual_id"]): r for r in metadata.get("tpu_residuals", [])}
    print(f"metadata: {len(cache_hits)} cache hits, {len(tpu_meta)} expected TPU residuals",
          flush=True)

    # Load TPU solutions (residual_id -> bridge moves), if provided
    tpu_paths: dict[int, list[str]] = {}
    if args.tpu_solutions and args.tpu_solutions.exists():
        tpu_paths = load_submission(args.tpu_solutions)
        print(f"loaded {len(tpu_paths)} TPU solutions from {args.tpu_solutions.name}",
              flush=True)
    else:
        print("no TPU solutions provided; processing cache hits only", flush=True)

    # Build wins-per-pid from cache hits + verified TPU solutions
    wins_by_pid: dict[int, list[dict]] = {}
    n_cache_wins = 0
    for c in cache_hits:
        win = {
            "i": int(c["i"]),
            "j": int(c["j"]),
            "window_len": int(c["window_len"]),
            "bridge_len": int(c["bridge_len"]),
            "bridge_moves": list(c["bridge_moves"]),
            "saved": int(c["saved"]),
            "source": "cache",
        }
        wins_by_pid.setdefault(int(c["pid"]), []).append(win)
        n_cache_wins += 1

    # Process TPU solutions: residual_id -> match metadata -> verify shorter than window
    n_tpu_wins = 0
    n_tpu_skipped_no_improvement = 0
    n_tpu_unmapped = 0
    for rid, path in tpu_paths.items():
        meta = tpu_meta.get(rid)
        if meta is None:
            n_tpu_unmapped += 1
            continue
        window_len = int(meta["window_len"])
        if len(path) >= window_len:
            n_tpu_skipped_no_improvement += 1
            continue
        win = {
            "i": int(meta["i"]),
            "j": int(meta["j"]),
            "window_len": window_len,
            "bridge_len": len(path),
            "bridge_moves": list(path),
            "saved": window_len - len(path),
            "source": "tpu",
            "residual_id": rid,
        }
        wins_by_pid.setdefault(int(meta["pid"]), []).append(win)
        n_tpu_wins += 1
    print(f"raw wins: cache {n_cache_wins}, tpu {n_tpu_wins}; "
          f"tpu unmapped {n_tpu_unmapped}, tpu no-improvement {n_tpu_skipped_no_improvement}",
          flush=True)

    # Per-pid: greedy non-overlapping + verify
    out_paths: dict[int, list[str]] = dict(base_paths)
    n_pids_improved = 0
    total_saved = 0
    harvest_fp = None
    if args.harvest is not None:
        args.harvest.parent.mkdir(parents=True, exist_ok=True)
        harvest_fp = open(args.harvest, "a", buffering=1)

    for pid, wins in wins_by_pid.items():
        if pid not in base_paths or pid not in test_states:
            print(f"  pid {pid}: missing in base/test, skipping", flush=True)
            continue
        # Verify each individual win actually applies cleanly. Residual-level math
        # guarantees apply_path(prefix[i], bridge) == prefix[j]; we re-check here
        # to defend against any encoding glitch.
        base_path = base_paths[pid]
        # Compute prefix to verify
        from megaminx.bridge import compute_prefix_states
        prefix = compute_prefix_states(test_states[pid], base_path, puzzle)
        valid_wins = []
        for w in wins:
            i, j = w["i"], w["j"]
            if j > len(base_path) or i >= j:
                continue
            # Verify bridge maps prefix[i] -> prefix[j].
            try:
                end = puzzle.apply_path(prefix[i], w["bridge_moves"])
                if tuple(end) != tuple(prefix[j]):
                    print(f"  pid {pid} win i={i} j={j} source={w['source']} "
                          f"FAILED bridge endpoint check", flush=True)
                    continue
            except Exception as e:
                print(f"  pid {pid} win i={i} j={j} EXCEPTION {e}", flush=True)
                continue
            valid_wins.append(w)
        if not valid_wins:
            continue

        selected = select_non_overlapping(valid_wins)
        new_path = splice_path(base_path, selected)
        vr = verify_path(puzzle, test_states[pid], new_path)
        if not vr.ok:
            print(f"  pid {pid}: spliced path FAILED final verify ({vr.reason}); skipping",
                  flush=True)
            continue
        if len(new_path) >= len(base_path):
            continue
        saved = len(base_path) - len(new_path)
        out_paths[pid] = new_path
        n_pids_improved += 1
        total_saved += saved
        sources = [w["source"] for w in selected]
        print(f"  pid {pid:4d}: {len(base_path)} -> {len(new_path)} "
              f"(saved {saved:+d}, applied {len(selected)} wins, sources={sources})",
              flush=True)
        if harvest_fp is not None:
            for w in selected:
                if w["source"] != "tpu":
                    continue  # cache hits already in harvest
                # Need the residual permutation; look it up from tpu metadata
                rid = w["residual_id"]
                meta = tpu_meta[rid]
                # Re-compute residual from prefix (cheaper than re-load)
                from megaminx.bridge import make_residual
                residual = make_residual(prefix[meta["i"]], prefix[meta["j"]]).tolist()
                harvest_fp.write(json.dumps({
                    "pid": pid, "i": w["i"], "j": w["j"],
                    "window_len": w["window_len"],
                    "bridge_len": w["bridge_len"],
                    "saved": w["saved"],
                    "residual": residual,
                    "bridge_moves": w["bridge_moves"],
                    "source": "tpu_b1m",
                }) + "\n")

    if harvest_fp is not None:
        harvest_fp.close()

    # Write final CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths.keys()):
            w.writerow([pid, ".".join(out_paths[pid])])
    new_total = sum(len(p) for p in out_paths.values())
    print(f"\nwrote {args.out}", flush=True)
    print(f"submission total: {base_total} -> {new_total} ({base_total - new_total:+d})",
          flush=True)
    print(f"pids improved: {n_pids_improved} / {len(wins_by_pid)} with wins",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
