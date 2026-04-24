"""Solve Megaminx test.csv with a trained model.

Features:
  - Beam escalation: pass 1 narrow (fast, catches easy), subsequent passes widen
    (for unsolved only) — total time bounded because the unsolved set shrinks.
  - NISS (inverse-scramble search): also solve invert_state(s); invert the path back.
    Directional anisotropy → solves a non-overlapping set vs forward.
  - int8 state encoding (default; passed explicitly for clarity).
  - Post-processing: same-face order-5 reduction + adjacent-inverse cancellation.
  - Fallback: pp_fallback.csv (post-processed sample) when model can't solve.

Examples:
  # Single-pass at beam 65k:
  python megaminx/scripts/03_solve.py --checkpoint <ckpt> --out <csv> --beams 65536 --max-steps 120

  # Two-pass escalation with NISS:
  python megaminx/scripts/03_solve.py --checkpoint <ckpt> --out <csv> \
      --beams 16384,65536 --max-steps 60,150 --niss --bf16
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
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission
from cayley.bfs_table import BfsTable
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _try_solve(puzzle, solver, cfg, state, bfs_table, bfs_max_window) -> list[str] | None:
    """One beam-search call + post-process + verify. Returns a valid path or None."""
    found, _, raw = solver.solve(state, cfg)
    if not found:
        return None
    path = full_post_process(raw, puzzle=puzzle, bfs_table=bfs_table, max_window=bfs_max_window)
    if not verify_path(puzzle, state, path).ok:
        return None
    return path


def _solve_with_niss(puzzle, solver, cfg, state, niss, bfs_table, bfs_max_window) -> list[str] | None:
    """Forward solve + optional NISS, return shorter valid path or None."""
    candidates: list[list[str]] = []
    fwd = _try_solve(puzzle, solver, cfg, state, bfs_table, bfs_max_window)
    if fwd is not None:
        candidates.append(fwd)
    if niss:
        inv_state = puzzle.invert_state(state)
        raw_inv_path = _try_solve(puzzle, solver, cfg, inv_state, bfs_table, bfs_max_window)
        if raw_inv_path is not None:
            path_for_orig = full_post_process(
                puzzle.invert_path(raw_inv_path),
                puzzle=puzzle, bfs_table=bfs_table, max_window=bfs_max_window,
            )
            if verify_path(puzzle, state, path_for_orig).ok:
                candidates.append(path_for_orig)
    return min(candidates, key=len) if candidates else None


def _solve_escalating(
    puzzle, solver, state, beams: list[int], max_steps_list: list[int],
    num_attempts: int, niss: bool, bfs_table, bfs_max_window,
) -> tuple[list[str] | None, int]:
    """Try each (beam, max_steps) in order; return first valid path + which pass solved.

    Returns (path_or_None, pass_idx). pass_idx = -1 if never solved.
    """
    for i, (b, ms) in enumerate(zip(beams, max_steps_list)):
        cfg = KhoruzhiiSearchConfig(beam_width=b, num_steps=ms, num_attempts=num_attempts)
        path = _solve_with_niss(puzzle, solver, cfg, state, niss, bfs_table, bfs_max_window)
        if path is not None:
            return path, i
    return None, -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beams", type=str, default="16384,65536",
                    help="comma-separated beam widths; each pass retries unsolved puzzles")
    ap.add_argument("--max-steps", type=str, default="60,150",
                    help="comma-separated max-steps matching --beams")
    ap.add_argument("--num-attempts", type=int, default=1,
                    help="stagnation retries per pass (1 means no retry)")
    ap.add_argument("--niss", action="store_true",
                    help="also solve invert_state; keep shorter")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratified", type=int, default=None,
                    help="K puzzles per 100-pid bucket (covers full difficulty range)")
    ap.add_argument("--strat-seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fallback", type=Path, default=PROJECT / "data" / "pp_bfs6_fallback.csv",
                    help="default: pp_bfs6_fallback.csv (414,678 floor; same-face + BFS-d6 "
                         "on raw sample). Previous: pp_bfs5_fallback.csv (415,521).")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--fp32-state", action="store_true",
                    help="disable int8 state encoding (debugging only; int8 is the default)")
    ap.add_argument("--bfs-table", type=Path, default=None,
                    help="path to bfs_table_d*.pkl; enables window-replacement post-processing")
    ap.add_argument("--bfs-max-window", type=int, default=None,
                    help="override max_window for BFS window replacement (default: d+1)")
    args = ap.parse_args()

    beams = _parse_int_list(args.beams)
    max_steps_list = _parse_int_list(args.max_steps)
    if len(beams) != len(max_steps_list):
        ap.error(f"--beams ({len(beams)}) and --max-steps ({len(max_steps_list)}) "
                 f"must have matching lengths")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    all_ids = sorted(states)
    if args.stratified is not None:
        import random
        rng = random.Random(args.strat_seed)
        buckets: dict[int, list[int]] = {}
        for pid in all_ids:
            buckets.setdefault(pid // 100, []).append(pid)
        picked: list[int] = []
        for b in sorted(buckets):
            picked.extend(sorted(rng.sample(buckets[b], min(args.stratified, len(buckets[b])))))
        solve_ids = picked
        print(f"stratified: {len(solve_ids)} pids across {len(buckets)} buckets "
              f"(seed={args.strat_seed})")
    elif args.limit is not None:
        solve_ids = all_ids[: args.limit]
    else:
        solve_ids = all_ids

    fallback = load_submission(args.fallback)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    if args.chunk_size is not None:
        base = getattr(model, "_orig_mod", model)
        base.inference_chunk_size = args.chunk_size

    state_dtype = torch.int32 if args.fp32_state else torch.int8
    solver = KhoruzhiiSolver(puzzle, model, device=args.device, state_dtype=state_dtype)

    bfs_table: BfsTable | None = None
    if args.bfs_table is not None:
        t_load = time.time()
        bfs_table = BfsTable.load(args.bfs_table)
        print(f"loaded BFS table {args.bfs_table.name}: {len(bfs_table.table):,} states, "
              f"max_depth={bfs_table.max_depth} ({time.time() - t_load:.1f}s)")

    print(f"beams={beams}  max_steps={max_steps_list}  niss={args.niss}  "
          f"state_dtype={state_dtype}  num_attempts={args.num_attempts}")

    rows: list[tuple[int, str]] = []
    stats = {"solved_by_model": 0, "fallback": 0, "total_moves": 0}
    pass_solves = [0] * len(beams)
    # Per-puzzle record: (pid, source, model_len_or_None, fallback_len, chosen_len, pass_idx)
    per_puzzle: list[tuple[int, str, int | None, int, int, int]] = []
    t0 = time.time()
    solve_ids_set = set(solve_ids)
    n_attempted = 0

    for pid in all_ids:
        state = states[pid]
        model_path: list[str] | None = None
        solved_pass = -1

        if pid in solve_ids_set:
            n_attempted += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model_path, solved_pass = _solve_escalating(
                puzzle, solver, state, beams, max_steps_list,
                args.num_attempts, args.niss, bfs_table, args.bfs_max_window,
            )
            if solved_pass >= 0:
                pass_solves[solved_pass] += 1

        # Fallback CSV is expected to already be post-processed (pp_bfs5_fallback.csv or
        # similar). Only run the cheap same-face + cancel passes here; skip the expensive
        # BFS window replacement to avoid redundant work on 900+-move paths for every pid.
        fb_path = full_post_process(fallback[pid])
        candidates: list[tuple[list[str], str]] = []
        if model_path is not None:
            candidates.append((model_path, "solved_by_model"))
        candidates.append((fb_path, "fallback"))
        best_path, best_source = min(candidates, key=lambda x: len(x[0]))
        stats[best_source] += 1
        stats["total_moves"] += len(best_path)
        rows.append((pid, ".".join(best_path)))
        per_puzzle.append((
            pid, best_source,
            len(model_path) if model_path is not None else None,
            len(fb_path), len(best_path), solved_pass,
        ))

        if pid in solve_ids_set and (n_attempted % 20 == 0 or pid == solve_ids[-1]):
            elapsed = time.time() - t0
            rate = n_attempted / elapsed if elapsed > 0 else 0
            pass_str = "/".join(str(n) for n in pass_solves)
            print(f"  pid={pid:4d} attempts={n_attempted}/{len(solve_ids)} "
                  f"total_moves={stats['total_moves']:,} "
                  f"(model:{stats['solved_by_model']} fb:{stats['fallback']} "
                  f"passes:{pass_str}) {elapsed:.1f}s ({rate:.2f} p/s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["initial_state_id", "path"])
        for pid, p in rows:
            writer.writerow([pid, p])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"wrote {args.out}")
    print(f"  stats: {stats}")
    print(f"  pass_solves: {dict(zip(beams, pass_solves))}")
    print(f"  verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves:,}")

    if args.stratified is not None or args.limit is not None:
        buckets: dict[int, list[tuple[int | None, int, int]]] = {}
        for pid, src, mlen, fblen, clen, _ in per_puzzle:
            if pid in solve_ids_set:
                buckets.setdefault(pid // 100, []).append((mlen, fblen, clen))
        print("\n  per-bucket (attempted only):")
        print(f"  {'bucket':>8s} {'n':>4s} {'solved':>7s} {'model_avg':>10s} {'fb_avg':>8s} "
              f"{'chosen_avg':>11s} {'saved':>8s}")
        for b_idx in sorted(buckets):
            rows_b = buckets[b_idx]
            n = len(rows_b)
            n_solved = sum(1 for m, _, _ in rows_b if m is not None)
            m_avg = (sum(m for m, _, _ in rows_b if m is not None) / n_solved) if n_solved else None
            fb_avg = sum(fb for _, fb, _ in rows_b) / n
            chosen_avg = sum(c for _, _, c in rows_b) / n
            saved = fb_avg - chosen_avg
            m_s = f"{m_avg:.1f}" if m_avg is not None else "-"
            print(f"  {b_idx * 100:>4d}-{b_idx * 100 + 99:<3d} {n:>4d} {n_solved:>3d}/{n:<3d} "
                  f"{m_s:>10s} {fb_avg:>8.1f} {chosen_avg:>11.1f} {saved:>+8.1f}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
