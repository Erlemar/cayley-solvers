"""Solve Megaminx test.csv with a trained model and write a submission CSV.

    python megaminx/scripts/03_solve.py --checkpoint megaminx/models/fast_first/epoch_0199.pt \
        --out megaminx/submissions/first.csv --beam 65536 --bf16

For each puzzle we run khoruzhii beam search. If the model fails, we fall back to the
sample_submission path for that puzzle (guaranteed valid, ~500 moves).
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
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--num-attempts", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="solve only the first N puzzles")
    ap.add_argument(
        "--stratified",
        type=int,
        default=None,
        help="solve K puzzles from each 100-pid bucket (covers full difficulty range); "
             "deterministic with --strat-seed",
    )
    ap.add_argument("--strat-seed", type=int, default=0, help="seed for --stratified sampling")
    ap.add_argument("--bf16", action="store_true", help="cast model to bfloat16")
    ap.add_argument(
        "--fallback",
        type=Path,
        default=PROJECT / "data" / "pp_fallback.csv",
        help="CSV to use for puzzles the model can't solve (default: post-processed sample)",
    )
    ap.add_argument("--chunk-size", type=int, default=None)
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    all_ids = sorted(states)
    if args.stratified is not None:
        import random
        rng = random.Random(args.strat_seed)
        picked: list[int] = []
        buckets: dict[int, list[int]] = {}
        for pid in all_ids:
            buckets.setdefault(pid // 100, []).append(pid)
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

    solver = KhoruzhiiSolver(puzzle, model, device=args.device)
    cfg = KhoruzhiiSearchConfig(
        beam_width=args.beam, num_steps=args.max_steps, num_attempts=args.num_attempts
    )

    rows: list[tuple[int, str]] = []
    stats = {"solved_by_model": 0, "fallback": 0, "total_moves": 0}
    # Per-puzzle record: (pid, source, model_len_or_None, fallback_len, chosen_len)
    per_puzzle: list[tuple[int, str, int | None, int, int]] = []
    t0 = time.time()
    solve_ids_set = set(solve_ids)
    n_attempted = 0

    for pid in all_ids:
        state = states[pid]
        model_path: list[str] | None = None

        if pid in solve_ids_set:
            n_attempted += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            found, length, raw_path = solver.solve(state, cfg)
            if found:
                p = full_post_process(raw_path)
                if verify_path(puzzle, state, p).ok:
                    model_path = p

        # Also post-process the fallback (same-face runs shortened).
        fb_path = full_post_process(fallback[pid])
        candidates: list[tuple[list[str], str]] = []
        if model_path is not None:
            candidates.append((model_path, "solved_by_model"))
        candidates.append((fb_path, "fallback"))
        best_path, best_source = min(candidates, key=lambda x: len(x[0]))
        stats[best_source] += 1
        stats["total_moves"] += len(best_path)
        rows.append((pid, ".".join(best_path)))
        per_puzzle.append(
            (pid, best_source, len(model_path) if model_path is not None else None,
             len(fb_path), len(best_path))
        )

        if pid in solve_ids_set and (n_attempted % 20 == 0 or pid == solve_ids[-1]):
            elapsed = time.time() - t0
            rate = n_attempted / elapsed if elapsed > 0 else 0
            print(f"  pid={pid:4d} attempts={n_attempted}/{len(solve_ids)} "
                  f"total_moves={stats['total_moves']:,} "
                  f"({stats['solved_by_model']} by model, {stats['fallback']} fallback) "
                  f"{elapsed:.1f}s ({rate:.2f} p/s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["initial_state_id", "path"])
        for pid, p in rows:
            writer.writerow([pid, p])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"wrote {args.out}")
    print(f"  stats: {stats}")
    print(f"  verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves:,}")

    # Per-bucket breakdown for puzzles we actually attempted.
    if args.stratified is not None or args.limit is not None:
        buckets: dict[int, list[tuple[int | None, int, int]]] = {}
        for pid, src, mlen, fblen, clen in per_puzzle:
            if pid in solve_ids_set:
                buckets.setdefault(pid // 100, []).append((mlen, fblen, clen))
        print("\n  per-bucket (attempted only):")
        print(f"  {'bucket':>8s} {'n':>4s} {'solved':>7s} {'model_avg':>10s} {'fb_avg':>8s} "
              f"{'chosen_avg':>11s} {'saved':>8s}")
        for b in sorted(buckets):
            rows_b = buckets[b]
            n = len(rows_b)
            n_solved = sum(1 for m, _, _ in rows_b if m is not None)
            m_avg = (sum(m for m, _, _ in rows_b if m is not None) / n_solved) if n_solved else None
            fb_avg = sum(fb for _, fb, _ in rows_b) / n
            chosen_avg = sum(c for _, _, c in rows_b) / n
            saved = fb_avg - chosen_avg
            m_s = f"{m_avg:.1f}" if m_avg is not None else "-"
            print(f"  {b * 100:>4d}-{b * 100 + 99:<3d} {n:>4d} {n_solved:>3d}/{n:<3d} "
                  f"{m_s:>10s} {fb_avg:>8.1f} {chosen_avg:>11.1f} {saved:>+8.1f}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
