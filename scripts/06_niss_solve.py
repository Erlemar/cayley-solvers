"""NISS (Inverse Scramble Search): solve sigma^-1 for each puzzle and convert back.

For each test puzzle sigma, we:
  1. Compute sigma^-1 = invert_state(sigma) — a valid, reachable puzzle state.
  2. Run the beam solver on sigma^-1 to get path P with apply(sigma^-1, P) = solved.
  3. Convert: invert_path(P) solves sigma.

This is orthogonal to the heuristic: the model was trained on forward random walks
from solved, so the loss landscape seen from sigma vs sigma^-1 differs. The FMC
community reports 0.5–1.5 moves saved per scramble. Output is a candidate CSV;
merge with existing candidates via `combine_submissions.py`.

    python scripts/06_niss_solve.py \\
        --checkpoint models/small_e5/epoch_7999.pt \\
        --out submissions/niss_e5_b65k.csv \\
        --beam 65536 --max-steps 50 --bf16 \\
        --fallback data/kociemba_fallback.csv
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

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.post_process import full_post_process
from cayley.puzzle import PictureCube
from cayley.search import SearchConfig, Solver, load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="NISS candidate CSV")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="solve only first N puzzles")
    ap.add_argument("--skip-post", action="store_true")
    ap.add_argument("--fallback", type=Path, default=PROJECT / "data" / "kociemba_fallback.csv")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument(
        "--searcher", default="khoruzhii", choices=["cayleypy", "khoruzhii"],
    )
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--mitm-depth", type=int, default=0, help="cayleypy only")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    all_ids = sorted(states)
    solve_ids = all_ids[: args.limit] if args.limit is not None else all_ids
    solve_ids_set = set(solve_ids)

    fallback = load_submission(args.fallback)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)

    if args.searcher == "khoruzhii":
        solver = KhoruzhiiSolver(puzzle, model, device=args.device)
        cfg = KhoruzhiiSearchConfig(
            beam_width=args.beam, num_steps=args.max_steps, num_attempts=args.num_attempts,
        )
    else:
        solver = Solver(puzzle, model, device=args.device, mitm_depth=args.mitm_depth)
        cfg = SearchConfig(beam_width=args.beam, max_steps=args.max_steps, beam_mode="simple")

    rows: list[tuple[int, str]] = []
    stats = {"niss_solved": 0, "niss_fallback": 0, "niss_verify_failed": 0, "total_moves": 0}
    t0 = time.time()

    for pid in all_ids:
        state = states[pid]
        niss_path: list[str] | None = None

        if pid in solve_ids_set:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sigma_inv = puzzle.invert_state(state)

            if args.searcher == "khoruzhii":
                found, _, raw = solver.solve(sigma_inv, cfg)
                ok = found
            else:
                res = solver.solve(sigma_inv, cfg)
                ok = res.found
                raw = res.path if ok else []

            if ok:
                converted = puzzle.invert_path(raw)
                if not args.skip_post:
                    converted = full_post_process(converted, state, puzzle)
                if verify_path(puzzle, state, converted).ok:
                    niss_path = converted
                else:
                    stats["niss_verify_failed"] += 1

        # Emit niss_path if valid, else fallback (a NISS run should never emit worse-than-fallback).
        if niss_path is not None:
            chosen = niss_path
            stats["niss_solved"] += 1
        else:
            chosen = fallback[pid]
            stats["niss_fallback"] += 1

        stats["total_moves"] += len(chosen)
        rows.append((pid, ".".join(chosen)))

        if (pid + 1) % 50 == 0 or pid == all_ids[-1]:
            elapsed = time.time() - t0
            print(
                f"  pid={pid:4d} solved={stats['niss_solved']} fallback={stats['niss_fallback']} "
                f"total={stats['total_moves']} ({elapsed:.1f}s)"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid, p in rows:
            w.writerow([pid, p])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"wrote {args.out}")
    print(f"  stats: {stats}")
    print(f"  verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
