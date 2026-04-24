"""Re-solve a selected subset of puzzles at a wider beam and emit a candidate CSV.

Pass either `--min-length N` (re-solve every puzzle whose current best path is >= N)
or `--pids a,b,c` (re-solve these explicit ids). Output is a candidate CSV with the
re-solved paths plus the existing paths for everything else. Combine with other
candidates via `combine_submissions.py` to pick the min per puzzle.

    python scripts/07_target_wider_beam.py \\
        --baseline submissions/ens_e6_niss_pp.csv \\
        --checkpoint models/e6/epoch_0499.pt \\
        --beam 131072 --max-steps 60 --bf16 \\
        --min-length 27 \\
        --try-niss \\
        --out submissions/wide_b131k_tail.csv
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
from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path, help="existing submission CSV")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--beam", type=int, default=131072)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--try-niss", action="store_true", help="also try inverse-scramble")
    ap.add_argument("--use-q-function", action="store_true",
                    help="force Q-function path (auto-detected from checkpoint if output_dim>1)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--min-length", type=int, help="re-solve puzzles with baseline path >= N")
    group.add_argument("--pids", type=str, help="comma-separated pid list")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    baseline = load_submission(args.baseline)

    if args.pids:
        target_pids = sorted(int(x) for x in args.pids.split(","))
    else:
        target_pids = sorted(pid for pid, p in baseline.items() if len(p) >= args.min_length)
    print(f"target count: {len(target_pids)}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    base = getattr(model, "_orig_mod", model)
    detected_q = getattr(base, "output_dim", 1) > 1
    use_q = args.use_q_function or detected_q
    if detected_q and not args.use_q_function:
        print(f"auto-detected Q-function (output_dim={base.output_dim}); using Q-solver path")
    solver = KhoruzhiiSolver(puzzle, model, device=args.device, use_q_function=use_q)
    cfg = KhoruzhiiSearchConfig(
        beam_width=args.beam, num_steps=args.max_steps, num_attempts=args.num_attempts,
    )

    new_paths: dict[int, list[str]] = {}
    stats = {"improved": 0, "same_or_worse": 0, "failed_solve": 0, "saved_moves": 0}
    t0 = time.time()

    for i, pid in enumerate(target_pids):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        original = baseline[pid]
        state = states[pid]
        candidates: list[list[str]] = [list(original)]

        # Forward re-solve
        found, _, raw = solver.solve(state, cfg)
        if found:
            pp = full_post_process(raw, state, puzzle)
            if verify_path(puzzle, state, pp).ok:
                candidates.append(pp)

        # NISS re-solve
        if args.try_niss:
            sigma_inv = puzzle.invert_state(state)
            found_n, _, raw_n = solver.solve(sigma_inv, cfg)
            if found_n:
                converted = puzzle.invert_path(raw_n)
                pp_n = full_post_process(converted, state, puzzle)
                if verify_path(puzzle, state, pp_n).ok:
                    candidates.append(pp_n)

        best = min(candidates, key=len)
        new_paths[pid] = best
        if len(best) < len(original):
            stats["improved"] += 1
            stats["saved_moves"] += len(original) - len(best)
        elif len(candidates) == 1:
            stats["failed_solve"] += 1
        else:
            stats["same_or_worse"] += 1

        if (i + 1) % 10 == 0 or i == len(target_pids) - 1:
            elapsed = time.time() - t0
            print(
                f"  {i + 1}/{len(target_pids)} pid={pid} orig={len(original)} "
                f"new={len(best)} improved={stats['improved']} saved={stats['saved_moves']} "
                f"({elapsed:.1f}s)"
            )

    # Emit full CSV: new path where re-solved, else baseline.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(states):
            path = new_paths.get(pid, baseline[pid])
            w.writerow([pid, ".".join(path)])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"wrote {args.out}")
    print(f"  stats: {stats}")
    print(f"  verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
