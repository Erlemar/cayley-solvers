"""Solve test.csv with a trained model and write a final submission CSV.

    python scripts/02_solve.py --checkpoint models/small/epoch_0099.pt \
        --out submissions/small_beam18.csv [--beam 262144] [--limit 20]

For each puzzle we run beam search, post-process the path, verify, and keep the shorter
of (our solution, fallback). The fallback (default: data/sample_fallback.csv) guarantees
we always emit a valid path even if the model fails a puzzle. Add `--extra-runs a.csv b.csv`
to ensemble across previous runs without re-solving.
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
    ap.add_argument("--out", required=True, type=Path, help="final submission CSV")
    ap.add_argument("--beam", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--mode", default="simple", choices=["simple", "advanced"])
    ap.add_argument("--history-depth", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="solve only the first N puzzles")
    ap.add_argument("--skip-post", action="store_true")
    ap.add_argument("--fallback", type=Path, default=PROJECT / "data" / "sample_fallback.csv")
    ap.add_argument("--extra-runs", nargs="*", type=Path, default=[], help="additional CSVs to ensemble")
    ap.add_argument("--chunk-size", type=int, default=None, help="override model.inference_chunk_size")
    ap.add_argument("--bf16", action="store_true", help="cast model to bfloat16 for ~2x inference speedup")
    ap.add_argument("--compile-inference", action="store_true", help="torch.compile the model for inference")
    ap.add_argument("--mitm-depth", type=int, default=0, help="BFS depth for MITM (0 = disabled)")
    ap.add_argument(
        "--searcher",
        default="cayleypy",
        choices=["cayleypy", "khoruzhii"],
        help="'cayleypy' (library) or 'khoruzhii' (our port, wider beams possible)",
    )
    ap.add_argument("--num-attempts", type=int, default=1, help="khoruzhii: retry on stagnation")
    ap.add_argument("--use-q-function", action="store_true",
                    help="khoruzhii: force Q-function head path (auto-detected from checkpoint if output_dim>1)")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    all_ids = sorted(states)
    solve_ids = all_ids[: args.limit] if args.limit is not None else all_ids

    fallback = load_submission(args.fallback)
    extra: dict[int, list[list[str]]] = {}
    for extra_csv in args.extra_runs:
        for pid, path in load_submission(extra_csv).items():
            extra.setdefault(pid, []).append(path)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(
        args.checkpoint, device=args.device, dtype=dtype, compile_inference=args.compile_inference
    )
    if args.chunk_size is not None:
        base = getattr(model, "_orig_mod", model)
        base.inference_chunk_size = args.chunk_size
    base = getattr(model, "_orig_mod", model)
    detected_q = getattr(base, "output_dim", 1) > 1
    use_q = args.use_q_function or detected_q
    if detected_q and not args.use_q_function:
        print(f"auto-detected Q-function (output_dim={base.output_dim}); using Q-solver path")
    if args.searcher == "khoruzhii":
        solver = KhoruzhiiSolver(puzzle, model, device=args.device, use_q_function=use_q)
        cfg = KhoruzhiiSearchConfig(
            beam_width=args.beam, num_steps=args.max_steps, num_attempts=args.num_attempts
        )
    else:
        solver = Solver(puzzle, model, device=args.device, mitm_depth=args.mitm_depth)
        cfg = SearchConfig(
            beam_width=args.beam, max_steps=args.max_steps, beam_mode=args.mode, history_depth=args.history_depth
        )

    rows: list[tuple[int, str]] = []
    stats = {"solved_by_model": 0, "fallback": 0, "from_extra": 0, "total_moves": 0}
    t0 = time.time()

    solve_ids_set = set(solve_ids)
    for pid in all_ids:
        state = states[pid]
        model_path: list[str] | None = None
        extra_paths: list[list[str]] = []

        if pid in solve_ids_set:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.searcher == "khoruzhii":
                found, length, raw_path = solver.solve(state, cfg)
                if found:
                    p = raw_path if args.skip_post else full_post_process(raw_path, state, puzzle)
                    if verify_path(puzzle, state, p).ok:
                        model_path = p
            else:
                res = solver.solve(state, cfg)
                if res.found:
                    p = res.path if args.skip_post else full_post_process(res.path, state, puzzle)
                    if verify_path(puzzle, state, p).ok:
                        model_path = p

        for p in extra.get(pid, []):
            if verify_path(puzzle, state, p).ok:
                extra_paths.append(p)

        # Gather all candidates with provenance so we can attribute the winner.
        candidates: list[tuple[list[str], str]] = []
        if model_path is not None:
            candidates.append((model_path, "solved_by_model"))
        for p in extra_paths:
            candidates.append((p, "from_extra"))
        candidates.append((fallback[pid], "fallback"))

        best_path, best_source = min(candidates, key=lambda x: len(x[0]))
        stats[best_source] += 1
        stats["total_moves"] += len(best_path)
        rows.append((pid, ".".join(best_path)))

        if (pid + 1) % 100 == 0 or pid == all_ids[-1]:
            elapsed = time.time() - t0
            print(f"  pid={pid:4d} total={stats['total_moves']} ({elapsed:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["initial_state_id", "path"])
        for pid, p in rows:
            writer.writerow([pid, p])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"wrote {args.out}")
    print(f"  stats: {stats}")
    print(f"  verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves}")
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
