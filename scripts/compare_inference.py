"""A/B test: fp32 baseline vs bf16+compile for the same model + puzzles.

Runs the same seed against the same set of puzzles under two inference configs and
prints per-puzzle solve/length comparison + aggregate stats.

    python scripts/compare_inference.py --checkpoint models/fast/epoch_0499.pt \\
        --beam 4096 --n-puzzles 15
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.post_process import full_post_process
from cayley.puzzle import PictureCube
from cayley.search import SearchConfig, Solver, load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path


def run(model_loader_kwargs: dict, label: str, puzzle, states, pids, cfg):
    torch.cuda.empty_cache()
    model = load_model_checkpoint(**model_loader_kwargs)
    solver = Solver(puzzle, model, device=model_loader_kwargs["device"])

    results: list[tuple[int, bool, int]] = []
    total_moves = 0
    solved = 0
    t0 = time.time()
    for pid in pids:
        torch.cuda.empty_cache()
        res = solver.solve(states[pid], cfg)
        ok = False
        if res.found:
            p = full_post_process(res.path, states[pid], puzzle)
            if verify_path(puzzle, states[pid], p).ok:
                ok = True
                solved += 1
                total_moves += len(p)
                results.append((pid, True, len(p)))
                continue
        results.append((pid, False, 0))
    elapsed = time.time() - t0
    print(f"\n[{label}] solved {solved}/{len(pids)}, total={total_moves}, {elapsed:.1f}s")
    return results, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--beam", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--n-puzzles", type=int, default=15)
    ap.add_argument("--chunk-baseline", type=int, default=2048)
    ap.add_argument("--chunk-optimized", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    sample = load_submission(PROJECT / "data" / "sample_submission.csv")

    # Same deterministic puzzle sample across difficulty.
    all_ids = sorted(states.keys(), key=lambda pid: len(sample[pid]))
    step = max(1, len(all_ids) // args.n_puzzles)
    pids = all_ids[::step][: args.n_puzzles]

    cfg = SearchConfig(beam_width=args.beam, max_steps=args.max_steps, beam_mode="simple")

    # Baseline: fp32, default chunk, no compile
    baseline_kwargs = dict(
        path=args.checkpoint, device=args.device, dtype=torch.float32, compile_inference=False
    )
    res_base, t_base = run(baseline_kwargs, "baseline fp32", puzzle, states, pids, cfg)
    # Set chunk size (through the loader's returned model) — do this via a second load because
    # our load_model_checkpoint doesn't expose it directly.
    # Instead: after-the-fact patch the model's inference_chunk_size before the Solver call.
    # For the baseline above, we left it at default (2048).

    # Optimized: bf16, larger chunk, compile
    opt_kwargs = dict(
        path=args.checkpoint, device=args.device, dtype=torch.bfloat16, compile_inference=True
    )
    res_opt, t_opt = run(opt_kwargs, "optimized bf16+compile", puzzle, states, pids, cfg)

    # Compare
    print("\n--- PER-PUZZLE COMPARISON ---")
    print(f"{'pid':>5} {'base':>6} {'opt':>6} {'delta':>6}")
    base_solved = 0
    opt_solved = 0
    both_solved = 0
    length_matches = 0
    total_delta = 0
    for (pid, b_ok, b_len), (_, o_ok, o_len) in zip(res_base, res_opt):
        bs = f"{b_len}" if b_ok else "FAIL"
        os_ = f"{o_len}" if o_ok else "FAIL"
        delta = ""
        if b_ok:
            base_solved += 1
        if o_ok:
            opt_solved += 1
        if b_ok and o_ok:
            both_solved += 1
            if b_len == o_len:
                length_matches += 1
            delta = f"{o_len - b_len:+d}"
            total_delta += (o_len - b_len)
        print(f"{pid:>5} {bs:>6} {os_:>6} {delta:>6}")

    print(f"\nboth solved: {both_solved}/{len(pids)}")
    print(f"length matches exactly on: {length_matches}/{both_solved if both_solved else 1}")
    print(f"total moves delta (opt - base): {total_delta:+d}")
    print(f"speedup: {t_base / t_opt:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
