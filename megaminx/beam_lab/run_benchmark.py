"""CLI: solve the 12 sample puzzles with one or more beam configs and print
a per-puzzle + aggregate timing table.

Three modes:
    --mode single   classic single-puzzle KhoruzhiiSolver (baseline)
    --mode batch    BatchedKhoruzhiiSolver — K puzzles in lockstep, batched model call
    --mode mitm     MitmKhoruzhiiSolver — single-puzzle, terminates on BFS-d6 shell hit

Use this as the harness for trying optimizations: keep the sample fixed and the
checkpoint fixed, vary the implementation. Compare wall time and path length.

Examples
--------
Baseline (single-puzzle, beam 131k):

    python run_benchmark.py \\
        --checkpoint ../models/m05_bellman_warm/epoch_0499.pt \\
        --beam 131072 --max-steps 120 --bf16

Sweep beam sizes:

    python run_benchmark.py --checkpoint ../models/m05_bellman_warm/epoch_0499.pt \\
        --beam 16384,65536,131072 --max-steps 120 --bf16

Multi-puzzle parallel (K=4):

    python run_benchmark.py --checkpoint ../models/m05_bellman_warm/epoch_0499.pt \\
        --mode batch --batch-size 4 --beam 131072 --max-steps 120 --bf16

MITM (with BFS-d6 shell):

    python run_benchmark.py --checkpoint ../models/m05_bellman_warm/epoch_0499.pt \\
        --mode mitm --bfs-table ../data/bfs_bytes_d6.pkl \\
        --beam 131072 --max-steps 120 --bf16

Single hard puzzle (pid 492):

    python run_benchmark.py --checkpoint ../models/m05_bellman_warm/epoch_0499.pt \\
        --beam 131072 --max-steps 120 --pid 492 --bf16
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from beam_search import (  # noqa: E402
    KhoruzhiiSearchConfig,
    KhoruzhiiSolver,
    Profile,
    setup_model_for_compile,
    setup_model_for_cuda_graphs,
    setup_model_for_inference,
)
from model import load_checkpoint  # noqa: E402
from puzzle import Megaminx  # noqa: E402

DEFAULT_PUZZLE_INFO = HERE / "data" / "puzzle_info.json"
DEFAULT_SAMPLES = HERE / "data" / "sample_puzzles.csv"


def load_samples(csv_path: Path, only_pid: int | None = None) -> list[tuple[int, tuple[int, ...]]]:
    rows = []
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pid = int(row["initial_state_id"])
            if only_pid is not None and pid != only_pid:
                continue
            state = tuple(int(x) for x in row["initial_state"].split(","))
            rows.append((pid, state))
    return rows


def fmt_s(x: float) -> str:
    return f"{x:6.2f}"


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple Metal Performance Shaders"
    return "cpu"


def verify_path(puzzle, initial_state, names) -> bool:
    """Apply each move; check we end at solved."""
    from puzzle import STATE_SIZE  # noqa: F401
    cur = tuple(initial_state)
    for m in names:
        cur = puzzle.apply_move(cur, m)
    return cur == puzzle.solved_state


def print_per_puzzle_row(pid, beam, found, path_len, prof: Profile, valid: bool | None = None):
    valid_mark = "" if valid is None else (" OK" if valid else " BAD")
    print(
        f"{pid:>4d} {beam:>7d} {int(found):>5d} {path_len:>5d}{valid_mark} "
        f"{fmt_s(prof.total_s)} {fmt_s(prof.model_s)} "
        f"{fmt_s(prof.neighbor_s)} {fmt_s(prof.dedup_s)} "
        f"{fmt_s(prof.apply_s)} {fmt_s(prof.topk_s)} "
        f"{fmt_s(prof.hash_s)} {prof.n_steps:>5d}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "batch", "mitm", "qshort"], default="single")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--puzzle-info", type=Path, default=DEFAULT_PUZZLE_INFO)
    ap.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    ap.add_argument("--beam", type=str, default="131072")
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--num-attempts", type=int, default=1)
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true",
                    help="cast model to float16 for inference (useful to A/B on MPS)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-incremental-hash", action="store_true",
                    help="use full child materialization for hash timing A/B")
    ap.add_argument("--reuse-full-neighbors", action="store_true",
                    help="with --no-incremental-hash, reuse generated neighbors as candidates")
    ap.add_argument("--compile", action="store_true",
                    help="wrap model with torch.compile + pad model forward to fixed "
                         "internal_batch_size. Avoids recompile thrash; expected 1.3-1.8x "
                         "speedup on model_s.")
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="wrap model in a CUDA graph capture (no torch.compile). "
                         "Lower per-call dispatch overhead than --compile. Mutually "
                         "exclusive with --compile. Pads forwards to internal-batch-size.")
    ap.add_argument("--tensorrt-engine", type=Path, default=None,
                    help="path to a pre-built TensorRT engine (.ts). Replaces the "
                         "model with the engine; engine is shape-fixed to "
                         "internal-batch-size. Mutually exclusive with --compile / "
                         "--cuda-graphs. Build with megaminx/beam_lab/export_tensorrt.py.")
    # beam-decay (geometric narrowing): per-step beam = max(min, B * decay**j)
    ap.add_argument("--beam-decay", type=float, default=1.0,
                    help="geometric beam narrowing per step (1.0=constant, 0.97=narrow ~70%% "
                         "by step 12). Lower wall but risk longer paths.")
    ap.add_argument("--min-beam-width", type=int, default=0,
                    help="floor for the decayed beam (default 0 = no floor).")
    # stochastic beam (softmax sampling)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0=greedy top-B (default); >0 = sample B without replacement from "
                         "softmax(-value/T) via Gumbel-top-k. Higher T = more exploration.")
    # async-mode flags
    ap.add_argument("--no-profile", action="store_true",
                    help="disable per-stage cuda.synchronize() calls. Lets CPU race ahead "
                         "and queue more GPU work; per-stage timings (model_s, neighbor_s) "
                         "become 0/inaccurate. Use for production runs.")
    ap.add_argument("--solved-check-every", type=int, default=1,
                    help="run the solved-position check (CPU sync) every K steps instead "
                         "of every step (default 1). K>1 trades a few wasted forwards for "
                         "fewer CPU syncs. Stagnation check still runs every step.")
    # batch-mode specific
    ap.add_argument("--batch-size", type=int, default=4,
                    help="K puzzles per batched solve (mode=batch only)")
    # mitm-mode specific
    ap.add_argument("--bfs-table", type=Path, default=None,
                    help="path to bfs_bytes_d*.pkl (mode=mitm only)")
    # qshort-mode specific
    ap.add_argument("--student", type=Path, default=None,
                    help="Q-head student checkpoint (mode=qshort only)")
    ap.add_argument("--alpha", type=float, default=2.0,
                    help="shortlist size = alpha * B (mode=qshort only). "
                         "Recall validation (09_eval_q_recall.py) sets the floor.")
    ap.add_argument("--verify", action="store_true",
                    help="apply each found path to its initial state to confirm it solves")
    args = ap.parse_args()

    device = args.device or default_device()
    print(f"device: {device} ({device_label(device)})")

    puzzle = Megaminx.load(args.puzzle_info)
    samples = load_samples(args.samples, only_pid=args.pid)
    print(f"samples: {len(samples)} puzzles ({[pid for pid, _ in samples]})")
    print(f"loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(str(args.checkpoint), device=device)
    if args.fp16:
        model = model.to(torch.float16)
        print("model cast to float16")
    elif args.bf16 and device == "cuda":
        model = model.to(torch.bfloat16)
        print("model cast to bfloat16")
    setup_model_for_inference(model)
    print("disabled model.inference_chunk_size (single-level chunking via internal_batch_size)")

    n_accel = sum([bool(args.compile), bool(args.cuda_graphs), args.tensorrt_engine is not None])
    if n_accel > 1:
        print("ERROR: --compile, --cuda-graphs, --tensorrt-engine are mutually exclusive",
              file=sys.stderr)
        return 2

    if args.compile:
        print(f"compiling model with torch.compile (pre-warm at batch={args.internal_batch_size})...")
        t0 = time.time()
        model = setup_model_for_compile(model, batch_size=args.internal_batch_size)
        print(f"compile + warmup done in {time.time() - t0:.1f}s")
    elif args.cuda_graphs:
        print(f"wrapping model in CUDA graph (capture at batch={args.internal_batch_size})...")
        t0 = time.time()
        model = setup_model_for_cuda_graphs(model, batch_size=args.internal_batch_size)
        print(f"cuda-graph capture + warmup done in {time.time() - t0:.1f}s")
    elif args.tensorrt_engine is not None:
        print(f"loading TensorRT engine: {args.tensorrt_engine}")
        # torch_tensorrt registers the runtime ops on import — needed even if we just
        # `torch.load` the engine.
        import torch_tensorrt  # noqa: F401
        t0 = time.time()
        model = torch.load(str(args.tensorrt_engine), weights_only=False)
        model.eval()
        # Sanity smoke: one forward at batch_size to make sure shapes & device match
        dummy = torch.zeros((args.internal_batch_size, 120), dtype=torch.int64, device=device)
        with torch.inference_mode():
            _ = model(dummy)
        print(f"engine loaded + warmed in {time.time() - t0:.1f}s")

    use_pad = bool(args.compile or args.cuda_graphs or args.tensorrt_engine is not None)
    beams = [int(x) for x in args.beam.split(",")]

    if args.mode == "mitm":
        if args.bfs_table is None:
            print("ERROR: --bfs-table required for --mode mitm", file=sys.stderr)
            return 2
        from beam_search_mitm import MitmKhoruzhiiSolver, load_bfs_table
        mitm_table = load_bfs_table(args.bfs_table)

    if args.mode == "qshort":
        if args.student is None:
            print("ERROR: --student required for --mode qshort", file=sys.stderr)
            return 2
        from beam_search_qshort import QShortlisterSolver
        student = load_checkpoint(str(args.student), device=device)
        if args.fp16:
            student = student.to(torch.float16)
        elif args.bf16 and device == "cuda":
            student = student.to(torch.bfloat16)
        setup_model_for_inference(student)
        print(f"loaded student: {sum(p.numel() for p in student.parameters()):,} params, "
              f"output_dim={student.output_dim}, alpha={args.alpha}")

    rows = []
    print()
    header = (
        f"{'pid':>4s} {'beam':>7s} {'found':>5s} {'path':>7s} "
        f"{'total':>6s} {'model':>6s} {'neigh':>6s} {'dedup':>6s} "
        f"{'apply':>6s} {'topk':>6s} {'hash':>6s} {'steps':>5s}"
    )
    print(header)
    print("-" * len(header))

    for beam_width in beams:
        cfg = KhoruzhiiSearchConfig(
            beam_width=beam_width,
            num_steps=args.max_steps,
            num_attempts=args.num_attempts,
            internal_batch_size=args.internal_batch_size,
            use_incremental_hash=not args.no_incremental_hash,
            reuse_full_neighbors=args.reuse_full_neighbors,
            beam_decay=args.beam_decay,
            min_beam_width=args.min_beam_width,
            temperature=args.temperature,
            solved_check_every=args.solved_check_every,
        )

        if args.mode == "single":
            solver = KhoruzhiiSolver(
                puzzle, model, device=device,
                internal_batch_size=args.internal_batch_size,
                random_seed=0,
                state_dtype=torch.int8,
                pad_to_batch_size=use_pad,
                profile=not args.no_profile,
            )
            for pid, state in samples:
                ok, plen, names, prof = solver.solve(state, cfg)
                valid = verify_path(puzzle, state, names) if (ok and args.verify) else None
                rows.append({
                    "pid": pid, "beam": beam_width, "mode": "single",
                    "found": int(prof.found),
                    "path_len": prof.path_len if prof.found else -1,
                    "valid": int(valid) if valid is not None else None,
                    "total_s": prof.total_s, "model_s": prof.model_s,
                    "neighbor_s": prof.neighbor_s, "dedup_s": prof.dedup_s,
                    "apply_s": prof.apply_s, "topk_s": prof.topk_s,
                    "hash_s": prof.hash_s, "n_steps": prof.n_steps,
                })
                print_per_puzzle_row(pid, beam_width, ok, plen, prof, valid)

        elif args.mode == "batch":
            from beam_search_batch import BatchedKhoruzhiiSolver
            solver = BatchedKhoruzhiiSolver(
                puzzle, model, device=device,
                internal_batch_size=args.internal_batch_size,
                random_seed=0,
                state_dtype=torch.int8,
                pad_to_batch_size=use_pad,
            )
            K = args.batch_size
            for batch_start in range(0, len(samples), K):
                batch = samples[batch_start : batch_start + K]
                states = [s for _, s in batch]
                results, bp = solver.solve_batch(states, cfg)
                print(f"  [batch] K={len(batch)} wall={bp.total_s:.2f}s phase1={bp.phase1_s:.2f}s "
                      f"model={bp.model_s:.2f}s phase3={bp.phase3_s:.2f}s solved={bp.n_solved}")
                for (pid, state), (ok, plen, names, prof) in zip(batch, results):
                    valid = verify_path(puzzle, state, names) if (ok and args.verify) else None
                    rows.append({
                        "pid": pid, "beam": beam_width, "mode": "batch",
                        "found": int(prof.found),
                        "path_len": prof.path_len if prof.found else -1,
                        "valid": int(valid) if valid is not None else None,
                        "total_s": prof.total_s, "model_s": prof.model_s,
                        "neighbor_s": prof.neighbor_s, "dedup_s": prof.dedup_s,
                        "apply_s": prof.apply_s, "topk_s": prof.topk_s,
                        "hash_s": prof.hash_s, "n_steps": prof.n_steps,
                    })
                    print_per_puzzle_row(pid, beam_width, ok, plen, prof, valid)

        elif args.mode == "qshort":
            solver = QShortlisterSolver(
                puzzle, teacher=model, student=student,
                device=device,
                internal_batch_size=args.internal_batch_size,
                random_seed=0,
                state_dtype=torch.int8,
                alpha=args.alpha,
                pad_to_batch_size=use_pad,
            )
            for pid, state in samples:
                ok, plen, names, prof = solver.solve(state, cfg)
                valid = verify_path(puzzle, state, names) if (ok and args.verify) else None
                rows.append({
                    "pid": pid, "beam": beam_width, "mode": "qshort",
                    "found": int(prof.found),
                    "path_len": prof.path_len if prof.found else -1,
                    "valid": int(valid) if valid is not None else None,
                    "total_s": prof.total_s, "model_s": prof.model_s,
                    "neighbor_s": prof.neighbor_s, "dedup_s": prof.dedup_s,
                    "apply_s": prof.apply_s, "topk_s": prof.topk_s,
                    "hash_s": prof.hash_s, "n_steps": prof.n_steps,
                })
                print_per_puzzle_row(pid, beam_width, ok, plen, prof, valid)

        elif args.mode == "mitm":
            solver = MitmKhoruzhiiSolver(
                puzzle, model, mitm_table=mitm_table,
                device=device,
                internal_batch_size=args.internal_batch_size,
                random_seed=0,
                state_dtype=torch.int8,
                pad_to_batch_size=use_pad,
            )
            for pid, state in samples:
                ok, plen, names, prof = solver.solve(state, cfg)
                valid = verify_path(puzzle, state, names) if (ok and args.verify) else None
                rows.append({
                    "pid": pid, "beam": beam_width, "mode": "mitm",
                    "found": int(prof.found),
                    "path_len": prof.path_len if prof.found else -1,
                    "valid": int(valid) if valid is not None else None,
                    "total_s": prof.total_s, "model_s": prof.model_s,
                    "neighbor_s": prof.neighbor_s, "dedup_s": prof.dedup_s,
                    "apply_s": prof.apply_s, "topk_s": prof.topk_s,
                    "hash_s": prof.hash_s, "n_steps": prof.n_steps,
                })
                print_per_puzzle_row(pid, beam_width, ok, plen, prof, valid)

    print()
    print("aggregate per beam (only solved puzzles counted):")
    for beam_width in beams:
        sub = [r for r in rows if r["beam"] == beam_width and r["found"]]
        if not sub:
            continue
        n = len(sub)
        total_s = sum(r["total_s"] for r in sub)
        avg_path = sum(r["path_len"] for r in sub) / n
        avg_total = total_s / n
        m = sum(r["model_s"] for r in sub) / max(total_s, 1e-9)
        ne = sum(r["neighbor_s"] for r in sub) / max(total_s, 1e-9)
        de = sum(r["dedup_s"] for r in sub) / max(total_s, 1e-9)
        ap_ = sum(r["apply_s"] for r in sub) / max(total_s, 1e-9)
        tk = sum(r["topk_s"] for r in sub) / max(total_s, 1e-9)
        ha = sum(r["hash_s"] for r in sub) / max(total_s, 1e-9)
        print(
            f"  mode={args.mode}  beam={beam_width:>6d}  solved={n}  wall={total_s:6.2f}s  "
            f"avg_path={avg_path:6.2f}  avg_total={avg_total:5.2f}s  "
            f"model={m:.0%} neighbor={ne:.0%} dedup={de:.0%} apply={ap_:.0%} topk={tk:.0%} hash={ha:.0%}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            cols = ["pid", "beam", "mode", "found", "path_len", "valid",
                    "total_s", "model_s", "neighbor_s", "dedup_s",
                    "apply_s", "topk_s", "hash_s", "n_steps"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
