"""Interior segment re-solve for verified Megaminx paths.

Tail re-solve only replaces a suffix. This script generalizes that idea to
interior windows: for states S_i and S_j along a verified path, build the
residual state whose solution is a bridge from S_i to S_j. If beam search finds
a shorter bridge than the original window path[i:j], splice it in and verify the
whole path.

The operation is additive: a candidate is accepted only when it is shorter and
the full puzzle path verifies.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.bridge import compute_prefix_states, make_residual
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _candidate_windows(
    n_moves: int,
    lengths: list[int],
    center_fracs: list[float],
    random_windows: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for length in lengths:
        if length <= 0 or length >= n_moves:
            continue
        for frac in center_fracs:
            center = int(round(frac * n_moves))
            start = max(0, min(n_moves - length, center - length // 2))
            out.add((start, start + length))
        for _ in range(random_windows):
            start = rng.randint(0, n_moves - length)
            out.add((start, start + length))
    return sorted(out, key=lambda x: (x[1] - x[0], x[0]))


def _try_window(
    puzzle: Megaminx,
    solver: KhoruzhiiSolver,
    initial_state: tuple[int, ...],
    path: list[str],
    prefix_states: list[tuple[int, ...]],
    i: int,
    j: int,
    beam: int,
    max_extra_steps: int,
    bfs_table,
    bfs_max_window: int,
) -> tuple[list[str] | None, int, list[str] | None]:
    original_len = j - i
    residual = tuple(int(x) for x in make_residual(prefix_states[i], prefix_states[j]))
    cfg = KhoruzhiiSearchConfig(
        beam_width=beam,
        num_steps=original_len + max_extra_steps,
        num_attempts=1,
    )
    found, _, raw_bridge = solver.solve(residual, cfg)
    if not found:
        return None, 0, None

    bridge = full_post_process(
        raw_bridge,
        puzzle=puzzle,
        bfs_table=bfs_table,
        max_window=bfs_max_window,
    )
    if len(bridge) >= original_len:
        return None, 0, bridge
    if tuple(puzzle.apply_path(prefix_states[i], bridge)) != tuple(prefix_states[j]):
        return None, 0, bridge

    candidate = list(path[:i]) + list(bridge) + list(path[j:])
    if len(candidate) >= len(path):
        return None, 0, bridge
    vr = verify_path(puzzle, initial_state, candidate)
    if not vr.ok:
        return None, 0, bridge
    return candidate, len(path) - len(candidate), bridge


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", default="megaminx/models/m_az_v4_v_only.pt")
    ap.add_argument("--bfs-table", default="megaminx/data/bfs_bytes_d6.pkl")
    ap.add_argument("--bfs-max-window", type=int, default=12)
    ap.add_argument("--beam", type=int, default=32768)
    ap.add_argument("--window-list", default="20,30,40,50,60")
    ap.add_argument("--center-fracs", default="0.25,0.5,0.75")
    ap.add_argument("--random-windows", type=int, default=2,
                    help="Extra random starts per window length and pid.")
    ap.add_argument("--max-extra-steps", type=int, default=16)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--pids", default="")
    ap.add_argument("--seed", type=int, default=117)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--internal-batch-size", type=int, default=2**14)
    ap.add_argument("--state-dtype", default="int8", choices=["int8", "int16", "int32", "int64"])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    window_lengths = _parse_ints(args.window_list)
    center_fracs = _parse_floats(args.center_fracs)
    state_dtype = {
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
    }[args.state_dtype]
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

    print("loading puzzle + paths + model ...", flush=True)
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    base = load_submission(args.base)
    base_total = sum(len(p) for p in base.values())
    print(f"  base rows={len(base)} total={base_total:,}", flush=True)

    bfs_path = Path(args.bfs_table)
    if bfs_path.exists():
        print(f"  loading bfs table {bfs_path} ...", flush=True)
        bfs_table = BfsBytesTable.load(bfs_path)
    else:
        print(f"  WARNING: no bfs table at {bfs_path}; post-proc will be cancel-only", flush=True)
        bfs_table = None

    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=model_dtype)
    solver = KhoruzhiiSolver(
        puzzle,
        model,
        device=args.device,
        internal_batch_size=args.internal_batch_size,
        state_dtype=state_dtype,
    )

    if args.pids.strip():
        pid_order = [int(p) for p in args.pids.split(",") if p.strip()]
    else:
        pid_order = sorted(base.keys(), key=lambda p: -len(base[p]))
        if args.top > 0:
            pid_order = pid_order[: args.top]

    print(
        f"\nprocessing {len(pid_order)} pids; windows={window_lengths}; "
        f"centers={center_fracs}; random={args.random_windows}; "
        f"passes={args.passes}; beam={args.beam}\n",
        flush=True,
    )

    out_paths: dict[int, list[str]] = {pid: list(path) for pid, path in base.items()}
    n_improved = 0
    total_savings = 0
    n_windows = 0
    n_found_short = 0
    t0 = time.time()

    for rank, pid in enumerate(pid_order, start=1):
        initial = states[pid]
        current = list(out_paths[pid])
        start_len = len(current)
        pid_savings = 0
        rng = random.Random(args.seed + pid * 1009)

        if not verify_path(puzzle, initial, current).ok:
            print(f"  pid={pid:4d} base path failed verification; skipping", flush=True)
            continue

        for pass_idx in range(args.passes):
            prefix_states = compute_prefix_states(initial, current, puzzle)
            windows = _candidate_windows(
                len(current),
                window_lengths,
                center_fracs,
                args.random_windows,
                rng,
            )
            best_candidate = None
            best_savings = 0
            best_window = None
            for i, j in windows:
                n_windows += 1
                candidate, savings, bridge = _try_window(
                    puzzle,
                    solver,
                    initial,
                    current,
                    prefix_states,
                    i,
                    j,
                    args.beam,
                    args.max_extra_steps,
                    bfs_table,
                    args.bfs_max_window,
                )
                if bridge is not None and len(bridge) < (j - i):
                    n_found_short += 1
                if candidate is not None and savings > best_savings:
                    best_candidate = candidate
                    best_savings = savings
                    best_window = (i, j)
            if best_candidate is None:
                break
            current = best_candidate
            pid_savings += best_savings
            if not args.quiet:
                i, j = best_window or (-1, -1)
                print(
                    f"  pid={pid:4d} pass={pass_idx+1} window={i}:{j} "
                    f"{len(current)+best_savings:3d}->{len(current):3d} "
                    f"(-{best_savings:2d})",
                    flush=True,
                )

        if pid_savings > 0:
            out_paths[pid] = current
            n_improved += 1
            total_savings += pid_savings
            if not args.quiet:
                print(
                    f"  pid={pid:4d} total {start_len:3d}->{len(current):3d} "
                    f"(-{pid_savings:2d}) [rank {rank}/{len(pid_order)}, "
                    f"global -{total_savings}]",
                    flush=True,
                )
        elif not args.quiet and (rank <= 10 or rank % 25 == 0):
            print(f"  pid={pid:4d} {start_len:3d} no improvement [rank {rank}/{len(pid_order)}]", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths):
            writer.writerow([pid, ".".join(out_paths[pid])])

    final_total = sum(len(p) for p in out_paths.values())
    elapsed = time.time() - t0
    print("\n=== summary ===", flush=True)
    print(f"  improved: {n_improved}/{len(pid_order)} pids", flush=True)
    print(f"  total savings: {total_savings} moves", flush=True)
    print(f"  windows tried: {n_windows:,}; short bridges found: {n_found_short:,}", flush=True)
    print(f"  walltime: {elapsed/60:.1f} min", flush=True)
    print(f"  wrote {args.out}", flush=True)
    print(f"  base total: {base_total:,}", flush=True)
    print(f"  new  total: {final_total:,} ({final_total - base_total:+d})", flush=True)

    if not args.no_verify:
        report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
        print(
            f"  verify: {report.n_valid}/{report.n_total}; "
            f"total={report.total_moves:,}",
            flush=True,
        )
        if not report.all_valid:
            for pid, reason in report.failures[:5]:
                print(f"    failure pid={pid}: {reason}", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
