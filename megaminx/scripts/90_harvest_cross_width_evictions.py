"""Harvest cross-width eviction labels from strong teacher paths.

This is the practical version of "cross-width expert iteration":

1. Take a verified teacher path for a pid, usually from a wider/diverse run or
   the current best merged CSV.
2. Run a *narrow* beam with the target model.
3. Find the first depth where the teacher prefix was alive at depth t-1 but the
   teacher child at depth t is missing from the narrow frontier.
4. Emit (parent, good_child, boundary_states) so a later delta/rank head can
   learn to keep that good child above the cutoff.

Unlike frontier_regret_v0, these labels are grounded in an externally verified
complete solution path and a real beam-pruning event. They are still a training
dataset, not a submission; pair this with broad solution-generating runs so we
do not spend the whole day on one famous hard pid.

Usage:
    .venv/Scripts/python.exe megaminx/scripts/90_harvest_cross_width_evictions.py ^
        --checkpoint megaminx/models/m_az_v4_v_only.pt ^
        --teacher-csv megaminx/submissions/relink_v16_top200_bfsd6.csv ^
        --out megaminx/data/cross_width_evictions_v0.pt ^
        --top-n-pids 50 --beam 2048 --bf16
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path
from megaminx.bridge import compute_prefix_states
from megaminx.puzzle import Megaminx


def _path_len(path: list[str]) -> int:
    return len(path)


def _frontier_contains(states: torch.Tensor, target: torch.Tensor) -> bool:
    if states.numel() == 0:
        return False
    return bool((states == target).all(dim=1).any().item())


def _load_lengths(csv_path: Path) -> dict[int, int]:
    out: dict[int, int] = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = row["path"].strip()
            out[int(row["initial_state_id"])] = 0 if not raw else len(raw.split("."))
    return out


def _save_dataset(
    out_path: Path,
    *,
    parents: list[torch.Tensor],
    goods: list[torch.Tensor],
    boundaries: list[torch.Tensor],
    meta_pid: list[int],
    meta_depth: list[int],
    meta_teacher_len: list[int],
    meta_narrow_len: list[int],
    meta_boundary_len: list[int],
    state_size: int,
    checkpoint: Path,
    teacher_csv: Path,
    beam: int,
    boundary_size: int,
    n_pids_requested: int,
    n_verified: int,
    n_no_eviction: int,
    n_teacher_survived: int,
    n_narrow_good_enough: int,
    elapsed_s: float,
) -> None:
    if parents:
        parent_t = torch.stack(parents).to(torch.int8)
        good_t = torch.stack(goods).to(torch.int8)
        max_b = max(b.size(0) for b in boundaries)
        boundary_t = torch.empty((len(boundaries), max_b, state_size), dtype=torch.int8)
        boundary_mask = torch.zeros((len(boundaries), max_b), dtype=torch.bool)
        for i, b in enumerate(boundaries):
            boundary_t[i, : b.size(0)] = b.to(torch.int8)
            boundary_mask[i, : b.size(0)] = True
    else:
        parent_t = torch.empty((0, state_size), dtype=torch.int8)
        good_t = torch.empty((0, state_size), dtype=torch.int8)
        boundary_t = torch.empty((0, boundary_size, state_size), dtype=torch.int8)
        boundary_mask = torch.empty((0, boundary_size), dtype=torch.bool)

    out = {
        "parents": parent_t,
        "goods": good_t,
        "boundaries": boundary_t,
        "boundary_mask": boundary_mask,
        "pid": torch.tensor(meta_pid, dtype=torch.int32),
        "depth": torch.tensor(meta_depth, dtype=torch.int32),
        "teacher_len": torch.tensor(meta_teacher_len, dtype=torch.int32),
        "narrow_len": torch.tensor(meta_narrow_len, dtype=torch.int32),
        "boundary_len": torch.tensor(meta_boundary_len, dtype=torch.int32),
        "meta": {
            "checkpoint": str(checkpoint),
            "teacher_csv": str(teacher_csv),
            "beam": beam,
            "boundary_size": boundary_size,
            "n_pids_requested": n_pids_requested,
            "n_teacher_verified": n_verified,
            "n_labels": len(parents),
            "n_no_eviction": n_no_eviction,
            "n_teacher_survived": n_teacher_survived,
            "n_narrow_good_enough": n_narrow_good_enough,
            "elapsed_s": round(elapsed_s, 2),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--teacher-csv", required=True, type=Path,
                    help="Verified strong paths to use as the teacher.")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--test-csv", type=Path, default=PROJECT / "data" / "test.csv")
    ap.add_argument("--puzzle-info", type=Path, default=PROJECT / "data" / "puzzle_info.json")
    ap.add_argument("--pids", default="", help="Comma-separated pid list. Overrides --top-n-pids.")
    ap.add_argument("--top-n-pids", type=int, default=50,
                    help="If --pids is empty, process the longest N teacher paths.")
    ap.add_argument("--min-teacher-len", type=int, default=0)
    ap.add_argument("--beam", type=int, default=2048)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="0 means teacher path length; otherwise cap the narrow trace.")
    ap.add_argument("--boundary-size", type=int, default=64,
                    help="Number of near-cutoff surviving frontier states to save.")
    ap.add_argument("--internal-batch-size", type=int, default=16384)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--keep-if-narrow-not-worse", action="store_true",
                    help="By default, drop labels when the narrow beam solves no longer "
                         "than the teacher. Keep them anyway with this flag.")
    ap.add_argument("--all-evictions", action="store_true",
                    help="Emit every teacher-child eviction instead of only the first one.")
    ap.add_argument("--force-teacher-prefix", action="store_true",
                    help="After each step, inject the teacher child into the frontier if "
                         "it was evicted. This creates labels for later teacher-path cuts "
                         "under the assumption earlier cuts will be fixed by the delta.")
    ap.add_argument("--save-every", type=int, default=100,
                    help="Checkpoint the dataset every N processed pids. 0 disables.")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    puzzle = Megaminx.load(args.puzzle_info)
    states_by_pid = load_test_states(args.test_csv)
    teacher_paths = load_submission(args.teacher_csv)
    teacher_lengths = _load_lengths(args.teacher_csv)

    if args.pids.strip():
        pid_list = [int(p) for p in args.pids.split(",") if p.strip()]
    else:
        eligible = [pid for pid, n in teacher_lengths.items() if n >= args.min_teacher_len]
        eligible.sort(key=lambda pid: teacher_lengths[pid], reverse=True)
        pid_list = eligible[: args.top_n_pids]

    print(f"device={args.device}; loading {args.checkpoint} as {dtype}", flush=True)
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    solver = KhoruzhiiSolver(
        puzzle,
        model,
        device=args.device,
        internal_batch_size=args.internal_batch_size,
        state_dtype=torch.int8,
    )

    parents: list[torch.Tensor] = []
    goods: list[torch.Tensor] = []
    boundaries: list[torch.Tensor] = []
    meta_pid: list[int] = []
    meta_depth: list[int] = []
    meta_teacher_len: list[int] = []
    meta_narrow_len: list[int] = []
    meta_boundary_len: list[int] = []

    n_verified = 0
    n_teacher_survived = 0
    n_narrow_good_enough = 0
    n_no_eviction = 0
    t_start = time.time()

    for rank, pid in enumerate(pid_list, start=1):
        if pid not in states_by_pid or pid not in teacher_paths:
            continue
        initial = states_by_pid[pid]
        teacher = teacher_paths[pid]
        if not teacher:
            continue
        vr = verify_path(puzzle, initial, teacher)
        if not vr.ok:
            print(f"[{rank}/{len(pid_list)}] pid {pid}: teacher invalid: {vr.reason}", flush=True)
            continue
        n_verified += 1

        prefix = compute_prefix_states(initial, teacher, puzzle)
        prefix_t = torch.tensor(np.asarray(prefix, dtype=np.int64), dtype=torch.int8, device=args.device)

        max_steps = len(teacher) if args.max_steps <= 0 else min(args.max_steps, len(teacher))
        cfg = KhoruzhiiSearchConfig(
            beam_width=args.beam,
            num_steps=max_steps,
            num_attempts=1,
            internal_batch_size=args.internal_batch_size,
        )

        state0 = torch.tensor(list(initial), dtype=torch.int8, device=args.device)
        states = state0.unsqueeze(0).clone()
        states_bad_hashed = torch.empty(0, dtype=torch.int64, device=args.device)
        narrow_len: int | None = None
        candidate_labels = []

        for depth in range(1, max_steps + 1):
            parent_target = prefix_t[depth - 1]
            child_target = prefix_t[depth]
            parent_alive = _frontier_contains(states, parent_target)
            states, values, _moves, _parents_idx = solver._do_greedy_step(
                states, states_bad_hashed, args.beam
            )
            if states.numel() == 0:
                break

            solved_mask = (states == solver.V0).all(dim=1)
            if solved_mask.any() and narrow_len is None:
                narrow_len = depth

            child_alive = _frontier_contains(states, child_target)
            if parent_alive and not child_alive:
                bsz = min(args.boundary_size, states.size(0))
                # _do_greedy_step returns states sorted by score ascending, so the
                # last rows are the near-cutoff survivors.
                boundary = states[-bsz:].detach().cpu()
                candidate_labels.append((
                    parent_target.detach().cpu(),
                    child_target.detach().cpu(),
                    boundary,
                    depth,
                    bsz,
                ))
                # Keep tracing a little longer so we know whether the narrow beam
                # solved no worse than the teacher via some other path.
                if not args.all_evictions and not args.keep_if_narrow_not_worse:
                    continue
                if not args.all_evictions:
                    break

            if child_alive and depth == max_steps:
                n_teacher_survived += 1
            if args.force_teacher_prefix and not child_alive and states.numel() > 0:
                # Future labels need the teacher parent to be alive. Replace the
                # worst retained state with the teacher child; this preserves beam
                # width while simulating "earlier evictions were fixed".
                states[-1] = child_target

        if not candidate_labels:
            n_no_eviction += 1
            status = "no-eviction"
        else:
            if (
                not args.force_teacher_prefix
                and not args.keep_if_narrow_not_worse
                and narrow_len is not None
                and narrow_len <= len(teacher)
            ):
                n_narrow_good_enough += 1
                status = f"drop:narrow_len={narrow_len}"
            else:
                for parent, good, boundary, depth, bsz in candidate_labels:
                    parents.append(parent)
                    goods.append(good)
                    boundaries.append(boundary)
                    meta_pid.append(pid)
                    meta_depth.append(depth)
                    meta_teacher_len.append(len(teacher))
                    meta_narrow_len.append(-1 if narrow_len is None else narrow_len)
                    meta_boundary_len.append(bsz)
                if len(candidate_labels) == 1:
                    status = f"label@{candidate_labels[0][3]}"
                else:
                    status = f"labels={len(candidate_labels)} first@{candidate_labels[0][3]}"

        if rank <= 10 or rank % 10 == 0:
            narrow_text = "none" if narrow_len is None else str(narrow_len)
            print(f"[{rank}/{len(pid_list)}] pid={pid:4d} teacher={len(teacher):3d} "
                  f"narrow={narrow_text:>4s} {status}", flush=True)

        if args.save_every > 0 and rank % args.save_every == 0:
            _save_dataset(
                args.out,
                parents=parents,
                goods=goods,
                boundaries=boundaries,
                meta_pid=meta_pid,
                meta_depth=meta_depth,
                meta_teacher_len=meta_teacher_len,
                meta_narrow_len=meta_narrow_len,
                meta_boundary_len=meta_boundary_len,
                state_size=len(puzzle.solved_state),
                checkpoint=args.checkpoint,
                teacher_csv=args.teacher_csv,
                beam=args.beam,
                boundary_size=args.boundary_size,
                n_pids_requested=len(pid_list),
                n_verified=n_verified,
                n_no_eviction=n_no_eviction,
                n_teacher_survived=n_teacher_survived,
                n_narrow_good_enough=n_narrow_good_enough,
                elapsed_s=time.time() - t_start,
            )
            print(f"  checkpointed {len(parents)} labels -> {args.out}", flush=True)

    _save_dataset(
        args.out,
        parents=parents,
        goods=goods,
        boundaries=boundaries,
        meta_pid=meta_pid,
        meta_depth=meta_depth,
        meta_teacher_len=meta_teacher_len,
        meta_narrow_len=meta_narrow_len,
        meta_boundary_len=meta_boundary_len,
        state_size=len(puzzle.solved_state),
        checkpoint=args.checkpoint,
        teacher_csv=args.teacher_csv,
        beam=args.beam,
        boundary_size=args.boundary_size,
        n_pids_requested=len(pid_list),
        n_verified=n_verified,
        n_no_eviction=n_no_eviction,
        n_teacher_survived=n_teacher_survived,
        n_narrow_good_enough=n_narrow_good_enough,
        elapsed_s=time.time() - t_start,
    )
    print("\n=== summary ===", flush=True)
    print(f"verified teachers: {n_verified}/{len(pid_list)}", flush=True)
    print(f"labels emitted: {len(parents)}", flush=True)
    print(f"no eviction: {n_no_eviction}", flush=True)
    print(f"narrow solved no-worse, dropped: {n_narrow_good_enough}", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
