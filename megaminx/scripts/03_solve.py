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

  # Q-shortlister (m05 teacher + m23 student, alpha=2):
  python megaminx/scripts/03_solve.py \
      --checkpoint models/m05_bellman_warm/epoch_0499.pt \
      --qshort-student models/m23_q_shortlister/epoch_0499.pt \
      --qshort-alpha 2 \
      --out submissions/qshort_524k.csv \
      --beams 524288 --max-steps 150 --bf16

  NOTE: beam_lab/run_benchmark.py is benchmarking-only; it does NOT save move
  sequences. Always use 03_solve.py for any solve you might submit. (See
  megaminx_gotchas.md memory entry for the 14h-burn that established this.)
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
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission
from cayley.bfs_table import BfsTable
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.mitm_solver import MitmKhoruzhiiSolver
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _apply_rotation_to_state(state, R, R_inv):
    """Compute R · state · R_inv. apply_rotation convention: out[i] = R[state[R_inv[i]]]."""
    return tuple(R[state[R_inv[i]]] for i in range(len(state)))


def _compute_conjugation_map(R, R_inv, generators_dict, gen_names):
    """For each generator name n in the rotated frame, find the original-frame
    name whose generator equals R_inv · g_n · R. Used to translate solver paths
    found on a rotated state back to the original state's frame."""
    n = len(R)
    perm_to_name = {tuple(g): nm for nm, g in generators_dict.items()}
    out = {}
    for nm in gen_names:
        g = generators_dict[nm]
        conj = tuple(R_inv[g[R[i]]] for i in range(n))
        if conj not in perm_to_name:
            raise ValueError(f"rotation is not a symmetry: R_inv * g_{nm} * R is not a generator")
        out[nm] = perm_to_name[conj]
    return out


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
    ap.add_argument("--pid-from", type=int, default=None,
                    help="solve pids >= this (inclusive). Pair with --pid-to for batching.")
    ap.add_argument("--pid-to", type=int, default=None,
                    help="solve pids < this (exclusive). With --pid-from, restricts to "
                         "a pid range; useful for batching a big run across machines or "
                         "for incremental resume after a crash.")
    ap.add_argument("--pids", type=str, default=None,
                    help="comma-separated explicit list of pids to solve (e.g. "
                         "'490,920,671'). Mutually exclusive with --pid-from/--pid-to "
                         "and --stratified.")
    ap.add_argument("--resume", action="store_true",
                    help="if --out already exists, skip pids already present and append; "
                         "makes mid-batch crashes cheap to recover from.")
    ap.add_argument("--stratified", type=int, default=None,
                    help="K puzzles per 100-pid bucket (covers full difficulty range)")
    ap.add_argument("--strat-seed", type=int, default=0)
    ap.add_argument("--strat-buckets", type=str, default=None,
                    help="Comma-separated bucket indices (0-10) to sample from. "
                         "Default: all buckets. E.g., '7,8,9,10' for hard-tail-only "
                         "eval (16 puzzles, top-30 percent by difficulty). Useful "
                         "when full strat-5 is saturated on solve count.")
    ap.add_argument("--quantile-reduce", default=None,
                    choices=["median", "mean", "lower-q25", "lower-q10", "min"],
                    help="If set, wrap the loaded model to reduce output_dim>1 "
                         "(quantile distribution) to a scalar V via the chosen "
                         "statistic. For distributional V models like m42.")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fallback", type=Path, default=PROJECT / "data" / "pp_bfs6_fallback.csv",
                    help="default: pp_bfs6_fallback.csv (414,678 floor; same-face + BFS-d6 "
                         "on raw sample). Previous: pp_bfs5_fallback.csv (415,521).")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--fp32-state", action="store_true",
                    help="disable int8 state encoding (debugging only; int8 is the default)")
    ap.add_argument("--bfs-table", type=Path, default=None,
                    help="path to bfs_table_d*.pkl OR bfs_bytes_d*.pkl; enables window-"
                         "replacement post-processing")
    ap.add_argument("--bfs-max-window", type=int, default=None,
                    help="override max_window for BFS window replacement (default: d+1)")
    ap.add_argument("--mitm", action="store_true",
                    help="enable inline MITM in beam search — beam terminates when any "
                         "state hits the BFS shell (requires --bfs-table pointing at a "
                         "bytes-keyed BfsBytesTable). Appends the known-optimal tail.")
    # Q-shortlister mode: m23 student picks αB candidates per step, m05 (teacher) reranks.
    # The qshort machinery lives in megaminx/beam_lab/ — it's API-compatible via a thin
    # adapter below. Use this for production Q-shortlister submissions; run_benchmark.py
    # in beam_lab is benchmark-only (does NOT save move sequences).
    ap.add_argument("--qshort-student", type=Path, default=None,
                    help="path to Q-head student checkpoint (m23). Activates qshort "
                         "mode: student shortlists alpha*B candidates, teacher (--checkpoint) "
                         "reranks. Mutually exclusive with --mitm.")
    ap.add_argument("--qshort-alpha", type=float, default=2.0,
                    help="shortlist multiplier (default 2.0). Validated with "
                         "09_eval_q_recall.py - recall must be >=99%% at chosen alpha.")
    ap.add_argument("--qshort-internal-batch-size", type=int, default=16384,
                    help="internal batch size for qshort solver (default 16384).")
    ap.add_argument("--macros", type=Path, default=None,
                    help="path to a curated_macros_*.pkl. When set, the solver's action "
                         "set is extended with the macros; their net 120-perms become "
                         "additional atoms with cost = len(word). Pairs with a Q-head "
                         "trained on n_actions = n_gen + n_macros (m24 / Macro-Q "
                         "shortlister). For V-only solves, macros work with cost-aware "
                         "scoring; the qshort student's output_dim must match n_actions.")
    ap.add_argument("--policy-model", type=Path, default=None,
                    help="path to a policy-head checkpoint (m_pi). Activates Idea 4 "
                         "scoring: score(child) = V(child) + lambda * (-log pi(a|parent)). "
                         "Output_dim must equal n_gen. Mutually exclusive with --macros "
                         "(v0 keeps Idea 1 and Idea 4 separate).")
    ap.add_argument("--lambda-policy", type=float, default=0.0,
                    help="weight on the policy penalty term (default 0.0 = policy off). "
                         "Sweep on strat-5 over {0.05, 0.1, 0.2, 0.5}.")
    ap.add_argument("--phs-cumulative", action="store_true",
                    help="PHS cumulative scoring: instead of the memoryless local "
                         "penalty (lambda * -log pi for one step), accumulate -log pi "
                         "along the whole path: score(child) = V(child) + lambda * "
                         "sum_t(-log pi(a_t|s_t)). Changes which partial paths survive "
                         "over depth. Requires --policy-model + --lambda-policy>0 + "
                         "--qshort-student (same beam_lab path as the local penalty). "
                         "Sweep --lambda-policy over {0.01,0.03,0.05,0.1,0.2}.")
    ap.add_argument("--cutoff-model", type=Path, default=None,
                    help="Optional scalar beam-retention head. V first overgenerates "
                         "top cutoff_pool_mult*B candidates, then this head reranks "
                         "only that near-cutoff pool.")
    ap.add_argument("--cutoff-lambda", type=float, default=0.0,
                    help="Weight for the cutoff head inside the overgenerated pool. "
                         "The head is z-normalized per step unless --no-cutoff-normalize.")
    ap.add_argument("--cutoff-pool-mult", type=float, default=1.0,
                    help="Overgeneration factor before cutoff reranking. 1 disables.")
    ap.add_argument("--no-cutoff-normalize", action="store_true",
                    help="Use raw cutoff-head outputs instead of per-step z-normalization.")
    ap.add_argument("--tensorrt-engine", type=Path, default=None,
                    help="path to a pre-built TensorRT engine (.ts) for the teacher "
                         "model. Replaces the eager teacher with the engine; engine is "
                         "shape-fixed to internal_batch_size. Build with "
                         "megaminx/beam_lab/export_tensorrt.py. Requires --qshort-student "
                         "(only the qshort path is wired through beam_lab/QShortlister).")
    # Multi-seed beam ensemble: run beam search N times with different RNG seeds,
    # take the shortest valid path. Each seed varies the dedup hash_vec, which
    # changes which duplicates survive at topk tie-breaks - small per-step
    # diversity that compounds across the search depth. Cost: N x wall.
    ap.add_argument("--ensemble-seeds", type=str, default="",
                    help="comma-separated list of seeds to ensemble (e.g. '0,1,2'). "
                         "Empty = single-seed (current behavior). When set, the solver is "
                         "rebuilt per seed and each puzzle is solved by every seed; "
                         "shortest valid path wins. Wall scales linearly with N seeds.")
    # Symmetry ensemble (rotations.npy from 19_symmetry_v3.py): for each pid,
    # apply K-1 random rotations + identity, solve each rotated puzzle, translate
    # paths back to the original frame, take shortest. Distance-preserving by
    # construction: the 360 rotations form A5 x C6 acting on the 120-state space.
    ap.add_argument("--sym-ensemble", type=int, default=0,
                    help="number of symmetry rotations per pid (0=disabled, 1=identity-only). "
                         "K=4-8 realistic. Wall scales K x len(--ensemble-seeds).")
    ap.add_argument("--sym-rotations", type=Path, default=PROJECT / "data" / "rotations.npy",
                    help="path to (N, state_size) int8 rotations array. Default: 360-element "
                         "icosahedral x C6 group from 19_symmetry_v3.py.")
    ap.add_argument("--sym-seed", type=int, default=0,
                    help="RNG seed for picking sym-ensemble rotations.")
    args = ap.parse_args()

    beams = _parse_int_list(args.beams)
    max_steps_list = _parse_int_list(args.max_steps)
    if len(beams) != len(max_steps_list):
        ap.error(f"--beams ({len(beams)}) and --max-steps ({len(max_steps_list)}) "
                 f"must have matching lengths")

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    all_ids = sorted(states)
    if args.pids is not None:
        if args.stratified is not None or args.pid_from is not None or args.pid_to is not None:
            ap.error("--pids is mutually exclusive with --stratified / --pid-from / --pid-to")
        explicit = [int(x) for x in args.pids.split(",") if x.strip()]
        all_ids_set = set(all_ids)
        missing = [p for p in explicit if p not in all_ids_set]
        if missing:
            ap.error(f"--pids contains pids not in test.csv: {missing[:5]}")
        solve_ids = sorted(explicit)
        print(f"pids: {len(solve_ids)} explicit pids ({solve_ids[:5]}{'...' if len(solve_ids) > 5 else ''})")
    elif args.stratified is not None:
        import random
        rng = random.Random(args.strat_seed)
        buckets: dict[int, list[int]] = {}
        for pid in all_ids:
            buckets.setdefault(pid // 100, []).append(pid)
        if args.strat_buckets:
            wanted = set(int(b) for b in args.strat_buckets.split(",") if b.strip())
        else:
            wanted = set(buckets.keys())
        picked: list[int] = []
        for b in sorted(buckets):
            if b not in wanted:
                continue
            picked.extend(sorted(rng.sample(buckets[b], min(args.stratified, len(buckets[b])))))
        solve_ids = picked
        n_used = sum(1 for b in buckets if b in wanted)
        print(f"stratified: {len(solve_ids)} pids across {n_used} buckets "
              f"(seed={args.strat_seed}, buckets={sorted(b for b in buckets if b in wanted)})")
    elif args.pid_from is not None or args.pid_to is not None:
        lo = args.pid_from if args.pid_from is not None else 0
        hi = args.pid_to if args.pid_to is not None else max(all_ids) + 1
        solve_ids = [p for p in all_ids if lo <= p < hi]
        print(f"pid range: [{lo}, {hi}) -> {len(solve_ids)} pids")
    elif args.limit is not None:
        solve_ids = all_ids[: args.limit]
    else:
        solve_ids = all_ids

    # Resume mode: if --out exists, load already-solved pids and skip them.
    already_done: dict[int, list[str]] = {}
    if args.resume and args.out.exists():
        already_done = load_submission(args.out)
        skip_attempted = [p for p in solve_ids if p in already_done]
        solve_ids = [p for p in solve_ids if p not in already_done]
        print(f"resume: {len(already_done)} rows in existing {args.out.name}, "
              f"skipping {len(skip_attempted)} already-attempted pids; "
              f"{len(solve_ids)} remaining")

    fallback = load_submission(args.fallback)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    if args.chunk_size is not None:
        def _set_inference_chunk_size(obj, chunk_size: int, seen: set[int] | None = None) -> None:
            if seen is None:
                seen = set()
            if obj is None or id(obj) in seen:
                return
            seen.add(id(obj))
            base_obj = getattr(obj, "_orig_mod", obj)
            if hasattr(base_obj, "inference_chunk_size"):
                base_obj.inference_chunk_size = chunk_size
            for child_name in ("base", "delta"):
                _set_inference_chunk_size(getattr(base_obj, child_name, None), chunk_size, seen)

        _set_inference_chunk_size(model, args.chunk_size)

    if args.quantile_reduce is not None:
        # Wrap a quantile-output model (output_dim > 1) into a scalar-V model
        # by reducing across the quantile axis. The wrapper preserves the
        # interface KhoruzhiiSolver expects.
        import torch.nn as _nn
        class _QuantileReduceWrapper(_nn.Module):
            def __init__(self, base, mode):
                super().__init__()
                self.base = base
                self.mode = mode
            def forward(self, x):
                q = self.base(x)
                if self.mode == "median":
                    return q[:, q.shape[1] // 2]
                if self.mode == "mean":
                    return q.mean(dim=1)
                if self.mode == "lower-q25":
                    return q[:, q.shape[1] // 4]
                if self.mode == "lower-q10":
                    return q[:, max(q.shape[1] // 10, 0)]
                if self.mode == "min":
                    return q.min(dim=1).values
                raise ValueError(f"unknown quantile-reduce: {self.mode}")
        print(f"  wrapping model with quantile reduction: {args.quantile_reduce}")
        model = _QuantileReduceWrapper(model, args.quantile_reduce).to(args.device).eval()

    state_dtype = torch.int32 if args.fp32_state else torch.int8

    bfs_table = None
    if args.bfs_table is not None:
        t_load = time.time()
        if "bytes" in args.bfs_table.name:
            bfs_table = BfsBytesTable.load(args.bfs_table)
        else:
            bfs_table = BfsTable.load(args.bfs_table)
        print(f"loaded BFS table {args.bfs_table.name}: {len(bfs_table.table):,} states, "
              f"max_depth={bfs_table.max_depth} ({time.time() - t_load:.1f}s)")

    # Auto-detect Q-function model from output_dim. KhoruzhiiSolver supports both V
    # (output_dim=1, scalar distance) and Q (output_dim=n_gen, neighbor scores).
    base = getattr(model, "_orig_mod", model)
    detected_q = getattr(base, "output_dim", 1) > 1
    if detected_q:
        print(f"auto-detected Q-function model (output_dim={base.output_dim}); "
              f"beam search will use Q-path (one fwd per parent)")

    if args.mitm and args.qshort_student is not None:
        ap.error("--mitm and --qshort-student are mutually exclusive")

    # Optional Q-shortlister student is loaded ONCE (model weights shared across
    # ensemble seeds; only hash_vec differs per seed).
    student = None
    teacher_for_qshort = model  # may be swapped for TRT engine below
    _BLConfig = None
    QShortlisterSolver = None
    if args.qshort_student is not None:
        if detected_q:
            ap.error("--qshort-student is incompatible with a Q-function teacher "
                     "(--checkpoint must be a V model; the student IS the Q-head)")
        sys.path.insert(0, str(PROJECT / "beam_lab"))
        from beam_search import (  # noqa: E402  (path injection above)
            KhoruzhiiSearchConfig as _BLConfig,
            setup_model_for_inference as _bl_setup_inference,
        )
        from beam_search_qshort import QShortlisterSolver  # noqa: E402
        student = load_model_checkpoint(args.qshort_student, device=args.device, dtype=dtype)
        student_base = getattr(student, "_orig_mod", student)
        student_dim = getattr(student_base, "output_dim", 1)
        if student_dim <= 1:
            ap.error(f"--qshort-student must be a Q-head (output_dim > 1); "
                     f"got output_dim={student_dim}")
        _bl_setup_inference(model)
        _bl_setup_inference(student)

        if args.tensorrt_engine is not None:
            # Replace eager teacher with the TRT engine. The engine is shape-fixed
            # to internal_batch_size — qshort's _model_predict already calls with
            # pad_to_batch_size=True so this works.
            print(f"loading TensorRT engine for teacher: {args.tensorrt_engine}")
            import torch_tensorrt  # noqa: F401  (registers TRT runtime ops)
            t0 = time.time()
            teacher_for_qshort = torch.load(str(args.tensorrt_engine), weights_only=False)
            teacher_for_qshort.eval()
            # Smoke: one forward at batch_size to confirm shape & device match
            dummy = torch.zeros(
                (args.qshort_internal_batch_size, 120),
                dtype=torch.int64, device=args.device,
            )
            with torch.inference_mode():
                _ = teacher_for_qshort(dummy)
            print(f"  engine loaded + warmed in {time.time() - t0:.1f}s")
    elif args.tensorrt_engine is not None:
        ap.error("--tensorrt-engine currently requires --qshort-student "
                 "(only the qshort path uses beam_lab where the engine wraps cleanly)")

    # Optional policy model for Idea 4 (Policy-guided V + π).
    policy_model_obj = None
    if args.policy_model is not None:
        if args.lambda_policy <= 0:
            print(f"WARNING: --policy-model set but --lambda-policy={args.lambda_policy} "
                  f"is non-positive; policy term disabled.", file=sys.stderr)
        if args.macros is not None:
            ap.error("--policy-model and --macros are not compatible in v0. "
                     "Pick one mechanism per solve.")
        policy_model_obj = load_model_checkpoint(args.policy_model, device=args.device, dtype=dtype)
        pol_base = getattr(policy_model_obj, "_orig_mod", policy_model_obj)
        pol_dim = getattr(pol_base, "output_dim", 1)
        n_gen_check = len(puzzle.move_names)
        if pol_dim != n_gen_check:
            ap.error(f"--policy-model output_dim={pol_dim} != n_gen={n_gen_check}; "
                     f"policy must output one logit per primitive generator.")
        print(f"loaded policy model from {args.policy_model.name}: "
              f"output_dim={pol_dim}, lambda={args.lambda_policy}, "
              f"phs_cumulative={args.phs_cumulative}")
    if args.phs_cumulative and (policy_model_obj is None or args.lambda_policy <= 0):
        ap.error("--phs-cumulative requires --policy-model and --lambda-policy>0 "
                 "(the cumulative term is lambda * sum of -log pi along the path).")

    cutoff_model_obj = None
    if args.cutoff_model is not None:
        if args.mitm or detected_q:
            ap.error("--cutoff-model requires a scalar V teacher (no mitm, no Q teacher)")
        if args.cutoff_lambda == 0.0 or args.cutoff_pool_mult <= 1.0:
            print(
                "WARNING: --cutoff-model set but cutoff_lambda=0 or cutoff_pool_mult<=1; "
                "cutoff rerank is effectively disabled.",
                file=sys.stderr,
            )
        cutoff_model_obj = load_model_checkpoint(args.cutoff_model, device=args.device, dtype=dtype)
        cutoff_base = getattr(cutoff_model_obj, "_orig_mod", cutoff_model_obj)
        if getattr(cutoff_base, "output_dim", 1) != 1:
            ap.error("--cutoff-model must be scalar output_dim=1")
        if args.chunk_size is not None and hasattr(cutoff_base, "inference_chunk_size"):
            cutoff_base.inference_chunk_size = args.chunk_size
        if args.qshort_student is not None:
            _bl_setup_inference(cutoff_model_obj)
        print(
            f"loaded cutoff model from {args.cutoff_model.name}: "
            f"lambda={args.cutoff_lambda}, pool_mult={args.cutoff_pool_mult}, "
            f"normalize={not args.no_cutoff_normalize}"
        )

    # Optional macro library — extends action set for the V-path / qshort path.
    # m24 (Macro-Q shortlister) needs n_actions = n_gen + n_macros to match.
    macros_for_solver: list[tuple[list[int], list[str]]] | None = None
    if args.macros is not None:
        import pickle as _pickle
        with open(args.macros, "rb") as f:
            _md = _pickle.load(f)
        _macro_entries = _md["macros"] if isinstance(_md, dict) and "macros" in _md else _md
        macros_for_solver = []
        for _m in _macro_entries:
            _perm = _m["perm"]
            if hasattr(_perm, "tolist"):
                _perm = _perm.tolist()
            _word = _m["word_names"]
            if isinstance(_word, str):
                _word = _word.split()
            macros_for_solver.append((list(_perm), list(_word)))
        print(f"loaded {len(macros_for_solver)} macros from {args.macros}")
        if args.mitm:
            ap.error("--macros and --mitm are not compatible (mitm does not "
                     "yet route macros through the BFS shell terminator).")

    class _QShortAdapter:
        """Adapt beam_lab's 4-tuple solver API to cayley's 3-tuple API."""
        def __init__(self, inner, internal_batch_size: int):
            self._inner = inner
            self._ibs = internal_batch_size

        def solve(self, state, cfg):
            bl_cfg = _BLConfig(
                beam_width=cfg.beam_width,
                num_steps=cfg.num_steps,
                num_attempts=cfg.num_attempts,
                internal_batch_size=self._ibs,
            )
            found, plen, names, _prof = self._inner.solve(state, bl_cfg)
            return found, plen, names

    def _build_solver(seed: int):
        if args.mitm:
            if not isinstance(bfs_table, BfsBytesTable):
                ap.error("--mitm requires --bfs-table pointing to a bfs_bytes_d*.pkl "
                         "(BfsBytesTable, not BfsTable)")
            if detected_q:
                ap.error("--mitm + Q-function not yet wired through MitmKhoruzhiiSolver")
            return MitmKhoruzhiiSolver(
                puzzle, model, mitm_table=bfs_table, device=args.device,
                state_dtype=state_dtype,
            )
        elif args.qshort_student is not None:
            inner = QShortlisterSolver(
                puzzle, teacher=teacher_for_qshort, student=student,
                device=args.device,
                internal_batch_size=args.qshort_internal_batch_size,
                random_seed=seed, state_dtype=state_dtype,
                alpha=args.qshort_alpha,
                pad_to_batch_size=(args.tensorrt_engine is not None),
                macros=macros_for_solver,
                policy_model=policy_model_obj,
                lambda_policy=args.lambda_policy,
                phs_cumulative=args.phs_cumulative,
                cutoff_model=cutoff_model_obj,
                cutoff_lambda=args.cutoff_lambda,
                cutoff_pool_mult=args.cutoff_pool_mult,
                cutoff_normalize=not args.no_cutoff_normalize,
            )
            return _QShortAdapter(inner, args.qshort_internal_batch_size)
        else:
            # Policy works on the cayley.khoruzhii_search.KhoruzhiiSolver path too,
            # but currently that V-only solver doesn't accept a policy_model parameter.
            # If users want policy in V-only mode, the right path is to use beam_lab
            # via --qshort-student (which is the production stack anyway).
            if policy_model_obj is not None:
                ap.error("--policy-model currently requires --qshort-student "
                         "(beam_lab path; the cayley V-only solver doesn't yet accept "
                         "policy_model parameter).")
            return KhoruzhiiSolver(
                puzzle, model, device=args.device, state_dtype=state_dtype,
                use_q_function=detected_q, random_seed=seed,
                macros=macros_for_solver,
                cutoff_model=cutoff_model_obj,
                cutoff_lambda=args.cutoff_lambda,
                cutoff_pool_mult=args.cutoff_pool_mult,
                cutoff_normalize=not args.no_cutoff_normalize,
            )

    # Parse ensemble seeds. Empty -> single-seed (current behavior, seed=0).
    ensemble_seeds: list[int] = (
        [int(s) for s in args.ensemble_seeds.split(",") if s.strip()]
        if args.ensemble_seeds.strip() else [0]
    )
    solvers = [_build_solver(s) for s in ensemble_seeds]
    if args.mitm:
        print(f"using MITM solver (terminates on shell of depth {bfs_table.max_depth})")
    elif args.qshort_student is not None:
        print(f"using Q-shortlister: teacher={args.checkpoint.name}, "
              f"student={args.qshort_student.name}, α={args.qshort_alpha}, "
              f"internal_bs={args.qshort_internal_batch_size}")
    if len(solvers) > 1:
        print(f"ensemble: {len(solvers)} solvers with seeds={ensemble_seeds}; "
              f"per-puzzle wall scales {len(solvers)}x")
    # Backwards-compat: single-solver paths use `solver` directly.
    solver = solvers[0]

    print(f"beams={beams}  max_steps={max_steps_list}  niss={args.niss}  "
          f"state_dtype={state_dtype}  num_attempts={args.num_attempts}")

    # Load symmetry rotations if requested and pick K of them (always include identity).
    sym_rotations: list[tuple[tuple[int, ...], tuple[int, ...], dict[str, str]]] = []
    if args.sym_ensemble > 0:
        rot_arr = np.load(args.sym_rotations)
        if rot_arr.shape[1] != len(puzzle.solved_state):
            ap.error(f"--sym-rotations cols={rot_arr.shape[1]} != state_size {len(puzzle.solved_state)}")
        K = args.sym_ensemble
        if K > rot_arr.shape[0]:
            ap.error(f"--sym-ensemble {K} > #rotations {rot_arr.shape[0]}")
        identity = np.arange(rot_arr.shape[1], dtype=rot_arr.dtype)
        identity_idx = None
        for i in range(rot_arr.shape[0]):
            if np.array_equal(rot_arr[i], identity):
                identity_idx = i; break
        if identity_idx is None:
            ap.error("rotations file missing identity row")
        rng_sym = np.random.default_rng(args.sym_seed)
        other_idxs = [i for i in range(rot_arr.shape[0]) if i != identity_idx]
        chosen_idxs = [identity_idx]
        if K > 1:
            chosen_idxs += list(rng_sym.choice(other_idxs, size=K - 1, replace=False).tolist())
        for idx in chosen_idxs:
            R = tuple(int(x) for x in rot_arr[idx])
            R_inv = tuple(int(x) for x in np.argsort(rot_arr[idx]))
            cm = _compute_conjugation_map(R, R_inv, puzzle.generators, puzzle.move_names)
            sym_rotations.append((R, R_inv, cm))
        print(f"sym-ensemble: {K} rotations selected from {rot_arr.shape[0]} "
              f"(idxs={chosen_idxs[:8]}{'...' if len(chosen_idxs) > 8 else ''})")

    # Decide output layout. In "batch mode" (--pid-from / --pid-to / --pids / --resume
    # appending) we write ONLY the attempted pids, no fallback-fill — the caller merges
    # batches. In normal mode, every pid gets a row (model solution or fallback).
    batch_mode = (args.pid_from is not None or args.pid_to is not None
                  or args.pids is not None)

    stats = {"solved_by_model": 0, "fallback": 0, "total_moves": 0}
    pass_solves = [0] * len(beams)
    per_puzzle: list[tuple[int, str, int | None, int, int, int]] = []
    t0 = time.time()
    solve_ids_set = set(solve_ids)
    n_attempted = 0

    # Open CSV once, write header if new, flush each row immediately. Append mode if
    # resuming; write mode otherwise.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_exists_nonempty = args.out.exists() and args.out.stat().st_size > 0
    open_mode = "a" if (args.resume and out_exists_nonempty) else "w"
    out_f = open(args.out, open_mode, newline="")
    writer = csv.writer(out_f)
    if open_mode == "w":
        writer.writerow(["initial_state_id", "path"])
        out_f.flush()

    # In non-batch mode (full / limit / stratified), we want every pid in the output —
    # fallback-fill for pids we don't attempt (and for already-done rows on resume).
    # In batch mode we omit fallback-fill entirely.
    if not batch_mode:
        # Write fallback rows for skipped-on-resume pids first (they're already in the
        # existing file if resuming; we only need to write the ones not in solve_ids).
        for pid in all_ids:
            if pid in already_done or pid in solve_ids_set:
                continue
            fb_path = full_post_process(fallback[pid])
            stats["fallback"] += 1
            stats["total_moves"] += len(fb_path)
            writer.writerow([pid, ".".join(fb_path)])
        out_f.flush()

    iter_ids = solve_ids  # only attempt these; the rest already recorded above
    for pid in iter_ids:
        state = states[pid]
        model_path: list[str] | None = None
        solved_pass = -1

        n_attempted += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Iterate (rotation, solver) pairs. Without sym-ensemble there's a single
        # "no-rotation" entry; with K rotations we always include identity.
        rot_iter = sym_rotations if sym_rotations else [(None, None, None)]
        ens_paths: list[tuple[list[str], int]] = []
        for R, R_inv, conj_map in rot_iter:
            s_to_solve = state if R is None else _apply_rotation_to_state(state, R, R_inv)
            for s_solver in solvers:
                p, pidx = _solve_escalating(
                    puzzle, s_solver, s_to_solve, beams, max_steps_list,
                    args.num_attempts, args.niss, bfs_table, args.bfs_max_window,
                )
                if p is None:
                    continue
                if R is not None:
                    p_orig = [conj_map[m] for m in p]
                    if not verify_path(puzzle, state, p_orig).ok:
                        # Rotation-translation produced an invalid path — silently skip.
                        # If this fires, math/data is wrong; the no-rotation pass below
                        # will still cover the pid.
                        continue
                    p = p_orig
                ens_paths.append((p, pidx))
        if ens_paths:
            model_path, solved_pass = min(ens_paths, key=lambda x: len(x[0]))
        else:
            model_path, solved_pass = None, -1
        if solved_pass >= 0:
            pass_solves[solved_pass] += 1

        fb_path = full_post_process(fallback[pid])
        candidates: list[tuple[list[str], str]] = []
        if model_path is not None:
            candidates.append((model_path, "solved_by_model"))
        if not batch_mode:
            # In batch mode we don't mix fallback in — the downstream merger does it.
            candidates.append((fb_path, "fallback"))
        if candidates:
            best_path, best_source = min(candidates, key=lambda x: len(x[0]))
        else:
            # batch_mode + model didn't solve: emit no row for this pid (caller merges fallback).
            best_path, best_source = None, None
        if best_path is not None:
            stats[best_source] += 1
            stats["total_moves"] += len(best_path)
            writer.writerow([pid, ".".join(best_path)])
            out_f.flush()
        per_puzzle.append((
            pid, best_source or "none",
            len(model_path) if model_path is not None else None,
            len(fb_path), len(best_path) if best_path is not None else -1, solved_pass,
        ))

        if n_attempted % 20 == 0 or pid == iter_ids[-1]:
            elapsed = time.time() - t0
            rate = n_attempted / elapsed if elapsed > 0 else 0
            pass_str = "/".join(str(n) for n in pass_solves)
            print(f"  pid={pid:4d} attempts={n_attempted}/{len(iter_ids)} "
                  f"total_moves={stats['total_moves']:,} "
                  f"(model:{stats['solved_by_model']} fb:{stats['fallback']} "
                  f"passes:{pass_str}) {elapsed:.1f}s ({rate:.2f} p/s)", flush=True)

    out_f.close()

    print(f"wrote {args.out}")
    print(f"  stats: {stats}")
    print(f"  pass_solves: {dict(zip(beams, pass_solves))}")
    if not batch_mode:
        # Only verify in full-submission mode — batch output is partial by design.
        report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
        print(f"  verify: {report.n_valid}/{report.n_total} valid, total {report.total_moves:,}")

    if args.stratified is not None or args.limit is not None or batch_mode:
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
    return 0 if (batch_mode or report.all_valid) else 1


if __name__ == "__main__":
    sys.exit(main())
