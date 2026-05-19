"""T1.6 — Simulated Annealing / Hill Climbing on completed paths.

Iterative local search to shorten BFS-d6-post-processed paths. Three mutation
operators that complement (not duplicate) BFS-d6 window replacement:

  1. CommutingSwap (cheap, length-preserving): pick a random adjacent pair
     (path[i], path[i+1]); if they commute as permutations, swap them, then
     re-run full_post_process. Net effect: drifts moves through commuting
     neighbors, occasionally bringing inverse pairs adjacent and triggering
     a cancellation cascade that BFS-d6 windows missed. Empirically weak on
     paths already at BFS-d6 fixpoint; useful as cheap exploration.

  2. TailResolve (expensive, can shorten): pick a random position i in
     [N//3, N-K_min]; re-solve path[i:] with a fresh beam. Generalization of
     T2.2 (which only re-solves last K moves) to arbitrary positions.

  3. MacroInsert (expensive, FMC-style, NOVEL vs T2.2): insert a random
     commutator C from data/commutator_table.pkl at a random position i,
     then re-solve the resulting state's suffix with a fresh beam. The
     inserted macro creates a "residue" the model must absorb; if the new
     residue is closer-to-solved than the natural mid-path state, the new
     suffix is shorter and total path < N. This is the FMC competitors'
     mechanism: insert a structured macro, solve to absorb its effect.
     Different from T2.2 because C != identity changes the residue
     structure, opening shortcuts T2.2 cannot find.

Acceptance: hill-climbing by default (only accept improvements). SA mode
(probabilistic) optional via --mode sa. Verifier guards correctness on every
accepted change.

Distinct from T2.2:
  - T2.2 only truncates the last K moves (fixed K options, end-anchored).
  - T1.6 also tries random-position truncations + structural shifts via
    commuting swaps. Designed to find wins T2.2 missed.

Usage:
  # Smoke on top 20 longest pids:
  .venv/Scripts/python.exe megaminx/scripts/27_path_sa.py \\
      --base megaminx/submissions/merge_plus_sym8_top20.csv \\
      --out  megaminx/submissions/path_sa_top20.csv \\
      --top 20 --n-iter 30 --beam 65536 --bf16

  # Full-1001 after T2.2 sweep finishes:
  .venv/Scripts/python.exe megaminx/scripts/27_path_sa.py \\
      --base megaminx/submissions/tail_resolve_full1001.csv \\
      --out  megaminx/submissions/sa_full1001.csv \\
      --n-iter 20 --beam 65536 --bf16
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver
from cayley.search import load_model_checkpoint
from cayley.verify import load_submission, load_test_states, verify_path
from megaminx.bfs_bytes import BfsBytesTable
from megaminx.post_process import full_post_process
from megaminx.puzzle import Megaminx


def compute_commuting_pairs(puzzle: Megaminx) -> dict[tuple[str, str], bool]:
    """Return {(a, b): commutes} for all ordered pairs of distinct generators.

    Two generators commute iff g_a . g_b = g_b . g_a as permutations. For
    megaminx: all opposite-face pairs commute; many same-face pairs trivially
    commute (e.g., U with -U). The dict is symmetric and indexed by name.
    """
    move_names = list(puzzle.move_names)
    gens = {n: np.array(puzzle.generators[n], dtype=np.int64) for n in move_names}
    pairs: dict[tuple[str, str], bool] = {}
    for a in move_names:
        for b in move_names:
            if a == b:
                continue
            ab = gens[a][gens[b]]
            ba = gens[b][gens[a]]
            pairs[(a, b)] = bool(np.array_equal(ab, ba))
    return pairs


# =================== Mutation operators ===================


def op_commuting_swap(
    path: list[str],
    initial_state,
    puzzle: Megaminx,
    bfs_table,
    bfs_max_window: int,
    commutes: dict[tuple[str, str], bool],
    rng: random.Random,
) -> list[str] | None:
    """Pick a random adjacent pair; if commuting, swap and re-post-process.

    Returns the post-processed candidate or None if no swap found / no change.
    The candidate may be SHORTER (cancellation triggered) or same length.
    """
    N = len(path)
    if N < 2:
        return None
    # Find indices where adjacent moves commute.
    swappable = [i for i in range(N - 1) if commutes.get((path[i], path[i + 1]), False)]
    if not swappable:
        return None
    i = rng.choice(swappable)
    cand = list(path)
    cand[i], cand[i + 1] = cand[i + 1], cand[i]
    cand = full_post_process(cand, puzzle=puzzle, bfs_table=bfs_table, max_window=bfs_max_window)
    if cand == path:
        return None
    return cand


def op_macro_insert(
    path: list[str],
    initial_state,
    puzzle: Megaminx,
    solver: KhoruzhiiSolver,
    bfs_table,
    bfs_max_window: int,
    macros: list[list[str]],
    beam: int,
    K_max: int,
    rng: random.Random,
) -> list[str] | None:
    """FMC-style: insert a random commutator at a random position, re-solve suffix.

    new_path = path[:i] + macro + new_suffix_re_solved
    Accept if total len < original len.

    `macros`: list of pre-converted commutator words (lists of generator-name strings).
    """
    N = len(path)
    if N < 5 or not macros:
        return None
    i = rng.randint(1, N - 1)
    macro = list(rng.choice(macros))
    # State after path[:i] + macro
    state_pre = puzzle.apply_path(initial_state, path[:i])
    state_after_macro = puzzle.apply_path(state_pre, macro)
    if puzzle.is_solved(state_after_macro):
        # Lucky — macro completed the solve. Total = i + len(macro).
        cand = list(path[:i]) + macro
        if len(cand) >= N:
            return None
        return cand
    # Re-solve from state_after_macro to solved.
    max_steps = K_max + 10
    cfg = KhoruzhiiSearchConfig(beam_width=beam, num_steps=max_steps, num_attempts=1)
    found, _, raw_suffix = solver.solve(state_after_macro, cfg)
    if not found:
        return None
    new_suffix = full_post_process(raw_suffix, puzzle=puzzle, bfs_table=bfs_table, max_window=bfs_max_window)
    cand = list(path[:i]) + macro + list(new_suffix)
    # Final boundary pass — catches cancellations at prefix/macro and macro/suffix joins.
    cand = full_post_process(cand, puzzle=puzzle, bfs_table=bfs_table, max_window=bfs_max_window)
    if len(cand) >= N:
        return None
    return cand


def op_tail_resolve(
    path: list[str],
    initial_state,
    puzzle: Megaminx,
    solver: KhoruzhiiSolver,
    bfs_table,
    bfs_max_window: int,
    beam: int,
    K_min: int,
    K_max: int,
    rng: random.Random,
) -> list[str] | None:
    """Pick a random position i in [N//3, N-K_min]; re-solve M[i:] with a fresh beam.

    Returns the candidate (M[:i] + new_tail) if shorter, else None.
    """
    N = len(path)
    if N < K_min + 5:
        return None
    i_min = max(1, N // 3)
    i_max = N - K_min
    if i_min > i_max:
        return None
    i = rng.randint(i_min, i_max)
    K_target = N - i  # how many moves currently solve from position i
    state_i = puzzle.apply_path(initial_state, path[:i])
    if puzzle.is_solved(state_i):
        # path was over-solving; just return the prefix
        return list(path[:i])
    max_steps = min(K_max + 20, K_target + 10)
    cfg = KhoruzhiiSearchConfig(beam_width=beam, num_steps=max_steps, num_attempts=1)
    found, _, raw = solver.solve(state_i, cfg)
    if not found:
        return None
    new_tail = full_post_process(raw, puzzle=puzzle, bfs_table=bfs_table, max_window=bfs_max_window)
    cand = list(path[:i]) + list(new_tail)
    # Final boundary pass — catches cancellations at prefix/tail join.
    cand = full_post_process(cand, puzzle=puzzle, bfs_table=bfs_table, max_window=bfs_max_window)
    if len(cand) >= N:
        return None
    return cand


# =================== SA / HC core ===================


def sa_search(
    initial_path: list[str],
    initial_state,
    puzzle: Megaminx,
    solver: KhoruzhiiSolver,
    bfs_table,
    bfs_max_window: int,
    commutes: dict[tuple[str, str], bool],
    macros: list[list[str]],
    n_iter: int,
    beam: int,
    K_min: int,
    K_max: int,
    op_weights: dict[str, float],
    accept_mode: str,
    T0: float,
    T_end: float,
    rng: random.Random,
    verbose: bool,
) -> tuple[list[str], dict]:
    """Iterative local search. Returns (best_path, stats_dict).

    stats_dict: {n_accepted, n_op_calls, n_op_accepts, best_len, initial_len}
    """
    cur = list(initial_path)
    cur_len = len(cur)
    best = list(cur)
    best_len = cur_len
    n_op_calls = {op: 0 for op in op_weights}
    n_op_accepts = {op: 0 for op in op_weights}
    n_accepted = 0

    op_names = list(op_weights.keys())
    op_w = [op_weights[n] for n in op_names]

    for it in range(n_iter):
        # Linear cooling.
        T = T0 + (T_end - T0) * (it / max(n_iter - 1, 1))
        op = rng.choices(op_names, weights=op_w, k=1)[0]
        n_op_calls[op] += 1

        if op == "commuting_swap":
            cand = op_commuting_swap(cur, initial_state, puzzle, bfs_table,
                                      bfs_max_window, commutes, rng)
        elif op == "tail_resolve":
            cand = op_tail_resolve(cur, initial_state, puzzle, solver, bfs_table,
                                    bfs_max_window, beam, K_min, K_max, rng)
        elif op == "macro_insert":
            cand = op_macro_insert(cur, initial_state, puzzle, solver, bfs_table,
                                    bfs_max_window, macros, beam, K_max, rng)
        else:
            cand = None

        if cand is None:
            continue

        delta = len(cand) - cur_len
        if accept_mode == "hc":
            accept = delta < 0
        else:  # sa
            if delta < 0:
                accept = True
            else:
                p = math.exp(-delta / max(T, 1e-9))
                accept = rng.random() < p

        if accept:
            if not verify_path(puzzle, initial_state, cand).ok:
                continue
            cur = cand
            cur_len = len(cur)
            n_accepted += 1
            n_op_accepts[op] += 1
            if cur_len < best_len:
                best = list(cur)
                best_len = cur_len
                if verbose:
                    print(f"      iter={it:3d} op={op:<14s} -> len {cur_len} (best!)")

    return best, {
        "n_accepted": n_accepted,
        "n_op_calls": n_op_calls,
        "n_op_accepts": n_op_accepts,
        "initial_len": len(initial_path),
        "best_len": best_len,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="megaminx/models/m05_bellman_warm/epoch_0499.pt")
    ap.add_argument("--bfs-table", default="megaminx/data/bfs_bytes_d6.pkl")
    ap.add_argument("--bfs-max-window", type=int, default=12)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--K-min", type=int, default=15)
    ap.add_argument("--K-max", type=int, default=60)
    ap.add_argument("--n-iter", type=int, default=20,
                    help="SA iterations per pid")
    ap.add_argument("--top", type=int, default=0,
                    help="Process only top-N longest-path pids (0 = all 1001)")
    ap.add_argument("--pids", default="",
                    help="Optional comma-separated explicit pid list (overrides --top)")
    ap.add_argument("--mode", choices=["hc", "sa"], default="hc")
    ap.add_argument("--T0", type=float, default=2.0, help="SA initial temperature (mode=sa)")
    ap.add_argument("--T-end", type=float, default=0.05, help="SA final temperature (mode=sa)")
    ap.add_argument("--w-swap", type=float, default=3.0,
                    help="Operator weight for cheap commuting_swap (default 3.0)")
    ap.add_argument("--w-tail", type=float, default=1.0,
                    help="Operator weight for expensive tail_resolve (default 1.0)")
    ap.add_argument("--w-macro", type=float, default=1.0,
                    help="Operator weight for expensive macro_insert (default 1.0)")
    ap.add_argument("--commutator-table", default="megaminx/data/commutator_table.pkl",
                    help="Pickled commutator library (used by macro_insert)")
    ap.add_argument("--max-macro-len", type=int, default=6,
                    help="Cap on commutator word length used as insertion macro")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--internal-batch-size", type=int, default=2**14)
    ap.add_argument("--state-dtype", default="int8",
                    choices=["int8", "int16", "int32", "int64"])
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    state_dtype = {"int8": torch.int8, "int16": torch.int16,
                   "int32": torch.int32, "int64": torch.int64}[args.state_dtype]
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"loading puzzle + model + commute table ...")
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    base = load_submission(args.base)
    print(f"  test states: {len(states)}, base submission: {len(base)} pids, "
          f"total moves: {sum(len(p) for p in base.values()):,}")

    bfs_path = Path(args.bfs_table)
    if bfs_path.exists():
        bfs_table = BfsBytesTable.load(bfs_path)
        print(f"  bfs-d6 table loaded: {bfs_path.name} ({len(bfs_table.table):,} entries)")
    else:
        bfs_table = None
        print(f"  WARN: no bfs table at {bfs_path}; post-proc will be cancel-only")

    commutes = compute_commuting_pairs(puzzle)
    n_commute = sum(1 for v in commutes.values() if v)
    print(f"  commuting pairs (ordered): {n_commute} / {len(commutes)} "
          f"({100 * n_commute / len(commutes):.1f}%)")

    # Load commutator library and convert to lists of generator names.
    # The pickled table maps perm -> tuple of generator INDEX. Convert to names.
    macros: list[list[str]] = []
    commutator_path = Path(args.commutator_table)
    if commutator_path.exists() and args.w_macro > 0:
        import pickle
        with open(commutator_path, "rb") as f:
            cm = pickle.load(f)
        idx_to_name = list(puzzle.move_names)
        n_skipped = 0
        for perm, word in cm["table"].items():
            if not word or len(word) > args.max_macro_len:
                n_skipped += 1
                continue
            try:
                names = [idx_to_name[i] for i in word]
                macros.append(names)
            except (IndexError, TypeError):
                n_skipped += 1
        print(f"  commutator macros: {len(macros)} (max_len={args.max_macro_len}; "
              f"skipped {n_skipped})")
    else:
        print(f"  commutator macros: 0 (table missing or w_macro=0)")

    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=model_dtype)
    solver = KhoruzhiiSolver(
        puzzle, model, device=args.device,
        internal_batch_size=args.internal_batch_size,
        state_dtype=state_dtype,
    )

    # Determine pid order.
    if args.pids.strip():
        pid_order = [int(p) for p in args.pids.split(",") if p.strip()]
    else:
        pid_order = sorted(base.keys(), key=lambda p: -len(base[p]))
        if args.top > 0:
            pid_order = pid_order[: args.top]

    op_weights = {
        "commuting_swap": args.w_swap,
        "tail_resolve": args.w_tail,
        "macro_insert": args.w_macro if macros else 0.0,
    }
    # Drop zero-weighted operators so weights normalize cleanly.
    op_weights = {k: v for k, v in op_weights.items() if v > 0}

    print(f"\nprocessing {len(pid_order)} pids; n_iter={args.n_iter}, mode={args.mode}, "
          f"beam={args.beam}, K=[{args.K_min},{args.K_max}], op_weights={op_weights}\n")

    out_paths: dict[int, list[str]] = dict(base)
    n_improved = 0
    total_savings = 0
    total_op_calls = {op: 0 for op in op_weights}
    total_op_accepts = {op: 0 for op in op_weights}
    t0 = time.time()
    rng = random.Random(args.seed)

    for rank, pid in enumerate(pid_order):
        path = list(base[pid])
        initial = states[pid]
        # Per-pid RNG so reruns are reproducible.
        prng = random.Random(args.seed * 1000003 + pid)
        best, stats = sa_search(
            path, initial, puzzle, solver, bfs_table, args.bfs_max_window,
            commutes, macros, args.n_iter, args.beam, args.K_min, args.K_max,
            op_weights, args.mode, args.T0, args.T_end, prng, args.verbose,
        )
        savings = stats["initial_len"] - stats["best_len"]
        for op in op_weights:
            total_op_calls[op] += stats["n_op_calls"][op]
            total_op_accepts[op] += stats["n_op_accepts"][op]
        if savings > 0:
            out_paths[pid] = best
            n_improved += 1
            total_savings += savings
            print(f"  pid={pid:4d} {stats['initial_len']:3d}->{stats['best_len']:3d} "
                  f"(-{savings:2d}) [rank {rank+1}/{len(pid_order)}, total -{total_savings}]")
        elif rank < 10 or rank % 50 == 0:
            print(f"  pid={pid:4d} {stats['initial_len']:3d}        no improvement "
                  f"[rank {rank+1}/{len(pid_order)}]")

    elapsed = time.time() - t0
    print(f"\n=== summary ===")
    print(f"  improved: {n_improved}/{len(pid_order)} pids")
    print(f"  total savings: {total_savings} moves")
    print(f"  walltime: {elapsed/60:.1f} min")
    print(f"  op call counts: {total_op_calls}")
    print(f"  op accept counts: {total_op_accepts}")
    for op, n_call in total_op_calls.items():
        if n_call > 0:
            rate = 100 * total_op_accepts[op] / n_call
            print(f"  {op:>14s} accept rate: {rate:.1f}% "
                  f"({total_op_accepts[op]}/{n_call})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(out_paths.keys()):
            w.writerow([pid, ".".join(out_paths[pid])])

    final_total = sum(len(p) for p in out_paths.values())
    base_total = sum(len(p) for p in base.values())
    print(f"\nwrote {out_path} ({len(out_paths)} rows)")
    print(f"  base total: {base_total:,} moves")
    print(f"  new  total: {final_total:,} moves  ({final_total - base_total:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
