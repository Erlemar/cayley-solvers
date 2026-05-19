"""T1.1 prototype: macro-augmented beam search.

For a small starter set of macros (filtered from the d=4 commutator library by
'support size' = number of stickers the macro permutes), wire each as an
extra action in `KhoruzhiiSolver`. Compare path lengths with vs without
macros on a handful of hard pids.

Macros tested:
- Filter d=4 commutators (240 entries) to those with small support
  (<=12 stickers moved). These are 'piece-cycle-like' permutations that
  cycle a small number of pieces — exactly the kind of move human
  speedcubers use for finishing layers.

Why this might work (vs window-replacement which scored 0):
- Window-replacement post-process scans EXISTING beam paths for windows whose
  net permutation matches a macro. Real beam paths don't land on macro atoms.
- BUT: at SEARCH time, having macros as actions lets the beam REACH states
  that gen-only beam can't reach within max_steps from the same V-budget.
  The beam's V function still drives the search; macros just give it more
  reachable states per step.

Caveat: macros count as 1 BEAM step but K REAL moves. Path is reconstructed
in real moves (correct LB scoring). So macros only help when the V function
correctly identifies macro-reachable states as "close to solved" — i.e.,
when the model's V on the post-macro state is meaningfully lower than V on
gen-only descendants.
"""
from __future__ import annotations

import csv
import pickle
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
from cayley.verify import load_test_states, verify_path
from megaminx.puzzle import Megaminx


def select_useful_macros(commutator_path: Path, puzzle: Megaminx,
                          max_support: int = 12) -> list:
    """Load commutator library and filter to macros with small support
    (= small number of stickers moved by the permutation). These are the
    piece-cycle-like macros most likely to help structured search."""
    with open(commutator_path, "rb") as f:
        d = pickle.load(f)
    table = d["table"]  # tuple_perm -> tuple_word_idxs
    move_names = list(puzzle.move_names)
    state_size = len(puzzle.solved_state)
    identity = tuple(range(state_size))

    out = []
    for perm, word_idx in table.items():
        if perm == identity:
            continue
        # Support = number of stickers moved by this permutation
        support = sum(1 for i, p in enumerate(perm) if p != i)
        if support > max_support:
            continue
        # Only keep d=4 macros for the starter — short, fast to apply
        if len(word_idx) != 4:
            continue
        word_names = [move_names[i] for i in word_idx]
        out.append((list(perm), word_names, support))
    # Sort by support ascending (smallest piece-cycles first)
    out.sort(key=lambda x: x[2])
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT / "models/m05_bellman_warm/epoch_0499.pt")
    ap.add_argument("--n-macros", type=int, default=30,
                    help="number of macros to use (sorted by support ascending)")
    ap.add_argument("--max-support", type=int, default=12,
                    help="max stickers a macro is allowed to move")
    ap.add_argument("--pids", type=str, default=None,
                    help="comma-separated pids to test; default = top 5 long-pids of best submission")
    ap.add_argument("--beam", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=120)
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"loading commutator library...")
    macros_full = select_useful_macros(
        PROJECT / "data" / "commutator_table.pkl", puzzle, args.max_support
    )
    print(f"  {len(macros_full)} d=4 macros with support <= {args.max_support}")
    if not macros_full:
        print("  WARNING: no macros match filter; aborting")
        return 1
    macros_used = [(p, w) for p, w, _ in macros_full[: args.n_macros]]
    print(f"  using top {len(macros_used)} (smallest support):")
    for i, (_, w, sup) in enumerate(macros_full[:5]):
        print(f"    macro {i}: support={sup}, word={'.'.join(w)}")

    # Load model + states
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_checkpoint(args.checkpoint, device=device, dtype=torch.bfloat16)
    states = load_test_states(PROJECT / "data" / "test.csv")

    # Pick test pids
    if args.pids is not None:
        pids = [int(x) for x in args.pids.split(",")]
    else:
        # Default: top 5 long-pids of current best submission
        sub_path = PROJECT / "submissions" / "merge_plus_sym8_top20.csv"
        rows = list(csv.DictReader(open(sub_path)))
        rows.sort(key=lambda r: -len(r["path"].split(".")))
        pids = [int(r["initial_state_id"]) for r in rows[:5]]
    print(f"\ntest pids: {pids}")

    # Build two solvers: baseline (no macros) and macro-augmented.
    cfg = KhoruzhiiSearchConfig(beam_width=args.beam, num_steps=args.max_steps)
    solver_base = KhoruzhiiSolver(
        puzzle, model, device=device, state_dtype=torch.int8, random_seed=0,
    )
    solver_macro = KhoruzhiiSolver(
        puzzle, model, device=device, state_dtype=torch.int8, random_seed=0,
        macros=macros_used,
    )
    print(f"\nbaseline solver: n_actions = {solver_base.n_actions}")
    print(f"macro solver:    n_actions = {solver_macro.n_actions}")

    # Run both solvers per pid
    print(f"\nrunning beam={args.beam} max_steps={args.max_steps}\n")
    print(f"{'pid':>5} | {'base':>10} | {'macro':>10} | {'delta':>6} | wall")
    print("-" * 55)
    total_base, total_macro = 0, 0
    for pid in pids:
        s = states[pid]
        t0 = time.time()
        found_b, plen_b, path_b = solver_base.solve(s, cfg)
        t1 = time.time()
        found_m, plen_m, path_m = solver_macro.solve(s, cfg)
        t2 = time.time()
        # Verify both paths
        if found_b and not verify_path(puzzle, s, path_b).ok:
            print(f"  pid={pid}: BASELINE PATH INVALID")
            continue
        if found_m and not verify_path(puzzle, s, path_m).ok:
            print(f"  pid={pid}: MACRO PATH INVALID")
            continue
        b_str = str(plen_b) if found_b else "NONE"
        m_str = str(plen_m) if found_m else "NONE"
        if found_b and found_m:
            delta = plen_m - plen_b
            d_str = f"{delta:+d}"
            total_base += plen_b
            total_macro += plen_m
        else:
            d_str = "n/a"
        print(f"{pid:>5} | {b_str:>10} | {m_str:>10} | {d_str:>6} | "
              f"base={t1-t0:.1f}s macro={t2-t1:.1f}s")

    if total_base > 0:
        print(f"\ntotals: base={total_base}, macro={total_macro}, delta={total_macro - total_base:+d}")
        if total_macro < total_base:
            print(f"  → macros HELPED ({total_base - total_macro} moves saved)")
        elif total_macro == total_base:
            print(f"  → macros TIED (no change)")
        else:
            print(f"  → macros HURT ({total_macro - total_base} moves longer)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
