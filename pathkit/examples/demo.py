"""End-to-end tour of every method, on a self-contained toy puzzle (no data files needed).

    PYTHONPATH=pathkit python pathkit/examples/demo.py

It builds a small submission, inflates it so there is something real to find, then runs the
whole pipeline in the recommended order and prints what each stage bought. Use it as a
template: swap `Puzzle.demo_lrx(...)` for `Puzzle.from_puzzle_info(...)` and the same calls
work on any of the competition puzzles.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathkit import bridge, cheap, merge, suffix, table, window          # noqa: E402
from pathkit.io import write_submission                                  # noqa: E402
from pathkit.ladder import BallSolver, Journal, apply_journal, run_ladder  # noqa: E402
from pathkit.puzzle import Puzzle                                        # noqa: E402

N_PIDS = 40


def build_corpus(pz, rng):
    """A `tests` dict plus two source files: a good one and a deliberately loose one."""
    tests, good, loose = {}, {}, {}
    for pid in range(N_PIDS):
        s, w = pz.random_scramble(22, rng)
        tests[pid] = s
        good[pid] = w
        # inflate: a move and its inverse, plus a detour that a ball sweep can collapse
        pad = [int(w[0]), int(pz.inv_move[w[0]])]
        loose[pid] = list(w[:3]) + pad + list(w[3:6]) + pad + list(w[6:])
    return tests, good, loose


def main():
    rng = np.random.default_rng(20260820)
    pz = Puzzle.demo_lrx(14)
    print(pz.describe())
    tests, good, loose = build_corpus(pz, rng)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        write_submission(td / "loose.csv", pz, loose, tests, log=lambda *a: None)
        write_submission(td / "good.csv", pz, good, tests, log=lambda *a: None)

        # 1. merge -----------------------------------------------------------
        best, rep = merge.merge(pz, tests, [td], verbose=False)
        print("\n1. merge      %d files -> %d moves over %d pids"
              % (rep["files_csv"], rep["total"], rep["covered"]))

        # everything below deliberately starts from the LOOSE file, so each stage
        # has something real to find
        paths = dict(loose)
        start = sum(len(v) for v in paths.values())

        # 2. cheap -----------------------------------------------------------
        paths, r = cheap.sweep(pz, tests, paths)
        print("2. cheap      %d -> %d (%+d)" % (r["before"], r["after"], -r["saved"]))

        # 3. ball window sweep ------------------------------------------------
        paths, r = window.sweep(pz, tests, paths, radius=3)
        print("3. window     " + window.certify_report(r))

        # 4. table sweep + extend ---------------------------------------------
        tb = table.BfsTable.build(pz, 5)
        paths, r = table.sweep(pz, tests, paths, tb, max_window=10, extend=2)
        print("4. table      %d -> %d (%+d), effective reach %d"
              % (r["before"], r["after"], -r["saved"], r["effective_reach"]))

        # 5. pooled bridging (pool the loose and the good path per pid) --------
        pools = {pid: [paths[pid], good[pid]] for pid in paths}
        paths, r = bridge.sweep(pz, tests, pools, radius=2)
        print("5. bridge     merged floor %d -> %d (%+d beyond the merge, %d pids bridged)"
              % (r["before_merged"], r["after"], -r["saved_beyond_merge"], r["bridged_pids"]))

        # 6. exact ladder with honest verdicts ---------------------------------
        journal = Journal(td / "ladder.jsonl")
        solver = BallSolver(pz, radius=3)
        run_ladder(pz, tests, paths, solver, journal, window_length=6, progress=0)
        paths, r = apply_journal(pz, tests, paths, journal)

        # 7. suffix diagnostic --------------------------------------------------
        print("\n7. suffix probe (is the solver ever able to beat this file?)")
        solve = suffix.make_table_solver(table.BfsTable.build(pz, 5, root=pz.solved), pz,
                                         extend=1)
        suffix.probe(pz, tests, paths, solve, ks=(6, 8, 10),
                     pids=sorted(paths)[:10], log=lambda m: print("   " + m))

        end = sum(len(v) for v in paths.values())
        print("\ntotal: %d -> %d (%+d moves, %.1f pct)"
              % (start, end, end - start, 100.0 * (start - end) / start))
        for pid in sorted(paths):
            assert pz.solves(tests[pid], paths[pid]), "pid %d broken" % pid
        print("every path replay-verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
