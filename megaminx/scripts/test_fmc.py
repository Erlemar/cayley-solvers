"""Self-test for megaminx.fmc piece decoder + residue classifier.

Strongest check: decode(move applied to solved) must reproduce the independently
built corner_coord corner_perm/corner_ori tables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.corner_coord import build_corner_tables
from megaminx.fmc import PieceModel, perm_cycles
from megaminx.puzzle import Megaminx


def main() -> int:
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    pm = PieceModel(puzzle)
    solved = puzzle.solved_state
    n_fail = 0

    # 1. solved -> identity
    ps = pm.decode(solved)
    assert np.array_equal(ps.corner_perm, np.arange(20)), "corner_perm(solved) != id"
    assert (ps.corner_ori == 0).all(), "corner_ori(solved) != 0"
    assert np.array_equal(ps.edge_perm, np.arange(30)), "edge_perm(solved) != id"
    assert (ps.edge_ori == 0).all(), "edge_ori(solved) != 0"
    assert pm.residue(solved).kind == "solved"
    print("[ok] solved decodes to identity, residue=solved")

    # 2. cross-check every move vs corner_coord tables
    ct = build_corner_tables(PROJECT / "data" / "puzzle_info.json")
    cc_names = ct["move_names"]
    for m in puzzle.move_names:
        st = puzzle.apply_move(solved, m)
        ps = pm.decode(st)
        ref_perm = ct["corner_perm"][cc_names.index(m)]
        ref_ori = ct["corner_ori"][cc_names.index(m)]
        if not np.array_equal(ps.corner_perm, ref_perm):
            print(f"[FAIL] {m}: corner_perm mismatch")
            n_fail += 1
        if not np.array_equal(ps.corner_ori, ref_ori):
            print(f"[FAIL] {m}: corner_ori mismatch")
            n_fail += 1
    if n_fail == 0:
        print(f"[ok] all {len(puzzle.move_names)} moves: corner_perm+ori match corner_coord")

    # 3. edge perms bijective for every move
    for m in puzzle.move_names:
        ps = pm.decode(puzzle.apply_move(solved, m))
        if sorted(ps.edge_perm.tolist()) != list(range(30)):
            print(f"[FAIL] {m}: edge_perm not a permutation")
            n_fail += 1
    print("[ok] all moves: edge_perm is a valid permutation")

    # 4. a commutator [U, R, -U, -R] -> inspect residue
    comm = ["U", "R", "-U", "-R"]
    st = puzzle.apply_path(solved, comm)
    r = pm.residue(st)
    print(f"[info] [U,R,-U,-R]: residue kind={r.kind} "
          f"n_corner_unsolved={r.n_corner_unsolved} n_edge_unsolved={r.n_edge_unsolved}")
    print(f"       corner_cycles={r.corner_cycles} edge_cycles={r.edge_cycles}")

    # 5. a few longer commutators/conjugates, report how often we get a clean 3-cycle
    import random
    rng = random.Random(0)
    names = list(puzzle.move_names)
    kinds = {"solved": 0, "corner_3cycle": 0, "edge_3cycle": 0, "other": 0}
    for _ in range(200):
        # random conjugated commutator X [A B A' B'] X'
        a, b = rng.choice(names), rng.choice(names)
        x = rng.choice(names)
        inv = {nm: ("-" + nm if not nm.startswith("-") else nm[1:]) for nm in names}
        word = [x, a, b, inv[a], inv[b], inv[x]]
        st = puzzle.apply_path(solved, word)
        kinds[pm.residue(st).kind] += 1
    print(f"[info] 200 random conj-commutators residue kinds: {kinds}")

    print("\nFAILURES:", n_fail)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
