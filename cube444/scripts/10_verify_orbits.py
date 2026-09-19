"""Verify every structural claim in `cube444.orbits` against the raw generators.

Run:  .venv/Scripts/python.exe cube444/scripts/10_verify_orbits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube444" / "src"))
sys.path.insert(0, str(ROOT / "src"))

from cube444.orbits import (  # noqa: E402
    MAX_DEFECTS, NONCORNER_SLOTS, is_reduced, outer_move_names,
    project_noncorner, reduction_defects, verify_structure,
)
from cube444.puzzle import Cube444  # noqa: E402

PUZZLE = ROOT / "cube444" / "data" / "puzzle_info.json"


def main() -> int:
    puz = Cube444.load(PUZZLE)
    gens = {k: np.array(v, dtype=np.int64) for k, v in puz.generators.items()}
    solved = np.array(puz.solved_state, dtype=np.int64)

    print("=== structural verification ===")
    report = verify_structure(gens, solved)
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("  ALL STRUCTURAL ASSERTIONS PASSED")

    rng = np.random.default_rng(0)
    names = list(gens)

    # defect statistics on random deep states
    batch = np.repeat(solved[None, :], 512, axis=0)
    for _ in range(60):
        picks = rng.integers(len(names), size=512)
        batch = np.take_along_axis(
            batch, np.stack([gens[names[p]] for p in picks]), axis=1
        )
    d = reduction_defects(batch)
    print("\n=== defect statistics, 512 random depth-60 states ===")
    print(f"  mean {d.mean():.2f}  min {d.min()}  max {d.max()}  (max possible {MAX_DEFECTS})")
    print(f"  fraction already reduced: {is_reduced(batch).mean():.4f}")

    # the phase-1 quotient claim: distance-to-R is independent of corner colours.
    # Scrambling corners alone must not change the defect count OR the defect count
    # of any descendant -- check the whole 24-child profile is identical.
    from cube444.orbits import CORNER_SLOTS
    s = batch[0].copy()
    t = s.copy()
    perm = rng.permutation(len(CORNER_SLOTS))
    t[CORNER_SLOTS] = s[CORNER_SLOTS][perm]
    prof_s = sorted(int(reduction_defects(s[gens[n]])) for n in names)
    prof_t = sorted(int(reduction_defects(t[gens[n]])) for n in names)
    print("\n=== phase-1 quotient claim ===")
    print(f"  corner-scrambled twin has identical child-defect profile: {prof_s == prof_t}")
    print(f"  non-corner projection identical: "
          f"{np.array_equal(project_noncorner(s), project_noncorner(t))}")
    assert prof_s == prof_t

    # outer-move reachability sanity: R is reachable and stays reachable
    outer = outer_move_names(names)
    print(f"\n=== move sets ===")
    print(f"  outer (phase 2): {len(outer)}  {' '.join(sorted(outer))}")
    print(f"  full  (phase 1): {len(names)}")
    print(f"  non-corner projection size: {len(NONCORNER_SLOTS)}")

    # a short full-group walk from solved: how fast does R break?
    print("\n=== R decay along a full-group walk from solved ===")
    s = solved.copy()
    marks = []
    for step in range(1, 21):
        s = s[gens[names[rng.integers(len(names))]]]
        marks.append((step, int(reduction_defects(s))))
    print("  " + "  ".join(f"d{st}={dv}" for st, dv in marks[:10]))
    print("  " + "  ".join(f"d{st}={dv}" for st, dv in marks[10:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
