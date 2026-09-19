"""FMC viability probe: how many moves is a PURE single 3-cycle on megaminx?

The whole FMC-insertion bet hinges on this. An insertion replaces the beam's
endgame on a 3-cycle with a commutator. If pure 3-cycles cost ~8 moves, insertions
(after join cancellation) beat the beam's endgame and FMC pays. If they cost ~14,
the beam already does comparably and FMC can't win.

We construct pure corner / edge 3-cycles (position-only, orientation 0), confirm
they decode as a single 3-cycle, and solve each with the V-beam to read off a
near-optimal word length.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "beam_lab"))

from beam_search import KhoruzhiiSearchConfig, KhoruzhiiSolver  # noqa: E402
from cayley.search import load_model_checkpoint  # noqa: E402
from cayley.verify import verify_path  # noqa: E402
from megaminx.fmc import PieceModel  # noqa: E402
from megaminx.puzzle import Megaminx  # noqa: E402


def make_3cycle(solved, slots, a, b, c):
    """State where the pieces at slots a,b,c are 3-cycled a->b->c->a (orientation 0)."""
    state = list(solved)
    sa, sb, sc = slots[a], slots[b], slots[c]
    for k in range(len(sa)):
        state[sb[k]] = sa[k]
        state[sc[k]] = sb[k]
        state[sa[k]] = sc[k]
    return tuple(state)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    pm = PieceModel(puzzle)
    solved = puzzle.solved_state

    teacher = PROJECT / "models" / "m_az_v4_v_only_e99.pt"
    model = load_model_checkpoint(teacher, device=device, dtype=torch.float32)
    if hasattr(model, "inference_chunk_size"):
        model.inference_chunk_size = None
    solver = KhoruzhiiSolver(puzzle, model, device=device,
                             internal_batch_size=16384, profile=False)
    cfg = KhoruzhiiSearchConfig(beam_width=16384, num_steps=30, num_attempts=1)

    cases = [
        ("corner", pm.corner_slots, [(0, 1, 2), (0, 5, 10), (0, 10, 19), (3, 8, 14)]),
        ("edge", pm.edge_slots, [(0, 1, 2), (0, 15, 29), (5, 12, 23), (1, 14, 27)]),
    ]
    print(f"{'type':>7s} {'cycle':>14s} {'kind':>14s} {'len':>5s} {'verify':>7s}")
    print("-" * 54)
    lens = {"corner": [], "edge": []}
    for typ, slots, triples in cases:
        for (a, b, c) in triples:
            st = make_3cycle(solved, slots, a, b, c)
            res = pm.residue(st)
            found, plen, names, _ = solver.solve(st, cfg)
            ok = found and verify_path(puzzle, st, names).ok
            if ok:
                lens[typ].append(plen)
            print(f"{typ:>7s} {str((a,b,c)):>14s} {res.kind:>14s} "
                  f"{(plen if found else -1):>5d} {str(ok):>7s}")
    print("-" * 54)
    for typ in ("corner", "edge"):
        if lens[typ]:
            xs = lens[typ]
            print(f"{typ} 3-cycle word length: min={min(xs)} max={max(xs)} "
                  f"mean={sum(xs)/len(xs):.1f}  (n={len(xs)})")
    print("\nFMC read: insertion cost ~= these lengths. <=8 strong, ~10 marginal, "
          ">=12 weak (beam endgame is comparable).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
