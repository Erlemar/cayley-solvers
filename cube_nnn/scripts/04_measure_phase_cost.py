"""MEASURE the cost of each reduction sub-phase. BIGCUBES_PLAN R5.

R5 estimates ~2.3 bits/move for moves that must preserve an already-solved orbit
and says outright "this number is an estimate, not a measurement -- measure it
early, it sets everything". It was never measured. Everything about the phase
ladder depends on it:

  step-price model      cost(phase k) = (bits of orbit k) / rate
  cumulative-price model cost(phase k) = (bits of orbits 1..k) / rate

Under step-price a 4-orbit centre solve costs ~4 x 16.3 = 65 moves; under
cumulative-price it costs 16.3+32.5+48.8+65.1 = 163. That is the difference
between a ~150-move solver and a ~400-move one.

This measures it directly: solve orbit 1, then orbit 2 GIVEN orbit 1 solved
(target = both), and so on, and reports achieved moves against each bound.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/04_measure_phase_cost.py
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.beam import Engine, beam_solve  # noqa: E402

DATA = ROOT / "cayley-py-666-cube" / "puzzle_info.json"
BITS_PER_ORBIT = math.log2(math.factorial(24))   # 79.04
B6 = math.log2(29.023)                            # 4.859 bits/move


def centre_orbits(cube: NCube) -> list[np.ndarray]:
    """The 4 centre sticker orbits (24 positions each), as index arrays."""
    orbits = cube.sticker_orbits()
    pieces = cube.pieces()
    centre_stickers = {s[0] for s in pieces.values() if len(s) == 1}
    out = [np.array(sorted(o), dtype=np.int64) for o in orbits
           if set(o) <= centre_stickers]
    assert len(out) == 4, f"expected 4 centre orbits, got {len(out)}"
    return out


def wing_orbits(cube: NCube) -> list[np.ndarray]:
    orbits = cube.sticker_orbits()
    pieces = cube.pieces()
    wing_stickers = {s for cell, ss in pieces.items() if len(ss) == 2 for s in ss}
    return [np.array(sorted(o), dtype=np.int64) for o in orbits
            if set(o) <= wing_stickers]


def random_state(cube: NCube, rng: np.random.Generator, length: int = 400) -> np.ndarray:
    st = np.array(cube.solved_state, dtype=np.int16)
    names = list(cube.move_names)
    for m in rng.choice(len(names), size=length):
        st = st[np.array(cube.generators[names[int(m)]], dtype=np.int64)]
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1 << 13)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=120)
    args = ap.parse_args()

    cube = NCube.from_puzzle_info(DATA)
    eng = Engine(cube, list(cube.move_names))
    cent = centre_orbits(cube)
    wings = wing_orbits(cube)
    print(f"centre orbits: {[len(o) for o in cent]}   wing orbits: {[len(o) for o in wings]}")
    print(f"beam width {args.width}, {args.trials} trials, max {args.max_steps} steps")
    print(f"one 24-orbit = {BITS_PER_ORBIT:.2f} bits = {BITS_PER_ORBIT / B6:.1f} moves (bound)")
    print()

    rng = np.random.default_rng(0)
    print(f"{'phase target':28s} {'bits':>7s} {'step-bd':>8s} {'cum-bd':>7s} "
          f"{'achieved':>20s} {'wall s':>7s}")
    print("-" * 86)

    cumulative: list[np.ndarray] = []
    for k in range(1, 5):
        cumulative.append(cent[k - 1])
        proj = np.concatenate(cumulative)
        cum_bits = BITS_PER_ORBIT * k
        step_bd = BITS_PER_ORBIT / B6
        cum_bd = cum_bits / B6

        lengths, walls = [], []
        for t in range(args.trials):
            st = random_state(cube, rng, 400)
            t0 = time.time()
            res = beam_solve(eng, st, proj, width=args.width,
                             max_steps=args.max_steps, seed=t)
            walls.append(time.time() - t0)
            lengths.append(len(res.path) if res.solved else None)
        got = [x for x in lengths if x is not None]
        desc = (f"{np.mean(got):.1f} ({min(got)}-{max(got)}) {len(got)}/{args.trials}"
                if got else f"UNSOLVED 0/{args.trials}")
        print(f"{'centres orbits 1..' + str(k):28s} {cum_bits:7.1f} {step_bd:8.1f} "
              f"{cum_bd:7.1f} {desc:>20s} {np.mean(walls):7.1f}")

    print()
    print("READING:")
    print("  achieved ~ step-bd  each time  -> step-price: phases are cheap, ladder wins")
    print("  achieved ~ cum-bd   each time  -> cumulative-price: preservation is full cost")
    print("  achieved blows up / unsolved   -> that width cannot hold the phase")


if __name__ == "__main__":
    main()
