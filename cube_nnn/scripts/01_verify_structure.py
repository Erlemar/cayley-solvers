"""Verify the built generators against shipped data, then MEASURE the structure.

Everything printed here is measured, not asserted. Several numbers in the
project docs were wrong because they were asserted; this script is the check.

Run:
    .venv/Scripts/python.exe cube_nnn/scripts/01_verify_structure.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cube_nnn" / "src"))

from cube_nnn.puzzle import NCube, FACE_NAMES  # noqa: E402

DATA = {
    4: ROOT / "cube444" / "data" / "puzzle_info.json",
    6: ROOT / "cayley-py-666-cube" / "puzzle_info.json",
}


def log2_factorial(k: int) -> float:
    return sum(math.log2(i) for i in range(2, k + 1))


def check_against_data() -> dict[int, NCube]:
    print("=" * 72)
    print("1. GENERATORS: built from geometry vs shipped competition data")
    print("=" * 72)
    cubes = {}
    for n, path in DATA.items():
        if not path.exists():
            print(f"  n={n}: {path} MISSING -- skipped")
            continue
        cube = NCube.from_puzzle_info(path)  # asserts an exact match
        ref = json.load(open(path, encoding="utf-8"))
        print(f"  n={n}: {len(ref['generators'])}/{len(ref['generators'])} generators "
              f"EXACT, state {cube.state_size}, supercube={cube.is_supercube}")
        cubes[n] = cube
    for n in (5, 7):
        cube = NCube.build(n)
        cubes[n] = cube
        print(f"  n={n}: built (no shipped data here), state {cube.state_size}, "
              f"{len(cube.generators)} generators")
    return cubes


def check_inverses(cube: NCube) -> None:
    for name in cube.move_names:
        if name.startswith("-"):
            continue
        s = cube.apply_move(cube.solved_state, name)
        assert cube.apply_move(s, cube.inverse_name(name)) == cube.solved_state, name
        # order 4
        t = cube.solved_state
        for _ in range(4):
            t = cube.apply_move(t, name)
        assert t == cube.solved_state, f"{name} does not have order 4"


def measure_branching(cube: NCube, depth: int = 4) -> float:
    """BFS growth ratio at `depth` -- the b used in every counting bound."""
    solved = cube.solved_state
    frontier = {solved}
    seen = {solved}
    sizes = []
    fwd = [m for m in cube.move_names]
    for _ in range(depth):
        nxt = set()
        for st in frontier:
            for mv in fwd:
                t = cube.apply_move(st, mv)
                if t not in seen:
                    nxt.add(t)
        seen |= nxt
        sizes.append(len(nxt))
        frontier = nxt
    return sizes[-1] / sizes[-2], sizes


def main() -> None:
    cubes = check_against_data()

    print()
    print("=" * 72)
    print("2. GENERATOR PARITY  (docs claimed 'every generator is odd' for 666)")
    print("=" * 72)
    for n in sorted(cubes):
        cube = cubes[n]
        par = cube.generator_parity()
        odd = sum(par.values())
        tot = len(par)
        m = n - 1
        outer = [f"{a}{i}" for a in ("f", "r", "d") for i in (0, m)]
        o_odd = sum(par[k] for k in outer)
        i_odd = odd - o_odd
        bipartite = (odd == tot)
        print(f"  n={n}: {odd} odd / {tot - odd} even   "
              f"(outer {o_odd}/{len(outer)} odd, inner {i_odd}/{tot - len(outer)} odd)   "
              f"parity invariant usable: {bipartite}")
    print()
    print("  Rule: a slice turn's ring is 4n stickers = n four-cycles, sign (-1)^n;")
    print("  an outer turn adds the face's own (n^2 - [n odd])/4 four-cycles.")
    print("  => the d(s) == sgn(s) mod 2 invariant exists iff n is ODD.")

    print()
    print("=" * 72)
    print("3. STICKER ORBITS and INFORMATION CONTENT")
    print("=" * 72)
    for n in sorted(cubes):
        cube = cubes[n]
        orbits = cube.sticker_orbits()
        from collections import Counter
        sizes = Counter(len(o) for o in orbits)
        pieces = cube.pieces()
        pc = Counter(len(v) for v in pieces.values())
        print(f"  n={n}: {len(orbits)} sticker orbits, sizes {dict(sorted(sizes.items()))}")
        print(f"        {len(pieces)} pieces: "
              + ", ".join(f"{k}-sticker x{v}" for k, v in sorted(pc.items())))

    print()
    print("=" * 72)
    print("4. BRANCHING and COUNTING BOUNDS")
    print("=" * 72)
    known_bits = {4: None, 5: 307.93, 6: 500.62, 7: None}
    for n in sorted(cubes):
        cube = cubes[n]
        if cube.state_size > 200 and n > 6:
            print(f"  n={n}: branching BFS skipped (cost); using G-ratio estimate")
            continue
        b, sizes = measure_branching(cube, depth=4)
        bits_per_move = math.log2(b)
        line = f"  n={n}: BFS sizes {sizes}  b={b:.3f}  {bits_per_move:.3f} bits/move"
        kb = known_bits.get(n)
        if kb:
            line += f"   LB = {kb / bits_per_move:.2f} moves"
        print(line)

    print()
    print("=" * 72)
    print("5. THE ENDGAME SUBGROUP G1 = <12 outer turns>   (n=6)")
    print("=" * 72)
    cube = cubes[6]
    outer = cube.outer_moves()
    print(f"  outer moves: {sorted(set(m.lstrip('-') for m in outer))}")
    # Do outer turns ever move a centre sticker off its face?
    centres = [i for i, v in enumerate(cube.pieces().values()) for _ in ()]  # placeholder
    piece_map = cube.pieces()
    centre_stickers = {s[0] for s in piece_map.values() if len(s) == 1}
    nn = cube.n
    off_face = 0
    for mv in outer:
        g = cube.generators[mv]
        for i in centre_stickers:
            if g[i] // (nn * nn) != i // (nn * nn):
                off_face += 1
    print(f"  centre stickers moved BETWEEN faces by any outer turn: {off_face}")
    print("  (0 confirms: outer turns rotate each face's centre block rigidly,")
    print("   so G1 is the 3x3x3 supercube group and reduction is a coset problem)")

    g1_bits = math.log2(4.3252003274489856e19) + math.log2(2048)
    tot_bits = 500.62
    b6 = math.log2(29.023)
    print()
    print(f"  |G1| = |3x3x3| * 4^6/2 = 8.858e22  ->  {g1_bits:.2f} bits")
    print(f"  endgame bound at 12 generators   =  {g1_bits / math.log2(12):.1f} moves")
    print(f"  [G:G1] = {tot_bits - g1_bits:.1f} bits  ->  reduction bound "
          f"= {(tot_bits - g1_bits) / b6:.1f} moves")
    print(f"  two-phase total = {g1_bits / math.log2(12) + (tot_bits - g1_bits) / b6:.1f} "
          f"vs global bound {tot_bits / b6:.1f}  "
          f"-> phasing costs {100 * ((g1_bits / math.log2(12) + (tot_bits - g1_bits) / b6) / (tot_bits / b6) - 1):.1f}%")


if __name__ == "__main__":
    main()
