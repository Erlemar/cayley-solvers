"""Parameterized NxNxN cube for the CayleyPy 555 / 666 / 777 competitions.

BIGCUBES_PLAN T1.1. Generators are BUILT from 3D geometry, not loaded, and the
build is validated bit-exact against the shipped competition data:

    n=4  cube444/data/puzzle_info.json          24/24 generators
    n=6  cayley-py-666-cube/puzzle_info.json    36/36 generators

so n=5 and n=7 come for free from the same code path.

LAYOUT (reverse-engineered from the data, then confirmed by the exact match)
---------------------------------------------------------------------------
    index = face*n*n + row*n + col
    faces  0=U  1=F  2=R  3=B  4=L  5=D
    moves  f0..f{n-1}  r0..r{n-1}  d0..d{n-1}  and the same with a '-' prefix
    f0 turns F(1), r0 turns R(2), d0 turns D(5); layer index counts inward
    from that face, so f{n-1} is the B layer, r{n-1} the L layer, d{n-1} the U.

    Cell coords (x right, y up, z toward viewer), each in 0..n-1. A sticker is
    (cell, outward normal); stickers sharing a cell are the same PIECE, which
    is where the piece/orbit decomposition below comes from for free.

CONVENTION
----------
    new_state[i] = state[gen[i]]        (gen is the SOURCE map)

    Quarter turns only, single slice only. No wide turns, no half turns, no
    whole-cube rotations. Price anything imported from the cubing world in this
    metric before believing its move count: `Rw2` is 4 moves here.

SUPERCUBE vs COLOUR CUBE
------------------------
    Odd n and n=6 competition data ship `central_state == identity`: every
    sticker distinct, so a state IS a group element and `invert_state` is well
    defined. cube444 ships a 6-colour state and is a Schreier coset graph
    instead. `is_supercube` records which one you have; `invert_state` raises
    on a colour cube rather than returning something wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

U, F, R, B, L, D = 0, 1, 2, 3, 4, 5
FACE_NAMES = ("U", "F", "R", "B", "L", "D")
MOVE_SEPARATOR = "."

# Rotation senses that reproduce the shipped data exactly (fitted at n=4, then
# confirmed at n=6). Do not change these without re-running verify_against_data.
_SIGN = {0: -1, 1: +1, 2: -1}  # axis 0=x (r moves), 1=y (d moves), 2=z (f moves)

_NORMALS = {U: (0, 1, 0), F: (0, 0, 1), R: (1, 0, 0),
            B: (0, 0, -1), L: (-1, 0, 0), D: (0, -1, 0)}


def _face_cell(face: int, r: int, c: int, n: int) -> tuple[int, int, int]:
    m = n - 1
    if face == U:
        return (c, m, r)
    if face == F:
        return (c, m - r, m)
    if face == R:
        return (m, m - r, m - c)
    if face == B:
        return (m - c, m - r, 0)
    if face == L:
        return (0, m - r, c)
    if face == D:
        return (c, 0, m - r)
    raise ValueError(f"bad face {face}")


def _rot_pos(v, axis: int, sign: int, n: int):
    x, y, z = v
    m = n - 1
    if axis == 0:
        return (x, m - z, y) if sign > 0 else (x, z, m - y)
    if axis == 1:
        return (z, y, m - x) if sign > 0 else (m - z, y, x)
    return (m - y, x, z) if sign > 0 else (y, m - x, z)


def _rot_dir(v, axis: int, sign: int):
    x, y, z = v
    if axis == 0:
        return (x, -z, y) if sign > 0 else (x, z, -y)
    if axis == 1:
        return (z, y, -x) if sign > 0 else (-z, y, x)
    return (-y, x, z) if sign > 0 else (y, -x, z)


def _sticker_maps(n: int):
    idx_of, info = {}, {}
    for face in range(6):
        for r in range(n):
            for c in range(n):
                cell = _face_cell(face, r, c, n)
                idx = face * n * n + r * n + c
                idx_of[(cell, _NORMALS[face])] = idx
                info[idx] = (cell, _NORMALS[face])
    return idx_of, info


def build_generators(n: int) -> dict[str, tuple[int, ...]]:
    """All 6n single-slice quarter-turn generators for the n-cube."""
    idx_of, info = _sticker_maps(n)
    m = n - 1
    gens: dict[str, tuple[int, ...]] = {}
    for axis, letter in ((2, "f"), (0, "r"), (1, "d")):
        sign = _SIGN[axis]
        for layer in range(n):
            if axis == 2:
                coord, want = 2, m - layer
            elif axis == 0:
                coord, want = 0, m - layer
            else:
                coord, want = 1, layer
            dest = list(range(6 * n * n))
            for i, (cell, nrm) in info.items():
                if cell[coord] != want:
                    continue
                dest[i] = idx_of[(_rot_pos(cell, axis, sign, n), _rot_dir(nrm, axis, sign))]
            src = [0] * len(dest)
            for i, j in enumerate(dest):
                src[j] = i
            gens[f"{letter}{layer}"] = tuple(src)
            gens[f"-{letter}{layer}"] = tuple(dest)
    return gens


def piece_of_sticker(n: int) -> list[tuple[int, int, int]]:
    """Sticker index -> the cell it sits on. Stickers sharing a cell are one piece."""
    _, info = _sticker_maps(n)
    return [info[i][0] for i in range(6 * n * n)]


@dataclass(frozen=True)
class NCube:
    n: int
    solved_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    move_names: tuple[str, ...]
    is_supercube: bool

    @classmethod
    def build(cls, n: int, solved_state: Sequence[int] | None = None) -> "NCube":
        gens = build_generators(n)
        size = 6 * n * n
        solved = tuple(solved_state) if solved_state is not None else tuple(range(size))
        names = tuple(sorted(gens, key=lambda s: (s.lstrip("-"), s.startswith("-"))))
        return cls(n=n, solved_state=solved, generators=gens, move_names=names,
                   is_supercube=len(set(solved)) == size)

    @classmethod
    def from_puzzle_info(cls, path: str | Path) -> "NCube":
        """Load the competition file and CHECK our built generators reproduce it."""
        with open(path, encoding="utf-8") as fh:
            info = json.load(fh)
        solved = tuple(info["central_state"])
        size = len(solved)
        n = round((size / 6) ** 0.5)
        assert 6 * n * n == size, f"state size {size} is not 6n^2"
        cube = cls.build(n, solved)
        ref = {k: tuple(v) for k, v in info["generators"].items()}
        assert set(ref) == set(cube.generators), "generator names differ from the data"
        bad = [k for k in ref if ref[k] != cube.generators[k]]
        assert not bad, f"built generators differ from the data: {bad[:5]}"
        return cube

    @property
    def state_size(self) -> int:
        return 6 * self.n * self.n

    def apply_move(self, state: Sequence[int], move: str) -> tuple[int, ...]:
        g = self.generators[move]
        return tuple(state[i] for i in g)

    def apply_path(self, state: Sequence[int], path: Iterable[str]) -> tuple[int, ...]:
        cur = tuple(state)
        for mv in path:
            g = self.generators[mv]
            cur = tuple(cur[i] for i in g)
        return cur

    def is_solved(self, state: Sequence[int]) -> bool:
        return tuple(state) == self.solved_state

    @staticmethod
    def inverse_name(move: str) -> str:
        return move[1:] if move.startswith("-") else "-" + move

    def invert_path(self, path: Iterable[str]) -> list[str]:
        return [self.inverse_name(m) for m in reversed(list(path))]

    def invert_state(self, state: Sequence[int]) -> tuple[int, ...]:
        """Only valid on a supercube, where the state IS the group element."""
        if not self.is_supercube:
            raise ValueError("invert_state is undefined on a colour cube (see cube444)")
        out = [0] * len(state)
        for i, v in enumerate(state):
            out[v] = i
        return tuple(out)

    def parse_path(self, s: str) -> list[str]:
        return s.split(MOVE_SEPARATOR) if s.strip() else []

    def format_path(self, path: Iterable[str]) -> str:
        return MOVE_SEPARATOR.join(path)

    # ---- structure -------------------------------------------------------

    def pieces(self) -> dict[tuple[int, int, int], list[int]]:
        """Cell -> the sticker indices on it. 3 = corner, 2 = wing/midge, 1 = centre."""
        out: dict[tuple[int, int, int], list[int]] = {}
        for i, cell in enumerate(piece_of_sticker(self.n)):
            out.setdefault(cell, []).append(i)
        return out

    def sticker_orbits(self) -> list[list[int]]:
        """Orbits of sticker POSITIONS under the full move group, via union-find."""
        size = self.state_size
        parent = list(range(size))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for g in self.generators.values():
            for i, j in enumerate(g):
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
        groups: dict[int, list[int]] = {}
        for i in range(size):
            groups.setdefault(find(i), []).append(i)
        return sorted(groups.values(), key=lambda g: (len(g), g[0]))

    def generator_parity(self) -> dict[str, int]:
        """Sign of each forward generator: 1 = odd permutation, 0 = even."""
        out = {}
        for name in sorted(self.generators):
            if name.startswith("-"):
                continue
            g = self.generators[name]
            seen = [False] * len(g)
            transpositions = 0
            for i in range(len(g)):
                if seen[i]:
                    continue
                length = 0
                j = i
                while not seen[j]:
                    seen[j] = True
                    j = g[j]
                    length += 1
                transpositions += length - 1
            out[name] = transpositions % 2
        return out

    def outer_moves(self) -> list[str]:
        """The 12 outer-layer turns. These generate G1 = the 3x3x3 supercube group:
        they rotate each face's centre block rigidly and move each edge's wing
        group rigidly, never mixing centres between faces."""
        m = self.n - 1
        base = [f"{a}{i}" for a in ("f", "r", "d") for i in (0, m)]
        return base + ["-" + b for b in base]
