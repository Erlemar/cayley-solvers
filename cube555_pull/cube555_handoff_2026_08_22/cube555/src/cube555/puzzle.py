"""CayleyPy 5x5x5 Cube -- state representation and move operators.

DIFFERENT IN KIND FROM cube444. Read this before porting anything.

`data/puzzle_info.json` has `central_state == [0, 1, ..., 149]`: every sticker is
DISTINCT. This is the PICTURE (super) cube, a true Cayley graph, not the colour cube
that cube444 was. Three consequences, all of which invert a cube444 rule:

  * `num_classes` is **150**, not 6. cube444's rule 1 ("always 6, never state_size")
    is a colour-cube rule; here the correct value IS state_size. Still pass it
    explicitly at every construction site.
  * A state IS a group element, so `invert_state` is well defined (below). Inverse
    frames, and d(s) == d(s^-1), are legal here. cube444 rule 2 does not transfer.
  * Symmetry is CONJUGATION, `sym(s, M) = Minv[s[M]]`, not a recolouring. There are
    no colour maps.

Moves: 30 generators -- axes {f, r, d} x layers {0..4} x directions {CW, CCW}.
  Forward names: f0 f1 f2 f3 f4 r0 ... d4;  inverses the same with a leading '-'.
  Name index convention (used by the symmetry derivation): axis*10 + layer*2 + sign.

Convention (matches the competition data and CayleyPy):
  apply(state, gen) -> new_state  where  new_state[i] = state[gen[i]]

so, reading a state as a permutation function, `apply` RIGHT-composes:
  s_0 = identity,  s_k = g_1 o g_2 o ... o g_k   with   (a o b)[i] = a[b[i]].

Measured structure (10_derive_symmetry / 11_build_bfs, 2026-08-15):
  |G| = 6.198e91 (exact, Schreier-Sims)   branching factor -> 24.03
  BFS levels 1..5 = 30, 735, 17760, 427877, 10292614
  counting bound 66.4  =>  diameter ~70-76  =>  k_max ~80
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

STATE_SIZE = 150
N_GENERATORS = 30
NUM_CLASSES = 150  # picture cube: this really is state_size. Pass it explicitly anyway.
N_AXES = 3
N_LAYERS = 5
MOVE_SEPARATOR = "."


def gen_index(axis: int, layer: int, sign: int) -> int:
    """Position of a generator in the puzzle_info name order: f0,-f0,f1,-f1,...,d4,-d4."""
    return axis * (2 * N_LAYERS) + layer * 2 + sign


@dataclass(frozen=True)
class Cube555:
    """Immutable puzzle definition loaded from puzzle_info.json."""

    solved_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    move_names: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Cube555":
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        gens = {k: tuple(int(x) for x in v) for k, v in info["generators"].items()}
        names = tuple(gens.keys())
        solved = tuple(int(x) for x in info["central_state"])
        if len(solved) != STATE_SIZE:
            raise ValueError(f"expected {STATE_SIZE} stickers, got {len(solved)}")
        if len(names) != N_GENERATORS:
            raise ValueError(f"expected {N_GENERATORS} generators, got {len(names)}")
        if tuple(sorted(solved)) != tuple(range(STATE_SIZE)):
            raise ValueError(
                "central_state is not a permutation of 0..149 -- this loader is for the "
                "PICTURE cube. A colour cube needs the cube444 module instead."
            )
        return cls(solved_state=solved, generators=gens, move_names=names)

    # ---- group operations ---------------------------------------------------------
    @staticmethod
    def apply(state: Sequence[int], gen: Sequence[int]) -> tuple[int, ...]:
        return tuple(state[i] for i in gen)

    @staticmethod
    def invert_state(state: Sequence[int]) -> tuple[int, ...]:
        """s^-1. Defined here (unlike cube444) because the state is a permutation.

        d(s) == d(s^-1) exactly, so solving s^-1 and mapping the path back --
        REVERSE the move list and INVERT each move -- is an exactly valid, and
        genuinely decorrelated, second search trajectory.
        """
        out = [0] * len(state)
        for i, v in enumerate(state):
            out[v] = i
        return tuple(out)

    def inverse_name(self, name: str) -> str:
        return name[1:] if name.startswith("-") else "-" + name

    def invert_path(self, path: Iterable[str]) -> list[str]:
        """Path that solves s, given a path that solves s^-1 (and vice versa)."""
        return [self.inverse_name(m) for m in reversed(list(path))]

    # ---- io -----------------------------------------------------------------------
    def format_path(self, moves: Iterable[str]) -> str:
        return MOVE_SEPARATOR.join(moves)

    def parse_path(self, s: str) -> list[str]:
        return [m for m in s.split(MOVE_SEPARATOR) if m]

    def replay(self, state: Sequence[int], moves: Iterable[str]) -> tuple[int, ...]:
        cur = tuple(state)
        for m in moves:
            cur = self.apply(cur, self.generators[m])
        return cur

    def is_solved(self, state: Sequence[int]) -> bool:
        return tuple(state) == self.solved_state
