"""CayleyPy 4x4x4 Cube -- state representation and move operators.

State: length-96 COLOR vector over {0..5}, 16 stickers of each color.
Solved state is `central_state` from puzzle_info.json (face-major: 16x color 0,
16x color 1, ...).

Moves: 24 generators -- axes {f, r, d} x layers {0,1,2,3} x directions {CW, CCW}.
  Forward names: f0 f1 f2 f3 r0 r1 r2 r3 d0 d1 d2 d3
  Inverse names: same with leading '-': -f0 -f1 ... -d3
All 12 layers turn independently; quarter turns only.

Convention (matches the competition data and CayleyPy):
  apply(state, gen) -> new_state  where  new_state[i] = state[gen[i]]

DIFFERENCE FROM THE OTHER PUZZLES IN THIS REPO
----------------------------------------------
`cayley.puzzle.PictureCube` and `megaminx.puzzle.Megaminx` both have
`solved_state == (0, 1, ..., N-1)`: every state is a permutation, so the state
determines the group element and `invert_state` is well-defined.

Here the state is a *coloring*, not a permutation. 16 stickers share each color,
so a state pins the group element only up to the stabiliser of the solved
coloring (the 4 same-colored centers of each face are interchangeable:
4!^6 = 191,102,976). This is a Schreier coset graph, not a Cayley graph.

Consequences, deliberately encoded below:
  * There is NO `invert_state`. NISS and inverse-frame sym-ensembling are not
    available. (Rule 11 drops NISS anyway.)
  * `num_classes` is 6, NOT `state_size`. Every model constructor in this repo
    defaults `num_classes` to the state size -- always pass it explicitly.

This class is otherwise structurally identical to the other two, so the shared
modules (`cayley.data`, `cayley.training`, `cayley.bellman`,
`cayley.khoruzhii_search`) accept it via duck typing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

STATE_SIZE = 96
N_GENERATORS = 24
NUM_CLASSES = 6
MOVE_SEPARATOR = "."


@dataclass(frozen=True)
class Cube444:
    """Immutable puzzle definition loaded from puzzle_info.json."""

    solved_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    move_names: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Cube444":
        with open(path, encoding="utf-8") as f:
            info = json.load(f)
        solved = tuple(info["central_state"])
        gens = {name: tuple(perm) for name, perm in info["generators"].items()}
        names = tuple(gens.keys())
        assert len(solved) == STATE_SIZE, f"expected state size {STATE_SIZE}, got {len(solved)}"
        assert len(gens) == N_GENERATORS, f"expected {N_GENERATORS} generators, got {len(gens)}"
        assert len(set(solved)) == NUM_CLASSES, f"expected {NUM_CLASSES} colors, got {len(set(solved))}"
        return cls(solved_state=solved, generators=gens, move_names=names)

    def inverse_name(self, name: str) -> str:
        return name[1:] if name.startswith("-") else "-" + name

    def apply_move(self, state: Sequence[int], move_name: str) -> tuple[int, ...]:
        gen = self.generators[move_name]
        return tuple(state[g] for g in gen)

    def apply_path(self, state: Sequence[int], path: Iterable[str]) -> tuple[int, ...]:
        cur = tuple(state)
        for m in path:
            cur = self.apply_move(cur, m)
        return cur

    def is_solved(self, state: Sequence[int]) -> bool:
        return tuple(state) == self.solved_state

    def parse_path(self, path_str: str) -> list[str]:
        if not path_str.strip():
            return []
        return path_str.split(MOVE_SEPARATOR)

    def format_path(self, path: Iterable[str]) -> str:
        return MOVE_SEPARATOR.join(path)

    def invert_path(self, path: Iterable[str]) -> list[str]:
        """Reversed path with each move replaced by its inverse.

        Identity: apply_path(s, path + invert_path(path)) == s. This IS well
        defined (it acts on move words, not on states) and is what turns a
        generating random walk into a solution.
        """
        return [self.inverse_name(m) for m in reversed(list(path))]

    def verify_inverse_pairs(self) -> None:
        """Assert every generator composed with its inverse is the identity."""
        for name in self.move_names:
            if name.startswith("-"):
                continue
            inv = self.inverse_name(name)
            assert inv in self.generators, f"missing inverse {inv} for {name}"
            forward = self.apply_move(self.solved_state, name)
            roundtrip = self.apply_move(forward, inv)
            assert roundtrip == self.solved_state, f"{name} . {inv} is not identity"

    # Deliberately absent: invert_state(). See module docstring.
