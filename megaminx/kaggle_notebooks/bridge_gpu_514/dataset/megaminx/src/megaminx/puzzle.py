"""Megaminx (dodecahedron, 12-face) — state representation and move operators.

State: length-120 permutation (int). Solved state is identity [0, 1, ..., 119].
Moves: 24 generators — 12 face rotations each with CW and CCW (inverse) variants.
  Forward names: U, D, F, B, L, R, DR, DL, FR, FL, BR, BL
  Inverse names: same with leading '-': -U, -D, ..., -BL

The data format matches the IHES picture cube: puzzle_info.json has keys
`central_state` (the solved-state vector) and `generators` (name -> permutation).

Convention (matches the competition data and CayleyPy):
  apply(state, gen) -> new_state  where  new_state[i] = state[gen[i]]

This class is structurally identical to `cayley.puzzle.PictureCube` so shared modules
(`cayley.data`, `cayley.training`, `cayley.khoruzhii_search`, etc.) accept it via
duck typing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

STATE_SIZE = 120
N_GENERATORS = 24
MOVE_SEPARATOR = "."


@dataclass(frozen=True)
class Megaminx:
    solved_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    move_names: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Megaminx":
        with open(path) as f:
            info = json.load(f)
        solved = tuple(info["central_state"])
        gens = {name: tuple(perm) for name, perm in info["generators"].items()}
        names = tuple(gens.keys())
        assert len(solved) == STATE_SIZE, f"expected state size {STATE_SIZE}, got {len(solved)}"
        assert len(gens) == N_GENERATORS, f"expected {N_GENERATORS} generators, got {len(gens)}"
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

    def verify_inverse_pairs(self) -> None:
        for name in self.move_names:
            if name.startswith("-"):
                continue
            inv = self.inverse_name(name)
            assert inv in self.generators, f"missing inverse {inv} for {name}"
            forward = self.apply_move(self.solved_state, name)
            roundtrip = self.apply_move(forward, inv)
            assert roundtrip == self.solved_state, f"{name} . {inv} is not identity"

    def invert_state(self, state: Sequence[int]) -> tuple[int, ...]:
        n = len(state)
        inv = [0] * n
        for i in range(n):
            inv[state[i]] = i
        return tuple(inv)

    def invert_path(self, path: Iterable[str]) -> list[str]:
        return [self.inverse_name(m) for m in reversed(list(path))]
