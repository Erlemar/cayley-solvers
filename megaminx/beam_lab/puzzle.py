"""Megaminx puzzle — state representation and move operators.

State: length-120 permutation. Solved = identity [0..119].
Moves: 24 generators (12 face rotations × CW/CCW).

This is the same class our main pipeline uses, copied here so beam_lab is
self-contained (no PYTHONPATH gymnastics needed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

STATE_SIZE = 120
N_GENERATORS = 24


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
        return cls(solved_state=solved, generators=gens, move_names=tuple(gens.keys()))

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
