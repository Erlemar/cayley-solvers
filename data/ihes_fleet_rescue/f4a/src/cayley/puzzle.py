"""Picture Cube (IHES SuperCube 3x3x3) — state representation and move operators.

State: length-72 permutation (int). Solved state is identity [0, 1, ..., 71].
Moves: 18 generators — axes {f, r, d} x layers {0, 1, 2} x directions {CW, CCW}.
  Forward names: f0 f1 f2 r0 r1 r2 d0 d1 d2
  Inverse names: same with leading '-': -f0 -f1 ... -d2

Convention (matches CayleyPy and puzzle_info.json):
  apply(state, gen) -> new_state  where  new_state[i] = state[gen[i]]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

STATE_SIZE = 72
N_GENERATORS = 18
MOVE_SEPARATOR = "."


@dataclass(frozen=True)
class PictureCube:
    """Immutable puzzle definition loaded from puzzle_info.json."""

    solved_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    move_names: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "PictureCube":
        with open(path) as f:
            info = json.load(f)
        solved = tuple(info["central_state"])
        gens = {name: tuple(perm) for name, perm in info["generators"].items()}
        # Deterministic move order — matches puzzle_info.json insertion order.
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
        """Assert every generator composed with its inverse is identity."""
        for name in self.move_names:
            if name.startswith("-"):
                continue
            inv = self.inverse_name(name)
            assert inv in self.generators, f"missing inverse {inv} for {name}"
            forward = self.apply_move(self.solved_state, name)
            roundtrip = self.apply_move(forward, inv)
            assert roundtrip == self.solved_state, f"{name} . {inv} is not identity"

    def invert_state(self, state: Sequence[int]) -> tuple[int, ...]:
        """Return the group-theoretic inverse permutation of `state`.

        If `state = apply_path(solved, Q)` for some move sequence Q, then
        `invert_state(state) = apply_path(solved, Q^{-1})`. Used by NISS to get the
        "inverse scramble" as a valid puzzle state.
        """
        n = len(state)
        inv = [0] * n
        for i in range(n):
            inv[state[i]] = i
        return tuple(inv)

    def invert_path(self, path: Iterable[str]) -> list[str]:
        """Return the inverse path: reversed, with each move replaced by its inverse.

        Identity: `apply_path(s, path + invert_path(path)) == s`. Used by NISS to
        convert a solution for σ^{-1} into a solution for σ.
        """
        return [self.inverse_name(m) for m in reversed(list(path))]
