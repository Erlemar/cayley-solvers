"""Professor Tetraminx (CayleyPy competition) -- state representation and moves.

State: length-88 permutation of ints. Solved state is identity [0, 1, ..., 87].
Moves: 24 generators -- axes {D, F, BL, BR} x layers {2, 3, 4} x directions.
  Forward names: 2D 3D 4D 2F 3F 4F 2BL 3BL 4BL 2BR 3BR 4BR
  Inverse names: same with leading '-'.
Every generator has order 3 (they are products of 3-cycles):
  layer 2 moves 21 facelets, layer 3 moves 15, layer 4 moves 9.

Convention (matches CayleyPy and puzzle_info.json):
  apply(state, gen) -> new_state  where  new_state[i] = state[gen[i]]

Duck-types cayley.puzzle.PictureCube so the shared src/cayley/* stack
(model, bellman, khoruzhii_search, data, post_process, verify) works unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

STATE_SIZE = 88
N_GENERATORS = 24
MOVE_SEPARATOR = "."


@dataclass(frozen=True)
class Tetraminx:
    """Immutable puzzle definition loaded from puzzle_info.json."""

    solved_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    move_names: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Tetraminx":
        with open(path, encoding="utf-8") as f:
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
        """Assert every generator composed with its inverse is the identity."""
        for name in self.move_names:
            if name.startswith("-"):
                continue
            inv = self.inverse_name(name)
            assert inv in self.generators, f"missing inverse {inv} for {name}"
            forward = self.apply_move(self.solved_state, name)
            roundtrip = self.apply_move(forward, inv)
            assert roundtrip == self.solved_state, f"{name} . {inv} is not identity"

    def verify_order_three(self) -> None:
        """Assert every generator has order 3 (tetrahedral vertex turns)."""
        for name in self.move_names:
            s = self.apply_move(self.apply_move(self.apply_move(self.solved_state, name), name), name)
            assert s == self.solved_state, f"{name} does not have order 3"

    def invert_state(self, state: Sequence[int]) -> tuple[int, ...]:
        """Return the group-theoretic inverse permutation of `state`.

        dist(s) == dist(s^-1) because every generator's inverse is also a
        generator, so inverting a length-L solution gives a length-L solution.
        This is the inverse-antisymmetry used to double the sym-ensemble.
        """
        inv = [0] * len(state)
        for i, v in enumerate(state):
            inv[v] = i
        return tuple(inv)
