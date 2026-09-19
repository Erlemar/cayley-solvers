"""I/O and move application for the exact-sticker 6x6x6 puzzle.

The competition uses CayleyPy's pullback convention::

    new_state[i] = old_state[generator[i]]

Keeping that convention explicit is important: composing the arrays in the other
order produces legal-looking permutations but invalid solution paths.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Cube666Puzzle:
    """An exact-sticker puzzle definition with named pullback permutations."""

    solved_state: tuple[int, ...]
    generators: dict[str, tuple[int, ...]]
    move_names: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "Cube666Puzzle":
        with open(path, encoding="utf-8") as handle:
            info = json.load(handle)

        try:
            solved = tuple(int(value) for value in info["central_state"])
            raw_generators = info["generators"]
        except KeyError as exc:
            raise ValueError(f"missing puzzle_info field: {exc.args[0]}") from exc

        generators = {
            str(name): tuple(int(value) for value in permutation)
            for name, permutation in raw_generators.items()
        }
        puzzle = cls(solved, generators, tuple(generators))
        puzzle.validate()
        return puzzle

    @property
    def size(self) -> int:
        return len(self.solved_state)

    def validate(self) -> None:
        n = self.size
        expected = set(range(n))
        if set(self.solved_state) != expected:
            raise ValueError("central_state must be a permutation of 0..N-1")
        if not self.generators:
            raise ValueError("puzzle has no generators")
        for name, permutation in self.generators.items():
            if len(permutation) != n or set(permutation) != expected:
                raise ValueError(f"generator {name!r} is not a permutation of 0..{n - 1}")

    @staticmethod
    def inverse_name(name: str) -> str:
        return name[1:] if name.startswith("-") else "-" + name

    def apply_move(self, state: Sequence[int], move_name: str) -> tuple[int, ...]:
        if len(state) != self.size:
            raise ValueError(f"expected state length {self.size}, got {len(state)}")
        permutation = self.generators[move_name]
        return tuple(state[index] for index in permutation)

    def apply_path(self, state: Sequence[int], path: Iterable[str]) -> tuple[int, ...]:
        current = tuple(state)
        for move_name in path:
            current = self.apply_move(current, move_name)
        return current

    def verify_inverse_pairs(self) -> None:
        for name in self.move_names:
            if name.startswith("-"):
                continue
            inverse = self.inverse_name(name)
            if inverse not in self.generators:
                raise ValueError(f"missing inverse {inverse!r} for {name!r}")
            roundtrip = self.apply_path(self.solved_state, (name, inverse))
            if roundtrip != self.solved_state:
                raise ValueError(f"{name!r} followed by {inverse!r} is not identity")

    @staticmethod
    def parse_state(text: str) -> tuple[int, ...]:
        """Parse the comma-separated state representation used by test.csv."""

        return tuple(int(token) for token in text.split(",") if token != "")

    def iter_test_states(self, path: str | Path) -> Iterator[tuple[str, tuple[int, ...]]]:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"initial_state_id", "initial_state"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"test CSV must contain columns {sorted(required)}")
            for row in reader:
                state = self.parse_state(row["initial_state"])
                if len(state) != self.size:
                    raise ValueError(
                        f"state {row['initial_state_id']} has length {len(state)}, expected {self.size}"
                    )
                yield row["initial_state_id"], state
