"""Length-preserving symmetry and inverse formulations for exact-sticker cubes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


def invert_state(state: Sequence[int]) -> tuple[int, ...]:
    """Invert a uniquely labelled permutation state."""

    size = len(state)
    if set(state) != set(range(size)):
        raise ValueError("inverse formulation requires a permutation of 0..N-1")
    inverse = [0] * size
    for position, sticker in enumerate(state):
        inverse[sticker] = position
    return tuple(inverse)


def inverse_move_name(name: str) -> str:
    return name[1:] if name.startswith("-") else "-" + name


def invert_path(path: Sequence[str]) -> tuple[str, ...]:
    return tuple(inverse_move_name(name) for name in reversed(path))


@dataclass(frozen=True)
class FrameTransform:
    """One spatial conjugation, optionally followed by group inversion.

    ``sticker_permutation`` and ``move_to_transformed`` describe a spatial
    automorphism.  A ``None`` permutation is the identity spatial frame.  The
    inverse formulation is applied after the spatial conjugation.
    """

    label: str
    move_names: tuple[str, ...]
    sticker_permutation: tuple[int, ...] | None = None
    move_to_transformed: Mapping[str, str] | None = None
    use_inverse: bool = False

    def __post_init__(self) -> None:
        expected_moves = set(self.move_names)
        if self.sticker_permutation is None:
            if self.move_to_transformed is not None:
                raise ValueError("identity spatial frame cannot have a relabel map")
        else:
            size = len(self.sticker_permutation)
            if set(self.sticker_permutation) != set(range(size)):
                raise ValueError("sticker_permutation is not bijective")
            if self.move_to_transformed is None:
                raise ValueError("spatial frame requires a move relabel map")
            if set(self.move_to_transformed) != expected_moves:
                raise ValueError("move relabel keys differ from the generator set")
            if set(self.move_to_transformed.values()) != expected_moves:
                raise ValueError("move relabel values are not bijective")

    @property
    def move_to_original(self) -> dict[str, str]:
        if self.move_to_transformed is None:
            return {name: name for name in self.move_names}
        return {transformed: original for original, transformed in self.move_to_transformed.items()}

    def _conjugate_state(self, state: Sequence[int]) -> tuple[int, ...]:
        if self.sticker_permutation is None:
            return tuple(state)
        permutation = self.sticker_permutation
        if len(state) != len(permutation):
            raise ValueError("state and frame sizes differ")
        inverse_positions = [0] * len(permutation)
        for old_position, new_position in enumerate(permutation):
            inverse_positions[new_position] = old_position
        # Exact sticker labels transform with positions, so identity remains
        # fixed under the spatial frame.
        return tuple(
            permutation[state[inverse_positions[new_position]]]
            for new_position in range(len(permutation))
        )

    def transform_state(self, state: Sequence[int]) -> tuple[int, ...]:
        transformed = self._conjugate_state(state)
        return invert_state(transformed) if self.use_inverse else transformed

    def transform_path(self, path: Sequence[str]) -> tuple[str, ...]:
        if self.move_to_transformed is None:
            transformed = tuple(path)
        else:
            transformed = tuple(self.move_to_transformed[name] for name in path)
        return invert_path(transformed) if self.use_inverse else transformed

    def restore_path(self, path: Sequence[str]) -> tuple[str, ...]:
        restored = invert_path(path) if self.use_inverse else tuple(path)
        move_to_original = self.move_to_original
        return tuple(move_to_original[name] for name in restored)

