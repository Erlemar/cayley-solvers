"""Adapters for Daniel Walton's conventional-colour NxNxN cube solver.

The IHES 6x6x6 instance uses 216 distinct sticker labels and the primitive
``f/r/d`` slice generators.  Walton's solver instead consumes a six-colour
``URFDLB`` face string and emits standard cube notation, including wide and
half turns.  This module contains the lossless move translation and the
deliberately lossy colour projection used by the macro-mining experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


CUBE_SIZE = 6
FACE_SIZE = CUBE_SIZE * CUBE_SIZE
STICKER_COUNT = 6 * FACE_SIZE

# Native IHES/Santa face-block order inferred from the identity target and the
# move geometry.  Walton accepts Kociemba's URFDLB order.
NATIVE_FACE_ORDER = ("U", "F", "R", "B", "L", "D")
WALTON_FACE_ORDER = ("U", "R", "F", "D", "L", "B")
_NATIVE_INDEX_BY_FACE = {face: index for index, face in enumerate(NATIVE_FACE_ORDER)}

_MOVE_RE = re.compile(r"^(?:(\d+))?([URFDLB])(w)?(2|')?$")


def project_exact_state_to_walton(
    state: Sequence[int],
    target: Sequence[int],
) -> str:
    """Collapse unique stickers to target-face colours and return ``URFDLB``.

    Sticker colour is defined by the face block containing its target slot, not
    by its numeric value.  This keeps the adapter correct for any permutation of
    the identity labels in ``central_state``.
    """

    if len(state) != STICKER_COUNT or len(target) != STICKER_COUNT:
        raise ValueError(f"expected {STICKER_COUNT} stickers")
    if len(set(target)) != STICKER_COUNT:
        raise ValueError("target stickers must be unique")
    target_position = {sticker: position for position, sticker in enumerate(target)}
    try:
        native_colours = tuple(
            NATIVE_FACE_ORDER[target_position[sticker] // FACE_SIZE]
            for sticker in state
        )
    except KeyError as exc:
        raise ValueError(f"state sticker {exc.args[0]!r} is absent from target") from exc

    blocks = {
        face: native_colours[index * FACE_SIZE : (index + 1) * FACE_SIZE]
        for face, index in _NATIVE_INDEX_BY_FACE.items()
    }
    return "".join("".join(blocks[face]) for face in WALTON_FACE_ORDER)


def exact_state_is_colour_solved(
    state: Sequence[int],
    target: Sequence[int],
) -> bool:
    """Whether every exact sticker is on its target face, ignoring face position."""

    if len(state) != STICKER_COUNT or len(target) != STICKER_COUNT:
        raise ValueError(f"expected {STICKER_COUNT} stickers")
    target_position = {sticker: position for position, sticker in enumerate(target)}
    if len(target_position) != STICKER_COUNT:
        raise ValueError("target stickers must be unique")
    try:
        return all(
            target_position[sticker] // FACE_SIZE == position // FACE_SIZE
            for position, sticker in enumerate(state)
        )
    except KeyError as exc:
        raise ValueError(f"state sticker {exc.args[0]!r} is absent from target") from exc


def _base_wide_turn(face: str, width: int) -> tuple[str, ...]:
    if not 1 <= width <= CUBE_SIZE:
        raise ValueError(f"wide-turn width must be in 1..{CUBE_SIZE}, got {width}")
    if face == "U":
        return tuple(f"-d{layer}" for layer in range(CUBE_SIZE - width, CUBE_SIZE))
    if face == "D":
        return tuple(f"d{layer}" for layer in range(width))
    if face == "R":
        return tuple(f"r{layer}" for layer in range(width))
    if face == "L":
        return tuple(f"-r{layer}" for layer in range(CUBE_SIZE - width, CUBE_SIZE))
    if face == "F":
        return tuple(f"f{layer}" for layer in range(width))
    if face == "B":
        return tuple(f"-f{layer}" for layer in range(CUBE_SIZE - width, CUBE_SIZE))
    raise ValueError(f"unsupported face {face!r}")


def _invert_primitive(name: str) -> str:
    return name[1:] if name.startswith("-") else "-" + name


def translate_walton_move(move: str) -> tuple[str, ...]:
    """Expand one Walton move into legal IHES primitive quarter turns.

    Walton's final compressed paths use outer moves, ``Uw``-style two-layer
    moves, and explicit-width moves such as ``3Rw``.  Numeric non-wide slice
    notation is intentionally rejected: Walton expands those internally before
    recording its solution, and silently treating them as wide turns would be
    incorrect.
    """

    match = _MOVE_RE.fullmatch(move)
    if match is None:
        raise ValueError(f"unsupported Walton move {move!r}")
    width_text, face, wide_marker, suffix = match.groups()
    if width_text is not None and wide_marker is None:
        raise ValueError(
            f"unexpanded Walton slice move {move!r}; expected the solver's recorded wide form"
        )
    width = int(width_text) if width_text is not None else (2 if wide_marker else 1)
    primitive = _base_wide_turn(face, width)
    if suffix == "'":
        primitive = tuple(_invert_primitive(name) for name in primitive)
    elif suffix == "2":
        primitive = primitive + primitive
    return primitive


def translate_walton_path(path: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        primitive
        for move in path
        for primitive in translate_walton_move(move)
    )


@dataclass(frozen=True)
class WaltonPhase:
    label: str
    standard_moves: tuple[str, ...]


def split_marked_solution(marked_solution: Sequence[str]) -> tuple[WaltonPhase, ...]:
    """Split ``solution_with_markers`` into the phase words Walton recorded."""

    phases: list[WaltonPhase] = []
    pending: list[str] = []
    unnamed = 0
    for token in marked_solution:
        if token.startswith("COMMENT"):
            label = token.removeprefix("COMMENT_") or f"phase_{unnamed}"
            phases.append(WaltonPhase(label, tuple(pending)))
            pending.clear()
            unnamed += 1
        else:
            pending.append(token)
    if pending:
        phases.append(WaltonPhase(f"unmarked_tail_{unnamed}", tuple(pending)))
    return tuple(phases)
