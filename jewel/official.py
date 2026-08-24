"""Adapter for the competition's official 48-position representation.

The learned/search code uses compact cubie coordinates.  This module infers
the coordinate relabelling from ``puzzle_info.json`` and makes every reported
path independently replayable with the exact Kaggle permutations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .puzzle import (
    ACTION_EDGE_FLIP,
    ACTION_EDGE_SRC,
    ACTION_NAMES,
    ACTION_TO_INDEX,
    ACTION_RING_DELTA,
    BASE_ACTION_NAMES,
    JewelState,
)


def _edge_tables(permutations: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Extract 12 edge-piece sources and flips from 48-position moves."""
    srcs = np.empty((len(permutations), 12), dtype=np.uint8)
    flips = np.empty_like(srcs)
    for action_i, permutation in enumerate(permutations):
        for slot in range(12):
            pair = permutation[2 * slot : 2 * slot + 2]
            if pair[0] // 2 != pair[1] // 2 or int(pair[0]) ^ int(pair[1]) != 1:
                raise ValueError(f"official move does not preserve edge pairs: action={action_i}, slot={slot}")
            srcs[action_i, slot] = pair[0] // 2
            flips[action_i, slot] = pair[0] & 1
    return srcs, flips


def _infer_edge_slot_map(
    official_src: np.ndarray, official_flip: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find edge conjugacy and the official-positive -> internal action map."""
    structural_candidates = 0
    for base_permutation in permutations(range(6)):
      for direction_mask in range(1 << 6):
        direction = [(direction_mask >> i) & 1 for i in range(6)]
        internal_indices = np.asarray(
            [2 * base_permutation[i] + direction[i] for i in range(6)], dtype=np.uint8
        )
        internal_src = ACTION_EDGE_SRC[internal_indices]
        internal_flip = ACTION_EDGE_FLIP[internal_indices]
        for seed_official in range(12):
            mapping = np.full(12, -1, dtype=np.int16)
            mapping[0] = seed_official
            queue = [0]
            consistent = True
            while queue and consistent:
                internal_slot = queue.pop()
                official_slot = int(mapping[internal_slot])
                for action_i in range(6):
                    internal_next = int(internal_src[action_i, internal_slot])
                    official_next = int(official_src[action_i, official_slot])
                    if mapping[internal_next] < 0:
                        mapping[internal_next] = official_next
                        queue.append(internal_next)
                    elif mapping[internal_next] != official_next:
                        consistent = False
                        break
            if not consistent or np.any(mapping < 0) or len(set(map(int, mapping))) != 12:
                continue
            for action_i in range(6):
                for internal_slot in range(12):
                    if mapping[int(internal_src[action_i, internal_slot])] != official_src[
                        action_i, int(mapping[internal_slot])
                    ]:
                        consistent = False
            if not consistent:
                continue
            structural_candidates += 1

            # A reference-face gauge h satisfies
            # flip_int(a,t) = flip_off(a,f(t)) xor h[t] xor h[src_int(a,t)].
            gauge = np.full(12, -1, dtype=np.int8)
            gauge[0] = 0
            queue = [0]
            while queue and consistent:
                internal_slot = queue.pop()
                for action_i in range(6):
                    internal_next = int(internal_src[action_i, internal_slot])
                    required = (
                        int(gauge[internal_slot])
                        ^ int(internal_flip[action_i, internal_slot])
                        ^ int(official_flip[action_i, int(mapping[internal_slot])])
                    )
                    if gauge[internal_next] < 0:
                        gauge[internal_next] = required
                        queue.append(internal_next)
                    elif gauge[internal_next] != required:
                        consistent = False
                        break
            if consistent and not np.any(gauge < 0):
                return (
                    mapping.astype(np.uint8),
                    gauge.astype(np.uint8),
                    internal_indices,
                )
    raise ValueError(
        f"no full edge-coordinate isomorphism was found ({structural_candidates} structural candidates)"
    )


def _infer_ring_map(base_permutations: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return internal-ring -> official-block map and all four block patterns."""
    ring_map = np.full(6, -1, dtype=np.int16)
    patterns = np.empty((6, 4, 4), dtype=np.uint8)
    central = np.arange(48, dtype=np.uint8)
    for action_i, permutation in enumerate(base_permutations):
        moved_blocks = []
        for block in range(6):
            sl = slice(24 + 4 * block, 28 + 4 * block)
            if not np.array_equal(permutation[sl], central[sl]):
                moved_blocks.append(block)
        if len(moved_blocks) != 1:
            raise ValueError(f"base action {action_i} moves {len(moved_blocks)} ring blocks")
        internal_ring = int(np.flatnonzero(ACTION_RING_DELTA[2 * action_i])[0])
        official_block = moved_blocks[0]
        ring_map[internal_ring] = official_block
        state = central.copy()
        sl = slice(24 + 4 * official_block, 28 + 4 * official_block)
        for power in range(4):
            patterns[internal_ring, power] = state[sl]
            state = state[permutation]
        if not np.array_equal(state, central):
            raise ValueError(f"official action {action_i} is not order four")
    if np.any(ring_map < 0) or len(set(map(int, ring_map))) != 6:
        raise ValueError("ring-coordinate mapping is incomplete")
    return ring_map.astype(np.uint8), patterns


@dataclass(slots=True)
class OfficialPuzzle:
    central_state: np.ndarray
    generators: Mapping[str, np.ndarray]
    edge_slot_map: np.ndarray
    edge_orientation_gauge: np.ndarray
    internal_to_official_action: np.ndarray
    ring_block_map: np.ndarray
    ring_patterns: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "OfficialPuzzle":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        central = np.asarray(raw["central_state"], dtype=np.uint8)
        generators = {name: np.asarray(move, dtype=np.uint8) for name, move in raw["generators"].items()}
        if not np.array_equal(central, np.arange(48, dtype=np.uint8)):
            raise ValueError("the adapter expects the official identity-labelled central state")
        missing = set(ACTION_NAMES) - set(generators)
        if missing:
            raise ValueError(f"missing official generators: {sorted(missing)}")

        action_permutations = [generators[name] for name in ACTION_NAMES]
        for action_i in range(0, 12, 2):
            forward = action_permutations[action_i]
            inverse = action_permutations[action_i + 1]
            if not np.array_equal(forward[inverse], central) or not np.array_equal(inverse[forward], central):
                raise ValueError(f"official inverse mismatch for {ACTION_NAMES[action_i]}")

        official_src, official_flip = _edge_tables(action_permutations[::2])
        edge_map, edge_gauge, official_positive_to_internal = _infer_edge_slot_map(
            official_src, official_flip
        )
        official_to_internal = np.empty(12, dtype=np.uint8)
        for official_base, internal_positive in enumerate(official_positive_to_internal):
            official_to_internal[2 * official_base] = internal_positive
            official_to_internal[2 * official_base + 1] = int(internal_positive) ^ 1
        internal_to_official = np.empty(12, dtype=np.uint8)
        internal_to_official[official_to_internal] = np.arange(12, dtype=np.uint8)
        official_for_internal_base = [
            action_permutations[int(internal_to_official[2 * base_i])] for base_i in range(6)
        ]
        ring_map, ring_patterns = _infer_ring_map(official_for_internal_base)
        puzzle = cls(
            central,
            generators,
            edge_map,
            edge_gauge,
            internal_to_official,
            ring_map,
            ring_patterns,
        )
        puzzle.validate_action_isomorphism()
        return puzzle

    def apply_action(self, state: np.ndarray, action: int | str) -> np.ndarray:
        name = (
            ACTION_NAMES[int(self.internal_to_official_action[int(action)])]
            if not isinstance(action, str)
            else action
        )
        return np.asarray(state, dtype=np.uint8)[self.generators[name]]

    def apply_path(self, state: np.ndarray, actions: Iterable[int | str]) -> np.ndarray:
        out = np.asarray(state, dtype=np.uint8)
        for action in actions:
            out = self.apply_action(out, action)
        return out

    def parse_path(self, path: str) -> list[int]:
        if not path:
            return []
        official_to_internal = np.empty(12, dtype=np.uint8)
        official_to_internal[self.internal_to_official_action] = np.arange(12, dtype=np.uint8)
        return [int(official_to_internal[ACTION_TO_INDEX[token]]) for token in path.split(".")]

    def format_path(self, actions: Sequence[int]) -> str:
        return ".".join(
            ACTION_NAMES[int(self.internal_to_official_action[int(action)])] for action in actions
        )

    def to_structured(self, stickers: Sequence[int] | np.ndarray) -> JewelState:
        stickers = np.asarray(stickers, dtype=np.uint8)
        if stickers.shape != (48,) or len(np.unique(stickers)) != 48:
            raise ValueError("official state must be a permutation of 0..47")
        inverse_edge_map = np.empty(12, dtype=np.uint8)
        inverse_edge_map[self.edge_slot_map] = np.arange(12, dtype=np.uint8)
        edge_perm = np.empty(12, dtype=np.uint8)
        edge_ori = np.empty(12, dtype=np.uint8)
        for internal_slot, official_slot in enumerate(self.edge_slot_map):
            pair = stickers[2 * int(official_slot) : 2 * int(official_slot) + 2]
            if pair[0] // 2 != pair[1] // 2 or int(pair[0]) ^ int(pair[1]) != 1:
                raise ValueError(f"invalid edge pair in official slot {official_slot}")
            internal_piece = int(inverse_edge_map[pair[0] // 2])
            edge_perm[internal_slot] = internal_piece
            edge_ori[internal_slot] = (
                int(pair[0] & 1)
                ^ int(self.edge_orientation_gauge[internal_slot])
                ^ int(self.edge_orientation_gauge[internal_piece])
            )

        ring_ori = np.empty(6, dtype=np.uint8)
        for internal_ring, official_block in enumerate(self.ring_block_map):
            sl = slice(24 + 4 * int(official_block), 28 + 4 * int(official_block))
            observed = stickers[sl]
            matches = [power for power in range(4) if np.array_equal(observed, self.ring_patterns[internal_ring, power])]
            if len(matches) != 1:
                raise ValueError(f"invalid official ring block {official_block}")
            ring_ori[internal_ring] = matches[0]
        state = JewelState(edge_perm, edge_ori, ring_ori)
        state.validate()
        return state

    def from_structured(self, state: JewelState) -> np.ndarray:
        state.validate()
        stickers = np.empty(48, dtype=np.uint8)
        for internal_slot, official_slot in enumerate(self.edge_slot_map):
            official_piece = int(self.edge_slot_map[int(state.edge_perm[internal_slot])])
            internal_piece = int(state.edge_perm[internal_slot])
            orientation = (
                int(state.edge_ori[internal_slot])
                ^ int(self.edge_orientation_gauge[internal_slot])
                ^ int(self.edge_orientation_gauge[internal_piece])
            )
            start = 2 * int(official_slot)
            stickers[start] = 2 * official_piece + orientation
            stickers[start + 1] = 2 * official_piece + (orientation ^ 1)
        for internal_ring, official_block in enumerate(self.ring_block_map):
            sl = slice(24 + 4 * int(official_block), 28 + 4 * int(official_block))
            stickers[sl] = self.ring_patterns[internal_ring, int(state.ring_ori[internal_ring])]
        return stickers

    def validate_action_isomorphism(self) -> None:
        """Prove all twelve named moves agree in both representations."""
        from .puzzle import SOLVED, apply_action

        if not np.array_equal(self.from_structured(SOLVED), self.central_state):
            raise ValueError("central-state coordinate conversion mismatch")
        for action_i in range(12):
            official = self.apply_action(self.central_state, action_i)
            structured = apply_action(SOLVED, action_i)
            if self.to_structured(official).rank() != structured.rank():
                raise ValueError(f"action-coordinate mismatch: {ACTION_NAMES[action_i]}")
            if not np.array_equal(self.from_structured(structured), official):
                raise ValueError(f"reverse action-coordinate mismatch: {ACTION_NAMES[action_i]}")


def parse_official_state(text: str) -> np.ndarray:
    return np.fromstring(text, sep=",", dtype=np.uint8)
