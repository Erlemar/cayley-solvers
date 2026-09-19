"""Exact optimal-QTM solver for the 6x6x6 corner projection.

Only the twelve directed outer face turns affect corners.  The module discovers
the eight physical corner cubies and their orientation frames from the 24 exact
sticker permutations, then solves the resulting 8! * 3**7 coordinate with IDA*.
Two small exact pattern databases (permutation and orientation) provide an
admissible heuristic.
"""

from __future__ import annotations

import math
from array import array
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

from cube666.classical import Cube666Decomposition, Permutation


N_CORNERS = 8
N_ORIENTATIONS = 3**7
N_PERMUTATIONS = math.factorial(N_CORNERS)
PATTERN_PIECES = 4
N_PATTERN_LOCATIONS = math.prod(range(N_CORNERS - PATTERN_PIECES + 1, N_CORNERS + 1))
N_PATTERN_STATES = N_PATTERN_LOCATIONS * (3**PATTERN_PIECES)


def _rank_permutation(permutation: Sequence[int]) -> int:
    if len(permutation) != N_CORNERS or set(permutation) != set(range(N_CORNERS)):
        raise ValueError("corner permutation must contain 0..7 exactly once")
    rank = 0
    for index in range(N_CORNERS - 1):
        smaller = sum(permutation[j] < permutation[index] for j in range(index + 1, N_CORNERS))
        rank += smaller * math.factorial(N_CORNERS - 1 - index)
    return rank


def _unrank_permutation(rank: int) -> tuple[int, ...]:
    if not 0 <= rank < N_PERMUTATIONS:
        raise ValueError(f"corner permutation rank must be in 0..{N_PERMUTATIONS - 1}")
    available = list(range(N_CORNERS))
    result: list[int] = []
    remainder = rank
    for positions_left in range(N_CORNERS, 0, -1):
        factorial = math.factorial(positions_left - 1)
        quotient, remainder = divmod(remainder, factorial)
        result.append(available.pop(quotient))
    return tuple(result)


def _rank_orientation(orientation: Sequence[int]) -> int:
    if len(orientation) != N_CORNERS or any(value not in (0, 1, 2) for value in orientation):
        raise ValueError("corner orientation must contain eight values in 0..2")
    if sum(orientation) % 3:
        raise ValueError("corner orientation sum must be zero modulo three")
    rank = 0
    multiplier = 1
    for value in orientation[:7]:
        rank += value * multiplier
        multiplier *= 3
    return rank


def _unrank_orientation(rank: int) -> tuple[int, ...]:
    if not 0 <= rank < N_ORIENTATIONS:
        raise ValueError(f"corner orientation rank must be in 0..{N_ORIENTATIONS - 1}")
    values: list[int] = []
    remainder = rank
    for _ in range(7):
        remainder, value = divmod(remainder, 3)
        values.append(value)
    values.append((-sum(values)) % 3)
    return tuple(values)


def _cyclic_shift(reference: Sequence[int], candidate: Sequence[int]) -> int | None:
    for shift in range(3):
        if all(candidate[index] == reference[(index + shift) % 3] for index in range(3)):
            return shift
    return None


def _rank_partial_locations(locations: Sequence[int]) -> int:
    if len(locations) != PATTERN_PIECES or len(set(locations)) != PATTERN_PIECES:
        raise ValueError(f"expected {PATTERN_PIECES} distinct corner locations")
    available = list(range(N_CORNERS))
    rank = 0
    for index, location in enumerate(locations):
        try:
            digit = available.index(location)
        except ValueError as exc:
            raise ValueError(f"invalid corner location {location}") from exc
        rank = rank * (N_CORNERS - index) + digit
        available.pop(digit)
    return rank


def _unrank_partial_locations(rank: int) -> tuple[int, ...]:
    if not 0 <= rank < N_PATTERN_LOCATIONS:
        raise ValueError(f"partial location rank must be in 0..{N_PATTERN_LOCATIONS - 1}")
    digits = [0] * PATTERN_PIECES
    remainder = rank
    for index in range(PATTERN_PIECES - 1, -1, -1):
        base = N_CORNERS - index
        remainder, digits[index] = divmod(remainder, base)
    available = list(range(N_CORNERS))
    locations: list[int] = []
    for digit in digits:
        locations.append(available.pop(digit))
    return tuple(locations)


def _rank_pattern(locations: Sequence[int], orientations: Sequence[int]) -> int:
    if len(orientations) != PATTERN_PIECES or any(value not in (0, 1, 2) for value in orientations):
        raise ValueError(f"expected {PATTERN_PIECES} corner orientations in 0..2")
    orientation_rank = 0
    multiplier = 1
    for value in orientations:
        orientation_rank += value * multiplier
        multiplier *= 3
    return _rank_partial_locations(locations) * (3**PATTERN_PIECES) + orientation_rank


def _unrank_pattern(rank: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not 0 <= rank < N_PATTERN_STATES:
        raise ValueError(f"pattern rank must be in 0..{N_PATTERN_STATES - 1}")
    location_rank, orientation_rank = divmod(rank, 3**PATTERN_PIECES)
    orientations: list[int] = []
    for _ in range(PATTERN_PIECES):
        orientation_rank, value = divmod(orientation_rank, 3)
        orientations.append(value)
    return _unrank_partial_locations(location_rank), tuple(orientations)


def _pattern_from_coordinate(
    permutation: Sequence[int],
    orientation: Sequence[int],
    tracked_pieces: Sequence[int],
) -> int:
    location_by_piece = [0] * N_CORNERS
    for location, piece in enumerate(permutation):
        location_by_piece[piece] = location
    locations = tuple(location_by_piece[piece] for piece in tracked_pieces)
    orientations = tuple(orientation[location] for location in locations)
    return _rank_pattern(locations, orientations)


@dataclass(frozen=True)
class CornerCoordinateSystem:
    """Move-derived physical corner blocks, frames, and coordinate transforms."""

    blocks: tuple[tuple[int, int, int], ...]
    frames: tuple[tuple[int, int, int], ...]
    move_names: tuple[str, ...]
    move_piece_sources: tuple[tuple[int, ...], ...]
    move_orientation_deltas: tuple[tuple[int, ...], ...]

    @classmethod
    def discover(
        cls,
        generators: Mapping[str, Sequence[int]],
        decomposition: Cube666Decomposition,
    ) -> "CornerCoordinateSystem":
        corner = decomposition.corner_orbit
        corner_set = set(corner)
        outer_names = tuple(
            name
            for name, permutation in generators.items()
            if any(permutation[position] != position for position in corner)
        )
        if len(outer_names) != 12:
            raise ValueError(f"expected 12 directed outer moves, found {len(outer_names)}")

        # Stickers on one physical corner have identical outer-move support.
        groups: dict[tuple[str, ...], list[int]] = {}
        for position in corner:
            signature = tuple(
                name for name in outer_names if generators[name][position] != position
            )
            groups.setdefault(signature, []).append(position)
        blocks = tuple(sorted((tuple(sorted(group)) for group in groups.values())))
        if len(blocks) != N_CORNERS or any(len(block) != 3 for block in blocks):
            raise ValueError(
                "could not discover eight three-sticker corner blocks from move support; "
                f"block_sizes={[len(block) for block in blocks]}"
            )

        block_by_position = {
            position: block_index
            for block_index, block in enumerate(blocks)
            for position in block
        }
        if set(block_by_position) != corner_set:
            raise AssertionError("corner blocks do not partition the corner orbit")

        # Propagate an oriented sticker frame through the transitive corner action.
        # Images may differ from an existing frame by a cyclic shift, but never a
        # reflection.  This recovers a consistent mod-3 cubie orientation system.
        frames: list[tuple[int, int, int] | None] = [None] * N_CORNERS
        frames[0] = blocks[0]
        queue = deque((0,))
        while queue:
            destination_block = queue.popleft()
            destination_frame = frames[destination_block]
            assert destination_frame is not None
            for name in outer_names:
                permutation = generators[name]
                image = tuple(permutation[position] for position in destination_frame)
                source_blocks = {block_by_position[position] for position in image}
                if len(source_blocks) != 1:
                    raise ValueError(f"move {name!r} splits a physical corner block")
                source_block = source_blocks.pop()
                old_frame = frames[source_block]
                if old_frame is None:
                    frames[source_block] = image
                    queue.append(source_block)
                elif _cyclic_shift(old_frame, image) is None:
                    raise ValueError(f"move {name!r} reflects a corner orientation frame")
        if any(frame is None for frame in frames):
            raise ValueError("outer moves are not transitive on the eight corner blocks")
        complete_frames = tuple(frame for frame in frames if frame is not None)

        piece_sources: list[tuple[int, ...]] = []
        orientation_deltas: list[tuple[int, ...]] = []
        for name in outer_names:
            permutation = generators[name]
            move_sources: list[int] = []
            move_deltas: list[int] = []
            for destination_frame in complete_frames:
                image = tuple(permutation[position] for position in destination_frame)
                source_block = block_by_position[image[0]]
                if any(block_by_position[position] != source_block for position in image):
                    raise ValueError(f"move {name!r} splits a corner block")
                shift = _cyclic_shift(complete_frames[source_block], image)
                if shift is None:
                    raise ValueError(f"move {name!r} has a reflected corner action")
                move_sources.append(source_block)
                move_deltas.append(shift)
            piece_sources.append(tuple(move_sources))
            orientation_deltas.append(tuple(move_deltas))

        system = cls(
            blocks=blocks,
            frames=complete_frames,
            move_names=outer_names,
            move_piece_sources=tuple(piece_sources),
            move_orientation_deltas=tuple(orientation_deltas),
        )
        system._validate_moves()
        return system

    def _validate_moves(self) -> None:
        move_index = {name: index for index, name in enumerate(self.move_names)}
        for name, index in move_index.items():
            inverse = name[1:] if name.startswith("-") else "-" + name
            if inverse not in move_index:
                raise ValueError(f"corner move {name!r} has no inverse")
            cp = tuple(range(N_CORNERS))
            co = (0,) * N_CORNERS
            cp, co = self.apply_coordinate_move(cp, co, index)
            cp, co = self.apply_coordinate_move(cp, co, move_index[inverse])
            if cp != tuple(range(N_CORNERS)) or co != (0,) * N_CORNERS:
                raise ValueError(f"corner moves {name!r}, {inverse!r} do not round-trip")

    def apply_coordinate_move(
        self,
        permutation: Sequence[int],
        orientation: Sequence[int],
        move_index: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        sources = self.move_piece_sources[move_index]
        deltas = self.move_orientation_deltas[move_index]
        new_permutation = tuple(permutation[source] for source in sources)
        new_orientation = tuple(
            (orientation[source] + delta) % 3
            for source, delta in zip(sources, deltas, strict=True)
        )
        return new_permutation, new_orientation

    def encode_state(
        self,
        state: Sequence[int],
        target: Sequence[int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if len(state) != len(target):
            raise ValueError("state and target lengths differ")
        target_position = {sticker: position for position, sticker in enumerate(target)}
        if len(target_position) != len(target):
            raise ValueError("target stickers must be unique")
        block_by_position = {
            position: block_index
            for block_index, block in enumerate(self.blocks)
            for position in block
        }
        frame_index = {
            position: orientation_index
            for frame in self.frames
            for orientation_index, position in enumerate(frame)
        }

        permutation: list[int] = []
        orientation: list[int] = []
        for destination_frame in self.frames:
            try:
                source_positions = tuple(target_position[state[position]] for position in destination_frame)
            except KeyError as exc:
                raise ValueError(f"state sticker {exc.args[0]!r} is absent from target") from exc
            source_blocks = {block_by_position.get(position) for position in source_positions}
            if None in source_blocks or len(source_blocks) != 1:
                raise ValueError("state splits stickers belonging to one physical corner")
            source_block = source_blocks.pop()
            assert source_block is not None
            shift = frame_index[source_positions[0]]
            expected = tuple(self.frames[source_block][(index + shift) % 3] for index in range(3))
            if source_positions != expected:
                raise ValueError("state reflects a physical corner orientation")
            permutation.append(source_block)
            orientation.append(shift)

        cp = tuple(permutation)
        co = tuple(orientation)
        _rank_permutation(cp)
        _rank_orientation(co)
        return cp, co


@dataclass
class ExactCornerSolver:
    """Optimal quarter-turn corner solver with two exact coordinate PDBs."""

    coordinates: CornerCoordinateSystem
    permutation_moves: array
    orientation_moves: array
    permutation_distance: array
    orientation_distance: array
    pattern_moves: array
    pattern_distance_first: array
    pattern_distance_second: array
    face_ids: tuple[int, ...]
    axis_ids: tuple[int, ...]
    negative_directions: tuple[bool, ...]

    @classmethod
    def build(cls, coordinates: CornerCoordinateSystem) -> "ExactCornerSolver":
        move_count = len(coordinates.move_names)
        permutation_moves = array("H", [0]) * (N_PERMUTATIONS * move_count)
        for rank in range(N_PERMUTATIONS):
            permutation = _unrank_permutation(rank)
            for move_index, sources in enumerate(coordinates.move_piece_sources):
                moved = tuple(permutation[source] for source in sources)
                permutation_moves[rank * move_count + move_index] = _rank_permutation(moved)

        orientation_moves = array("H", [0]) * (N_ORIENTATIONS * move_count)
        identity_permutation = tuple(range(N_CORNERS))
        for rank in range(N_ORIENTATIONS):
            orientation = _unrank_orientation(rank)
            for move_index in range(move_count):
                _, moved = coordinates.apply_coordinate_move(
                    identity_permutation,
                    orientation,
                    move_index,
                )
                orientation_moves[rank * move_count + move_index] = _rank_orientation(moved)

        permutation_distance = cls._build_distance_table(
            N_PERMUTATIONS,
            move_count,
            permutation_moves,
        )
        orientation_distance = cls._build_distance_table(
            N_ORIENTATIONS,
            move_count,
            orientation_moves,
        )

        # A combined location+orientation PDB for four labelled corners is much
        # stronger than the separate 8-corner permutation/orientation bounds.
        # The transition graph is identical for either group of four pieces, so
        # one table supports two complementary goal-distance tables.
        pattern_moves = array("I", [0]) * (N_PATTERN_STATES * move_count)
        inverse_sources_by_move: list[tuple[int, ...]] = []
        for sources in coordinates.move_piece_sources:
            new_location_by_old = [0] * N_CORNERS
            for new_location, old_location in enumerate(sources):
                new_location_by_old[old_location] = new_location
            inverse_sources_by_move.append(tuple(new_location_by_old))
        for rank in range(N_PATTERN_STATES):
            locations, orientations = _unrank_pattern(rank)
            for move_index, new_location_by_old in enumerate(inverse_sources_by_move):
                moved_locations = tuple(new_location_by_old[location] for location in locations)
                deltas = coordinates.move_orientation_deltas[move_index]
                moved_orientations = tuple(
                    (orientation + deltas[new_location]) % 3
                    for orientation, new_location in zip(
                        orientations,
                        moved_locations,
                        strict=True,
                    )
                )
                pattern_moves[rank * move_count + move_index] = _rank_pattern(
                    moved_locations,
                    moved_orientations,
                )

        identity = tuple(range(N_CORNERS))
        zero_orientation = (0,) * N_CORNERS
        first_pieces = tuple(range(PATTERN_PIECES))
        second_pieces = tuple(range(PATTERN_PIECES, N_CORNERS))
        first_goal = _pattern_from_coordinate(identity, zero_orientation, first_pieces)
        second_goal = _pattern_from_coordinate(identity, zero_orientation, second_pieces)
        pattern_distance_first = cls._build_distance_table(
            N_PATTERN_STATES,
            move_count,
            pattern_moves,
            goal=first_goal,
        )
        pattern_distance_second = cls._build_distance_table(
            N_PATTERN_STATES,
            move_count,
            pattern_moves,
            goal=second_goal,
        )

        face_by_name: dict[str, int] = {}
        axis_by_name: dict[str, int] = {}
        face_ids: list[int] = []
        axis_ids: list[int] = []
        negative: list[bool] = []
        for name in coordinates.move_names:
            base = name[1:] if name.startswith("-") else name
            if base not in face_by_name:
                face_by_name[base] = len(face_by_name)
            axis = base[0]
            if axis not in axis_by_name:
                axis_by_name[axis] = len(axis_by_name)
            face_ids.append(face_by_name[base])
            axis_ids.append(axis_by_name[axis])
            negative.append(name.startswith("-"))
        if len(face_by_name) != 6:
            raise ValueError(f"expected six outer faces, found {sorted(face_by_name)}")
        if len(axis_by_name) != 3:
            raise ValueError(f"expected three axes, found {sorted(axis_by_name)}")

        return cls(
            coordinates=coordinates,
            permutation_moves=permutation_moves,
            orientation_moves=orientation_moves,
            permutation_distance=permutation_distance,
            orientation_distance=orientation_distance,
            pattern_moves=pattern_moves,
            pattern_distance_first=pattern_distance_first,
            pattern_distance_second=pattern_distance_second,
            face_ids=tuple(face_ids),
            axis_ids=tuple(axis_ids),
            negative_directions=tuple(negative),
        )

    @staticmethod
    def _build_distance_table(
        size: int,
        move_count: int,
        transitions: array,
        goal: int = 0,
    ) -> array:
        distance = array("b", [-1]) * size
        distance[goal] = 0
        queue = deque((goal,))
        while queue:
            coordinate = queue.popleft()
            next_distance = distance[coordinate] + 1
            offset = coordinate * move_count
            for move_index in range(move_count):
                image = transitions[offset + move_index]
                if distance[image] == -1:
                    distance[image] = next_distance
                    queue.append(image)
        if any(value < 0 for value in distance):
            raise ValueError("corner coordinate projection is not fully reachable")
        return distance

    def solve(
        self,
        state: Sequence[int],
        target: Sequence[int],
        max_depth: int = 14,
    ) -> tuple[str, ...]:
        permutation, orientation = self.coordinates.encode_state(state, target)
        permutation_rank = _rank_permutation(permutation)
        orientation_rank = _rank_orientation(orientation)
        if permutation_rank == 0 and orientation_rank == 0:
            return ()

        move_count = len(self.coordinates.move_names)
        first_pattern = _pattern_from_coordinate(
            permutation,
            orientation,
            tuple(range(PATTERN_PIECES)),
        )
        second_pattern = _pattern_from_coordinate(
            permutation,
            orientation,
            tuple(range(PATTERN_PIECES, N_CORNERS)),
        )
        path: list[int] = []
        on_path = {(permutation_rank, orientation_rank)}

        def heuristic(cp_rank: int, co_rank: int, first_rank: int, second_rank: int) -> int:
            return max(
                self.permutation_distance[cp_rank],
                self.orientation_distance[co_rank],
                self.pattern_distance_first[first_rank],
                self.pattern_distance_second[second_rank],
            )

        def search(
            cp_rank: int,
            co_rank: int,
            first_rank: int,
            second_rank: int,
            remaining: int,
            last_face: int,
            last_negative: bool,
            same_face_run: int,
        ) -> bool:
            if heuristic(cp_rank, co_rank, first_rank, second_rank) > remaining:
                return False
            if remaining == 0:
                return cp_rank == 0 and co_rank == 0

            cp_offset = cp_rank * move_count
            co_offset = co_rank * move_count
            first_offset = first_rank * move_count
            second_offset = second_rank * move_count
            for move_index in range(move_count):
                face = self.face_ids[move_index]
                axis = self.axis_ids[move_index]
                negative = self.negative_directions[move_index]
                if face == last_face:
                    # Opposite directions cancel; three equal quarter turns are
                    # one inverse turn.  Neither can occur in an optimal QTM path.
                    if negative != last_negative or same_face_run >= 2:
                        continue
                    next_run = same_face_run + 1
                else:
                    # Turns of opposite faces on one axis commute.  Keep only one
                    # of the two orders to remove a large duplicate subtree.
                    if last_face >= 0 and axis == self.axis_ids[path[-1]] and face < last_face:
                        continue
                    next_run = 1

                next_cp = self.permutation_moves[cp_offset + move_index]
                next_co = self.orientation_moves[co_offset + move_index]
                next_first = self.pattern_moves[first_offset + move_index]
                next_second = self.pattern_moves[second_offset + move_index]
                coordinate = (next_cp, next_co)
                if coordinate in on_path:
                    continue
                on_path.add(coordinate)
                path.append(move_index)
                if search(
                    next_cp,
                    next_co,
                    next_first,
                    next_second,
                    remaining - 1,
                    face,
                    negative,
                    next_run,
                ):
                    return True
                path.pop()
                on_path.remove(coordinate)
            return False

        lower_bound = heuristic(
            permutation_rank,
            orientation_rank,
            first_pattern,
            second_pattern,
        )
        for depth in range(lower_bound, max_depth + 1):
            if search(
                permutation_rank,
                orientation_rank,
                first_pattern,
                second_pattern,
                depth,
                -1,
                False,
                0,
            ):
                return tuple(self.coordinates.move_names[index] for index in path)
        raise ValueError(f"corner state has no solution at depth <= {max_depth}")
