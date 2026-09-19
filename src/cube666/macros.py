"""Commutator enumeration and effect analysis for the classical 666 solver."""

from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from cube666.classical import (
    Cube666Decomposition,
    Orbit,
    cluster_permutation,
    permutation_cycles,
    permutation_parity,
)


Permutation = tuple[int, ...]
AlternativeThreeCyclePaths = Mapping[
    tuple[int, tuple[int, int, int]],
    Sequence[tuple[str, ...]],
]


def compose_pullbacks(first: Sequence[int], second: Sequence[int]) -> Permutation:
    """Effect of applying ``first`` and then ``second`` under pullback convention."""

    if len(first) != len(second):
        raise ValueError("cannot compose permutations of different sizes")
    return tuple(first[second[index]] for index in range(len(first)))


def inverse_permutation(permutation: Sequence[int]) -> Permutation:
    inverse = [0] * len(permutation)
    for index, image in enumerate(permutation):
        inverse[image] = index
    if set(inverse) != set(range(len(permutation))):
        raise ValueError("input is not a permutation")
    return tuple(inverse)


def path_effect(
    generators: Mapping[str, Sequence[int]],
    path: Sequence[str],
) -> Permutation:
    if not generators:
        raise ValueError("at least one generator is required")
    size = len(next(iter(generators.values())))
    effect = tuple(range(size))
    for name in path:
        effect = compose_pullbacks(effect, generators[name])
    return effect


def invert_move_name(name: str) -> str:
    return name[1:] if name.startswith("-") else "-" + name


def invert_path(path: Sequence[str]) -> tuple[str, ...]:
    return tuple(invert_move_name(name) for name in reversed(path))


def commutator(left: str, right: str) -> tuple[str, ...]:
    """Return [left, right] = left right left^-1 right^-1."""

    return (left, right, invert_move_name(left), invert_move_name(right))


def _restriction(permutation: Sequence[int], orbit: Orbit) -> Permutation:
    local_index = {position: index for index, position in enumerate(orbit)}
    try:
        return tuple(local_index[permutation[position]] for position in orbit)
    except KeyError as exc:
        raise ValueError("permutation does not preserve a physical cluster") from exc


@dataclass(frozen=True)
class MacroEffect:
    path: tuple[str, ...]
    effect: Permutation
    cluster_permutations: tuple[Permutation, ...]
    cluster_cycles: tuple[tuple[tuple[int, ...], ...], ...]

    @property
    def corner_fixed(self) -> bool:
        # Set by analyze_macro through the filtered constructor contract.  Kept
        # as a semantic property for report code.
        return True

    @property
    def total_nontrivial_cycles(self) -> int:
        return sum(len(cycles) for cycles in self.cluster_cycles)

    @property
    def total_three_cycles(self) -> int:
        return sum(
            len(cycle) == 3
            for cycles in self.cluster_cycles
            for cycle in cycles
        )

    @property
    def pure_three_cycles(self) -> bool:
        return self.total_nontrivial_cycles > 0 and all(
            len(cycle) == 3
            for cycles in self.cluster_cycles
            for cycle in cycles
        )

    @property
    def unrestricted_three_cycle_units(self) -> int | None:
        """Algebraic 3-cycle length when every cluster restriction is even."""

        for cycles in self.cluster_cycles:
            even_cycle_count = sum(len(cycle) % 2 == 0 for cycle in cycles)
            if even_cycle_count % 2:
                return None
        return sum(
            len(cycle) // 2
            for cycles in self.cluster_cycles
            for cycle in cycles
        )

    @property
    def active_cluster_mask(self) -> str:
        return "".join("1" if cycles else "0" for cycles in self.cluster_cycles)


def analyze_corner_fixing_macro(
    path: Sequence[str],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
) -> MacroEffect:
    effect = path_effect(generators, path)
    if any(effect[position] != position for position in decomposition.corner_orbit):
        raise ValueError("macro does not fix the corner stickers")
    cluster_permutations = tuple(
        _restriction(effect, orbit)
        for orbit in decomposition.physical_clusters
    )
    cluster_cycles = tuple(permutation_cycles(permutation) for permutation in cluster_permutations)
    return MacroEffect(tuple(path), effect, cluster_permutations, cluster_cycles)


def _analyze_corner_fixing_effect(
    path: Sequence[str],
    effect: Permutation,
    decomposition: Cube666Decomposition,
) -> MacroEffect:
    if any(effect[position] != position for position in decomposition.corner_orbit):
        raise ValueError("macro does not fix the corner stickers")
    cluster_permutations = tuple(
        _restriction(effect, orbit)
        for orbit in decomposition.physical_clusters
    )
    cluster_cycles = tuple(permutation_cycles(permutation) for permutation in cluster_permutations)
    return MacroEffect(tuple(path), effect, cluster_permutations, cluster_cycles)


def enumerate_basic_inner_commutators(
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
) -> tuple[MacroEffect, ...]:
    """Enumerate unique four-turn commutators of perpendicular inner slices."""

    shortest_by_effect: dict[Permutation, MacroEffect] = {}
    identity = tuple(range(decomposition.size))
    for left in decomposition.inner_move_names:
        left_axis = left.lstrip("-")[0]
        for right in decomposition.inner_move_names:
            right_axis = right.lstrip("-")[0]
            if left_axis == right_axis:
                continue
            macro = analyze_corner_fixing_macro(
                commutator(left, right),
                generators,
                decomposition,
            )
            if macro.effect == identity:
                continue
            old = shortest_by_effect.get(macro.effect)
            if old is None or macro.path < old.path:
                shortest_by_effect[macro.effect] = macro
    return tuple(sorted(shortest_by_effect.values(), key=lambda macro: macro.path))


def enumerate_basic_corner_fixing_commutators(
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
) -> tuple[MacroEffect, ...]:
    """Enumerate four-turn commutators with at least one inner-slice operand.

    If either operand fixes corners pointwise, its commutator with any primitive
    face move also fixes corners.  Perpendicular inner-outer pairs supply the
    wing-moving macro family absent from inner-inner commutators.
    """

    inner = set(decomposition.inner_move_names)
    shortest_by_effect: dict[Permutation, MacroEffect] = {}
    identity = tuple(range(decomposition.size))
    for left in generators:
        left_axis = left.lstrip("-")[0]
        for right in generators:
            if left not in inner and right not in inner:
                continue
            right_axis = right.lstrip("-")[0]
            if left_axis == right_axis:
                continue
            macro = analyze_corner_fixing_macro(
                commutator(left, right),
                generators,
                decomposition,
            )
            if macro.effect == identity:
                continue
            old = shortest_by_effect.get(macro.effect)
            if old is None or macro.path < old.path:
                shortest_by_effect[macro.effect] = macro
    return tuple(sorted(shortest_by_effect.values(), key=lambda macro: macro.path))


def conjugate_macro(
    macro: MacroEffect,
    move_name: str,
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
) -> MacroEffect:
    """Return ``move macro move^-1``, preserving the corner-fixing property."""

    inverse_name = invert_move_name(move_name)
    path = (move_name,) + macro.path + (inverse_name,)
    effect = compose_pullbacks(
        compose_pullbacks(generators[move_name], macro.effect),
        generators[inverse_name],
    )
    return _analyze_corner_fixing_effect(path, effect, decomposition)


def enumerate_conjugated_macros(
    base_macros: Sequence[MacroEffect],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    max_conjugator_depth: int,
) -> tuple[MacroEffect, ...]:
    """Close a base macro set under short primitive conjugators.

    A depth-d result has an unsimplified word length of at most ``4 + 2*d`` for
    four-turn bases.  Effects are deduplicated globally at their shortest depth.
    """

    if max_conjugator_depth < 0:
        raise ValueError("max_conjugator_depth must be nonnegative")
    by_effect = {bytes(macro.effect): macro for macro in base_macros}
    frontier = list(base_macros)
    for _ in range(max_conjugator_depth):
        next_frontier: list[MacroEffect] = []
        for macro in frontier:
            for move_name in generators:
                conjugated = conjugate_macro(
                    macro,
                    move_name,
                    generators,
                    decomposition,
                )
                key = bytes(conjugated.effect)
                if key in by_effect:
                    continue
                by_effect[key] = conjugated
                next_frontier.append(conjugated)
        frontier = next_frontier
        if not frontier:
            break
    return tuple(by_effect.values())


def enumerate_nested_corner_fixing_commutators(
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    inner_conjugator_depth: int = 1,
) -> tuple[MacroEffect, ...]:
    """Enumerate ``[A, C B C^-1]`` with primitive A/B and short C.

    B is an inner slice, so every conjugate C B C^-1 fixes corners.  Its
    commutator with any primitive A therefore fixes corners as well.  At depth 1
    these words have length at most eight and can create isolated cycle geometry
    that whole-macro conjugation cannot.
    """

    if inner_conjugator_depth < 0:
        raise ValueError("inner_conjugator_depth must be nonnegative")
    inner_algorithms = tuple(
        analyze_corner_fixing_macro((name,), generators, decomposition)
        for name in decomposition.inner_move_names
    )
    right_algorithms = enumerate_conjugated_macros(
        inner_algorithms,
        generators,
        decomposition,
        max_conjugator_depth=inner_conjugator_depth,
    )

    identity = tuple(range(decomposition.size))
    shortest_by_effect: dict[bytes, MacroEffect] = {}
    for left_name, left_effect_raw in generators.items():
        left_effect = tuple(left_effect_raw)
        left_inverse = tuple(generators[invert_move_name(left_name)])
        for right in right_algorithms:
            right_inverse = inverse_permutation(right.effect)
            effect = compose_pullbacks(
                compose_pullbacks(
                    compose_pullbacks(left_effect, right.effect),
                    left_inverse,
                ),
                right_inverse,
            )
            if effect == identity:
                continue
            path = (
                (left_name,)
                + right.path
                + (invert_move_name(left_name),)
                + invert_path(right.path)
            )
            macro = _analyze_corner_fixing_effect(path, effect, decomposition)
            key = bytes(effect)
            old = shortest_by_effect.get(key)
            if old is None or (len(macro.path), macro.path) < (len(old.path), old.path):
                shortest_by_effect[key] = macro
    return tuple(shortest_by_effect.values())


def build_isolated_three_cycle_library(
    seed_macros: Sequence[MacroEffect],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    max_conjugator_depth: int = 1,
) -> dict[tuple[int, tuple[int, int, int]], MacroEffect]:
    """Build shortest known macros for isolated directed cluster 3-cycles."""

    if max_conjugator_depth < 0:
        raise ValueError("max_conjugator_depth must be nonnegative")

    def isolated_key(macro: MacroEffect) -> tuple[int, tuple[int, int, int]] | None:
        if macro.total_nontrivial_cycles != 1 or macro.total_three_cycles != 1:
            return None
        for cluster_index, cycles in enumerate(macro.cluster_cycles):
            if cycles:
                cycle = cycles[0]
                return cluster_index, (cycle[0], cycle[1], cycle[2])
        raise AssertionError("isolated macro has no active cluster")

    library: dict[tuple[int, tuple[int, int, int]], MacroEffect] = {}
    for macro in seed_macros:
        key = isolated_key(macro)
        if key is None:
            continue
        old = library.get(key)
        if old is None or (len(macro.path), macro.path) < (len(old.path), old.path):
            library[key] = macro

    frontier = list(library.values())
    for _ in range(max_conjugator_depth):
        next_frontier: dict[tuple[int, tuple[int, int, int]], MacroEffect] = {}
        for macro in frontier:
            for move_name in generators:
                conjugated = conjugate_macro(
                    macro,
                    move_name,
                    generators,
                    decomposition,
                )
                key = isolated_key(conjugated)
                if key is None:
                    raise AssertionError("conjugation did not preserve an isolated 3-cycle")
                old = library.get(key)
                if old is not None and (len(old.path), old.path) <= (
                    len(conjugated.path),
                    conjugated.path,
                ):
                    continue
                library[key] = conjugated
                next_frontier[key] = conjugated
        frontier = list(next_frontier.values())
        if not frontier:
            break
    return library


def _generator_digest(generators: Mapping[str, Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for name, permutation in generators.items():
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes(permutation))
    return digest.hexdigest()


def save_three_cycle_library(
    path: str | Path,
    library: Mapping[tuple[int, tuple[int, int, int]], MacroEffect],
    generators: Mapping[str, Sequence[int]],
) -> None:
    """Persist a finisher library as auditable JSON paths, not opaque pickle."""

    output_path = Path(path)
    payload = {
        "format_version": 1,
        "generator_sha256": _generator_digest(generators),
        "entries": [
            {
                "cluster": cluster,
                "cycle": list(cycle),
                "path": list(macro.path),
            }
            for (cluster, cycle), macro in sorted(library.items())
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def load_three_cycle_library(
    path: str | Path,
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
) -> dict[tuple[int, tuple[int, int, int]], MacroEffect]:
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported finisher library format in {input_path}")
    if payload.get("generator_sha256") != _generator_digest(generators):
        raise ValueError("finisher library generator digest does not match puzzle_info")

    library: dict[tuple[int, tuple[int, int, int]], MacroEffect] = {}
    for entry in payload["entries"]:
        stored_key = (
            int(entry["cluster"]),
            tuple(int(value) for value in entry["cycle"]),
        )
        macro = analyze_corner_fixing_macro(entry["path"], generators, decomposition)
        actual_key = isolated_three_cycle_key(macro)
        if actual_key != stored_key:
            raise ValueError(f"finisher entry key mismatch: stored={stored_key}, actual={actual_key}")
        library[actual_key] = macro
    return library


def isolated_three_cycle_key(
    macro: MacroEffect,
) -> tuple[int, tuple[int, int, int]]:
    if macro.total_nontrivial_cycles != 1 or macro.total_three_cycles != 1:
        raise ValueError("macro is not an isolated 3-cycle")
    for cluster_index, cycles in enumerate(macro.cluster_cycles):
        if cycles:
            cycle = cycles[0]
            return cluster_index, (cycle[0], cycle[1], cycle[2])
    raise AssertionError("isolated macro has no active cluster")


def _three_cycle_units_fast(permutation: Sequence[int]) -> int:
    """Unchecked 3-cycle length for a known-even permutation."""

    seen = [False] * len(permutation)
    units = 0
    for start in range(len(permutation)):
        if seen[start]:
            continue
        length = 0
        position = start
        while not seen[position]:
            seen[position] = True
            length += 1
            position = permutation[position]
        units += length // 2
    return units


def cluster_residual_cost(cluster_permutations: Sequence[Sequence[int]]) -> int:
    """Exact unrestricted 3-cycle cost for six even cluster permutations."""

    total = 0
    for permutation in cluster_permutations:
        if permutation_parity(permutation):
            raise ValueError("cluster residual cost requires even permutations")
        total += _three_cycle_units_fast(permutation)
    return total


def state_cluster_permutations(
    state: Sequence[int],
    target: Sequence[int],
    decomposition: Cube666Decomposition,
) -> tuple[Permutation, ...]:
    return tuple(
        cluster_permutation(state, target, orbit)
        for orbit in decomposition.physical_clusters
    )


def apply_macro_to_clusters(
    current: Sequence[Sequence[int]],
    macro: MacroEffect,
) -> tuple[Permutation, ...]:
    if len(current) != len(macro.cluster_permutations):
        raise ValueError("cluster counts differ")
    return tuple(
        compose_pullbacks(permutation, macro_permutation)
        for permutation, macro_permutation in zip(
            current,
            macro.cluster_permutations,
            strict=True,
        )
    )


@dataclass(frozen=True)
class GreedyMacroResult:
    path: tuple[str, ...]
    macro_count: int
    initial_cost: int
    final_cost: int
    cost_trajectory: tuple[int, ...]


@dataclass(frozen=True)
class InsertedFinisherResult:
    path: tuple[str, ...]
    macro_count: int
    initial_cost: int
    final_cost: int
    path_length_trajectory: tuple[int, ...]


def greedy_macro_reduce(
    initial_cluster_permutations: Sequence[Sequence[int]],
    macros: Sequence[MacroEffect],
    max_macros: int = 100,
) -> GreedyMacroResult:
    """Repeatedly apply the macro with the largest exact residual decrease."""

    if max_macros < 0:
        raise ValueError("max_macros must be nonnegative")
    current = tuple(tuple(permutation) for permutation in initial_cluster_permutations)
    initial_cost = cluster_residual_cost(current)
    current_cost = initial_cost
    trajectory = [current_cost]
    path: list[str] = []
    macro_count = 0

    for _ in range(max_macros):
        best_macro: MacroEffect | None = None
        best_state: tuple[Permutation, ...] | None = None
        best_cost = current_cost
        for macro in macros:
            candidate = apply_macro_to_clusters(current, macro)
            candidate_cost = sum(_three_cycle_units_fast(permutation) for permutation in candidate)
            if candidate_cost < best_cost or (
                candidate_cost == best_cost
                and best_macro is not None
                and (len(macro.path), macro.path) < (len(best_macro.path), best_macro.path)
            ):
                best_macro = macro
                best_state = candidate
                best_cost = candidate_cost
        if best_macro is None or best_state is None or best_cost >= current_cost:
            break
        current = best_state
        current_cost = best_cost
        trajectory.append(current_cost)
        path.extend(best_macro.path)
        macro_count += 1
        if current_cost == 0:
            break

    return GreedyMacroResult(
        path=tuple(path),
        macro_count=macro_count,
        initial_cost=initial_cost,
        final_cost=current_cost,
        cost_trajectory=tuple(trajectory),
    )


def reduce_quarter_turn_path(path: Sequence[str]) -> tuple[str, ...]:
    """Combine adjacent powers of one primitive face modulo four."""

    stack: list[tuple[str, int]] = []
    for name in path:
        base = name[1:] if name.startswith("-") else name
        amount = -1 if name.startswith("-") else 1
        if stack and stack[-1][0] == base:
            old_base, old_amount = stack.pop()
            combined = (old_amount + amount) % 4
            if combined:
                stack.append((old_base, combined))
        else:
            stack.append((base, amount % 4))

    reduced: list[str] = []
    for base, amount in stack:
        if amount == 1:
            reduced.append(base)
        elif amount == 2:
            reduced.extend((base, base))
        elif amount == 3:
            reduced.append("-" + base)
        else:
            raise AssertionError("zero power remained on reduction stack")
    return tuple(reduced)


def reduce_commuting_quarter_turn_path(path: Sequence[str]) -> tuple[str, ...]:
    """Canonicalize consecutive turns of parallel slices modulo four.

    Every layer on one cube axis commutes with every other layer on that axis.
    Treating a same-axis block as an exponent vector exposes cancellations such
    as ``f1.f3.-f1 -> f3`` that the adjacent-identical reducer cannot see.
    Empty blocks are removed online, allowing cancellation to cascade into the
    neighboring axis block.
    """

    blocks: list[tuple[str, dict[str, int]]] = []
    for name in path:
        base = name[1:] if name.startswith("-") else name
        if not base:
            raise ValueError("empty move name")
        axis = base[0]
        amount = -1 if name.startswith("-") else 1
        if not blocks or blocks[-1][0] != axis:
            blocks.append((axis, {}))
        powers = blocks[-1][1]
        combined = (powers.get(base, 0) + amount) % 4
        if combined:
            powers[base] = combined
        else:
            powers.pop(base, None)
        if not powers:
            blocks.pop()

    reduced: list[str] = []
    for _, powers in blocks:
        for base in sorted(powers, key=lambda value: (int(value[1:]), value)):
            amount = powers[base]
            if amount == 1:
                reduced.append(base)
            elif amount == 2:
                reduced.extend((base, base))
            elif amount == 3:
                reduced.append("-" + base)
            else:
                raise AssertionError("zero power remained in a parallel block")
    return tuple(reduced)


def _suffix_effects(
    path: Sequence[str],
    generators: Mapping[str, Sequence[int]],
) -> tuple[Permutation, ...]:
    if not generators:
        raise ValueError("at least one generator is required")
    size = len(next(iter(generators.values())))
    suffixes: list[Permutation] = [tuple()] * (len(path) + 1)
    suffixes[-1] = tuple(range(size))
    for index in range(len(path) - 1, -1, -1):
        suffixes[index] = compose_pullbacks(generators[path[index]], suffixes[index + 1])
    return tuple(suffixes)


def finish_with_inserted_three_cycles(
    initial_cluster_permutations: Sequence[Sequence[int]],
    initial_path: Sequence[str],
    library: Mapping[tuple[int, tuple[int, int, int]], MacroEffect],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
) -> InsertedFinisherResult:
    """Solve even cluster residuals using shortest 3-cycles inserted anywhere.

    For a desired end-of-path macro M and a split P=A B, insertion uses
    ``X = B M B^-1`` so that ``A X B = A B M``.  Complete directed 3-cycle
    coverage therefore lets every algebraic step choose the insertion time with
    the best adjacent quarter-turn cancellations.
    """

    current = tuple(tuple(permutation) for permutation in initial_cluster_permutations)
    initial_cost = cluster_residual_cost(current)
    current_cost = initial_cost
    path = reduce_quarter_turn_path(initial_path)
    path_lengths = [len(path)]
    macro_count = 0

    library_by_cluster: list[list[MacroEffect]] = [
        [] for _ in decomposition.physical_clusters
    ]
    for (cluster_index, _), macro in library.items():
        library_by_cluster[cluster_index].append(macro)
    for macros in library_by_cluster:
        macros.sort(key=lambda macro: (len(macro.path), macro.path))

    while current_cost:
        active_cluster = next(
            index
            for index, permutation in enumerate(current)
            if _three_cycle_units_fast(permutation)
        )
        cluster_cost = _three_cycle_units_fast(current[active_cluster])
        desired_macro: MacroEffect | None = None
        desired_cluster_state: Permutation | None = None
        for macro in library_by_cluster[active_cluster]:
            candidate = compose_pullbacks(
                current[active_cluster],
                macro.cluster_permutations[active_cluster],
            )
            if _three_cycle_units_fast(candidate) == cluster_cost - 1:
                desired_macro = macro
                desired_cluster_state = candidate
                break
        if desired_macro is None or desired_cluster_state is None:
            raise ValueError(f"no reducing 3-cycle found for cluster {active_cluster}")

        suffixes = _suffix_effects(path, generators)
        best_path: tuple[str, ...] | None = None
        for insertion_index, suffix in enumerate(suffixes):
            conjugated_effect = compose_pullbacks(
                compose_pullbacks(suffix, desired_macro.effect),
                inverse_permutation(suffix),
            )
            conjugated = _analyze_corner_fixing_effect((), conjugated_effect, decomposition)
            key = isolated_three_cycle_key(conjugated)
            insertion_macro = library.get(key)
            if insertion_macro is None:
                raise ValueError(f"finisher library lacks conjugated cycle {key}")
            candidate_path = reduce_quarter_turn_path(
                path[:insertion_index]
                + insertion_macro.path
                + path[insertion_index:]
            )
            if best_path is None or (len(candidate_path), candidate_path) < (
                len(best_path),
                best_path,
            ):
                best_path = candidate_path
        if best_path is None:
            raise AssertionError("no insertion position considered")

        updated = list(current)
        updated[active_cluster] = desired_cluster_state
        current = tuple(updated)
        current_cost -= 1
        path = best_path
        path_lengths.append(len(path))
        macro_count += 1

    return InsertedFinisherResult(
        path=path,
        macro_count=macro_count,
        initial_cost=initial_cost,
        final_cost=current_cost,
        path_length_trajectory=tuple(path_lengths),
    )


def _canonical_directed_three_cycle(cycle: Sequence[int]) -> tuple[int, int, int]:
    """Rotate a directed 3-cycle so its smallest element is first."""

    if len(cycle) != 3 or len(set(cycle)) != 3:
        raise ValueError("expected three distinct cycle elements")
    values = tuple(int(value) for value in cycle)
    start = values.index(min(values))
    return tuple(values[(start + offset) % 3] for offset in range(3))  # type: ignore[return-value]


def _local_suffix_effects(
    path: Sequence[str],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
) -> tuple[tuple[Permutation, ...], ...]:
    """Return each physical cluster's local effect for every path suffix."""

    cluster_count = len(decomposition.physical_clusters)
    identities = tuple(tuple(range(24)) for _ in range(cluster_count))
    suffixes: list[tuple[Permutation, ...]] = [identities] * (len(path) + 1)
    local_moves = {
        name: tuple(_restriction(permutation, orbit) for orbit in decomposition.physical_clusters)
        for name, permutation in generators.items()
    }
    suffixes[-1] = identities
    for index in range(len(path) - 1, -1, -1):
        suffixes[index] = tuple(
            compose_pullbacks(move, suffix)
            for move, suffix in zip(
                local_moves[path[index]],
                suffixes[index + 1],
                strict=True,
            )
        )
    return tuple(suffixes)


def finish_with_best_inserted_three_cycles(
    initial_cluster_permutations: Sequence[Sequence[int]],
    initial_path: Sequence[str],
    library: Mapping[tuple[int, tuple[int, int, int]], MacroEffect],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    alternative_paths: AlternativeThreeCyclePaths | None = None,
) -> InsertedFinisherResult:
    """Greedily choose the best reducing cycle *and* insertion position.

    ``finish_with_inserted_three_cycles`` fixes a deterministic algebraic
    3-cycle first and only optimizes where that one cycle is inserted.  That is
    fast, but it discards hundreds of equally valid cost-reducing cycles whose
    concrete words can cancel much more strongly against the current path.

    This variant considers every exact cost-reducing directed 3-cycle in every
    active cluster and every insertion position, then takes the globally
    shortest resulting quarter-turn path.  Conjugation is performed on the
    24-position local cluster action, avoiding a 216-position permutation
    reconstruction for every candidate.
    """

    current = tuple(tuple(permutation) for permutation in initial_cluster_permutations)
    initial_cost = cluster_residual_cost(current)
    current_cost = initial_cost
    path = reduce_quarter_turn_path(initial_path)
    path_lengths = [len(path)]
    macro_count = 0

    library_by_cluster: list[list[tuple[tuple[int, int, int], MacroEffect]]] = [
        [] for _ in decomposition.physical_clusters
    ]
    for (cluster_index, cycle), macro in library.items():
        library_by_cluster[cluster_index].append((cycle, macro))
    for entries in library_by_cluster:
        entries.sort(key=lambda entry: (len(entry[1].path), entry[0], entry[1].path))

    while current_cost:
        suffixes = _local_suffix_effects(path, generators, decomposition)
        best_key: tuple[int, tuple[str, ...], int, tuple[int, int, int]] | None = None
        best_path: tuple[str, ...] | None = None
        best_cluster = -1
        best_cluster_state: Permutation | None = None

        for cluster_index, permutation in enumerate(current):
            cluster_cost = _three_cycle_units_fast(permutation)
            if cluster_cost == 0:
                continue
            for desired_cycle, desired_macro in library_by_cluster[cluster_index]:
                candidate_cluster = compose_pullbacks(
                    permutation,
                    desired_macro.cluster_permutations[cluster_index],
                )
                if _three_cycle_units_fast(candidate_cluster) != cluster_cost - 1:
                    continue
                for insertion_index, suffix_set in enumerate(suffixes):
                    local_suffix = suffix_set[cluster_index]
                    insertion_cycle = _canonical_directed_three_cycle(
                        tuple(local_suffix[position] for position in desired_cycle)
                    )
                    insertion_key = (cluster_index, insertion_cycle)
                    default_path = library[insertion_key].path
                    choices = (
                        alternative_paths.get(insertion_key, (default_path,))
                        if alternative_paths is not None
                        else (default_path,)
                    )
                    for insertion_path in choices:
                        candidate_path = reduce_quarter_turn_path(
                            path[:insertion_index]
                            + tuple(insertion_path)
                            + path[insertion_index:]
                        )
                        key = (
                            len(candidate_path),
                            candidate_path,
                            cluster_index,
                            desired_cycle,
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_path = candidate_path
                            best_cluster = cluster_index
                            best_cluster_state = candidate_cluster

        if best_path is None or best_cluster_state is None or best_cluster < 0:
            raise ValueError("no reducing inserted 3-cycle found")
        updated = list(current)
        updated[best_cluster] = best_cluster_state
        current = tuple(updated)
        current_cost -= 1
        path = best_path
        path_lengths.append(len(path))
        macro_count += 1

    return InsertedFinisherResult(
        path=path,
        macro_count=macro_count,
        initial_cost=initial_cost,
        final_cost=current_cost,
        path_length_trajectory=tuple(path_lengths),
    )


InsertionCandidate = tuple[
    int,
    tuple[str, ...],
    int,
    tuple[int, int, int],
    Permutation,
    int,
]

InsertionDescriptor = tuple[
    int,
    int,
    tuple[int, int, int],
    int,
    Permutation,
    tuple[str, ...],
]


@dataclass(frozen=True)
class InsertionContextSummary:
    """Bounded description of the useful insertions available on a rough word.

    The residual permutations alone do not determine the final primitive path
    length: inserting the same algebraic 3-cycle at different points of two
    words that reach the same state can expose different cancellations.  This
    summary keeps the best exact one-step insertion lengths together with their
    cluster and path-position context so a ranker can observe that distinction.
    """

    path_length: int
    residual_cost: int
    descriptor_limit: int
    descriptors_returned: int
    added_lengths: tuple[int, ...]
    cluster_indices: tuple[int, ...]
    insertion_indices: tuple[int, ...]
    cluster_best_added_lengths: tuple[int | None, ...]
    cluster_descriptor_counts: tuple[int, ...]
    position_descriptor_counts: tuple[int, ...]
    distinct_desired_cycles: int


def _insertion_reduced_length(
    path: tuple[str, ...],
    insertion_index: int,
    inserted_path: tuple[str, ...],
) -> int:
    """Length after insertion under the adjacent-same-slice reducer.

    A reduced path has runs of at most two identical primitive slices.  Inserting
    a word can only change the left boundary run, the inserted word, and the
    right boundary run; every other token contributes a fixed amount.  This
    evaluates a candidate in O(len(inserted_path)) instead of rebuilding the
    whole path.
    """

    if not 0 <= insertion_index <= len(path):
        raise IndexError("insertion index out of range")
    # A cancellation can expose another inserted token to the next original
    # run.  Each such crossing consumes at least one inserted token, so a margin
    # slightly larger than the inserted word is sufficient; if the whole word
    # disappears, the untouched original path simply rejoins in its already
    # reduced form.
    margin = len(inserted_path) + 3
    left_start = max(0, insertion_index - margin)
    if left_start:
        left_base = path[left_start].lstrip("-")
        while left_start and path[left_start - 1].lstrip("-") == left_base:
            left_start -= 1
    right_end = min(len(path), insertion_index + margin)
    if right_end < len(path) and right_end:
        right_base = path[right_end - 1].lstrip("-")
        while right_end < len(path) and path[right_end].lstrip("-") == right_base:
            right_end += 1
    local = path[left_start:insertion_index] + inserted_path + path[insertion_index:right_end]
    return left_start + len(reduce_quarter_turn_path(local)) + (len(path) - right_end)


def _top_reducing_insertion_descriptors(
    current: tuple[Permutation, ...],
    path: tuple[str, ...],
    library: Mapping[tuple[int, tuple[int, int, int]], MacroEffect],
    library_by_cluster: Sequence[Sequence[tuple[tuple[int, int, int], MacroEffect]]],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    limit: int,
) -> list[InsertionDescriptor]:
    if limit <= 0:
        return []
    suffixes = _local_suffix_effects(path, generators, decomposition)

    def descriptors():
        for cluster_index, permutation in enumerate(current):
            cluster_cost = _three_cycle_units_fast(permutation)
            if cluster_cost == 0:
                continue
            for desired_cycle, desired_macro in library_by_cluster[cluster_index]:
                candidate_cluster = compose_pullbacks(
                    permutation,
                    desired_macro.cluster_permutations[cluster_index],
                )
                if _three_cycle_units_fast(candidate_cluster) != cluster_cost - 1:
                    continue
                for insertion_index, suffix_set in enumerate(suffixes):
                    local_suffix = suffix_set[cluster_index]
                    insertion_cycle = _canonical_directed_three_cycle(
                        tuple(local_suffix[position] for position in desired_cycle)
                    )
                    insertion_path = library[(cluster_index, insertion_cycle)].path
                    yield (
                        _insertion_reduced_length(path, insertion_index, insertion_path),
                        cluster_index,
                        desired_cycle,
                        insertion_index,
                        candidate_cluster,
                        insertion_path,
                    )

    return heapq.nsmallest(limit, descriptors())


def summarize_reducing_insertion_context(
    initial_cluster_permutations: Sequence[Sequence[int]],
    initial_path: Sequence[str],
    library: Mapping[tuple[int, tuple[int, int, int]], MacroEffect],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    *,
    descriptor_limit: int = 512,
    position_bins: int = 8,
) -> InsertionContextSummary:
    """Return exact, bounded one-step insertion/cancellation features.

    Candidate lengths use the same local exact reducer as the Python insertion
    beam.  Only the globally shortest ``descriptor_limit`` candidates are kept;
    this makes feature extraction deterministic and bounded while preserving
    the cancellation frontier most relevant to completion cost.
    """

    if descriptor_limit <= 0:
        raise ValueError("descriptor_limit must be positive")
    if position_bins <= 0:
        raise ValueError("position_bins must be positive")
    current = tuple(tuple(permutation) for permutation in initial_cluster_permutations)
    if len(current) != len(decomposition.physical_clusters):
        raise ValueError("cluster counts differ")
    path = reduce_quarter_turn_path(initial_path)
    residual_cost = cluster_residual_cost(current)

    library_by_cluster: list[list[tuple[tuple[int, int, int], MacroEffect]]] = [
        [] for _ in decomposition.physical_clusters
    ]
    for (cluster_index, cycle), macro in library.items():
        library_by_cluster[cluster_index].append((cycle, macro))
    for entries in library_by_cluster:
        entries.sort(key=lambda entry: (len(entry[1].path), entry[0], entry[1].path))

    descriptors = _top_reducing_insertion_descriptors(
        current,
        path,
        library,
        library_by_cluster,
        generators,
        decomposition,
        descriptor_limit,
    )
    if residual_cost and not descriptors:
        raise ValueError("nonzero residual has no reducing 3-cycle insertion")

    added_lengths = tuple(descriptor[0] - len(path) for descriptor in descriptors)
    cluster_indices = tuple(descriptor[1] for descriptor in descriptors)
    insertion_indices = tuple(descriptor[3] for descriptor in descriptors)
    cluster_best: list[int | None] = [None] * len(current)
    cluster_counts = [0] * len(current)
    position_counts = [0] * position_bins
    desired_cycles: set[tuple[int, tuple[int, int, int]]] = set()
    for descriptor, added_length in zip(descriptors, added_lengths, strict=True):
        _, cluster_index, desired_cycle, insertion_index, _, _ = descriptor
        cluster_counts[cluster_index] += 1
        best = cluster_best[cluster_index]
        if best is None or added_length < best:
            cluster_best[cluster_index] = added_length
        position_bin = min(
            position_bins - 1,
            insertion_index * position_bins // (len(path) + 1),
        )
        position_counts[position_bin] += 1
        desired_cycles.add((cluster_index, desired_cycle))

    return InsertionContextSummary(
        path_length=len(path),
        residual_cost=residual_cost,
        descriptor_limit=descriptor_limit,
        descriptors_returned=len(descriptors),
        added_lengths=added_lengths,
        cluster_indices=cluster_indices,
        insertion_indices=insertion_indices,
        cluster_best_added_lengths=tuple(cluster_best),
        cluster_descriptor_counts=tuple(cluster_counts),
        position_descriptor_counts=tuple(position_counts),
        distinct_desired_cycles=len(desired_cycles),
    )


def _materialize_insertion_descriptor(
    current: tuple[Permutation, ...],
    path: tuple[str, ...],
    descriptor: InsertionDescriptor,
) -> tuple[tuple[Permutation, ...], tuple[str, ...]]:
    expected_length, cluster_index, _, insertion_index, candidate_cluster, insertion_path = descriptor
    candidate_path = reduce_quarter_turn_path(
        path[:insertion_index] + insertion_path + path[insertion_index:]
    )
    if len(candidate_path) != expected_length:
        raise AssertionError(
            f"local insertion length {expected_length} disagrees with materialized {len(candidate_path)}"
        )
    updated = list(current)
    updated[cluster_index] = candidate_cluster
    return tuple(updated), candidate_path


def _iter_reducing_insertions(
    current: tuple[Permutation, ...],
    path: tuple[str, ...],
    library: Mapping[tuple[int, tuple[int, int, int]], MacroEffect],
    library_by_cluster: Sequence[Sequence[tuple[tuple[int, int, int], MacroEffect]]],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    insertion_positions: Sequence[int] | None = None,
):
    """Yield exact single-step reducing insertions ordered by an external key."""

    suffixes = _local_suffix_effects(path, generators, decomposition)
    positions = range(len(path) + 1) if insertion_positions is None else insertion_positions
    for cluster_index, permutation in enumerate(current):
        cluster_cost = _three_cycle_units_fast(permutation)
        if cluster_cost == 0:
            continue
        for desired_cycle, desired_macro in library_by_cluster[cluster_index]:
            candidate_cluster = compose_pullbacks(
                permutation,
                desired_macro.cluster_permutations[cluster_index],
            )
            if _three_cycle_units_fast(candidate_cluster) != cluster_cost - 1:
                continue
            for insertion_index in positions:
                local_suffix = suffixes[insertion_index][cluster_index]
                insertion_cycle = _canonical_directed_three_cycle(
                    tuple(local_suffix[position] for position in desired_cycle)
                )
                insertion_macro = library[(cluster_index, insertion_cycle)]
                candidate_path = reduce_quarter_turn_path(
                    path[:insertion_index]
                    + insertion_macro.path
                    + path[insertion_index:]
                )
                yield (
                    len(candidate_path),
                    candidate_path,
                    cluster_index,
                    desired_cycle,
                    candidate_cluster,
                    insertion_index,
                )


def finish_with_lookahead_inserted_three_cycles(
    initial_cluster_permutations: Sequence[Sequence[int]],
    initial_path: Sequence[str],
    library: Mapping[tuple[int, tuple[int, int, int]], MacroEffect],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    first_width: int = 24,
    local_window: int = 6,
    global_width: int = 2,
) -> InsertedFinisherResult:
    """Exact greedy finisher with a bounded one-step insertion lookahead.

    At every residual level, retain the ``first_width`` shortest legal first
    insertions.  Rank those choices by the shortest legal second insertion near
    the first insertion site; for the best ``global_width`` first choices, scan
    every second insertion position as an additional safeguard.  Only the first
    insertion is committed, so algebraic residual cost decreases monotonically
    and every final path remains exactly replayable.
    """

    if first_width <= 0 or local_window < 0 or global_width < 0:
        raise ValueError("invalid lookahead width/window")
    current = tuple(tuple(permutation) for permutation in initial_cluster_permutations)
    initial_cost = cluster_residual_cost(current)
    current_cost = initial_cost
    path = reduce_quarter_turn_path(initial_path)
    path_lengths = [len(path)]
    macro_count = 0

    library_by_cluster: list[list[tuple[tuple[int, int, int], MacroEffect]]] = [
        [] for _ in decomposition.physical_clusters
    ]
    for (cluster_index, cycle), macro in library.items():
        library_by_cluster[cluster_index].append((cycle, macro))
    for entries in library_by_cluster:
        entries.sort(key=lambda entry: (len(entry[1].path), entry[0], entry[1].path))

    while current_cost:
        first_candidates = heapq.nsmallest(
            first_width,
            _iter_reducing_insertions(
                current,
                path,
                library,
                library_by_cluster,
                generators,
                decomposition,
            ),
        )
        if not first_candidates:
            raise ValueError("no reducing inserted 3-cycle found")

        best_rank: tuple[int, int, tuple[str, ...]] | None = None
        best_first: InsertionCandidate | None = None
        for first_rank, candidate in enumerate(first_candidates):
            _, candidate_path, cluster_index, _, candidate_cluster, insertion_index = candidate
            updated = list(current)
            updated[cluster_index] = candidate_cluster
            next_current = tuple(updated)
            if current_cost == 1:
                lookahead_length = len(candidate_path)
            else:
                if first_rank < global_width:
                    positions: Sequence[int] | None = None
                else:
                    lower = max(0, insertion_index - local_window)
                    upper = min(len(candidate_path), insertion_index + local_window)
                    positions = tuple(range(lower, upper + 1))
                second = min(
                    _iter_reducing_insertions(
                        next_current,
                        candidate_path,
                        library,
                        library_by_cluster,
                        generators,
                        decomposition,
                        positions,
                    ),
                    default=None,
                )
                lookahead_length = len(candidate_path) if second is None else second[0]
            rank = (lookahead_length, len(candidate_path), candidate_path)
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_first = candidate

        if best_first is None:
            raise AssertionError("lookahead produced no first insertion")
        _, path, cluster_index, _, candidate_cluster, _ = best_first
        updated = list(current)
        updated[cluster_index] = candidate_cluster
        current = tuple(updated)
        current_cost -= 1
        path_lengths.append(len(path))
        macro_count += 1

    return InsertedFinisherResult(
        path=path,
        macro_count=macro_count,
        initial_cost=initial_cost,
        final_cost=current_cost,
        path_length_trajectory=tuple(path_lengths),
    )


def finish_with_beam_inserted_three_cycles(
    initial_cluster_permutations: Sequence[Sequence[int]],
    initial_path: Sequence[str],
    library: Mapping[tuple[int, tuple[int, int, int]], MacroEffect],
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
    beam_width: int = 64,
    branch_width: int = 12,
) -> InsertedFinisherResult:
    """Beam search over exact residual-reducing 3-cycle insertions.

    Every level uses exactly one algebraic 3-cycle, so all states at a level have
    the same remaining exact cost.  Path length is therefore the natural beam
    key.  Per-parent branching is restricted to its shortest immediate
    insertions; the beam preserves alternate cycle orders and cancellation
    contexts that one-step greedy permanently discards.
    """

    if beam_width <= 0 or branch_width <= 0:
        raise ValueError("beam_width and branch_width must be positive")
    initial_current = tuple(tuple(permutation) for permutation in initial_cluster_permutations)
    initial_cost = cluster_residual_cost(initial_current)
    initial_reduced_path = reduce_quarter_turn_path(initial_path)

    library_by_cluster: list[list[tuple[tuple[int, int, int], MacroEffect]]] = [
        [] for _ in decomposition.physical_clusters
    ]
    for (cluster_index, cycle), macro in library.items():
        library_by_cluster[cluster_index].append((cycle, macro))
    for entries in library_by_cluster:
        entries.sort(key=lambda entry: (len(entry[1].path), entry[0], entry[1].path))

    beam: list[tuple[tuple[Permutation, ...], tuple[str, ...]]] = [
        (initial_current, initial_reduced_path)
    ]
    best_lengths = [len(initial_reduced_path)]
    for _ in range(initial_cost):
        children: list[tuple[tuple[Permutation, ...], tuple[str, ...]]] = []
        for current, path in beam:
            descriptors = _top_reducing_insertion_descriptors(
                current,
                path,
                library,
                library_by_cluster,
                generators,
                decomposition,
                branch_width,
            )
            children.extend(
                _materialize_insertion_descriptor(current, path, descriptor)
                for descriptor in descriptors
            )
        if not children:
            raise ValueError("3-cycle insertion beam exhausted")

        children.sort(key=lambda item: (len(item[1]), item[1]))
        next_beam: list[tuple[tuple[Permutation, ...], tuple[str, ...]]] = []
        seen: set[tuple[bytes, ...]] = set()
        for current, path in children:
            # For one algebraic residual, the shortest path dominates an exact
            # duplicate.  Different residuals are retained even at equal path.
            state_key = tuple(bytes(permutation) for permutation in current)
            if state_key in seen:
                continue
            seen.add(state_key)
            next_beam.append((current, path))
            if len(next_beam) == beam_width:
                break
        beam = next_beam
        best_lengths.append(len(beam[0][1]))

    final_current, final_path = min(beam, key=lambda item: (len(item[1]), item[1]))
    if cluster_residual_cost(final_current) != 0:
        raise AssertionError("3-cycle insertion beam ended with nonzero residual")
    return InsertedFinisherResult(
        path=final_path,
        macro_count=initial_cost,
        initial_cost=initial_cost,
        final_cost=0,
        path_length_trajectory=tuple(best_lengths),
    )
