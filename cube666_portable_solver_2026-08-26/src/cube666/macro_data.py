"""Exact action tables and teacher data for the 6x6x6 bulk-macro policy.

The learned policy is only a shortlist generator.  Labels and evaluation costs
come from exact permutation algebra, so no model prediction is trusted without
re-ranking the proposed macros with :func:`MacroActionTable.action_costs`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from cube666.classical import Cube666Decomposition
from cube666.macros import MacroEffect, analyze_corner_fixing_macro, inverse_permutation


CLUSTER_COUNT = 6
CLUSTER_SIZE = 24


def _validate_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.uint8)
    if states.shape[-2:] != (CLUSTER_COUNT, CLUSTER_SIZE):
        raise ValueError(
            f"expected (..., {CLUSTER_COUNT}, {CLUSTER_SIZE}) cluster permutations; "
            f"got {states.shape}"
        )
    expected = np.arange(CLUSTER_SIZE, dtype=np.uint8)
    if not np.all(np.sort(states, axis=-1) == expected):
        raise ValueError("every cluster row must be a permutation of 0..23")
    return states


def _three_cycle_units_batch_unchecked(permutations: np.ndarray) -> np.ndarray:
    original_shape = permutations.shape[:-1]
    flat = permutations.reshape(-1, CLUSTER_SIZE)
    visited = np.zeros_like(flat, dtype=np.bool_)
    units = np.zeros(flat.shape[0], dtype=np.int16)
    rows = np.arange(flat.shape[0])

    for start in range(CLUSTER_SIZE):
        active = ~visited[:, start]
        positions = np.full(flat.shape[0], start, dtype=np.uint8)
        lengths = np.zeros(flat.shape[0], dtype=np.uint8)
        while np.any(active):
            active_rows = rows[active]
            visited[active_rows, positions[active]] = True
            positions[active] = flat[active_rows, positions[active]]
            lengths[active] += 1
            active &= positions != start
        units += lengths // 2
    return units.reshape(original_shape)


def _three_cycle_units_one_unchecked(permutation: np.ndarray) -> int:
    seen = np.zeros(CLUSTER_SIZE, dtype=np.bool_)
    units = 0
    for start in range(CLUSTER_SIZE):
        if seen[start]:
            continue
        length = 0
        position = start
        while not seen[position]:
            seen[position] = True
            length += 1
            position = int(permutation[position])
        units += length // 2
    return units


def three_cycle_units_batch(permutations: np.ndarray) -> np.ndarray:
    """Return ``sum(floor(cycle_length / 2))`` for arrays of permutations."""

    permutations = np.asarray(permutations, dtype=np.uint8)
    if permutations.shape[-1] != CLUSTER_SIZE:
        raise ValueError(f"expected permutations of size {CLUSTER_SIZE}")
    expected = np.arange(CLUSTER_SIZE, dtype=np.uint8)
    if not np.all(np.sort(permutations, axis=-1) == expected):
        raise ValueError("every row must be a permutation of 0..23")
    return _three_cycle_units_batch_unchecked(permutations)


def cluster_costs(states: np.ndarray) -> np.ndarray:
    """Exact unrestricted 3-cycle costs, one value per physical cluster."""

    states = _validate_states(states)
    return three_cycle_units_batch(states)


def compose_cluster_states(states: np.ndarray, effects: np.ndarray) -> np.ndarray:
    """Apply pullback macro effects to one state or a matched batch of states."""

    states = _validate_states(states)
    effects = _validate_states(effects)
    try:
        shape = np.broadcast_shapes(states.shape, effects.shape)
    except ValueError as exc:
        raise ValueError(f"state/effect shapes do not broadcast: {states.shape}, {effects.shape}") from exc
    states = np.broadcast_to(states, shape)
    effects = np.broadcast_to(effects, shape)
    return np.take_along_axis(states, effects, axis=-1)


@dataclass(frozen=True)
class MacroActionTable:
    """A stable policy-index-to-exact-macro mapping."""

    effects: np.ndarray
    paths: tuple[tuple[str, ...], ...]
    inverse_indices: np.ndarray
    digest: str

    @classmethod
    def from_macros(cls, macros: Sequence[MacroEffect]) -> "MacroActionTable":
        if not macros:
            raise ValueError("at least one macro is required")
        effects = np.asarray(
            [macro.cluster_permutations for macro in macros],
            dtype=np.uint8,
        )
        _validate_states(effects)
        paths = tuple(tuple(macro.path) for macro in macros)
        by_effect = {
            effect.tobytes(): index
            for index, effect in enumerate(effects)
        }
        if len(by_effect) != len(macros):
            raise ValueError("macro action table contains duplicate cluster effects")

        inverse_indices = np.empty(len(macros), dtype=np.int32)
        for index, effect in enumerate(effects):
            inverse = np.asarray(
                [inverse_permutation(permutation) for permutation in effect],
                dtype=np.uint8,
            )
            try:
                inverse_indices[index] = by_effect[inverse.tobytes()]
            except KeyError as exc:
                raise ValueError(f"macro {index} has no inverse in the action table") from exc

        digest = hashlib.sha256(effects.tobytes()).hexdigest()
        return cls(effects, paths, inverse_indices, digest)

    @property
    def action_count(self) -> int:
        return int(self.effects.shape[0])

    def apply(self, state: np.ndarray, action: int) -> np.ndarray:
        if action < 0 or action >= self.action_count:
            raise IndexError(action)
        return compose_cluster_states(state, self.effects[action])

    def action_cluster_costs(
        self,
        state: np.ndarray,
        *,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Exact per-cluster residual after every action."""

        state = _validate_states(np.asarray(state))
        if state.shape != (CLUSTER_COUNT, CLUSTER_SIZE):
            raise ValueError("action scoring accepts exactly one state")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        result = np.empty((self.action_count, CLUSTER_COUNT), dtype=np.int16)
        for start in range(0, self.action_count, chunk_size):
            stop = min(start + chunk_size, self.action_count)
            effects = self.effects[start:stop]
            candidates = np.take_along_axis(
                np.broadcast_to(state, effects.shape),
                effects,
                axis=-1,
            )
            result[start:stop] = _three_cycle_units_batch_unchecked(candidates)
        return result

    def action_costs(self, state: np.ndarray, *, chunk_size: int = 4096) -> np.ndarray:
        return self.action_cluster_costs(state, chunk_size=chunk_size).sum(axis=1)

    def exact_rerank(
        self,
        state: np.ndarray,
        proposed_actions: Sequence[int],
    ) -> tuple[int, int]:
        """Choose the lowest exact-cost action from a model-generated shortlist."""

        proposed = np.asarray(proposed_actions, dtype=np.int64)
        if proposed.ndim != 1 or proposed.size == 0:
            raise ValueError("proposed_actions must be a non-empty one-dimensional sequence")
        if np.any(proposed < 0) or np.any(proposed >= self.action_count):
            raise IndexError("proposed action is outside the action table")
        candidates = compose_cluster_states(state, self.effects[proposed])
        costs = cluster_costs(candidates).sum(axis=1)
        best_position = min(
            range(len(proposed)),
            key=lambda position: (
                int(costs[position]),
                len(self.paths[int(proposed[position])]),
                self.paths[int(proposed[position])],
            ),
        )
        return int(proposed[best_position]), int(costs[best_position])


def validate_factorized_action_table(table: MacroActionTable) -> int:
    """Verify six contiguous, identically ordered isolated action blocks."""

    identity = np.arange(CLUSTER_SIZE, dtype=np.uint8)
    active = np.any(table.effects != identity[None, None, :], axis=2)
    if not np.all(active.sum(axis=1) == 1):
        raise ValueError("factorized policy requires actions that affect exactly one cluster")
    active_clusters = active.argmax(axis=1)
    counts = np.bincount(active_clusters, minlength=CLUSTER_COUNT)
    if len(set(int(value) for value in counts)) != 1:
        raise ValueError("factorized action blocks have unequal sizes")
    local_count = int(counts[0])
    expected_clusters = np.repeat(np.arange(CLUSTER_COUNT), local_count)
    if not np.array_equal(active_clusters, expected_clusters):
        raise ValueError("factorized action blocks are not contiguous by cluster")
    reference = table.effects[:local_count, 0]
    for cluster in range(1, CLUSTER_COUNT):
        start = cluster * local_count
        if not np.array_equal(table.effects[start : start + local_count, cluster], reference):
            raise ValueError("factorized action blocks do not share local action order")
    return local_count


def save_macro_action_library(
    path: str | Path,
    macros: Sequence[MacroEffect],
) -> MacroActionTable:
    """Persist an ordered action vocabulary as auditable primitive paths."""

    table = MacroActionTable.from_macros(macros)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "action_digest": table.digest,
        "format_version": 1,
        "paths": [list(path) for path in table.paths],
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(output)
    return table


def load_macro_action_library(
    path: str | Path,
    generators: Mapping[str, Sequence[int]],
    decomposition: Cube666Decomposition,
) -> tuple[tuple[MacroEffect, ...], MacroActionTable]:
    """Replay and verify every path in a persisted macro action vocabulary."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != 1:
        raise ValueError("unsupported macro action library format")
    macros = tuple(
        analyze_corner_fixing_macro(raw_path, generators, decomposition)
        for raw_path in payload["paths"]
    )
    table = MacroActionTable.from_macros(macros)
    if table.digest != payload.get("action_digest"):
        raise ValueError("macro action library digest mismatch")
    return macros, table


@dataclass(frozen=True)
class MacroTeacherDataset:
    states: np.ndarray
    teacher_actions: np.ndarray
    teacher_action_counts: np.ndarray
    cluster_cost_targets: np.ndarray
    teacher_next_costs: np.ndarray
    walk_depths: np.ndarray
    action_digest: str
    allow_non_improving: bool = False
    search_value_targets: np.ndarray | None = None
    search_cluster_targets: np.ndarray | None = None

    def validate(self, action_count: int | None = None) -> None:
        states = _validate_states(self.states)
        sample_count = len(states)
        if self.teacher_actions.ndim != 2 or len(self.teacher_actions) != sample_count:
            raise ValueError("teacher_actions must have shape (samples, max_labels)")
        if self.teacher_action_counts.shape != (sample_count,):
            raise ValueError("teacher_action_counts has the wrong shape")
        if self.cluster_cost_targets.shape != (sample_count, CLUSTER_COUNT):
            raise ValueError("cluster_cost_targets has the wrong shape")
        if self.teacher_next_costs.shape != (sample_count,):
            raise ValueError("teacher_next_costs has the wrong shape")
        if self.walk_depths.shape != (sample_count,):
            raise ValueError("walk_depths has the wrong shape")
        if (self.search_value_targets is None) != (self.search_cluster_targets is None):
            raise ValueError("search value targets must be supplied together")
        if self.search_value_targets is not None:
            if self.search_value_targets.shape != (sample_count,):
                raise ValueError("search_value_targets has the wrong shape")
            if self.search_cluster_targets.shape != (sample_count, CLUSTER_COUNT):
                raise ValueError("search_cluster_targets has the wrong shape")
            if (
                not np.all(np.isfinite(self.search_value_targets))
                or not np.all(np.isfinite(self.search_cluster_targets))
                or np.any(self.search_value_targets < 0)
                or np.any(self.search_cluster_targets < 0)
            ):
                raise ValueError("search value targets must be finite and nonnegative")
            if not np.allclose(
                self.search_cluster_targets.sum(axis=1),
                self.search_value_targets,
                rtol=1e-5,
                atol=1e-5,
            ):
                raise ValueError("cluster search targets must sum to total targets")
        if np.any(self.teacher_action_counts <= 0):
            raise ValueError("every sample needs at least one teacher action")
        mask = np.arange(self.teacher_actions.shape[1])[None, :] < self.teacher_action_counts[:, None]
        if np.any(self.teacher_actions[mask] < 0):
            raise ValueError("active teacher labels must be nonnegative")
        if np.any(self.teacher_actions[~mask] != -1):
            raise ValueError("unused teacher label slots must be -1")
        if action_count is not None and np.any(self.teacher_actions[mask] >= action_count):
            raise ValueError("teacher label exceeds the action table")
        if (
            not self.allow_non_improving
            and np.any(self.teacher_next_costs >= self.cluster_cost_targets.sum(axis=1))
        ):
            raise ValueError("every teacher action must strictly reduce exact residual cost")
        if np.any(self.teacher_next_costs < 0):
            raise ValueError("teacher next costs must be nonnegative")

    @property
    def sample_count(self) -> int:
        return int(self.states.shape[0])

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        metadata = json.dumps(
            {
                "action_digest": self.action_digest,
                "allow_non_improving": self.allow_non_improving,
                "format_version": 3,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        with open(temporary, "wb") as handle:
            np.savez_compressed(
                handle,
                states=self.states,
                teacher_actions=self.teacher_actions,
                teacher_action_counts=self.teacher_action_counts,
                cluster_cost_targets=self.cluster_cost_targets,
                teacher_next_costs=self.teacher_next_costs,
                walk_depths=self.walk_depths,
                search_value_targets=(
                    self.search_value_targets
                    if self.search_value_targets is not None
                    else self.cluster_cost_targets.sum(axis=1).astype(np.float32)
                ),
                search_cluster_targets=(
                    self.search_cluster_targets
                    if self.search_cluster_targets is not None
                    else self.cluster_cost_targets.astype(np.float32)
                ),
                metadata=np.asarray(metadata),
            )
        temporary.replace(output)

    @classmethod
    def load(cls, path: str | Path) -> "MacroTeacherDataset":
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            if metadata.get("format_version") not in (1, 2, 3):
                raise ValueError("unsupported macro teacher dataset format")
            cluster_cost_targets = payload["cluster_cost_targets"].astype(
                np.int16, copy=False
            )
            if metadata.get("format_version") == 3:
                search_value_targets = payload["search_value_targets"].astype(
                    np.float32, copy=False
                )
                search_cluster_targets = payload["search_cluster_targets"].astype(
                    np.float32, copy=False
                )
            else:
                search_cluster_targets = cluster_cost_targets.astype(np.float32)
                search_value_targets = search_cluster_targets.sum(axis=1)
            dataset = cls(
                states=payload["states"].astype(np.uint8, copy=False),
                teacher_actions=payload["teacher_actions"].astype(np.int32, copy=False),
                teacher_action_counts=payload["teacher_action_counts"].astype(np.int16, copy=False),
                cluster_cost_targets=cluster_cost_targets,
                teacher_next_costs=payload["teacher_next_costs"].astype(np.int16, copy=False),
                walk_depths=payload["walk_depths"].astype(np.int16, copy=False),
                action_digest=str(metadata["action_digest"]),
                allow_non_improving=bool(metadata.get("allow_non_improving", False)),
                search_value_targets=search_value_targets,
                search_cluster_targets=search_cluster_targets,
            )
        dataset.validate()
        return dataset


def generate_random_walk_teacher(
    table: MacroActionTable,
    *,
    sample_count: int,
    max_walk_depth: int,
    max_labels: int = 32,
    seed: int = 666,
    score_chunk_size: int = 4096,
) -> MacroTeacherDataset:
    """Create exact greedy labels on macro-walk states.

    This is the bootstrap distribution.  A later DAgger pass should append
    normalized competition states and states visited by the learned rollout.
    """

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if max_walk_depth <= 0:
        raise ValueError("max_walk_depth must be positive")
    if max_labels <= 0:
        raise ValueError("max_labels must be positive")

    rng = np.random.default_rng(seed)
    identity = np.broadcast_to(
        np.arange(CLUSTER_SIZE, dtype=np.uint8),
        (CLUSTER_COUNT, CLUSTER_SIZE),
    ).copy()
    states = np.empty((sample_count, CLUSTER_COUNT, CLUSTER_SIZE), dtype=np.uint8)
    teacher_actions = np.full((sample_count, max_labels), -1, dtype=np.int32)
    teacher_action_counts = np.empty(sample_count, dtype=np.int16)
    cost_targets = np.empty((sample_count, CLUSTER_COUNT), dtype=np.int16)
    next_costs = np.empty(sample_count, dtype=np.int16)
    walk_depths = np.empty(sample_count, dtype=np.int16)

    state = identity
    previous_action = -1
    depth = 0
    sample = 0
    attempts = 0
    max_attempts = sample_count * 100
    while sample < sample_count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("could not find enough strictly improving teacher states")
        if depth >= max_walk_depth:
            state = identity
            previous_action = -1
            depth = 0

        action = int(rng.integers(table.action_count))
        if previous_action >= 0 and action == int(table.inverse_indices[previous_action]):
            action = (action + 1) % table.action_count
        state = table.apply(state, action)
        previous_action = action
        depth += 1

        current_cluster_costs = cluster_costs(state)
        action_costs = table.action_costs(state, chunk_size=score_chunk_size)
        best_cost = int(action_costs.min())
        if best_cost >= int(current_cluster_costs.sum()):
            continue
        best = np.flatnonzero(action_costs == best_cost)
        if len(best) > max_labels:
            best = np.sort(rng.choice(best, size=max_labels, replace=False))

        states[sample] = state
        teacher_actions[sample, : len(best)] = best
        teacher_action_counts[sample] = len(best)
        cost_targets[sample] = current_cluster_costs
        next_costs[sample] = best_cost
        walk_depths[sample] = depth
        sample += 1

    dataset = MacroTeacherDataset(
        states=states,
        teacher_actions=teacher_actions,
        teacher_action_counts=teacher_action_counts,
        cluster_cost_targets=cost_targets,
        teacher_next_costs=next_costs,
        walk_depths=walk_depths,
        action_digest=table.digest,
    )
    dataset.validate(table.action_count)
    return dataset


def generate_geodesic_macro_teacher(
    table: MacroActionTable,
    *,
    sample_count: int,
    max_depth: int = 68,
    max_labels: int = 16,
    label_candidates: int = 256,
    seed: int = 666,
    max_attempts_per_step: int = 20_000,
) -> MacroTeacherDataset:
    """Generate exact geodesic macro walks with certified reducing labels.

    This is intended for isolated 3-cycle action tables.  Each accepted outward
    action must increase the exact residual by one; consequently its inverse is
    a shortest-path action and the walk depth is the exact distance to solved.
    Additional exact reducers are sampled from active clusters so the policy is
    not asked to imitate an arbitrary inverse when many shortest moves exist.
    """

    if sample_count <= 0 or max_depth <= 0:
        raise ValueError("sample_count and max_depth must be positive")
    if max_labels <= 0 or label_candidates < 0:
        raise ValueError("max_labels must be positive and label_candidates nonnegative")
    if max_attempts_per_step <= 0:
        raise ValueError("max_attempts_per_step must be positive")
    per_action_cluster_cost = _three_cycle_units_batch_unchecked(table.effects)
    action_cluster_cost = per_action_cluster_cost.sum(axis=1)
    if not np.all(action_cluster_cost == 1):
        raise ValueError("geodesic teacher requires isolated unit-cost 3-cycle actions")
    active_cluster_by_action = per_action_cluster_cost.argmax(axis=1)
    actions_by_cluster = tuple(
        np.flatnonzero(active_cluster_by_action == cluster)
        for cluster in range(CLUSTER_COUNT)
    )
    actions_per_cluster = {len(actions) for actions in actions_by_cluster}
    if len(actions_per_cluster) != 1:
        raise ValueError("every cluster must have the same isolated action count")
    action_matrix = np.stack(actions_by_cluster)

    rng = np.random.default_rng(seed)
    identity = np.broadcast_to(
        np.arange(CLUSTER_SIZE, dtype=np.uint8),
        (CLUSTER_COUNT, CLUSTER_SIZE),
    ).copy()
    states = np.empty((sample_count, CLUSTER_COUNT, CLUSTER_SIZE), dtype=np.uint8)
    labels = np.full((sample_count, max_labels), -1, dtype=np.int32)
    counts = np.empty(sample_count, dtype=np.int16)
    cost_targets = np.empty((sample_count, CLUSTER_COUNT), dtype=np.int16)
    next_costs = np.empty(sample_count, dtype=np.int16)
    depths = np.empty(sample_count, dtype=np.int16)

    state = identity
    current_cluster_costs = np.zeros(CLUSTER_COUNT, dtype=np.int16)
    depth = 0
    for sample in range(sample_count):
        if depth >= max_depth:
            state = identity
            current_cluster_costs = np.zeros(CLUSTER_COUNT, dtype=np.int16)
            depth = 0

        for _ in range(max_attempts_per_step):
            eligible_clusters = np.flatnonzero(current_cluster_costs < 12)
            cluster = int(rng.choice(eligible_clusters))
            outward_action = int(rng.choice(actions_by_cluster[cluster]))
            effect = table.effects[outward_action]
            candidate_cluster = state[cluster][effect[cluster]]
            candidate_cost = _three_cycle_units_one_unchecked(candidate_cluster)
            if candidate_cost == int(current_cluster_costs[cluster]) + 1:
                break
        else:
            raise RuntimeError(
                f"failed to extend an exact geodesic walk at depth {depth}; "
                "lower max_depth or inspect the action table"
            )

        state = state.copy()
        state[cluster] = candidate_cluster
        current_cluster_costs = current_cluster_costs.copy()
        current_cluster_costs[cluster] = candidate_cost
        depth += 1
        states[sample] = state
        reducing = [int(table.inverse_indices[outward_action])]
        if max_labels > 1 and label_candidates > 0:
            active_clusters = np.flatnonzero(current_cluster_costs > 0)
            sampled_clusters = rng.choice(
                active_clusters,
                size=label_candidates,
                replace=True,
            )
            sampled_offsets = rng.integers(
                action_matrix.shape[1],
                size=label_candidates,
            )
            sampled_actions = action_matrix[sampled_clusters, sampled_offsets]
            sampled_effects = table.effects[
                sampled_actions,
                sampled_clusters,
            ]
            sampled_states = np.take_along_axis(
                state[sampled_clusters],
                sampled_effects,
                axis=1,
            )
            sampled_costs = _three_cycle_units_batch_unchecked(sampled_states)
            is_reducing = sampled_costs == current_cluster_costs[sampled_clusters] - 1
            for action in sampled_actions[is_reducing]:
                action_id = int(action)
                if action_id not in reducing:
                    reducing.append(action_id)
                    if len(reducing) == max_labels:
                        break
        labels[sample, : len(reducing)] = reducing
        counts[sample] = len(reducing)
        cost_targets[sample] = current_cluster_costs
        next_costs[sample] = depth - 1
        depths[sample] = depth

    dataset = MacroTeacherDataset(
        states=states,
        teacher_actions=labels,
        teacher_action_counts=counts,
        cluster_cost_targets=cost_targets,
        teacher_next_costs=next_costs,
        walk_depths=depths,
        action_digest=table.digest,
    )
    dataset.validate(table.action_count)
    return dataset
