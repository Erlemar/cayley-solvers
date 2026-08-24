"""Batched GFlowNet beam and best-first search for Christopher's Jewel.

The GFlowNet supplies backward action probabilities.  Search never trusts the
network for correctness: candidates are completed through the exact ball and
the returned path is replay-verified by the caller.  The PDB/ring heuristic is
used for ranking and incumbent pruning, but these algorithms are satisficing;
``optimal`` is true only when a solution length meets the admissible root bound.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np

from .ball import ExactBall
from .gflownet import GFNBackwardPolicy
from .pdb import EdgePatternDatabase, combined_heuristic
from .puzzle import (
    INVERSE_ACTION,
    JewelState,
    apply_action,
    apply_path,
    permutation_parity,
)


@dataclass(slots=True)
class GFNSearchResult:
    algorithm: str
    path: list[int] | None
    optimal: bool
    width: int
    expanded_states: int
    generated_states: int
    seconds: float
    reason: str
    lower_bound: int


def _admissible_h(state: JewelState, pdbs: Sequence[EdgePatternDatabase]) -> int:
    raw = combined_heuristic(state, pdbs)
    parity = permutation_parity(state.edge_perm)
    return raw + ((parity - raw) & 1)


def _direct_result(
    algorithm: str,
    start: JewelState,
    ball: ExactBall,
    lower_bound: int,
    incumbent_length: int | None,
    started: float,
) -> GFNSearchResult | None:
    direct = ball.exact_path(start)
    if direct is None:
        return None
    if incumbent_length is not None and len(direct) >= incumbent_length:
        return GFNSearchResult(
            algorithm,
            None,
            False,
            0,
            0,
            0,
            perf_counter() - started,
            "incumbent_is_exact",
            lower_bound,
        )
    return GFNSearchResult(
        algorithm,
        direct,
        True,
        0,
        0,
        0,
        perf_counter() - started,
        "exact_ball",
        len(direct),
    )


def _better_bound(best_path: list[int] | None, incumbent_length: int | None) -> int | None:
    if best_path is None:
        return incumbent_length
    if incumbent_length is None:
        return len(best_path)
    return min(len(best_path), incumbent_length)


def gfn_policy_beam_search(
    start: JewelState,
    policy: GFNBackwardPolicy,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    width: int,
    branch_actions: int = 4,
    max_learned_steps: int = 20,
    heuristic_weight: float = 0.35,
    depth_weight: float = 0.0,
    incumbent_length: int | None = None,
    probability_floor: float = 1e-12,
) -> GFNSearchResult:
    """Layered beam search using cumulative backward-policy surprise."""
    if width < 1:
        raise ValueError("width must be positive")
    if not 1 <= branch_actions <= 12:
        raise ValueError("branch_actions must be in 1..12")
    started = perf_counter()
    lower_bound = _admissible_h(start, pdbs)
    direct = _direct_result(
        "gfn_beam", start, ball, lower_bound, incumbent_length, started
    )
    if direct is not None:
        direct.width = width
        return direct

    # state, path, cumulative negative log probability, last action
    frontier: list[tuple[JewelState, list[int], float, int]] = [
        (start, [], 0.0, -1)
    ]
    selected_ranks = {start.rank()}
    expanded = 0
    generated = 0
    best_path: list[int] | None = None

    for _step in range(max_learned_steps):
        states = [item[0] for item in frontier]
        probabilities = np.asarray(policy.batch(states), dtype=np.float64)
        if probabilities.shape != (len(states), 12):
            raise ValueError("GFlowNet policy returned the wrong batch shape")
        log_probabilities = np.log(np.maximum(probabilities, probability_floor))
        expanded += len(frontier)
        # priority, state, path, last action, cumulative neural cost
        dedup: dict[int, tuple[float, JewelState, list[int], int, float]] = {}

        for parent_i, (state, path, neural_cost, last_action) in enumerate(frontier):
            order = np.argsort(-probabilities[parent_i])
            used = 0
            for action_value in order:
                action = int(action_value)
                if last_action >= 0 and action == int(INVERSE_ACTION[last_action]):
                    continue
                used += 1
                if used > branch_actions:
                    break
                child = apply_action(state, action)
                generated += 1
                child_path = path + [action]
                suffix = ball.exact_path(child)
                if suffix is not None:
                    candidate = child_path + suffix
                    bound = _better_bound(best_path, incumbent_length)
                    if bound is None or len(candidate) < bound:
                        best_path = candidate
                    continue

                child_h = _admissible_h(child, pdbs)
                bound = _better_bound(best_path, incumbent_length)
                if bound is not None and len(child_path) + child_h >= bound:
                    continue
                rank = child.rank()
                if rank in selected_ranks:
                    continue
                child_cost = neural_cost - float(log_probabilities[parent_i, action])
                priority = (
                    child_cost
                    + heuristic_weight * child_h
                    + depth_weight * len(child_path)
                )
                previous = dedup.get(rank)
                if previous is None or priority < previous[0]:
                    dedup[rank] = (
                        priority,
                        child,
                        child_path,
                        action,
                        child_cost,
                    )

        if best_path is not None and len(best_path) == lower_bound:
            assert apply_path(start, best_path).rank() == 0
            return GFNSearchResult(
                "gfn_beam",
                best_path,
                True,
                width,
                expanded,
                generated,
                perf_counter() - started,
                "solved_lower_bound",
                lower_bound,
            )
        if not dedup:
            break
        selected = heapq.nsmallest(width, dedup.values(), key=lambda item: item[0])
        frontier = [
            (state, path, neural_cost, last_action)
            for _, state, path, last_action, neural_cost in selected
        ]
        selected_ranks.update(state.rank() for state, _, _, _ in frontier)

    if best_path is not None:
        assert apply_path(start, best_path).rank() == 0
        return GFNSearchResult(
            "gfn_beam",
            best_path,
            len(best_path) == lower_bound,
            width,
            expanded,
            generated,
            perf_counter() - started,
            "solved" if incumbent_length is None else "improved_incumbent",
            lower_bound,
        )
    return GFNSearchResult(
        "gfn_beam",
        None,
        False,
        width,
        expanded,
        generated,
        perf_counter() - started,
        "search_exhausted",
        lower_bound,
    )


def gfn_adaptive_beam_search(
    start: JewelState,
    policy: GFNBackwardPolicy,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    widths: Sequence[int] = (1, 4, 16, 64, 256),
    branch_actions: int = 4,
    max_learned_steps: int = 20,
    heuristic_weight: float = 0.35,
    depth_weight: float = 0.0,
    incumbent_length: int | None = None,
    try_all_widths: bool = True,
) -> GFNSearchResult:
    """Try increasingly wide beams and keep the shortest verified candidate."""
    if not widths:
        raise ValueError("at least one beam width is required")
    started = perf_counter()
    lower_bound = _admissible_h(start, pdbs)
    total_expanded = 0
    total_generated = 0
    best: GFNSearchResult | None = None
    for width in widths:
        branch = 1 if width == 1 else branch_actions
        result = gfn_policy_beam_search(
            start,
            policy,
            ball,
            pdbs,
            width=int(width),
            branch_actions=branch,
            max_learned_steps=max_learned_steps,
            heuristic_weight=heuristic_weight,
            depth_weight=depth_weight,
            incumbent_length=incumbent_length,
        )
        total_expanded += result.expanded_states
        total_generated += result.generated_states
        if result.path is not None and (
            best is None or len(result.path) < len(best.path)
        ):
            best = result
        if best is not None and (
            len(best.path) == lower_bound or not try_all_widths
        ):
            break
    if best is not None:
        best.algorithm = "gfn_adaptive_beam"
        best.expanded_states = total_expanded
        best.generated_states = total_generated
        best.seconds = perf_counter() - started
        return best
    return GFNSearchResult(
        "gfn_adaptive_beam",
        None,
        False,
        int(widths[-1]),
        total_expanded,
        total_generated,
        perf_counter() - started,
        "all_widths_failed",
        lower_bound,
    )


def gfn_batched_best_first_search(
    start: JewelState,
    policy: GFNBackwardPolicy,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    batch_size: int = 256,
    node_limit: int = 100_000,
    time_limit: float = 30.0,
    max_depth: int = 32,
    branch_actions: int = 12,
    heuristic_weight: float = 0.35,
    depth_weight: float = 0.0,
    incumbent_length: int | None = None,
    probability_floor: float = 1e-12,
) -> GFNSearchResult:
    """Global approximate best-first search with batched neural scoring.

    Up to ``batch_size`` currently best nodes are expanded together.  This is
    substantially faster on a GPU than scalar best-first search, but nodes made
    by the first parent in a batch are considered in the next batch, so it is a
    batched approximation to strict one-at-a-time best-first order.
    """
    if batch_size < 1 or node_limit < 1:
        raise ValueError("batch_size and node_limit must be positive")
    started = perf_counter()
    lower_bound = _admissible_h(start, pdbs)
    direct = _direct_result(
        "gfn_best_first", start, ball, lower_bound, incumbent_length, started
    )
    if direct is not None:
        return direct

    # priority, tie, state, path, cumulative neural cost, last action
    queue: list[tuple[float, int, JewelState, list[int], float, int]] = [
        (heuristic_weight * lower_bound, 0, start, [], 0.0, -1)
    ]
    best_priority = {start.rank(): heuristic_weight * lower_bound}
    closed: set[int] = set()
    tie = 1
    expanded = 0
    generated = 0
    best_path: list[int] | None = None
    reason = "queue_exhausted"

    while queue:
        if perf_counter() - started >= time_limit:
            reason = "time_limit"
            break
        batch: list[tuple[float, int, JewelState, list[int], float, int]] = []
        while queue and len(batch) < batch_size and expanded + len(batch) < node_limit:
            entry = heapq.heappop(queue)
            state = entry[2]
            rank = state.rank()
            if rank in closed:
                continue
            if entry[0] > best_priority.get(rank, math.inf) + 1e-12:
                continue
            closed.add(rank)
            if len(entry[3]) >= max_depth:
                continue
            batch.append(entry)
        if not batch:
            if expanded >= node_limit:
                reason = "node_limit"
                break
            continue

        probabilities = np.asarray(
            policy.batch([entry[2] for entry in batch]), dtype=np.float64
        )
        log_probabilities = np.log(np.maximum(probabilities, probability_floor))
        expanded += len(batch)
        if expanded >= node_limit:
            reason = "node_limit"

        for parent_i, (_, _, state, path, neural_cost, last_action) in enumerate(batch):
            order = np.argsort(-probabilities[parent_i])
            used = 0
            for action_value in order:
                action = int(action_value)
                if last_action >= 0 and action == int(INVERSE_ACTION[last_action]):
                    continue
                used += 1
                if used > branch_actions:
                    break
                child = apply_action(state, action)
                generated += 1
                child_path = path + [action]
                suffix = ball.exact_path(child)
                if suffix is not None:
                    candidate = child_path + suffix
                    bound = _better_bound(best_path, incumbent_length)
                    if bound is None or len(candidate) < bound:
                        best_path = candidate
                    continue
                child_h = _admissible_h(child, pdbs)
                bound = _better_bound(best_path, incumbent_length)
                if bound is not None and len(child_path) + child_h >= bound:
                    continue
                rank = child.rank()
                if rank in closed:
                    continue
                child_cost = neural_cost - float(log_probabilities[parent_i, action])
                priority = (
                    child_cost
                    + heuristic_weight * child_h
                    + depth_weight * len(child_path)
                )
                if priority + 1e-12 >= best_priority.get(rank, math.inf):
                    continue
                best_priority[rank] = priority
                heapq.heappush(
                    queue,
                    (priority, tie, child, child_path, child_cost, action),
                )
                tie += 1

        if best_path is not None and len(best_path) == lower_bound:
            reason = "solved_lower_bound"
            break
        if expanded >= node_limit:
            break

    if best_path is not None:
        assert apply_path(start, best_path).rank() == 0
        return GFNSearchResult(
            "gfn_best_first",
            best_path,
            len(best_path) == lower_bound,
            batch_size,
            expanded,
            generated,
            perf_counter() - started,
            reason if reason == "solved_lower_bound" else "solved",
            lower_bound,
        )
    return GFNSearchResult(
        "gfn_best_first",
        None,
        False,
        batch_size,
        expanded,
        generated,
        perf_counter() - started,
        reason,
        lower_bound,
    )


def gfn_hybrid_search(
    start: JewelState,
    policy: GFNBackwardPolicy,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    widths: Sequence[int] = (1, 4, 16, 64),
    max_learned_steps: int = 20,
    beam_branch_actions: int = 4,
    heuristic_weight: float = 0.35,
    depth_weight: float = 0.0,
    incumbent_length: int | None = None,
    best_first_batch_size: int = 256,
    best_first_node_limit: int = 100_000,
    best_first_time_limit: float = 30.0,
    best_first_max_depth: int = 32,
    best_first_branch_actions: int = 12,
) -> GFNSearchResult:
    """Adaptive beam followed by a batched best-first fallback."""
    started = perf_counter()
    beam = gfn_adaptive_beam_search(
        start,
        policy,
        ball,
        pdbs,
        widths=widths,
        branch_actions=beam_branch_actions,
        max_learned_steps=max_learned_steps,
        heuristic_weight=heuristic_weight,
        depth_weight=depth_weight,
        incumbent_length=incumbent_length,
        try_all_widths=True,
    )
    if beam.path is not None and incumbent_length is None:
        beam.algorithm = "gfn_hybrid"
        return beam
    fallback_incumbent = incumbent_length
    if beam.path is not None:
        fallback_incumbent = min(
            len(beam.path),
            incumbent_length if incumbent_length is not None else len(beam.path),
        )
    best_first = gfn_batched_best_first_search(
        start,
        policy,
        ball,
        pdbs,
        batch_size=best_first_batch_size,
        node_limit=best_first_node_limit,
        time_limit=best_first_time_limit,
        max_depth=best_first_max_depth,
        branch_actions=best_first_branch_actions,
        heuristic_weight=heuristic_weight,
        depth_weight=depth_weight,
        incumbent_length=fallback_incumbent,
    )
    candidate = beam
    if best_first.path is not None and (
        candidate.path is None or len(best_first.path) < len(candidate.path)
    ):
        candidate = best_first
    candidate.algorithm = "gfn_hybrid"
    candidate.expanded_states = beam.expanded_states + best_first.expanded_states
    candidate.generated_states = beam.generated_states + best_first.generated_states
    candidate.seconds = perf_counter() - started
    if candidate.path is None:
        candidate.reason = "beam_and_best_first_failed"
    return candidate


__all__ = [
    "GFNSearchResult",
    "gfn_adaptive_beam_search",
    "gfn_batched_best_first_search",
    "gfn_hybrid_search",
    "gfn_policy_beam_search",
]
