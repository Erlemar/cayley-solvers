"""IDA* with exact-ball completion and pattern-database lower bounds."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .ball import ExactBall
from .pdb import EdgePatternDatabase, combined_heuristic
from .puzzle import INVERSE_ACTION, JewelState, apply_action, apply_path, permutation_parity


Policy = Callable[[JewelState], np.ndarray]


@dataclass(slots=True)
class ExactSearchResult:
    path: list[int] | None
    optimal: bool
    nodes: int
    thresholds: list[int]
    seconds: float
    reason: str


class _SearchLimit(RuntimeError):
    pass


def solve_ida_star(
    start: JewelState,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    policy: Policy | None = None,
    max_depth: int = 30,
    node_limit: int = 5_000_000,
    time_limit: float = 120.0,
) -> ExactSearchResult:
    started = perf_counter()
    direct = ball.exact_path(start)
    if direct is not None:
        return ExactSearchResult(direct, True, 0, [len(direct)], perf_counter() - started, "exact_ball")

    state_parity = permutation_parity(start.edge_perm)
    first_h = combined_heuristic(start, pdbs)
    threshold = first_h + ((state_parity - first_h) & 1)
    thresholds: list[int] = []
    nodes = 0
    found: list[int] | None = None

    try:
        while threshold <= max_depth:
            thresholds.append(threshold)
            transposition: dict[int, int] = {}

            def dfs(state: JewelState, g: int, last_action: int, path: list[int]) -> bool:
                nonlocal nodes, found
                nodes += 1
                if nodes > node_limit:
                    raise _SearchLimit("node_limit")
                if perf_counter() - started > time_limit:
                    raise _SearchLimit("time_limit")

                rank = state.rank()
                ball_distance = ball.distance_rank(rank)
                if ball_distance is not None:
                    if g + ball_distance <= threshold:
                        suffix = ball.exact_path(state)
                        assert suffix is not None
                        found = path + suffix
                        return True
                    return False

                h = combined_heuristic(state, pdbs)
                if g + h > threshold:
                    return False
                best_g = transposition.get(rank)
                if best_g is not None and best_g <= g:
                    return False
                transposition[rank] = g

                candidates: list[tuple[float, int, JewelState]] = []
                policy_scores = policy(state) if policy is not None else None
                for action in range(12):
                    if last_action >= 0 and action == int(INVERSE_ACTION[last_action]):
                        continue
                    child = apply_action(state, action)
                    child_h = combined_heuristic(child, pdbs)
                    policy_tiebreak = -float(policy_scores[action]) if policy_scores is not None else 0.0
                    candidates.append((child_h + 1e-3 * policy_tiebreak, action, child))
                candidates.sort(key=lambda item: item[0])
                for _, action, child in candidates:
                    path.append(action)
                    if dfs(child, g + 1, action, path):
                        return True
                    path.pop()
                return False

            if dfs(start, 0, -1, []):
                assert found is not None and apply_path(start, found).rank() == 0
                return ExactSearchResult(found, True, nodes, thresholds, perf_counter() - started, "solved")
            threshold += 2
    except _SearchLimit as exc:
        return ExactSearchResult(None, False, nodes, thresholds, perf_counter() - started, str(exc))

    return ExactSearchResult(None, False, nodes, thresholds, perf_counter() - started, "max_depth")

