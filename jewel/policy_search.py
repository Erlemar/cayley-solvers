"""Policy-guided exact IDA* and Policy-guided Heuristic Search.

``policy_ida_star`` is path-length optimal whenever it returns ``optimal=True``:
the neural policy changes successor order only, while pruning uses admissible
PDB/ring/exact-ball bounds.

``phs_search`` implements PHSh and PHS* priorities.  PHS optimizes search effort,
not path length, so a returned PHS path is valid but is only called optimal when
its length meets the root admissible lower bound.  ``phs_then_exact`` uses that
path as an incumbent and runs IDA* to improve or certify it.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal, Sequence

import numpy as np
import torch

from .ball import ExactBall
from .model import JewelTransformer
from .pdb import EdgePatternDatabase, combined_heuristic
from .puzzle import (
    INVERSE_ACTION,
    JewelState,
    apply_action,
    apply_path,
    permutation_parity,
)


Policy = Callable[[JewelState], np.ndarray]


def uniform_policy(_state: JewelState) -> np.ndarray:
    return np.full(12, 1.0 / 12.0, dtype=np.float64)


class TransformerPolicy:
    """Cached adapter for the existing structured Transformer's policy head."""

    def __init__(
        self,
        model: JewelTransformer,
        *,
        device: str | torch.device | None = None,
        cache: bool = True,
    ) -> None:
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.cache_enabled = cache
        self.cache: dict[int, np.ndarray] = {}

    @torch.no_grad()
    def __call__(self, state: JewelState) -> np.ndarray:
        rank = state.rank()
        if self.cache_enabled and rank in self.cache:
            return self.cache[rank]
        ep = torch.from_numpy(state.edge_perm.astype(np.int64))[None].to(self.device)
        eo = torch.from_numpy(state.edge_ori.astype(np.int64))[None].to(self.device)
        ro = torch.from_numpy(state.ring_ori.astype(np.int64))[None].to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(ep, eo, ro)["policy_logits"]
        probabilities = logits.float().softmax(dim=-1)[0].cpu().numpy()
        if self.cache_enabled:
            self.cache[rank] = probabilities
        return probabilities

    @torch.no_grad()
    def batch(self, states: Sequence[JewelState]) -> np.ndarray:
        """Score a collection of frontier states in one neural forward pass."""
        if not states:
            return np.empty((0, 12), dtype=np.float32)
        edge_perm = torch.from_numpy(
            np.stack([state.edge_perm for state in states]).astype(np.int64)
        ).to(self.device)
        edge_ori = torch.from_numpy(
            np.stack([state.edge_ori for state in states]).astype(np.int64)
        ).to(self.device)
        ring_ori = torch.from_numpy(
            np.stack([state.ring_ori for state in states]).astype(np.int64)
        ).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(edge_perm, edge_ori, ring_ori)["policy_logits"]
        probabilities = logits.float().softmax(dim=-1).cpu().numpy()
        if self.cache_enabled:
            for state, probs in zip(states, probabilities):
                self.cache[state.rank()] = probs
        return probabilities


@dataclass(slots=True)
class PolicySearchResult:
    algorithm: str
    path: list[int] | None
    optimal: bool
    expanded: int
    generated: int
    seconds: float
    reason: str
    lower_bound: int
    thresholds: list[int]
    incumbent_length: int | None = None


class _Limit(RuntimeError):
    pass


def _admissible_h(
    state: JewelState,
    pdbs: Sequence[EdgePatternDatabase],
    ball: ExactBall | None = None,
) -> int:
    if ball is not None:
        exact = ball.distance(state)
        if exact is not None:
            return exact
    raw = combined_heuristic(state, pdbs)
    parity = permutation_parity(state.edge_perm)
    return raw + ((parity - raw) & 1)


def _policy_probabilities(policy: Policy, state: JewelState, floor: float = 1e-12) -> np.ndarray:
    probabilities = np.asarray(policy(state), dtype=np.float64)
    if probabilities.shape != (12,) or not np.all(np.isfinite(probabilities)):
        raise ValueError("policy must return twelve finite scores/probabilities")
    probabilities = np.maximum(probabilities, floor)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("policy has no positive mass")
    return probabilities / total


def greedy_policy_path(
    start: JewelState,
    ball: ExactBall,
    *,
    policy: Policy,
    max_steps: int = 64,
) -> list[int] | None:
    """Fast deterministic policy rollout with cycle and exact-endgame guards."""
    state = start
    path: list[int] = []
    seen = {state.rank()}
    for _ in range(max_steps):
        suffix = ball.exact_path(state)
        if suffix is not None:
            result = path + suffix
            assert apply_path(start, result).rank() == 0
            return result
        action = int(np.argmax(_policy_probabilities(policy, state)))
        state = apply_action(state, action)
        path.append(action)
        rank = state.rank()
        if rank in seen:
            return None
        seen.add(rank)
    return None


def policy_ida_star(
    start: JewelState,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    policy: Policy = uniform_policy,
    ordering: Literal["policy", "heuristic", "hybrid"] = "policy",
    hybrid_policy_weight: float = 0.35,
    incumbent_path: Sequence[int] | None = None,
    max_depth: int = 30,
    node_limit: int = 5_000_000,
    time_limit: float = 120.0,
) -> PolicySearchResult:
    started = perf_counter()
    lower_bound = _admissible_h(start, pdbs, ball)
    incumbent = list(incumbent_path) if incumbent_path is not None else None
    if incumbent is not None and apply_path(start, incumbent).rank() != 0:
        raise ValueError("incumbent_path does not solve the supplied state")

    direct = ball.exact_path(start)
    if direct is not None:
        return PolicySearchResult(
            "policy_ida_star", direct, True, 0, 0, perf_counter() - started,
            "exact_ball", len(direct), [len(direct)], len(incumbent) if incumbent is not None else None,
        )

    state_parity = permutation_parity(start.edge_perm)
    threshold = lower_bound
    if (threshold & 1) != state_parity:
        threshold += 1
    thresholds: list[int] = []
    expanded = 0
    generated = 0
    found: list[int] | None = None
    incumbent_length = len(incumbent) if incumbent is not None else None

    try:
        while threshold <= max_depth:
            # All smaller reachable depths have already been exhausted.  A valid
            # incumbent at this threshold is therefore an optimality certificate.
            if incumbent is not None and threshold >= len(incumbent):
                return PolicySearchResult(
                    "policy_ida_star", incumbent, True, expanded, generated,
                    perf_counter() - started, "incumbent_proved", lower_bound,
                    thresholds, incumbent_length,
                )
            thresholds.append(threshold)
            transposition: dict[int, int] = {}
            path_ranks = {start.rank()}

            def dfs(state: JewelState, g: int, last_action: int, path: list[int]) -> bool:
                nonlocal expanded, generated, found
                if perf_counter() - started > time_limit:
                    raise _Limit("time_limit")
                expanded += 1
                if expanded > node_limit:
                    raise _Limit("node_limit")

                rank = state.rank()
                exact_distance = ball.distance_rank(rank)
                if exact_distance is not None:
                    if g + exact_distance <= threshold:
                        suffix = ball.exact_path(state)
                        assert suffix is not None
                        found = path + suffix
                        return True
                    return False

                h = _admissible_h(state, pdbs)
                if g + h > threshold:
                    return False
                if incumbent is not None and g + h >= len(incumbent):
                    return False
                best_g = transposition.get(rank)
                if best_g is not None and best_g <= g:
                    return False
                transposition[rank] = g

                probabilities = _policy_probabilities(policy, state)
                candidates: list[tuple[tuple[float, ...], int, JewelState]] = []
                for action in range(12):
                    if last_action >= 0 and action == int(INVERSE_ACTION[last_action]):
                        continue
                    child = apply_action(state, action)
                    child_rank = child.rank()
                    if child_rank in path_ranks:
                        continue
                    generated += 1
                    child_h = _admissible_h(child, pdbs, ball)
                    neg_log_probability = -math.log(max(float(probabilities[action]), 1e-300))
                    if ordering == "policy":
                        key = (neg_log_probability, float(child_h), float(action))
                    elif ordering == "heuristic":
                        key = (float(child_h), neg_log_probability, float(action))
                    elif ordering == "hybrid":
                        key = (
                            float(child_h) + hybrid_policy_weight * neg_log_probability,
                            float(child_h),
                            float(action),
                        )
                    else:
                        raise ValueError(f"unknown ordering: {ordering}")
                    candidates.append((key, action, child))
                candidates.sort(key=lambda item: item[0])
                for _, action, child in candidates:
                    path.append(action)
                    child_rank = child.rank()
                    path_ranks.add(child_rank)
                    if dfs(child, g + 1, action, path):
                        return True
                    path_ranks.remove(child_rank)
                    path.pop()
                return False

            if dfs(start, 0, -1, []):
                assert found is not None and apply_path(start, found).rank() == 0
                return PolicySearchResult(
                    "policy_ida_star", found, True, expanded, generated,
                    perf_counter() - started, "solved", lower_bound, thresholds,
                    incumbent_length,
                )
            threshold += 2
    except _Limit as exc:
        return PolicySearchResult(
            "policy_ida_star", incumbent, False, expanded, generated,
            perf_counter() - started, str(exc), lower_bound, thresholds,
            incumbent_length,
        )

    return PolicySearchResult(
        "policy_ida_star", incumbent, False, expanded, generated,
        perf_counter() - started, "max_depth", lower_bound, thresholds,
        incumbent_length,
    )


@dataclass(slots=True)
class _PHSNode:
    state: JewelState
    path: list[int]
    last_action: int
    log_path_probability: float
    log_phi_plus: float
    probabilities: np.ndarray | None = None


def _phs_log_priority(
    depth: int,
    h: int,
    log_path_probability: float,
    variant: Literal["phsh", "phsstar", "levin"],
) -> float:
    # Unit expansion loss includes the root, hence g = depth + 1.
    g = float(depth + 1)
    if variant == "levin":
        return math.log(g) - log_path_probability
    if variant == "phsh":
        return math.log(g + h) - log_path_probability
    if variant == "phsstar":
        return math.log(g + h) - (1.0 + h / g) * log_path_probability
    raise ValueError(f"unknown PHS variant: {variant}")


def phs_search(
    start: JewelState,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    policy: Policy = uniform_policy,
    variant: Literal["phsh", "phsstar", "levin"] = "phsh",
    exact_endgame: bool = True,
    deduplicate: bool = True,
    incumbent_path: Sequence[int] | None = None,
    max_depth: int = 40,
    node_limit: int = 500_000,
    time_limit: float = 120.0,
    probability_floor: float = 1e-12,
    policy_batch_size: int = 256,
) -> PolicySearchResult:
    started = perf_counter()
    lower_bound = _admissible_h(start, pdbs, ball)
    incumbent = list(incumbent_path) if incumbent_path is not None else None
    if incumbent is not None and apply_path(start, incumbent).rank() != 0:
        raise ValueError("incumbent_path does not solve the supplied state")
    incumbent_length = len(incumbent) if incumbent is not None else None

    direct = ball.exact_path(start) if exact_endgame else None
    if direct is not None:
        return PolicySearchResult(
            variant, direct, True, 0, 0, perf_counter() - started,
            "exact_ball", len(direct), [], incumbent_length,
        )

    root_priority = _phs_log_priority(0, lower_bound, 0.0, variant)
    root = _PHSNode(start, [], -1, 0.0, root_priority)
    queue: list[tuple[float, int, _PHSNode]] = [(root_priority, 0, root)]
    tie = 1
    expanded = 0
    generated = 0
    visited: set[int] = set()

    while queue:
        if perf_counter() - started > time_limit:
            reason = "time_limit"
            break
        _, _, node = heapq.heappop(queue)
        rank = node.state.rank()
        if deduplicate and rank in visited:
            continue
        if deduplicate:
            visited.add(rank)
        expanded += 1
        if expanded > node_limit:
            reason = "node_limit"
            break

        depth = len(node.path)
        if rank == 0:
            optimal = depth == lower_bound
            return PolicySearchResult(
                variant, node.path, optimal, expanded, generated,
                perf_counter() - started, "solved_lower_bound" if optimal else "solved",
                lower_bound, [], incumbent_length,
            )
        if exact_endgame:
            suffix = ball.exact_path(node.state)
            if suffix is not None:
                result_path = node.path + suffix
                if incumbent is None or len(result_path) < len(incumbent):
                    optimal = len(result_path) == lower_bound
                    return PolicySearchResult(
                        variant, result_path, optimal, expanded, generated,
                        perf_counter() - started,
                        "solved_exact_endgame_lower_bound" if optimal else "solved_exact_endgame",
                        lower_bound, [], incumbent_length,
                    )
        if depth >= max_depth:
            continue

        if node.probabilities is None:
            batch_method = getattr(policy, "batch", None)
            if callable(batch_method) and policy_batch_size > 1:
                # Scoring does not change a node's priority.  We can therefore
                # pre-score other unscored frontier nodes, put them back, and
                # retain exactly the same best-first expansion order.
                batch_entries: list[tuple[float, int, _PHSNode]] = []
                held_entries: list[tuple[float, int, _PHSNode]] = []
                while queue and len(batch_entries) < policy_batch_size - 1:
                    entry = heapq.heappop(queue)
                    candidate = entry[2]
                    if candidate.probabilities is None and len(candidate.path) < max_depth:
                        batch_entries.append(entry)
                    else:
                        held_entries.append(entry)
                score_nodes = [node, *(entry[2] for entry in batch_entries)]
                batch_probabilities = np.asarray(
                    batch_method([candidate.state for candidate in score_nodes]),
                    dtype=np.float64,
                )
                if batch_probabilities.shape != (len(score_nodes), 12):
                    raise ValueError("batched policy must return shape (batch, 12)")
                for candidate, probs in zip(score_nodes, batch_probabilities):
                    candidate.probabilities = probs
                for entry in (*held_entries, *batch_entries):
                    heapq.heappush(queue, entry)
            else:
                node.probabilities = _policy_probabilities(
                    policy, node.state, probability_floor
                )
        probabilities = _policy_probabilities(
            lambda _state: node.probabilities, node.state, probability_floor
        )
        allowed = np.ones(12, dtype=bool)
        if node.last_action >= 0:
            allowed[int(INVERSE_ACTION[node.last_action])] = False
        conditioned = np.where(allowed, probabilities, 0.0)
        conditioned_sum = float(conditioned.sum())
        if conditioned_sum <= 0.0:
            continue
        conditioned /= conditioned_sum

        for action in range(12):
            if not allowed[action]:
                continue
            child = apply_action(node.state, action)
            generated += 1
            child_depth = depth + 1
            child_h = _admissible_h(child, pdbs, ball if exact_endgame else None)
            if incumbent is not None and child_depth + child_h >= len(incumbent):
                continue
            child_log_probability = node.log_path_probability + math.log(
                max(float(conditioned[action]), probability_floor)
            )
            raw_priority = _phs_log_priority(
                child_depth, child_h, child_log_probability, variant
            )
            monotone_priority = max(node.log_phi_plus, raw_priority)
            child_node = _PHSNode(
                child,
                node.path + [action],
                action,
                child_log_probability,
                monotone_priority,
                None,
            )
            heapq.heappush(queue, (monotone_priority, tie, child_node))
            tie += 1
    else:
        reason = "queue_exhausted"

    return PolicySearchResult(
        variant, incumbent, False, expanded, generated, perf_counter() - started,
        reason, lower_bound, [], incumbent_length,
    )


@dataclass(slots=True)
class PHSExactResult:
    phs: PolicySearchResult
    exact: PolicySearchResult

    @property
    def path(self) -> list[int] | None:
        return self.exact.path

    @property
    def optimal(self) -> bool:
        return self.exact.optimal


def phs_then_exact(
    start: JewelState,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    policy: Policy,
    phs_variant: Literal["phsh", "phsstar", "levin"] = "phsh",
    ida_ordering: Literal["policy", "heuristic", "hybrid"] = "heuristic",
    phs_node_limit: int = 100_000,
    phs_time_limit: float = 30.0,
    phs_policy_batch_size: int = 256,
    ida_node_limit: int = 5_000_000,
    ida_time_limit: float = 120.0,
    max_depth: int = 30,
    greedy_incumbent: bool = True,
    greedy_max_steps: int = 64,
) -> PHSExactResult:
    initial_incumbent = (
        greedy_policy_path(
            start, ball, policy=policy, max_steps=greedy_max_steps
        )
        if greedy_incumbent
        else None
    )
    phs_result = phs_search(
        start,
        ball,
        pdbs,
        policy=policy,
        variant=phs_variant,
        incumbent_path=initial_incumbent,
        node_limit=phs_node_limit,
        time_limit=phs_time_limit,
        policy_batch_size=phs_policy_batch_size,
        max_depth=max_depth,
    )
    exact_result = policy_ida_star(
        start,
        ball,
        pdbs,
        policy=policy,
        ordering=ida_ordering,
        incumbent_path=phs_result.path,
        node_limit=ida_node_limit,
        time_limit=ida_time_limit,
        max_depth=max_depth,
    )
    return PHSExactResult(phs_result, exact_result)
