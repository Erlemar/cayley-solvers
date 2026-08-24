"""Adaptive neural policy search with exact endgame completion."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import numpy as np
import torch

from .ball import ExactBall
from .model import JewelTransformer
from .pdb import EdgePatternDatabase, combined_heuristic
from .puzzle import INVERSE_ACTION, JewelState, apply_action, apply_path


@dataclass(slots=True)
class InteractiveResult:
    path: list[int] | None
    width: int
    expanded_states: int
    generated_states: int
    seconds: float
    reason: str


@torch.no_grad()
def predict_batch(model: JewelTransformer, states: Sequence[JewelState], device: torch.device) -> dict[str, np.ndarray]:
    ep = torch.from_numpy(np.stack([s.edge_perm for s in states])).to(device)
    eo = torch.from_numpy(np.stack([s.edge_ori for s in states])).to(device)
    ro = torch.from_numpy(np.stack([s.ring_ori for s in states])).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(ep, eo, ro)
    return {key: value.float().cpu().numpy() for key, value in outputs.items()}


def policy_beam_search(
    start: JewelState,
    model: JewelTransformer,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    width: int,
    branch_actions: int = 4,
    max_learned_steps: int = 18,
    heuristic_weight: float = 0.35,
    regret_weight: float = 0.15,
    incumbent_length: int | None = None,
    device: torch.device | None = None,
) -> InteractiveResult:
    started = perf_counter()
    direct = ball.exact_path(start)
    if direct is not None:
        if incumbent_length is None or len(direct) < incumbent_length:
            return InteractiveResult(direct, width, 0, 0, perf_counter() - started, "exact_ball")
        return InteractiveResult(None, width, 0, 0, perf_counter() - started, "incumbent_is_exact")
    if device is None:
        device = next(model.parameters()).device
    # state, path, cumulative neural cost, last action
    frontier: list[tuple[JewelState, list[int], float, int]] = [(start, [], 0.0, -1)]
    expanded = 0
    generated = 0
    best_path: list[int] | None = None

    for _step in range(max_learned_steps):
        states = [item[0] for item in frontier]
        predictions = predict_batch(model, states, device)
        logits = predictions["policy_logits"]
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        regrets = predictions["regret"]
        expanded += len(frontier)
        dedup: dict[int, tuple[float, JewelState, list[int], int]] = {}

        for parent_i, (state, path, cost, last_action) in enumerate(frontier):
            order = np.argsort(-log_probs[parent_i])
            used = 0
            for action_np in order:
                action = int(action_np)
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
                    candidate_path = child_path + suffix
                    bound = len(best_path) if best_path is not None else incumbent_length
                    if bound is None or len(candidate_path) < bound:
                        best_path = candidate_path
                    continue
                child_h = combined_heuristic(child, pdbs)
                bound = len(best_path) if best_path is not None else incumbent_length
                if bound is not None and len(child_path) + child_h >= bound:
                    continue
                child_cost = (
                    cost
                    - float(log_probs[parent_i, action])
                    + regret_weight * max(float(regrets[parent_i, action]), 0.0)
                )
                priority = child_cost + heuristic_weight * child_h
                rank = child.rank()
                previous = dedup.get(rank)
                if previous is None or priority < previous[0]:
                    dedup[rank] = (priority, child, child_path, action)

        if best_path is not None and (
            incumbent_length is None or len(best_path) == combined_heuristic(start, pdbs)
        ):
            assert apply_path(start, best_path).rank() == 0
            return InteractiveResult(best_path, width, expanded, generated, perf_counter() - started, "solved")
        if not dedup:
            break
        selected = sorted(dedup.values(), key=lambda item: item[0])[:width]
        frontier = [(state, path, priority - heuristic_weight * combined_heuristic(state, pdbs), last) for priority, state, path, last in selected]

    if best_path is not None:
        assert apply_path(start, best_path).rank() == 0
        return InteractiveResult(best_path, width, expanded, generated, perf_counter() - started, "improved_incumbent")
    return InteractiveResult(None, width, expanded, generated, perf_counter() - started, "search_exhausted")


def interactive_solve(
    start: JewelState,
    model: JewelTransformer,
    ball: ExactBall,
    pdbs: Sequence[EdgePatternDatabase],
    *,
    widths: Sequence[int] = (1, 4, 16, 64, 256),
    max_learned_steps: int = 18,
    device: torch.device | None = None,
    try_all_widths: bool = True,
    heuristic_weight: float = 0.35,
    regret_weight: float = 0.15,
    incumbent_length: int | None = None,
) -> InteractiveResult:
    total_expanded = 0
    total_generated = 0
    started = perf_counter()
    best: InteractiveResult | None = None
    lower_bound = combined_heuristic(start, pdbs)
    for width in widths:
        branch = 1 if width == 1 else min(4, max(2, width))
        result = policy_beam_search(
            start,
            model,
            ball,
            pdbs,
            width=width,
            branch_actions=branch,
            max_learned_steps=max_learned_steps,
            device=device,
            heuristic_weight=heuristic_weight,
            regret_weight=regret_weight,
            incumbent_length=incumbent_length,
        )
        total_expanded += result.expanded_states
        total_generated += result.generated_states
        if result.path is not None:
            if best is None or len(result.path) < len(best.path):
                best = result
            if len(result.path) == lower_bound or not try_all_widths:
                break
    if best is not None:
        best.expanded_states = total_expanded
        best.generated_states = total_generated
        best.seconds = perf_counter() - started
        return best
    return InteractiveResult(None, int(widths[-1]), total_expanded, total_generated, perf_counter() - started, "all_widths_failed")
