"""Greedy Tetraminx inference with confidence-triggered local beam recovery.

The regular solver applies a global beam at every depth.  This module instead
uses the Q head as a policy and pays for a bounded lookahead only when the top
two actions are close or the greedy action would revisit a state.  The same
lookahead exposes per-root-action costs and is therefore also the teacher used
by ``70_build_frontier_regret.py``.

All scores are distances/costs: lower is better.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from cayley.khoruzhii_search import (
    KhoruzhiiSearchConfig,
    _q_predict,
    _qv_predict,
    _state_hash,
)


@dataclass
class GreedyRecoveryConfig:
    max_steps: int = 45
    confidence_gap: float = 0.25
    recovery_beam: int = 256
    recovery_depth: int = 4
    no_backtrack: bool = True
    recovery_to_goal: bool = False
    max_handoffs: int = 1


def q_policy_scores(solver, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw Q and the inference-adjusted policy score for ``states``.

    This deliberately mirrors KhoruzhiiSolver's Q/V-consistency adjustment so
    confidence is measured in the same score space used by the submitted beam.
    """
    if not solver.use_q_function:
        raise ValueError("greedy policy requires an all-actions Q model")
    if solver.qv_consistency_lambda != 0.0 and getattr(solver.model, "has_value_head", False):
        base = getattr(solver.model, "_orig_mod", solver.model)
        old_return_value = bool(getattr(base, "return_value", False))
        base.return_value = True
        try:
            q, value = _qv_predict(
                solver.model, states, solver.internal_batch_size, solver.n_actions)
        finally:
            base.return_value = old_return_value
        score = q.float()
        expected = (value.float() - 1.0).unsqueeze(1)
        score = score + solver.qv_consistency_lambda * (score - expected).abs()
    else:
        q = _q_predict(solver.model, states, solver.internal_batch_size, solver.n_actions)
        score = q.float()
    if solver.action_cost is not None:
        score = score + (solver.action_cost.float() - 1.0).unsqueeze(0)
    return q.float(), score


def _group_min(values: torch.Tensor, groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    out = torch.full((n_groups,), float("inf"), dtype=torch.float32, device=values.device)
    if values.numel():
        out.scatter_reduce_(0, groups, values.float(), reduce="amin", include_self=True)
    return out


def lookahead_root_costs(
    solver,
    state: torch.Tensor,
    *,
    beam_width: int,
    depth: int,
    goal_depth_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    forbidden_root_actions: torch.Tensor | None = None,
    forbidden_root_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Estimate solution cost for every first action with a bounded pooled beam.

    The first layer contains one child per legal action.  Later beam children
    carry their root-action label.  At the horizon, each action receives the
    best ``depth_so_far + predicted_remaining_distance`` among its surviving
    descendants.  Exact endgame hits instead receive ``depth_so_far + exact_d``.

    Actions eliminated by the pooled beam remain ``+inf``.  Dataset builders
    should treat these as censored labels, not as proven losing actions.
    """
    if state.dim() != 1:
        raise ValueError(f"state must have shape (S,), got {tuple(state.shape)}")
    if beam_width < 1 or depth < 1:
        raise ValueError("beam_width and depth must be positive")

    device = state.device
    n_actions = solver.n_actions
    parent = state.unsqueeze(0)
    raw_q, adjusted = q_policy_scores(solver, parent)
    one_step_costs = 1.0 + adjusted[0]
    root_costs = (one_step_costs.clone() if depth == 1 else
                  torch.full_like(one_step_costs, float("inf")))

    actions = torch.arange(n_actions, dtype=torch.int64, device=device)
    children = solver._apply_move(parent.expand(n_actions, -1), actions)
    legal = torch.ones(n_actions, dtype=torch.bool, device=device)
    if forbidden_root_actions is not None and forbidden_root_actions.numel():
        legal[forbidden_root_actions.to(device=device, dtype=torch.int64)] = False
    if forbidden_root_states is not None and forbidden_root_states.numel():
        child_hash = _state_hash(children, solver.hash_vec, solver.internal_batch_size)
        bad_hash = _state_hash(
            forbidden_root_states.to(device=device, dtype=solver.state_dtype),
            solver.hash_vec,
            solver.internal_batch_size,
        )
        legal &= ~torch.isin(child_hash, bad_hash)
    root_costs = root_costs.masked_fill(~legal, float("inf"))

    actions = actions[legal]
    states = children[legal]
    roots = actions.clone()
    frontier_value = raw_q[0, legal]
    terminal_roots: set[int] = set()

    if states.numel() == 0:
        return root_costs, {"surviving_roots": 0, "terminal_roots": 0, "expanded_layers": 0}

    if goal_depth_fn is not None:
        exact_d = goal_depth_fn(states).to(torch.int64)
        hit = exact_d >= 0
        if bool(hit.any()):
            exact_cost = 1.0 + exact_d[hit].float()
            root_costs[roots[hit]] = torch.minimum(root_costs[roots[hit]], exact_cost)
            terminal_roots.update(int(x) for x in roots[hit].detach().cpu().tolist())

    expanded_layers = 1
    # Prevent the lookahead from immediately returning to its root.  This is a
    # local analogue of history_depth=1 and removes a large number of order-3
    # oscillations without injecting hidden path history into the policy target.
    excluded = _state_hash(parent, solver.hash_vec, solver.internal_batch_size)
    for layer in range(2, depth + 1):
        states, frontier_value, _, parent_idx = solver._do_greedy_step(
            states, excluded, beam_width)
        if states.numel() == 0:
            break
        roots = roots.index_select(0, parent_idx)
        expanded_layers = layer
        if goal_depth_fn is not None:
            exact_d = goal_depth_fn(states).to(torch.int64)
            hit = exact_d >= 0
            if bool(hit.any()):
                hit_roots = roots[hit]
                exact_cost = float(layer) + exact_d[hit].float()
                exact_by_root = _group_min(exact_cost, hit_roots, n_actions)
                root_costs = torch.minimum(root_costs, exact_by_root)
                terminal_roots.update(int(x) for x in hit_roots.detach().cpu().tolist())

    if states.numel():
        horizon_cost = float(expanded_layers) + frontier_value.float()
        by_root = _group_min(horizon_cost, roots, n_actions)
        # An exact terminal estimate is authoritative.  For other roots the
        # deeper lookahead replaces the one-step estimate when that root survived.
        survived = torch.isfinite(by_root)
        if terminal_roots:
            terminal_mask = torch.zeros(n_actions, dtype=torch.bool, device=device)
            terminal_mask[list(terminal_roots)] = True
            survived &= ~terminal_mask
        root_costs[survived] = by_root[survived]

    return root_costs, {
        "surviving_roots": int(torch.unique(roots).numel()) if roots.numel() else 0,
        "terminal_roots": len(terminal_roots),
        "expanded_layers": expanded_layers,
    }


def greedy_recovery_solve(
    initial_state,
    solver,
    cfg: GreedyRecoveryConfig,
    *,
    policy_solver=None,
    goal_depth_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    stats: dict | None = None,
) -> tuple[bool, int, list[str]]:
    """Roll out the Q policy and call bounded lookahead only at uncertain steps.

    The returned path stops at the first state accepted by ``goal_depth_fn``.
    The caller can append an exact endgame descent, exactly as ``30_solve.py``
    already does for ordinary beam search.
    """
    policy_solver = policy_solver or solver
    if policy_solver.device != solver.device or policy_solver.n_actions != solver.n_actions:
        raise ValueError("policy and recovery solvers must share device and action space")
    state = torch.as_tensor(
        initial_state, dtype=solver.state_dtype, device=solver.device).clone()
    if goal_depth_fn is None:
        goal_depth_fn = lambda s: torch.where(
            (s == solver.V0).all(dim=1),
            torch.zeros(s.size(0), dtype=torch.int64, device=s.device),
            torch.full((s.size(0),), -1, dtype=torch.int64, device=s.device),
        )
    if int(goal_depth_fn(state.unsqueeze(0))[0]) >= 0:
        return True, 0, []

    inverse = torch.empty(solver.n_gen, dtype=torch.int64, device=solver.device)
    name_to_idx = {name: i for i, name in enumerate(solver.move_names)}
    for i, name in enumerate(solver.move_names):
        inv_name = name[1:] if name.startswith("-") else "-" + name
        inverse[i] = name_to_idx[inv_name]

    visited = [state.clone()]
    path_idx: list[int] = []
    recovery_calls = 0
    recovery_changes = 0
    handoff_calls = 0
    handoff_success = 0
    gaps: list[float] = []

    for _ in range(cfg.max_steps):
        _, score = q_policy_scores(policy_solver, state.unsqueeze(0))
        score = score[0]
        children = policy_solver._apply_move(
            state.unsqueeze(0).expand(policy_solver.n_actions, -1),
            torch.arange(policy_solver.n_actions, device=policy_solver.device),
        )
        legal = torch.ones(policy_solver.n_actions, dtype=torch.bool, device=policy_solver.device)
        if cfg.no_backtrack and path_idx:
            legal[inverse[path_idx[-1]]] = False
        visited_t = torch.stack(visited)
        child_hash = _state_hash(
            children, policy_solver.hash_vec, policy_solver.internal_batch_size)
        visited_hash = _state_hash(
            visited_t, policy_solver.hash_vec, policy_solver.internal_batch_size)
        legal &= ~torch.isin(child_hash, visited_hash)
        order = torch.argsort(score.masked_fill(~legal, float("inf")))
        finite = order[torch.isfinite(score[order]) & legal[order]]
        if finite.numel() == 0:
            break
        greedy_action = int(finite[0])
        gap = (float(score[finite[1]] - score[finite[0]])
               if finite.numel() > 1 else float("inf"))
        gaps.append(gap)
        action = greedy_action

        if gap < cfg.confidence_gap and cfg.recovery_beam > 1:
            if cfg.recovery_to_goal and handoff_calls < cfg.max_handoffs:
                handoff_calls += 1

                def handoff_goal(s):
                    d = goal_depth_fn(s)
                    hit = d >= 0
                    if not bool(hit.any()):
                        return hit
                    return hit & (d == d[hit].min())

                search_cfg = KhoruzhiiSearchConfig(
                    beam_width=cfg.recovery_beam,
                    num_steps=max(1, cfg.max_steps - len(path_idx)),
                    internal_batch_size=solver.internal_batch_size,
                    history_depth=1,
                )
                found, _, suffix = solver.solve(
                    state.detach().cpu().numpy(), search_cfg,
                    goal_check_fn=handoff_goal)
                if found:
                    handoff_success = 1
                    names = [solver.move_names[a] for a in path_idx] + suffix
                    if stats is not None:
                        stats.update(
                            recovery_calls=recovery_calls,
                            recovery_changes=recovery_changes,
                            handoff_calls=handoff_calls,
                            handoff_success=handoff_success,
                            mean_gap=sum(gaps) / max(len(gaps), 1),
                            greedy_steps=len(path_idx),
                        )
                    return True, len(names), names
            elif not cfg.recovery_to_goal and cfg.recovery_depth > 1:
                forbidden_actions = (inverse[path_idx[-1]].view(1)
                                     if cfg.no_backtrack and path_idx else None)
                root_costs, _ = lookahead_root_costs(
                    solver,
                    state,
                    beam_width=cfg.recovery_beam,
                    depth=cfg.recovery_depth,
                    goal_depth_fn=goal_depth_fn,
                    forbidden_root_actions=forbidden_actions,
                    forbidden_root_states=visited_t,
                )
                ranked = torch.argsort(root_costs)
                ranked = ranked[torch.isfinite(root_costs[ranked])]
                if ranked.numel():
                    action = int(ranked[0])
                    recovery_calls += 1
                    recovery_changes += int(action != greedy_action)

        path_idx.append(action)
        state = children[action]
        visited.append(state.clone())
        if int(goal_depth_fn(state.unsqueeze(0))[0]) >= 0:
            names = [solver.move_names[a] for a in path_idx]
            if stats is not None:
                stats.update(
                    recovery_calls=recovery_calls,
                    recovery_changes=recovery_changes,
                    handoff_calls=handoff_calls,
                    handoff_success=handoff_success,
                    mean_gap=sum(gaps) / max(len(gaps), 1),
                    greedy_steps=len(path_idx),
                )
            return True, len(names), names

    if stats is not None:
        stats.update(
            recovery_calls=recovery_calls,
            recovery_changes=recovery_changes,
            handoff_calls=handoff_calls,
            handoff_success=handoff_success,
            mean_gap=sum(gaps) / max(len(gaps), 1),
            greedy_steps=len(path_idx),
        )
    return False, 0, []
