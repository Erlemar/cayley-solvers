"""Model-guided beam search over exact 6x6x6 cluster macro actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from cube666.macro_data import CLUSTER_COUNT, CLUSTER_SIZE, MacroActionTable, cluster_costs
from cube666.macro_policy import (
    MacroEffectPolicyValueNet,
    MacroPolicyValueNet,
    score_macro_effect_candidates,
)


@dataclass(frozen=True)
class MacroBeamResult:
    solved: bool
    actions: tuple[int, ...]
    macro_steps: int
    expanded_states: int
    generated_states: int
    final_state: np.ndarray
    final_predicted_value: float
    final_exact_cost: int


@dataclass(frozen=True)
class RankedMacroAction:
    """One model-proposed action ranked by predicted downstream move cost."""

    action: int
    state: np.ndarray
    predicted_value: float
    exact_cost: int
    primitive_moves: int
    policy_nll: float
    rank: float


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def _policy_topk(
    model: MacroPolicyValueNet | MacroEffectPolicyValueNet,
    states: np.ndarray,
    last_actions: np.ndarray,
    inverse_indices: np.ndarray,
    action_effects: np.ndarray,
    extra_action_indices: np.ndarray,
    *,
    branch_width: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    action_batches: list[np.ndarray] = []
    log_probability_batches: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        stop = min(start + batch_size, len(states))
        state_tensor = torch.from_numpy(states[start:stop]).to(device)
        previous = last_actions[start:stop]
        has_previous = previous >= 0
        forbidden_actions = np.full(len(previous), -1, dtype=np.int32)
        forbidden_actions[has_previous] = inverse_indices[previous[has_previous]]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            policy_output, _, _ = model(state_tensor)
        if isinstance(model, MacroEffectPolicyValueNet):
            best_scores = torch.empty((len(previous), 0), device=device)
            best_actions = torch.empty(
                (len(previous), 0), dtype=torch.long, device=device
            )
            effect_logits = policy_output
            effect_tensor = torch.from_numpy(action_effects).to(device)
            action_chunk_size = 1024
            for action_start in range(0, model.config.action_count, action_chunk_size):
                action_stop = min(
                    action_start + action_chunk_size,
                    model.config.action_count,
                )
                candidates = effect_tensor[action_start:action_stop][None].expand(
                    len(previous), -1, -1, -1
                )
                scores = score_macro_effect_candidates(effect_logits, candidates)
                forbidden_in_chunk = (
                    (forbidden_actions >= action_start)
                    & (forbidden_actions < action_stop)
                )
                if np.any(forbidden_in_chunk):
                    rows = torch.from_numpy(np.flatnonzero(forbidden_in_chunk)).to(device)
                    columns = torch.from_numpy(
                        forbidden_actions[forbidden_in_chunk] - action_start
                    ).to(device)
                    scores[rows, columns] = -torch.inf
                action_ids = torch.arange(action_start, action_stop, device=device)
                action_ids = action_ids[None].expand(len(previous), -1)
                merged_scores = torch.cat((best_scores, scores), dim=1)
                merged_actions = torch.cat((best_actions, action_ids), dim=1)
                keep = min(branch_width, merged_scores.shape[1])
                best_scores, positions = merged_scores.topk(keep, dim=1)
                best_actions = merged_actions.gather(1, positions)
            values, actions = best_scores, best_actions
        else:
            logits = policy_output
            model_forbidden = has_previous & (forbidden_actions < logits.shape[1])
            if np.any(model_forbidden):
                rows = torch.from_numpy(np.flatnonzero(model_forbidden)).to(device)
                forbidden = torch.from_numpy(forbidden_actions[model_forbidden]).to(device)
                logits[rows, forbidden] = -torch.inf
            log_probabilities = torch.log_softmax(logits.float(), dim=1)
            values, actions = log_probabilities.topk(branch_width, dim=1)
        batch_actions = actions.cpu().numpy().astype(np.int32, copy=False)
        batch_values = values.cpu().numpy()
        if extra_action_indices.size:
            extra_actions = np.broadcast_to(
                extra_action_indices,
                (len(previous), len(extra_action_indices)),
            ).copy()
            extra_values = np.broadcast_to(
                batch_values[:, -1:],
                extra_actions.shape,
            ).copy()
            extra_values[extra_actions == forbidden_actions[:, None]] = -1.0e9
            batch_actions = np.concatenate((batch_actions, extra_actions), axis=1)
            batch_values = np.concatenate((batch_values, extra_values), axis=1)
        action_batches.append(batch_actions)
        log_probability_batches.append(batch_values)
    return np.concatenate(action_batches), np.concatenate(log_probability_batches)


@torch.no_grad()
def _predict_values(
    model: MacroPolicyValueNet | MacroEffectPolicyValueNet,
    states: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        state_tensor = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            total_values, cluster_values = model.value_only(state_tensor)
        total_values = total_values.float() * 72.0
        cluster_values = cluster_values.float().sum(dim=1) * 12.0
        batches.append(((total_values + cluster_values) * 0.5).cpu().numpy())
    return np.concatenate(batches)


@torch.no_grad()
def rank_one_step_macro_actions(
    initial_state: np.ndarray,
    model: MacroPolicyValueNet | MacroEffectPolicyValueNet,
    table: MacroActionTable,
    *,
    proposal_width: int = 256,
    limit: int = 16,
    policy_nll_weight: float = 0.02,
    move_cost_weight: float = 1.0,
    model_batch_size: int = 1024,
) -> tuple[RankedMacroAction, ...]:
    """Return distinct one-step children in the same order used by learned beam."""

    state = np.asarray(initial_state, dtype=np.uint8)
    if state.shape != (CLUSTER_COUNT, CLUSTER_SIZE):
        raise ValueError("initial_state must have shape (6, 24)")
    if proposal_width <= 0 or proposal_width > int(model.config.action_count):
        raise ValueError("invalid proposal_width")
    if limit <= 0 or policy_nll_weight < 0 or move_cost_weight < 0:
        raise ValueError("invalid one-step ranking options")
    device = _model_device(model)
    model.eval()
    actions, policy_scores = _policy_topk(
        model,
        state[None],
        np.asarray([-1], dtype=np.int32),
        table.inverse_indices,
        table.effects,
        np.empty(0, dtype=np.int32),
        branch_width=proposal_width,
        batch_size=model_batch_size,
        device=device,
    )
    flat_actions = actions[0]
    children = np.take_along_axis(
        np.broadcast_to(state, (len(flat_actions),) + state.shape),
        table.effects[flat_actions],
        axis=-1,
    )
    predicted_values = _predict_values(
        model,
        children,
        batch_size=model_batch_size,
        device=device,
    )
    exact_costs = cluster_costs(children).sum(axis=1)
    primitive_moves = np.fromiter(
        (len(table.paths[int(action)]) for action in flat_actions),
        dtype=np.int32,
        count=len(flat_actions),
    )
    policy_nll = -policy_scores[0].astype(np.float32, copy=False)
    ranks = (
        predicted_values
        + move_cost_weight * primitive_moves
        + policy_nll_weight * policy_nll
    )
    order = np.lexsort((policy_nll, primitive_moves, ranks))
    ranked: list[RankedMacroAction] = []
    seen: set[bytes] = set()
    for position in order:
        key = children[position].tobytes()
        if key in seen:
            continue
        seen.add(key)
        ranked.append(
            RankedMacroAction(
                action=int(flat_actions[position]),
                state=children[position].copy(),
                predicted_value=float(predicted_values[position]),
                exact_cost=int(exact_costs[position]),
                primitive_moves=int(primitive_moves[position]),
                policy_nll=float(policy_nll[position]),
                rank=float(ranks[position]),
            )
        )
        if len(ranked) == limit:
            break
    return tuple(ranked)


def learned_macro_beam_search(
    initial_state: np.ndarray,
    model: MacroPolicyValueNet | MacroEffectPolicyValueNet,
    table: MacroActionTable,
    *,
    beam_width: int = 512,
    branch_width: int = 16,
    max_steps: int = 72,
    policy_nll_weight: float = 0.02,
    exact_cost_weight: float | None = None,
    move_cost_weight: float = 0.0,
    model_batch_size: int = 1024,
    extra_action_indices: Sequence[int] = (),
    initial_solution_actions: Sequence[int] = (),
    exact_close_actions: Mapping[bytes, int] | None = None,
    fallback_exact_cost: bool = False,
    stop_on_first_solution: bool = True,
    ranking_value_predictor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MacroBeamResult:
    """Search to identity using policy proposals and learned or exact-cost ranking.

    When ``exact_cost_weight`` is supplied, beam rank is accumulated primitive
    macro length plus that weight times the exact unrestricted 3-cycle residual.
    Otherwise ``move_cost_weight`` trades predicted remaining macro steps against
    the exact primitive length already accumulated.  The model still chooses
    which actions are expanded.
    """

    state = np.asarray(initial_state, dtype=np.uint8)
    if state.shape != (CLUSTER_COUNT, CLUSTER_SIZE):
        raise ValueError("initial_state must have shape (6, 24)")
    if beam_width <= 0 or branch_width <= 0 or max_steps < 0:
        raise ValueError("invalid beam search dimensions")
    if exact_cost_weight is not None and exact_cost_weight <= 0:
        raise ValueError("exact_cost_weight must be positive when supplied")
    if move_cost_weight < 0:
        raise ValueError("move_cost_weight must be non-negative")
    if exact_cost_weight is not None and move_cost_weight != 0:
        raise ValueError("move_cost_weight is only valid with learned-value ranking")
    model_action_count = int(model.config.action_count)
    if branch_width > model_action_count:
        raise ValueError("branch_width exceeds the model action count")
    extras = np.asarray(extra_action_indices, dtype=np.int32)
    if extras.ndim != 1:
        raise ValueError("extra_action_indices must be one-dimensional")
    if len(set(int(action) for action in extras)) != len(extras):
        raise ValueError("extra_action_indices contains duplicates")
    if np.any(extras < 0) or np.any(extras >= table.action_count):
        raise ValueError("extra actions must be valid table action indices")
    initial_solution = tuple(int(action) for action in initial_solution_actions)
    if any(action < 0 or action >= table.action_count for action in initial_solution):
        raise ValueError("initial solution actions must be valid table action indices")
    identity = np.broadcast_to(
        np.arange(CLUSTER_SIZE, dtype=np.uint8),
        (CLUSTER_COUNT, CLUSTER_SIZE),
    )
    if np.array_equal(state, identity):
        return MacroBeamResult(True, (), 0, 0, 0, state.copy(), 0.0, 0)

    device = _model_device(model)
    model.eval()
    states = state[None].copy()
    paths: list[tuple[int, ...]] = [()]
    cumulative_nll = np.zeros(1, dtype=np.float32)
    cumulative_move_cost = np.zeros(1, dtype=np.int32)
    last_actions = np.full(1, -1, dtype=np.int32)
    best_move_cost = {state.tobytes(): 0}
    expanded_states = 0
    generated_states = 0
    final_predicted_value = float("inf")
    best_solution: tuple[int, float, tuple[int, ...]] | None = None
    if initial_solution:
        replay = state.copy()
        for action in initial_solution:
            replay = np.take_along_axis(replay, table.effects[action], axis=-1)
        if not np.array_equal(replay, identity):
            raise ValueError("initial_solution_actions do not solve initial_state")
        best_solution = (
            sum(len(table.paths[action]) for action in initial_solution),
            float("inf"),
            initial_solution,
        )

    for step in range(1, max_steps + 1):
        proposed, log_probabilities = _policy_topk(
            model,
            states,
            last_actions,
            table.inverse_indices,
            table.effects,
            extras,
            branch_width=branch_width,
            batch_size=model_batch_size,
            device=device,
        )
        parent_count = len(states)
        actions_per_parent = proposed.shape[1]
        expanded_states += parent_count
        flat_actions = proposed.reshape(-1)
        parent_indices = np.repeat(np.arange(parent_count), actions_per_parent)
        parent_states = states[parent_indices]
        effects = table.effects[flat_actions]
        children = np.take_along_axis(parent_states, effects, axis=-1)
        child_nll = cumulative_nll[parent_indices] - log_probabilities.reshape(-1)
        child_move_cost = cumulative_move_cost[parent_indices] + np.fromiter(
            (len(table.paths[int(action)]) for action in flat_actions),
            dtype=np.int32,
            count=len(flat_actions),
        )
        generated_states += len(children)

        solved_mask = np.all(children == identity, axis=(1, 2))
        if np.any(solved_mask):
            solved_positions = np.flatnonzero(solved_mask)
            best = min(
                solved_positions,
                key=lambda index: (
                    int(child_move_cost[index]),
                    float(child_nll[index]),
                ),
            )
            parent = int(parent_indices[best])
            actions = paths[parent] + (int(flat_actions[best]),)
            candidate_solution = (
                int(child_move_cost[best]),
                float(child_nll[best]),
                actions,
            )
            if best_solution is None or candidate_solution < best_solution:
                best_solution = candidate_solution
            if stop_on_first_solution:
                return MacroBeamResult(
                    True,
                    actions,
                    step,
                    expanded_states,
                    generated_states,
                    children[best].copy(),
                    0.0,
                    0,
                )

        if exact_close_actions:
            inverse_children = np.empty_like(children)
            inverse_positions = np.broadcast_to(
                np.arange(CLUSTER_SIZE, dtype=np.uint8), children.shape
            )
            np.put_along_axis(
                inverse_children,
                children,
                inverse_positions,
                axis=-1,
            )
            for candidate, inverse_child in enumerate(inverse_children):
                if solved_mask[candidate]:
                    continue
                close_action = exact_close_actions.get(inverse_child.tobytes())
                if close_action is None:
                    continue
                close_action = int(close_action)
                close_cost = int(child_move_cost[candidate]) + len(
                    table.paths[close_action]
                )
                parent = int(parent_indices[candidate])
                actions = (
                    paths[parent]
                    + (int(flat_actions[candidate]), close_action)
                )
                candidate_solution = (
                    close_cost,
                    float(child_nll[candidate]),
                    actions,
                )
                if best_solution is None or candidate_solution < best_solution:
                    best_solution = candidate_solution

        # At the final search layer an already verified solution is sufficient;
        # scoring and sorting the children cannot improve it because no child
        # will be expanded again.  This matters for wide exact-close passes,
        # where the model can shortlist thousands of possible first macros.
        if step == max_steps and best_solution is not None:
            break

        if exact_cost_weight is None:
            predicted_values = (
                _predict_values(
                    model,
                    children,
                    batch_size=model_batch_size,
                    device=device,
                )
                if ranking_value_predictor is None
                else np.asarray(ranking_value_predictor(children), dtype=np.float32)
            )
            if predicted_values.shape != (len(children),):
                raise ValueError("ranking_value_predictor returned the wrong shape")
            ranks = (
                predicted_values
                + move_cost_weight * child_move_cost
                + policy_nll_weight * child_nll
            )
        else:
            exact_child_costs = cluster_costs(children).sum(axis=1)
            ranks = (
                child_move_cost
                + exact_cost_weight * exact_child_costs
                + policy_nll_weight * child_nll
            )
            predicted_values = exact_child_costs.astype(np.float32, copy=False)
        order = np.lexsort((child_nll, child_move_cost, ranks))
        next_states: list[np.ndarray] = []
        next_paths: list[tuple[int, ...]] = []
        next_nll: list[float] = []
        next_move_cost: list[int] = []
        next_last_actions: list[int] = []
        next_values: list[float] = []
        for candidate in order:
            if solved_mask[candidate]:
                continue
            move_cost = int(child_move_cost[candidate])
            if best_solution is not None and move_cost >= best_solution[0]:
                continue
            key = children[candidate].tobytes()
            previous_cost = best_move_cost.get(key)
            if previous_cost is not None and previous_cost <= move_cost:
                continue
            best_move_cost[key] = move_cost
            parent = int(parent_indices[candidate])
            action = int(flat_actions[candidate])
            next_states.append(children[candidate])
            next_paths.append(paths[parent] + (action,))
            next_nll.append(float(child_nll[candidate]))
            next_move_cost.append(move_cost)
            next_last_actions.append(action)
            next_values.append(float(predicted_values[candidate]))
            if len(next_states) == beam_width:
                break
        if not next_states:
            break
        states = np.stack(next_states)
        paths = next_paths
        cumulative_nll = np.asarray(next_nll, dtype=np.float32)
        cumulative_move_cost = np.asarray(next_move_cost, dtype=np.int32)
        last_actions = np.asarray(next_last_actions, dtype=np.int32)
        final_predicted_value = next_values[0]

    if best_solution is not None:
        return MacroBeamResult(
            True,
            best_solution[2],
            len(best_solution[2]),
            expanded_states,
            generated_states,
            identity.copy(),
            0.0,
            0,
        )

    exact_costs = cluster_costs(states).sum(axis=1)
    predicted_values = (
        _predict_values(
            model,
            states,
            batch_size=model_batch_size,
            device=device,
        )
        if ranking_value_predictor is None
        else np.asarray(ranking_value_predictor(states), dtype=np.float32)
    )
    if predicted_values.shape != (len(states),):
        raise ValueError("ranking_value_predictor returned the wrong shape")
    if fallback_exact_cost:
        best_index = min(
            range(len(states)),
            key=lambda index: (
                int(exact_costs[index]),
                int(cumulative_move_cost[index]),
                float(predicted_values[index]),
                float(cumulative_nll[index]),
            ),
        )
    elif exact_cost_weight is not None:
        best_index = min(
            range(len(states)),
            key=lambda index: (
                int(cumulative_move_cost[index])
                + exact_cost_weight * int(exact_costs[index]),
                int(cumulative_move_cost[index]),
                float(cumulative_nll[index]),
            ),
        )
    else:
        learned_ranks = (
            predicted_values
            + move_cost_weight * cumulative_move_cost
            + policy_nll_weight * cumulative_nll
        )
        best_index = min(
            range(len(states)),
            key=lambda index: (
                float(learned_ranks[index]),
                int(cumulative_move_cost[index]),
                float(cumulative_nll[index]),
            ),
        )
    return MacroBeamResult(
        False,
        paths[best_index],
        len(paths[best_index]),
        expanded_states,
        generated_states,
        states[best_index].copy(),
        float(predicted_values[best_index]),
        int(exact_costs[best_index]),
    )
