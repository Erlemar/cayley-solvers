"""Cost-aware beam gate for autoregressively proposed verified short macros."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macro_autoregressive import load_macro_autoregressive_checkpoint  # noqa: E402
from cube666.macro_factorized_value import load_macro_factorized_value_checkpoint  # noqa: E402
from cube666.macro_policy import build_macro_policy_model  # noqa: E402


@dataclass
class TrieNode:
    children: dict[int, int] = field(default_factory=dict)
    actions: list[int] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path)
    parser.add_argument("--value-checkpoint-2", type=Path)
    parser.add_argument("--value-secondary-weight", type=float, default=0.5)
    parser.add_argument("--eval-puzzles", type=int, default=4)
    parser.add_argument("--pids", default="")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--branch", type=int, default=256)
    parser.add_argument("--decode-beam", type=int, default=512)
    parser.add_argument("--maximum-steps", type=int, default=30)
    parser.add_argument("--policy-weight", type=float, default=0.02)
    parser.add_argument("--oracle-teacher-actions", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def build_trie(tokens: np.ndarray, eos: int) -> list[TrieNode]:
    nodes = [TrieNode()]
    for action, row in enumerate(tokens):
        if row[0] < 0:
            continue
        node = 0
        for raw_token in row:
            token = int(raw_token)
            if token < 0:
                break
            if token == eos:
                nodes[node].actions.append(action)
                break
            child = nodes[node].children.get(token)
            if child is None:
                child = len(nodes)
                nodes[node].children[token] = child
                nodes.append(TrieNode())
            node = child
    return nodes


def trie_tensors(
    trie: list[TrieNode], action_count: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    transitions = np.full((len(trie), action_count), -1, dtype=np.int32)
    terminals = np.full(len(trie), -1, dtype=np.int32)
    for node_id, node in enumerate(trie):
        for token, child in node.children.items():
            transitions[node_id, token] = child
        if node.actions:
            terminals[node_id] = node.actions[0]
    return torch.from_numpy(transitions).to(device), torch.from_numpy(terminals).to(device)


@torch.inference_mode()
def decode(
    model: torch.nn.Module,
    state: np.ndarray,
    trie: list[TrieNode],
    *,
    beam_width: int,
    topk: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    state_tensor = torch.from_numpy(state[None]).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state_hidden = model.encode(state_tensor)
    hidden = model.decoder_initial(state_hidden)
    node_ids = [0]
    last_tokens = torch.full((1,), model.config.bos_token, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)
    completed: list[tuple[float, int]] = []
    for _ in range(model.config.maximum_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, next_hidden = model.decode_step(last_tokens, hidden)
        logp = torch.log_softmax(logits.float(), dim=1)
        candidate_scores: list[torch.Tensor] = []
        candidate_nodes: list[int] = []
        candidate_tokens: list[int] = []
        candidate_parents: list[int] = []
        for parent, node_id in enumerate(node_ids):
            node = trie[node_id]
            if node.actions:
                terminal_score = float(
                    scores[parent] + logp[parent, model.config.eos_token]
                )
                completed.extend((terminal_score, action) for action in node.actions)
            for token, child in node.children.items():
                candidate_scores.append(scores[parent] + logp[parent, token])
                candidate_nodes.append(child)
                candidate_tokens.append(token)
                candidate_parents.append(parent)
        if not candidate_scores:
            break
        stacked = torch.stack(candidate_scores)
        keep = min(beam_width, len(candidate_scores))
        kept_scores, positions = stacked.topk(keep)
        selected = positions.cpu().numpy()
        parents = torch.as_tensor(
            [candidate_parents[position] for position in selected], device=device
        )
        hidden = next_hidden[:, parents]
        scores = kept_scores
        node_ids = [candidate_nodes[position] for position in selected]
        last_tokens = torch.as_tensor(
            [candidate_tokens[position] for position in selected], device=device
        )
    completed.sort(reverse=True)
    actions: list[int] = []
    action_scores: list[float] = []
    seen: set[int] = set()
    for score, action in completed:
        if action in seen:
            continue
        seen.add(action)
        actions.append(action)
        action_scores.append(score)
        if len(actions) == topk:
            break
    return np.asarray(actions, dtype=np.int32), np.asarray(action_scores, dtype=np.float32)


@torch.inference_mode()
def decode_batch(
    model: torch.nn.Module,
    states: np.ndarray,
    transitions: torch.Tensor,
    terminals: torch.Tensor,
    *,
    beam_width: int,
    topk: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size = len(states)
    state_tensor = torch.from_numpy(states).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state_hidden = model.encode(state_tensor)
    initial = model.decoder_initial(state_hidden)
    hidden = initial[:, :, None].expand(-1, -1, beam_width, -1).clone()
    nodes = torch.full(
        (batch_size, beam_width), -1, dtype=torch.long, device=device
    )
    nodes[:, 0] = 0
    scores = torch.full((batch_size, beam_width), -torch.inf, device=device)
    scores[:, 0] = 0
    last_tokens = torch.full(
        (batch_size, beam_width),
        model.config.bos_token,
        dtype=torch.long,
        device=device,
    )
    completed_scores = torch.full(
        (batch_size, topk), -torch.inf, device=device
    )
    completed_actions = torch.full(
        (batch_size, topk), -1, dtype=torch.long, device=device
    )
    for _ in range(model.config.maximum_tokens):
        flat_tokens = last_tokens.flatten()
        flat_hidden = hidden.reshape(
            model.config.decoder_layers, batch_size * beam_width, model.config.hidden_dim
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, next_hidden_flat = model.decode_step(flat_tokens, flat_hidden)
        logp = torch.log_softmax(logits.float(), dim=1).reshape(
            batch_size, beam_width, -1
        )
        safe_nodes = nodes.clamp_min(0)
        terminal_actions = terminals[safe_nodes]
        terminal_scores = scores + logp[:, :, model.config.eos_token]
        terminal_scores = terminal_scores.masked_fill(
            (nodes < 0) | (terminal_actions < 0), -torch.inf
        )
        merged_scores = torch.cat((completed_scores, terminal_scores), dim=1)
        merged_actions = torch.cat((completed_actions, terminal_actions), dim=1)
        completed_scores, completed_positions = merged_scores.topk(topk, dim=1)
        completed_actions = merged_actions.gather(1, completed_positions)

        child_nodes = transitions[safe_nodes]
        candidate_scores = scores[:, :, None] + logp[:, :, : model.config.primitive_action_count]
        candidate_scores = candidate_scores.masked_fill(
            (nodes[:, :, None] < 0) | (child_nodes < 0), -torch.inf
        )
        scores, positions = candidate_scores.flatten(1).topk(beam_width, dim=1)
        parent_positions = torch.div(
            positions, model.config.primitive_action_count, rounding_mode="floor"
        )
        last_tokens = torch.remainder(
            positions, model.config.primitive_action_count
        )
        nodes = child_nodes.flatten(1).gather(1, positions)
        next_hidden = next_hidden_flat.reshape(
            model.config.decoder_layers,
            batch_size,
            beam_width,
            model.config.hidden_dim,
        )
        hidden = next_hidden.gather(
            2,
            parent_positions[None, :, :, None].expand(
                model.config.decoder_layers,
                -1,
                -1,
                model.config.hidden_dim,
            ),
        )
    return (
        completed_actions.cpu().numpy().astype(np.int32, copy=False),
        completed_scores.cpu().numpy().astype(np.float32, copy=False),
    )


@torch.inference_mode()
def predict_values(
    model: object,
    states: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if isinstance(model, tuple):
        primary, secondary, secondary_weight = model
        return (
            predict_values(primary, states, batch_size, device)
            + float(secondary_weight)
            * predict_values(secondary, states, batch_size, device)
        ) / (1.0 + float(secondary_weight))
    values: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        state_tensor = torch.from_numpy(states[start : start + batch_size]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if hasattr(model, "value_only"):
                total, clusters = model.value_only(state_tensor)
                moves = 0.5 * (
                    total.float() * 72.0
                    + clusters.float().sum(dim=1) * 12.0
                )
            elif hasattr(model.config, "total_scale"):
                total, _ = model(state_tensor)
                moves = total.float() * model.config.total_scale
            else:
                hidden = model.encode(state_tensor)
                scaled = model.value_head(hidden).squeeze(1)
                moves = scaled.float() * model.config.value_scale
        values.append(moves.clamp_min(0).cpu().numpy())
    return np.concatenate(values)


def reconstruct_teacher_paths(
    states: np.ndarray,
    labels: np.ndarray,
    counts: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    effects: np.ndarray,
    costs: np.ndarray,
    maximum_cost: int,
) -> dict[int, tuple[int, ...]]:
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24))
    by_pid_state = {
        (int(groups[index]), states[index].tobytes()): index
        for index in range(len(states))
    }
    memo: dict[int, tuple[int, ...] | None] = {}

    def solve(index: int) -> tuple[int, ...] | None:
        cached = memo.get(index, ...)
        if cached is not ...:
            return cached
        memo[index] = None
        best: tuple[int, tuple[int, ...]] | None = None
        for raw_action in labels[index, : int(counts[index])]:
            action = int(raw_action)
            if action < 0 or costs[action] > maximum_cost:
                continue
            child = np.take_along_axis(states[index], effects[action], axis=-1)
            if np.array_equal(child, identity):
                suffix: tuple[int, ...] = ()
            else:
                child_index = by_pid_state.get((int(groups[index]), child.tobytes()))
                if child_index is None or values[child_index] >= values[index] - 1e-5:
                    continue
                child_path = solve(child_index)
                if child_path is None:
                    continue
                suffix = child_path
            path = (action,) + suffix
            candidate = (sum(int(costs[item]) for item in path), path)
            if best is None or candidate < best:
                best = candidate
        memo[index] = None if best is None else best[1]
        return memo[index]

    output: dict[int, tuple[int, ...]] = {}
    for index in range(len(states)):
        path = solve(index)
        if path is not None:
            output[index] = path
    return output


@torch.inference_mode()
def teacher_step_ranks(
    model: torch.nn.Module,
    value_model: object,
    initial: np.ndarray,
    teacher_path: tuple[int, ...],
    trie: list[TrieNode],
    effects: np.ndarray,
    costs: np.ndarray,
    *,
    branch: int,
    decode_beam: int,
    inference_batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[int | None]]:
    state = initial.copy()
    ranks: list[int] = []
    proposal_ranks: list[int | None] = []
    for teacher_action in teacher_path:
        actions, policy_scores = decode(
            model,
            state,
            trie,
            beam_width=decode_beam,
            topk=branch,
            device=device,
        )
        teacher_matches = np.flatnonzero(actions == teacher_action)
        proposal_ranks.append(
            None if not len(teacher_matches) else 1 + int(teacher_matches[0])
        )
        if not len(teacher_matches):
            actions = np.concatenate((actions, np.asarray([teacher_action], dtype=np.int32)))
        children = np.take_along_axis(
            np.broadcast_to(state, (len(actions),) + state.shape),
            effects[actions],
            axis=-1,
        )
        predicted = predict_values(value_model, children, inference_batch_size, device)
        scores = costs[actions] + predicted
        teacher_position = int(np.flatnonzero(actions == teacher_action)[0])
        ranks.append(1 + int(np.sum(scores < scores[teacher_position])))
        state = np.take_along_axis(state, effects[teacher_action], axis=-1)
    return ranks, proposal_ranks


@torch.inference_mode()
def beam_search(
    model: torch.nn.Module,
    value_model: object,
    initial: np.ndarray,
    trie: list[TrieNode],
    trie_transitions: torch.Tensor,
    trie_terminals: torch.Tensor,
    effects: np.ndarray,
    costs: np.ndarray,
    *,
    beam_width: int,
    branch: int,
    decode_beam: int,
    maximum_steps: int,
    policy_weight: float,
    inference_batch_size: int,
    device: torch.device,
    oracle_actions: dict[bytes, tuple[int, ...]] | None,
    tracked_path: tuple[int, ...] | None = None,
) -> dict[str, object]:
    identity = np.broadcast_to(np.arange(24, dtype=np.uint8), (6, 24))
    states = initial[None].copy()
    paths: list[tuple[int, ...]] = [()]
    path_costs = np.zeros(1, dtype=np.int32)
    path_nll = np.zeros(1, dtype=np.float32)
    best_g = {initial.tobytes(): 0}
    best_solution: tuple[int, float, tuple[int, ...]] | None = None
    tracked_states: list[np.ndarray] = []
    tracked_diagnostics: list[dict[str, object]] = []
    if tracked_path is not None:
        tracked_state = initial.copy()
        for action in tracked_path:
            tracked_state = np.take_along_axis(
                tracked_state, effects[int(action)], axis=-1
            )
            tracked_states.append(tracked_state.copy())
    generated = 0
    started = time.perf_counter()
    for step in range(1, maximum_steps + 1):
        proposed, policy_scores = decode_batch(
            model,
            states,
            trie_transitions,
            trie_terminals,
            beam_width=decode_beam,
            topk=branch,
            device=device,
        )
        if np.any(proposed < 0):
            raise RuntimeError("batched trie decoder returned an incomplete proposal row")
        if oracle_actions is not None:
            proposed_parts: list[np.ndarray] = []
            score_parts: list[np.ndarray] = []
            parent_parts: list[np.ndarray] = []
            for parent, state in enumerate(states):
                actions = proposed[parent]
                scores_row = policy_scores[parent]
                extras = [
                    action
                    for action in oracle_actions.get(state.tobytes(), ())
                    if action not in set(actions.tolist())
                ]
                if extras:
                    actions = np.concatenate((actions, np.asarray(extras, dtype=np.int32)))
                    scores_row = np.concatenate(
                        (scores_row, np.zeros(len(extras), dtype=np.float32))
                    )
                proposed_parts.append(actions)
                score_parts.append(scores_row)
                parent_parts.append(np.full(len(actions), parent, dtype=np.int32))
            flat_actions = np.concatenate(proposed_parts)
            flat_policy_scores = np.concatenate(score_parts)
            parents = np.concatenate(parent_parts)
        else:
            flat_actions = proposed.reshape(-1)
            flat_policy_scores = policy_scores.reshape(-1)
            parents = np.repeat(np.arange(len(states)), branch)
        children = np.take_along_axis(states[parents], effects[flat_actions], axis=-1)
        child_costs = path_costs[parents] + costs[flat_actions]
        child_nll = path_nll[parents] - flat_policy_scores
        generated += len(children)
        solved = np.all(children == identity, axis=(1, 2))
        for position in np.flatnonzero(solved):
            parent = int(parents[position])
            candidate = (
                int(child_costs[position]),
                float(child_nll[position]),
                paths[parent] + (int(flat_actions[position]),),
            )
            if best_solution is None or candidate < best_solution:
                best_solution = candidate
        predicted = predict_values(value_model, children, inference_batch_size, device)
        ranks = child_costs + predicted + policy_weight * child_nll
        order = np.lexsort((child_nll, child_costs, ranks))
        next_states: list[np.ndarray] = []
        next_paths: list[tuple[int, ...]] = []
        next_costs: list[int] = []
        next_nll: list[float] = []
        for position in order:
            if solved[position]:
                continue
            cost = int(child_costs[position])
            if best_solution is not None and cost >= best_solution[0]:
                continue
            key = children[position].tobytes()
            old = best_g.get(key)
            if old is not None and old <= cost:
                continue
            best_g[key] = cost
            parent = int(parents[position])
            next_states.append(children[position])
            next_paths.append(paths[parent] + (int(flat_actions[position]),))
            next_costs.append(cost)
            next_nll.append(float(child_nll[position]))
            if len(next_states) == beam_width:
                break
        if step <= len(tracked_states):
            target_key = tracked_states[step - 1].tobytes()
            candidate_positions = [
                int(position)
                for position in order
                if children[int(position)].tobytes() == target_key
            ]
            selected_keys = {state.tobytes() for state in next_states}
            tracked_diagnostics.append(
                {
                    "available": bool(candidate_positions),
                    "global_rank": None
                    if not candidate_positions
                    else 1
                    + int(
                        np.flatnonzero(order == candidate_positions[0])[0]
                    ),
                    "retained": target_key in selected_keys,
                    "step": step,
                }
            )
        if step == 1 or step % 5 == 0:
            print(
                json.dumps(
                    {
                        "beam": len(next_states),
                        "best_solution": None if best_solution is None else best_solution[0],
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "generated": generated,
                        "minimum_rank": None if not len(order) else round(float(ranks[order[0]]), 3),
                        "step": step,
                    }
                ),
                flush=True,
            )
        if not next_states:
            break
        states = np.stack(next_states)
        paths = next_paths
        path_costs = np.asarray(next_costs, dtype=np.int32)
        path_nll = np.asarray(next_nll, dtype=np.float32)
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "generated": generated,
        "macro_steps": None if best_solution is None else len(best_solution[2]),
        "path": None if best_solution is None else list(best_solution[2]),
        "primitive_cost": None if best_solution is None else best_solution[0],
        "solved": best_solution is not None,
        "tracked_path_diagnostics": tracked_diagnostics,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    model, checkpoint = load_macro_autoregressive_checkpoint(args.checkpoint, device)
    value_model: torch.nn.Module = model
    if args.value_checkpoint is not None:
        value_payload = torch.load(
            args.value_checkpoint, map_location="cpu", weights_only=False
        )
        if "factorized_value_config" in value_payload:
            value_model, _ = load_macro_factorized_value_checkpoint(
                args.value_checkpoint, device
            )
        else:
            value_model = build_macro_policy_model(value_payload["model_config"]).to(device)
            value_model.load_state_dict(value_payload["model_state_dict"])
            value_model.eval()
    if args.value_checkpoint_2 is not None:
        secondary_model, _ = load_macro_factorized_value_checkpoint(
            args.value_checkpoint_2, device
        )
        value_model = (value_model, secondary_model, args.value_secondary_weight)
    action_tokens = np.asarray(checkpoint["action_tokens"], dtype=np.int16)
    costs = np.load(args.teacher_dir / "action_costs.npy", allow_pickle=False).astype(np.int32)
    effects = np.load(args.teacher_dir / "action_effects.npy", allow_pickle=False).astype(np.uint8)
    trie = build_trie(action_tokens, model.config.eos_token)
    trie_transitions, trie_terminals = trie_tensors(
        trie, model.config.primitive_action_count, device
    )
    with np.load(args.teacher_dir / "teacher.npz", allow_pickle=False) as teacher:
        states = teacher["states"].astype(np.uint8, copy=False)
        labels = teacher["teacher_actions"].astype(np.int32, copy=False)
        counts = teacher["teacher_action_counts"].astype(np.int16, copy=False)
        values = teacher["search_value_targets"].astype(np.float32, copy=False)
    groups = np.load(args.teacher_dir / "source_state_ids.npy", allow_pickle=False)
    teacher_paths = reconstruct_teacher_paths(
        states, labels, counts, values, groups, effects, costs, 14
    )
    heldout = [index for index in teacher_paths if int(groups[index]) % 10 == 0]
    best_by_pid: dict[int, int] = {}
    for index in heldout:
        pid = int(groups[index])
        current = best_by_pid.get(pid)
        if current is None or (values[index], -index) > (values[current], -current):
            best_by_pid[pid] = index
    requested_pids = {
        int(value) for value in args.pids.split(",") if value.strip()
    }
    if requested_pids:
        best_by_pid = {
            pid: index for pid, index in best_by_pid.items() if pid in requested_pids
        }
    selected = sorted(
        best_by_pid.values(),
        key=lambda index: (-values[index], int(groups[index]), index),
    )[: args.eval_puzzles]
    rows: list[dict[str, object]] = []
    for index in selected:
        pid = int(groups[index])
        oracle = None
        if args.oracle_teacher_actions:
            oracle = {}
            for row in range(len(states)):
                if int(groups[row]) != pid:
                    continue
                valid = tuple(
                    int(action)
                    for action in labels[row, : int(counts[row])]
                    if action >= 0 and costs[int(action)] <= 14
                )
                if valid:
                    oracle[states[row].tobytes()] = valid
        teacher_path = teacher_paths[index]
        local_teacher_ranks, teacher_proposal_ranks = teacher_step_ranks(
            model,
            value_model,
            states[index],
            teacher_path,
            trie,
            effects,
            costs,
            branch=args.branch,
            decode_beam=args.decode_beam,
            inference_batch_size=args.inference_batch_size,
            device=device,
        )
        result = beam_search(
            model,
            value_model,
            states[index],
            trie,
            trie_transitions,
            trie_terminals,
            effects,
            costs,
            beam_width=args.beam,
            branch=args.branch,
            decode_beam=args.decode_beam,
            maximum_steps=args.maximum_steps,
            policy_weight=args.policy_weight,
            inference_batch_size=args.inference_batch_size,
            device=device,
            oracle_actions=oracle,
            tracked_path=teacher_path if args.oracle_teacher_actions else None,
        )
        teacher_cost = sum(int(costs[action]) for action in teacher_path)
        result.update(
            {
                "pid": int(groups[index]),
                "row": index,
                "teacher_macro_steps": len(teacher_path),
                "teacher_primitive_cost": teacher_cost,
                "teacher_value": float(values[index]),
                "teacher_step_ranks": local_teacher_ranks,
                "teacher_step_rank_maximum": max(local_teacher_ranks),
                "teacher_proposal_ranks": teacher_proposal_ranks,
                "teacher_proposal_recall": float(
                    np.mean([rank is not None for rank in teacher_proposal_ranks])
                ),
            }
        )
        rows.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    report = {
        "beam": args.beam,
        "branch": args.branch,
        "decode_beam": args.decode_beam,
        "oracle_teacher_actions": args.oracle_teacher_actions,
        "value_checkpoint": None if args.value_checkpoint is None else str(args.value_checkpoint),
        "value_checkpoint_2": None if args.value_checkpoint_2 is None else str(args.value_checkpoint_2),
        "value_secondary_weight": args.value_secondary_weight,
        "rows": rows,
        "solved": sum(bool(row["solved"]) for row in rows),
        "teacher_short_path_rows": len(teacher_paths),
        "trie_nodes": len(trie),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
