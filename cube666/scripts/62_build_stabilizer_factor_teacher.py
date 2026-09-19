"""Build exact stage teachers with dense Schreier--Sims factorization.

Unlike random-walk depth labels, every trajectory produced here solves an
arbitrary reachable projected state.  The dense verified macro vocabulary is
given to SymPy as the original generator set.  Only a handful of derived strong
generators then need one-time expansion; subsequent arbitrary-state factors are
fast and are replay-checked before becoming labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, permutation_parity  # noqa: E402
from cube666.macro_data import load_macro_action_library  # noqa: E402
from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--puzzle-info",
        type=Path,
        default=PROJECT / "cayley-py-666-cube" / "puzzle_info.json",
    )
    parser.add_argument(
        "--action-library",
        type=Path,
        default=(
            PROJECT
            / "cube666"
            / "training"
            / "kmc_macro_teacher_allruns_sym8_v3"
            / "action_library.json"
        ),
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        default=PROJECT / "cube666" / "artifacts" / "stabilizer_ladder_v1.json",
    )
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--samples", type=int, default=1_000)
    parser.add_argument("--terminal-walk-depth", type=int, default=200)
    parser.add_argument("--seed", type=int, default=62666)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def even_random_permutation(rng: np.random.Generator) -> np.ndarray:
    permutation = rng.permutation(24).astype(np.uint8)
    if permutation_parity(permutation.tolist()):
        permutation[0], permutation[1] = permutation[1], permutation[0]
    return permutation


def eligible_actions(
    active_cluster: int,
    locked_clusters: tuple[int, ...],
    active_masks: np.ndarray,
    effects: np.ndarray,
    paths: tuple[tuple[str, ...], ...],
) -> tuple[np.ndarray, tuple[tuple[str, ...], ...], np.ndarray]:
    mask = active_masks[:, active_cluster].copy()
    if locked_clusters:
        mask &= ~active_masks[:, locked_clusters].any(axis=1)
    shortest: dict[bytes, tuple[np.ndarray, tuple[str, ...], int]] = {}
    for source_action in np.flatnonzero(mask):
        effect = effects[source_action, active_cluster]
        path = paths[int(source_action)]
        candidate = (effect.copy(), path, int(source_action))
        old = shortest.get(effect.tobytes())
        if old is None or (len(path), path, int(source_action)) < (
            len(old[1]),
            old[1],
            old[2],
        ):
            shortest[effect.tobytes()] = candidate
    ordered = sorted(shortest.values(), key=lambda item: (len(item[1]), item[1], item[2]))
    return (
        np.asarray([item[0] for item in ordered], dtype=np.uint8),
        tuple(item[1] for item in ordered),
        np.asarray([item[2] for item in ordered], dtype=np.int32),
    )


def inverse_indices(effects: np.ndarray) -> np.ndarray:
    by_effect = {effect.tobytes(): index for index, effect in enumerate(effects)}
    result = np.empty(len(effects), dtype=np.int32)
    for index, effect in enumerate(effects):
        inverse = np.empty_like(effect)
        inverse[effect] = np.arange(24, dtype=np.uint8)
        result[index] = by_effect[inverse.tobytes()]
    return result


def terminal_random_walk(
    rng: np.random.Generator,
    effects: np.ndarray,
    inverse: np.ndarray,
    depth: int,
) -> np.ndarray:
    state = np.arange(24, dtype=np.uint8)
    last = -1
    for _ in range(depth):
        action = int(rng.integers(len(effects)))
        if last >= 0 and action == int(inverse[last]):
            action = (action + 1) % len(effects)
        state = state[effects[action]]
        last = action
    return state


def permutation_key(permutation: Permutation) -> bytes:
    return bytes(permutation.array_form)


def main() -> None:
    args = parse_args()
    if min(args.samples, args.terminal_walk_depth) <= 0:
        raise ValueError("sample and walk settings must be positive")
    rng = np.random.default_rng(args.seed)
    puzzle = Cube666Puzzle.load(args.puzzle_info)
    decomposition = build_decomposition(puzzle.generators)
    _, table = load_macro_action_library(
        args.action_library,
        puzzle.generators,
        decomposition,
    )
    ladder = json.loads(args.ladder.read_text(encoding="utf-8"))
    stage = ladder["stages"][args.stage]
    identity = np.arange(24, dtype=np.uint8)
    active_masks = np.any(table.effects != identity[None, None, :], axis=2)
    effects, paths, source_actions = eligible_actions(
        int(stage["active_cluster"]),
        tuple(int(value) for value in stage["locked_clusters"]),
        active_masks,
        table.effects,
        table.paths,
    )
    inverse = inverse_indices(effects)
    effect_to_action = {effect.tobytes(): index for index, effect in enumerate(effects)}

    started = time.perf_counter()
    group = PermutationGroup(
        [Permutation(effect.tolist(), size=24) for effect in effects]
    )
    group_order = int(group.order())
    expected_order = int(stage["projected_group_order"])
    if group_order != expected_order:
        raise ValueError(f"projected group order {group_order} != expected {expected_order}")
    bsgs_seconds = time.perf_counter() - started

    strong_words: dict[bytes, tuple[int, ...]] = {}
    derived_rows = []
    for strong_index, strong in enumerate(group.strong_gens):
        key = permutation_key(strong)
        direct = effect_to_action.get(key)
        if direct is not None:
            word = (direct,)
        else:
            expansion_started = time.perf_counter()
            original = group.generator_product(strong, original=True)
            original_actions = tuple(
                effect_to_action[permutation_key(generator)]
                for generator in original
            )
            candidate_words = (original_actions, tuple(reversed(original_actions)))
            word = ()
            for candidate_word in candidate_words:
                candidate_replay = identity.copy()
                for action in candidate_word:
                    candidate_replay = candidate_replay[effects[action]]
                if candidate_replay.tobytes() == key:
                    word = candidate_word
                    break
            if not word:
                raise AssertionError(
                    f"strong generator {strong_index} original expansion has wrong orientation"
                )
            derived_rows.append(
                {
                    "original_action_count": len(word),
                    "seconds": round(time.perf_counter() - expansion_started, 6),
                    "strong_index": strong_index,
                }
            )
        replay = identity.copy()
        for action in word:
            replay = replay[effects[action]]
        if replay.tobytes() != key:
            raise AssertionError(f"strong generator {strong_index} expansion replay failed")
        strong_words[key] = word

    teacher_states: list[np.ndarray] = []
    teacher_actions: list[int] = []
    remaining_macro_steps: list[int] = []
    remaining_primitive_moves: list[int] = []
    source_sample_ids: list[int] = []
    solution_rows = []
    factor_started = time.perf_counter()

    for sample_id in range(args.samples):
        if args.stage == len(ladder["stages"]) - 1:
            initial = terminal_random_walk(
                rng,
                effects,
                inverse,
                args.terminal_walk_depth,
            )
        else:
            initial = even_random_permutation(rng)
        target = np.empty_like(initial)
        target[initial] = identity
        product = group.generator_product(
            Permutation(target.tolist(), size=24),
            original=False,
        )
        solution: list[int] | None = None
        for ordered_product in (product, list(reversed(product))):
            candidate_solution: list[int] = []
            for strong in ordered_product:
                candidate_solution.extend(strong_words[permutation_key(strong)])
            candidate_state = initial.copy()
            for action in candidate_solution:
                candidate_state = candidate_state[effects[action]]
            if np.array_equal(candidate_state, identity):
                solution = candidate_solution
                break
        if solution is None:
            raise AssertionError(f"sample {sample_id} factor replay failed")

        state = initial.copy()
        trajectory = [state.copy()]
        for action in solution:
            state = state[effects[action]]
            trajectory.append(state.copy())

        suffix_primitive = [0] * (len(solution) + 1)
        suffix_path: tuple[str, ...] = ()
        for position in range(len(solution) - 1, -1, -1):
            suffix_path = reduce_commuting_quarter_turn_path(
                paths[solution[position]] + suffix_path
            )
            suffix_primitive[position] = len(suffix_path)

        for position, action in enumerate(solution):
            teacher_states.append(trajectory[position])
            teacher_actions.append(action)
            remaining_macro_steps.append(len(solution) - position)
            remaining_primitive_moves.append(suffix_primitive[position])
            source_sample_ids.append(sample_id)
        solution_rows.append(
            {
                "macro_steps": len(solution),
                "primitive_moves": suffix_primitive[0],
                "sample_id": sample_id,
                "strong_steps": len(product),
            }
        )
        if sample_id == 0 or (sample_id + 1) % max(1, args.samples // 20) == 0:
            print(json.dumps(solution_rows[-1]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_digest = hashlib.sha256(effects.tobytes()).hexdigest()
    dataset_path = args.output_dir / "teacher.npz"
    np.savez_compressed(
        dataset_path,
        action_digest=np.asarray(action_digest),
        effects=effects,
        inverse_indices=inverse,
        remaining_macro_steps=np.asarray(remaining_macro_steps, dtype=np.int16),
        remaining_primitive_moves=np.asarray(remaining_primitive_moves, dtype=np.int32),
        source_actions=source_actions,
        source_sample_ids=np.asarray(source_sample_ids, dtype=np.int32),
        states=np.asarray(teacher_states, dtype=np.uint8),
        teacher_actions=np.asarray(teacher_actions, dtype=np.int32),
    )
    actions_path = args.output_dir / "actions.json"
    actions_path.write_text(
        json.dumps(
            {
                "action_digest": action_digest,
                "active_cluster": int(stage["active_cluster"]),
                "format_version": 1,
                "locked_clusters": stage["locked_clusters"],
                "paths": [list(path) for path in paths],
                "source_actions": source_actions.tolist(),
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    macro_lengths = [row["macro_steps"] for row in solution_rows]
    primitive_lengths = [row["primitive_moves"] for row in solution_rows]
    report = {
        "action_count": len(effects),
        "action_digest": action_digest,
        "actions": str(actions_path.resolve()),
        "bsgs_seconds": round(bsgs_seconds, 6),
        "dataset": str(dataset_path.resolve()),
        "derived_strong_generators": derived_rows,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "examples": len(teacher_states),
        "factor_seconds": round(time.perf_counter() - factor_started, 6),
        "format_version": 1,
        "group_order": group_order,
        "macro_steps": {
            "maximum": max(macro_lengths),
            "mean": round(float(np.mean(macro_lengths)), 6),
            "median": round(float(np.median(macro_lengths)), 6),
            "minimum": min(macro_lengths),
        },
        "primitive_moves": {
            "maximum": max(primitive_lengths),
            "mean": round(float(np.mean(primitive_lengths)), 6),
            "median": round(float(np.median(primitive_lengths)), 6),
            "minimum": min(primitive_lengths),
        },
        "replay_verified_solutions": len(solution_rows),
        "samples": args.samples,
        "seed": args.seed,
        "stage_index": args.stage,
        "strong_generator_count": len(group.strong_gens),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
