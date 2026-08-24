"""Build short-horizon exact teachers from Schreier transversals.

The earlier exact factor teacher expanded every factor into the dense macro
vocabulary before emitting labels.  That made an arbitrary A24 solution 70--
300 decisions long.  A Schreier--Sims chain already supplies a much smaller
decision problem: one transversal choice per base point.  This script treats
each unique transversal (and its inverse) as an atomic action, records its
verified expansion into dense macros, and emits exact trajectories whose
horizon is at most the length of the group base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=64666)
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


def permutation_key(permutation: Permutation) -> bytes:
    return bytes(permutation.array_form)


def inverse_effect(effect: np.ndarray) -> np.ndarray:
    inverse = np.empty_like(effect)
    inverse[effect] = np.arange(len(effect), dtype=effect.dtype)
    return inverse


def verified_original_word(
    group: PermutationGroup,
    permutation: Permutation,
    dense_effects: np.ndarray,
    dense_effect_to_action: dict[bytes, int],
) -> tuple[int, ...]:
    """Expand a group element into original generators with checked orientation."""
    original = group.generator_product(permutation, original=True)
    raw = tuple(dense_effect_to_action[permutation_key(generator)] for generator in original)
    target = permutation_key(permutation)
    identity = np.arange(24, dtype=np.uint8)
    for word in (raw, tuple(reversed(raw))):
        replay = identity.copy()
        for action in word:
            replay = replay[dense_effects[action]]
        if replay.tobytes() == target:
            return word
    raise AssertionError("original-generator expansion has the wrong orientation")


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
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
    dense_effects, dense_paths, dense_source_actions = eligible_actions(
        int(stage["active_cluster"]),
        tuple(int(value) for value in stage["locked_clusters"]),
        active_masks,
        table.effects,
        table.paths,
    )
    dense_effect_to_action = {
        effect.tobytes(): index for index, effect in enumerate(dense_effects)
    }

    started = time.perf_counter()
    group = PermutationGroup(
        [Permutation(effect.tolist(), size=24) for effect in dense_effects]
    )
    group_order = int(group.order())
    expected_order = int(stage["projected_group_order"])
    if group_order != expected_order:
        raise ValueError(f"projected group order {group_order} != expected {expected_order}")
    bsgs_seconds = time.perf_counter() - started

    # The union of all basic transversals is a compact, fixed action vocabulary.
    # Make it inverse-closed so the beam can suppress immediate backtracking.
    atom_effects_by_key: dict[bytes, np.ndarray] = {}
    for transversal in group.basic_transversals:
        for permutation in transversal.values():
            effect = np.asarray(permutation.array_form, dtype=np.uint8)
            if np.array_equal(effect, identity):
                continue
            atom_effects_by_key[effect.tobytes()] = effect
            inverse = inverse_effect(effect)
            atom_effects_by_key[inverse.tobytes()] = inverse
    ordered_atom_keys = sorted(atom_effects_by_key)
    atom_effects = np.asarray(
        [atom_effects_by_key[key] for key in ordered_atom_keys], dtype=np.uint8
    )
    atom_index = {key: index for index, key in enumerate(ordered_atom_keys)}
    inverse_indices = np.asarray(
        [atom_index[inverse_effect(effect).tobytes()] for effect in atom_effects],
        dtype=np.int32,
    )

    atom_dense_words: list[tuple[int, ...]] = []
    atom_paths: list[tuple[str, ...]] = []
    for effect in atom_effects:
        permutation = Permutation(effect.tolist(), size=24)
        word = verified_original_word(
            group,
            permutation,
            dense_effects,
            dense_effect_to_action,
        )
        replay = identity.copy()
        primitive_path: tuple[str, ...] = ()
        for action in word:
            replay = replay[dense_effects[action]]
            primitive_path = reduce_commuting_quarter_turn_path(
                primitive_path + dense_paths[action]
            )
        if not np.array_equal(replay, effect):
            raise AssertionError("atom expansion replay failed")
        atom_dense_words.append(word)
        atom_paths.append(primitive_path)

    teacher_states: list[np.ndarray] = []
    teacher_actions: list[int] = []
    remaining_steps: list[int] = []
    remaining_primitive_moves: list[int] = []
    source_sample_ids: list[int] = []
    solution_rows = []
    factor_started = time.perf_counter()

    for sample_id in range(args.samples):
        if args.stage == len(ladder["stages"]) - 1:
            # Sampling dense words guarantees membership in the final invariant subgroup.
            initial = identity.copy()
            for _ in range(200):
                initial = initial[dense_effects[int(rng.integers(len(dense_effects)))]]
        else:
            initial = even_random_permutation(rng)
        target = np.empty_like(initial)
        target[initial] = identity
        factors = group.coset_factor(Permutation(target.tolist(), size=24))
        if factors == [] and not np.array_equal(target, identity):
            raise AssertionError(f"sample {sample_id} is outside the projected group")
        nonidentity = [factor for factor in factors if not factor.is_identity]
        solution: list[int] | None = None
        for ordered_factors in (nonidentity, list(reversed(nonidentity))):
            candidate = [atom_index[permutation_key(factor)] for factor in ordered_factors]
            replay = initial.copy()
            for action in candidate:
                replay = replay[atom_effects[action]]
            if np.array_equal(replay, identity):
                solution = candidate
                break
        if solution is None:
            raise AssertionError(f"sample {sample_id} transversal factor replay failed")

        suffix_primitive = [0] * (len(solution) + 1)
        suffix_path: tuple[str, ...] = ()
        for position in range(len(solution) - 1, -1, -1):
            suffix_path = reduce_commuting_quarter_turn_path(
                atom_paths[solution[position]] + suffix_path
            )
            suffix_primitive[position] = len(suffix_path)

        state = initial.copy()
        for position, action in enumerate(solution):
            teacher_states.append(state.copy())
            teacher_actions.append(action)
            remaining_steps.append(len(solution) - position)
            remaining_primitive_moves.append(suffix_primitive[position])
            source_sample_ids.append(sample_id)
            state = state[atom_effects[action]]
        if not np.array_equal(state, identity):
            raise AssertionError(f"sample {sample_id} trajectory replay failed")
        solution_rows.append(
            {
                "primitive_moves": suffix_primitive[0],
                "sample_id": sample_id,
                "transversal_steps": len(solution),
            }
        )
        if sample_id == 0 or (sample_id + 1) % max(1, args.samples // 20) == 0:
            print(json.dumps(solution_rows[-1]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_digest = hashlib.sha256(atom_effects.tobytes()).hexdigest()
    dataset_path = args.output_dir / "teacher.npz"
    np.savez_compressed(
        dataset_path,
        action_digest=np.asarray(action_digest),
        effects=atom_effects,
        inverse_indices=inverse_indices,
        remaining_macro_steps=np.asarray(remaining_steps, dtype=np.int16),
        remaining_primitive_moves=np.asarray(remaining_primitive_moves, dtype=np.int32),
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
                "atom_dense_action_words": [list(word) for word in atom_dense_words],
                "atom_paths": [list(path) for path in atom_paths],
                "dense_paths": [list(path) for path in dense_paths],
                "dense_source_actions": dense_source_actions.tolist(),
                "format_version": 1,
                "locked_clusters": stage["locked_clusters"],
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    step_lengths = [row["transversal_steps"] for row in solution_rows]
    primitive_lengths = [row["primitive_moves"] for row in solution_rows]
    atom_primitive_lengths = [len(path) for path in atom_paths]
    report = {
        "action_count": len(atom_effects),
        "action_digest": action_digest,
        "actions": str(actions_path.resolve()),
        "base": list(group.base),
        "base_length": len(group.base),
        "bsgs_seconds": round(bsgs_seconds, 6),
        "dataset": str(dataset_path.resolve()),
        "dense_action_count": len(dense_effects),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "examples": len(teacher_states),
        "factor_seconds": round(time.perf_counter() - factor_started, 6),
        "format_version": 1,
        "group_order": group_order,
        "atom_primitive_moves": {
            "maximum": max(atom_primitive_lengths),
            "mean": round(float(np.mean(atom_primitive_lengths)), 6),
            "median": round(float(np.median(atom_primitive_lengths)), 6),
            "minimum": min(atom_primitive_lengths),
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
        "transversal_steps": {
            "maximum": max(step_lengths),
            "mean": round(float(np.mean(step_lengths)), 6),
            "median": round(float(np.median(step_lengths)), 6),
            "minimum": min(step_lengths),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
