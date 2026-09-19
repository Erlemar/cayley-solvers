"""Compile Schreier-transversal actions into short dense macro words.

Every transversal atom has a correct (but often very long) Schreier expansion.
The macro library is dense enough that many of those same permutations may be
realized by one or two macros.  This script exhaustively checks all one- and
two-macro realizations with hash lookups, minimizes the reduced primitive path,
replays every winner, and reports the end-to-end teacher cost after replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
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
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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


def replay_word(effects: np.ndarray, word: tuple[int, ...]) -> np.ndarray:
    state = np.arange(24, dtype=np.uint8)
    for action in word:
        state = state[effects[action]]
    return state


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
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
    dense_by_effect = {effect.tobytes(): index for index, effect in enumerate(dense_effects)}
    dense_inverses = np.empty_like(dense_effects)
    positions = np.arange(24, dtype=np.uint8)
    for index, effect in enumerate(dense_effects):
        dense_inverses[index, effect] = positions

    teacher = np.load(args.teacher)
    atom_effects = teacher["effects"].astype(np.uint8, copy=False)
    teacher_actions = teacher["teacher_actions"].astype(np.int64, copy=False)
    source_sample_ids = teacher["source_sample_ids"].astype(np.int64, copy=False)
    metadata = json.loads(args.actions.read_text(encoding="utf-8"))
    baseline_words = [tuple(int(value) for value in word) for word in metadata["atom_dense_action_words"]]
    baseline_paths = [tuple(path) for path in metadata["atom_paths"]]
    if len(atom_effects) != len(baseline_words):
        raise ValueError("teacher and action metadata disagree on atom count")

    compiled_words: list[tuple[int, ...]] = []
    compiled_paths: list[tuple[str, ...]] = []
    depths: list[int] = []
    improvements = []
    for atom_index, effect in enumerate(atom_effects):
        best_word = baseline_words[atom_index]
        best_path = baseline_paths[atom_index]
        direct = dense_by_effect.get(effect.tobytes())
        if direct is not None:
            candidate_path = dense_paths[direct]
            if (len(candidate_path), candidate_path) < (len(best_path), best_path):
                best_word = (direct,)
                best_path = candidate_path

        for left in range(len(dense_effects)):
            # dense_effects[left][right] == effect
            needed_right = dense_inverses[left][effect]
            right = dense_by_effect.get(needed_right.tobytes())
            if right is None:
                continue
            candidate_path = reduce_commuting_quarter_turn_path(
                dense_paths[left] + dense_paths[right]
            )
            candidate_word = (left, right)
            if (len(candidate_path), candidate_path, candidate_word) < (
                len(best_path),
                best_path,
                best_word,
            ):
                best_word = candidate_word
                best_path = candidate_path
        if not np.array_equal(replay_word(dense_effects, best_word), effect):
            raise AssertionError(f"compiled atom {atom_index} failed replay")
        compiled_words.append(best_word)
        compiled_paths.append(best_path)
        depths.append(len(best_word))
        improvements.append(len(baseline_paths[atom_index]) - len(best_path))
        if atom_index == 0 or (atom_index + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "atom": atom_index,
                        "baseline": len(baseline_paths[atom_index]),
                        "compiled": len(best_path),
                        "depth": len(best_word),
                    }
                ),
                flush=True,
            )

    sample_costs = []
    baseline_sample_costs = []
    for sample_id in np.unique(source_sample_ids):
        rows = np.flatnonzero(source_sample_ids == sample_id)
        compiled_path: tuple[str, ...] = ()
        baseline_path: tuple[str, ...] = ()
        for row in rows:
            action = int(teacher_actions[row])
            compiled_path = reduce_commuting_quarter_turn_path(
                compiled_path + compiled_paths[action]
            )
            baseline_path = reduce_commuting_quarter_turn_path(
                baseline_path + baseline_paths[action]
            )
        sample_costs.append(len(compiled_path))
        baseline_sample_costs.append(len(baseline_path))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_digest = hashlib.sha256(atom_effects.tobytes()).hexdigest()
    actions_path = args.output_dir / "actions.json"
    actions_path.write_text(
        json.dumps(
            {
                "action_digest": action_digest,
                "active_cluster": int(stage["active_cluster"]),
                "atom_dense_action_words": [list(word) for word in compiled_words],
                "atom_paths": [list(path) for path in compiled_paths],
                "dense_paths": [list(path) for path in dense_paths],
                "dense_source_actions": dense_source_actions.tolist(),
                "format_version": 2,
                "locked_clusters": stage["locked_clusters"],
                "source_actions": str(args.actions.resolve()),
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "action_count": len(atom_effects),
        "action_digest": action_digest,
        "actions": str(actions_path.resolve()),
        "baseline_teacher_primitive_moves": {
            "maximum": max(baseline_sample_costs),
            "mean": round(float(np.mean(baseline_sample_costs)), 6),
            "median": round(float(np.median(baseline_sample_costs)), 6),
            "minimum": min(baseline_sample_costs),
        },
        "compiled_atom_depth_counts": {
            str(depth): int(np.count_nonzero(np.asarray(depths) == depth))
            for depth in sorted(set(depths))
        },
        "compiled_atom_primitive_moves": {
            "maximum": max(map(len, compiled_paths)),
            "mean": round(float(np.mean(list(map(len, compiled_paths)))), 6),
            "median": round(float(np.median(list(map(len, compiled_paths)))), 6),
            "minimum": min(map(len, compiled_paths)),
        },
        "compiled_teacher_primitive_moves": {
            "maximum": max(sample_costs),
            "mean": round(float(np.mean(sample_costs)), 6),
            "median": round(float(np.median(sample_costs)), 6),
            "minimum": min(sample_costs),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "format_version": 1,
        "improved_atoms": int(np.count_nonzero(np.asarray(improvements) > 0)),
        "mean_primitive_saving_per_atom": round(float(np.mean(improvements)), 6),
        "replay_verified_atoms": len(atom_effects),
        "samples": len(sample_costs),
        "stage_index": args.stage,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
