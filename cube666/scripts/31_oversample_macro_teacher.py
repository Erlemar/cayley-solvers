"""Append repeated, effect-remapped focus labels to a merged macro teacher.

This keeps the original PID group IDs, so validation exclusion remains leakage
safe while a small on-policy DAgger corpus can occupy a controlled fraction of
fine-tuning batches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import MacroTeacherDataset, load_macro_action_library  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT / "cayley-py-666-cube",
    )
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--focus-dir", type=Path, required=True)
    parser.add_argument("--focus-repeats", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.focus_repeats <= 0:
        raise ValueError("focus-repeats must be positive")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    _, base_table = load_macro_action_library(
        args.base_dir / "action_library.json",
        puzzle.generators,
        decomposition,
    )
    _, focus_table = load_macro_action_library(
        args.focus_dir / "action_library.json",
        puzzle.generators,
        decomposition,
    )
    base = MacroTeacherDataset.load(args.base_dir / "teacher.npz")
    focus = MacroTeacherDataset.load(args.focus_dir / "teacher.npz")
    base.validate(base_table.action_count)
    focus.validate(focus_table.action_count)
    if base.action_digest != base_table.digest or focus.action_digest != focus_table.digest:
        raise ValueError("teacher and action-library digests differ")
    base_action_by_effect = {
        effect.tobytes(): index for index, effect in enumerate(base_table.effects)
    }
    focus_remap = np.asarray(
        [base_action_by_effect[effect.tobytes()] for effect in focus_table.effects],
        dtype=np.int32,
    )
    remapped_focus_actions = focus.teacher_actions.copy()
    active = remapped_focus_actions >= 0
    remapped_focus_actions[active] = focus_remap[remapped_focus_actions[active]]
    maximum_labels = max(base.teacher_actions.shape[1], focus.teacher_actions.shape[1])

    def pad(actions: np.ndarray) -> np.ndarray:
        if actions.shape[1] == maximum_labels:
            return actions
        result = np.full((len(actions), maximum_labels), -1, dtype=np.int32)
        result[:, : actions.shape[1]] = actions
        return result

    repeats = args.focus_repeats
    dataset = MacroTeacherDataset(
        states=np.concatenate((base.states, np.tile(focus.states, (repeats, 1, 1)))),
        teacher_actions=np.concatenate(
            (pad(base.teacher_actions), np.tile(pad(remapped_focus_actions), (repeats, 1)))
        ),
        teacher_action_counts=np.concatenate(
            (base.teacher_action_counts, np.tile(focus.teacher_action_counts, repeats))
        ),
        cluster_cost_targets=np.concatenate(
            (base.cluster_cost_targets, np.tile(focus.cluster_cost_targets, (repeats, 1)))
        ),
        teacher_next_costs=np.concatenate(
            (base.teacher_next_costs, np.tile(focus.teacher_next_costs, repeats))
        ),
        walk_depths=np.concatenate(
            (base.walk_depths, np.tile(focus.walk_depths, repeats))
        ),
        action_digest=base_table.digest,
        allow_non_improving=base.allow_non_improving or focus.allow_non_improving,
        search_value_targets=np.concatenate(
            (
                base.search_value_targets,
                np.tile(focus.search_value_targets, repeats),
            )
        ),
        search_cluster_targets=np.concatenate(
            (
                base.search_cluster_targets,
                np.tile(focus.search_cluster_targets, (repeats, 1)),
            )
        ),
    )
    dataset.validate(base_table.action_count)
    base_groups = np.load(args.base_dir / "source_state_ids.npy", allow_pickle=False)
    focus_groups = np.load(args.focus_dir / "source_state_ids.npy", allow_pickle=False)
    if base_groups.shape != (base.sample_count,) or focus_groups.shape != (focus.sample_count,):
        raise ValueError("source group shapes do not match their teachers")
    groups = np.concatenate((base_groups, np.tile(focus_groups, repeats))).astype(
        np.int32,
        copy=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save(args.out_dir / "teacher.npz")
    temporary = (args.out_dir / "source_state_ids.npy").with_suffix(".npy.tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, groups)
    temporary.replace(args.out_dir / "source_state_ids.npy")
    effects_path = args.out_dir / "action_effects.npy"
    effects_temporary = effects_path.with_suffix(effects_path.suffix + ".tmp")
    with open(effects_temporary, "wb") as handle:
        np.save(handle, base_table.effects)
    effects_temporary.replace(effects_path)
    report = {
        "action_count": base_table.action_count,
        "action_digest": base_table.digest,
        "base_samples": base.sample_count,
        "focus_fraction": (repeats * focus.sample_count) / dataset.sample_count,
        "focus_repeats": repeats,
        "focus_samples": focus.sample_count,
        "non_improving_samples_with_repeats": int(
            np.count_nonzero(
                dataset.teacher_next_costs >= dataset.cluster_cost_targets.sum(axis=1)
            )
        ),
        "samples": dataset.sample_count,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
