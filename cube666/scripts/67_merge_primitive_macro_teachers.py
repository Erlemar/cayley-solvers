"""Merge cube666 teachers while preserving primitive-cost/action coupling.

The older generic merge script selects rows by the post-action three-cycle
residual and can then replace the value target independently.  That is valid
for an exact-residual policy teacher, but it can pair an action from one path
with a primitive cost-to-go from another path.  This scorer-specific merge
instead selects the shortest primitive suffix first and only unions actions
that start suffixes with exactly that same cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    MacroActionTable,
    MacroTeacherDataset,
    load_macro_action_library,
    save_macro_action_library,
)
from cube666.macros import MacroEffect  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


VALUE_TOLERANCE = 1.0e-5


@dataclass
class MergedRow:
    state: np.ndarray
    actions: set[int]
    cluster_costs: np.ndarray
    next_cost: int
    walk_depth: int
    primitive_value: float
    winning_sources: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT / "cayley-py-666-cube",
    )
    parser.add_argument("--source-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def validate_primitive_source(
    source_dir: Path,
    table: MacroActionTable,
    dataset: MacroTeacherDataset,
) -> None:
    if dataset.search_value_targets is None:
        raise ValueError(f"{source_dir}: missing primitive search_value_targets")
    values = dataset.search_value_targets.astype(np.float32, copy=False)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{source_dir}: primitive values must be finite and positive")
    action_costs = np.fromiter(
        (len(path) for path in table.paths),
        dtype=np.int32,
        count=table.action_count,
    )
    slots = np.arange(dataset.teacher_actions.shape[1])[None, :]
    valid = slots < dataset.teacher_action_counts[:, None]
    labels = dataset.teacher_actions.clip(min=0)
    violations = valid & (action_costs[labels] > values[:, None] + VALUE_TOLERANCE)
    if np.any(violations):
        rows, columns = np.nonzero(violations)
        row = int(rows[0])
        column = int(columns[0])
        action = int(labels[row, column])
        raise ValueError(
            f"{source_dir}: value target is not primitive cost-to-go at row {row}; "
            f"action {action} costs {action_costs[action]} > target {values[row]}"
        )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)

    sources: list[
        tuple[Path, tuple[MacroEffect, ...], MacroActionTable, MacroTeacherDataset, np.ndarray]
    ] = []
    macro_by_effect: dict[bytes, MacroEffect] = {}
    for source_dir in args.source_dirs:
        macros, table = load_macro_action_library(
            source_dir / "action_library.json",
            puzzle.generators,
            decomposition,
        )
        dataset = MacroTeacherDataset.load(source_dir / "teacher.npz")
        groups = np.load(source_dir / "source_state_ids.npy", allow_pickle=False)
        dataset.validate(table.action_count)
        if dataset.action_digest != table.digest:
            raise ValueError(f"{source_dir}: teacher and action digests differ")
        if groups.shape != (dataset.sample_count,):
            raise ValueError(f"{source_dir}: source_state_ids shape differs")
        validate_primitive_source(source_dir, table, dataset)
        sources.append((source_dir, macros, table, dataset, groups))
        for macro, effect in zip(macros, table.effects, strict=True):
            key = effect.tobytes()
            old = macro_by_effect.get(key)
            if old is None or (len(macro.path), macro.path) < (len(old.path), old.path):
                macro_by_effect[key] = macro

    macros = tuple(
        sorted(
            macro_by_effect.values(),
            key=lambda macro: (len(macro.path), macro.path, macro.cluster_permutations),
        )
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_table = save_macro_action_library(args.out_dir / "action_library.json", macros)
    merged_action_by_effect = {
        effect.tobytes(): action
        for action, effect in enumerate(merged_table.effects)
    }

    merged_rows: dict[tuple[int, bytes], MergedRow] = {}
    raw_samples = 0
    source_wins: Counter[str] = Counter()
    for source_dir, _, table, dataset, groups in sources:
        action_remap = np.asarray(
            [merged_action_by_effect[effect.tobytes()] for effect in table.effects],
            dtype=np.int32,
        )
        assert dataset.search_value_targets is not None
        for index in range(dataset.sample_count):
            raw_samples += 1
            state = dataset.states[index]
            pid = int(groups[index])
            key = (pid, state.tobytes())
            count = int(dataset.teacher_action_counts[index])
            old_actions = dataset.teacher_actions[index, :count]
            actions = {int(action_remap[int(action)]) for action in old_actions}
            primitive_value = float(dataset.search_value_targets[index])
            next_cost = int(dataset.teacher_next_costs[index])
            walk_depth = int(dataset.walk_depths[index])
            source_name = source_dir.name
            current = merged_rows.get(key)
            if current is None or primitive_value < current.primitive_value - VALUE_TOLERANCE:
                merged_rows[key] = MergedRow(
                    state=state.copy(),
                    actions=actions,
                    cluster_costs=dataset.cluster_cost_targets[index].copy(),
                    next_cost=next_cost,
                    walk_depth=walk_depth,
                    primitive_value=primitive_value,
                    winning_sources={source_name},
                )
            elif abs(primitive_value - current.primitive_value) <= VALUE_TOLERANCE:
                if not np.array_equal(current.state, state):
                    raise AssertionError("state-key collision")
                if not np.array_equal(
                    current.cluster_costs,
                    dataset.cluster_cost_targets[index],
                ):
                    raise ValueError("identical state has inconsistent exact cluster cost")
                current.actions.update(actions)
                current.next_cost = min(current.next_cost, next_cost)
                current.walk_depth = min(current.walk_depth, walk_depth)
                current.winning_sources.add(source_name)

    ordered = sorted(
        merged_rows.items(),
        key=lambda item: (item[0][0], item[1].walk_depth, item[0][1]),
    )
    for _, row in ordered:
        for source_name in row.winning_sources:
            source_wins[source_name] += 1
    maximum_labels = max(len(row.actions) for _, row in ordered)
    states = np.stack([row.state for _, row in ordered])
    labels = np.full((len(ordered), maximum_labels), -1, dtype=np.int32)
    counts = np.empty(len(ordered), dtype=np.int16)
    for index, (_, row) in enumerate(ordered):
        actions = sorted(row.actions)
        labels[index, : len(actions)] = actions
        counts[index] = len(actions)
    primitive_values = np.asarray(
        [row.primitive_value for _, row in ordered],
        dtype=np.float32,
    )
    search_clusters = np.broadcast_to(
        (primitive_values / 6.0)[:, None],
        (len(primitive_values), 6),
    ).copy()
    dataset = MacroTeacherDataset(
        states=states,
        teacher_actions=labels,
        teacher_action_counts=counts,
        cluster_cost_targets=np.stack([row.cluster_costs for _, row in ordered]),
        teacher_next_costs=np.asarray(
            [row.next_cost for _, row in ordered],
            dtype=np.int16,
        ),
        walk_depths=np.asarray(
            [row.walk_depth for _, row in ordered],
            dtype=np.int16,
        ),
        action_digest=merged_table.digest,
        allow_non_improving=any(
            row.next_cost >= int(row.cluster_costs.sum())
            for _, row in ordered
        ),
        search_value_targets=primitive_values,
        search_cluster_targets=search_clusters,
    )
    dataset.validate(merged_table.action_count)

    action_costs = np.fromiter(
        (len(path) for path in merged_table.paths),
        dtype=np.int16,
        count=merged_table.action_count,
    )
    slots = np.arange(labels.shape[1])[None, :]
    valid = slots < counts[:, None]
    label_costs = action_costs[labels.clip(min=0)]
    violations = valid & (label_costs > primitive_values[:, None] + VALUE_TOLERANCE)
    if np.any(violations):
        raise AssertionError("primitive merge produced a label/value cost violation")

    dataset.save(args.out_dir / "teacher.npz")
    atomic_npy(
        args.out_dir / "source_state_ids.npy",
        np.asarray([key[0] for key, _ in ordered], dtype=np.int32),
    )
    atomic_npy(args.out_dir / "action_effects.npy", merged_table.effects)
    atomic_npy(args.out_dir / "action_costs.npy", action_costs)

    report = {
        "action_count": merged_table.action_count,
        "action_digest": merged_table.digest,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "label_action_cost_maximum": int(label_costs[valid].max()),
        "label_action_cost_mean": float(label_costs[valid].mean()),
        "label_cost_violations": int(np.count_nonzero(violations)),
        "maximum_primitive_value": float(primitive_values.max()),
        "mean_primitive_value": float(primitive_values.mean()),
        "minimum_primitive_value": float(primitive_values.min()),
        "non_improving_samples": int(
            np.count_nonzero(
                dataset.teacher_next_costs
                >= dataset.cluster_cost_targets.sum(axis=1)
            )
        ),
        "raw_samples": raw_samples,
        "samples": dataset.sample_count,
        "source_dirs": [str(path) for path, *_ in sources],
        "source_pids": len({key[0] for key, _ in ordered}),
        "source_winning_rows": dict(sorted(source_wins.items())),
        "value_unit": "primitive_moves",
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
