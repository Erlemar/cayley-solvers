"""Merge exact macro teachers and action libraries by effect and source PID."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    MacroTeacherDataset,
    load_macro_action_library,
    save_macro_action_library,
)
from cube666.macros import MacroEffect  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


@dataclass
class MergedRow:
    state: np.ndarray
    actions: set[int]
    cluster_costs: np.ndarray
    next_cost: int
    walk_depth: int
    search_value: float
    search_clusters: np.ndarray


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


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)

    sources: list[tuple[Path, tuple[MacroEffect, ...], object, MacroTeacherDataset, np.ndarray]] = []
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
    for _, _, table, dataset, groups in sources:
        action_remap = np.asarray(
            [merged_action_by_effect[effect.tobytes()] for effect in table.effects],
            dtype=np.int32,
        )
        for index in range(dataset.sample_count):
            raw_samples += 1
            state = dataset.states[index]
            pid = int(groups[index])
            key = (pid, state.tobytes())
            count = int(dataset.teacher_action_counts[index])
            old_actions = dataset.teacher_actions[index, :count]
            actions = {int(action_remap[int(action)]) for action in old_actions}
            next_cost = int(dataset.teacher_next_costs[index])
            current = merged_rows.get(key)
            if current is None or next_cost < current.next_cost:
                merged_rows[key] = MergedRow(
                    state=state.copy(),
                    actions=actions,
                    cluster_costs=dataset.cluster_cost_targets[index].copy(),
                    next_cost=next_cost,
                    walk_depth=int(dataset.walk_depths[index]),
                    search_value=float(dataset.search_value_targets[index]),
                    search_clusters=dataset.search_cluster_targets[index].copy(),
                )
            elif next_cost == current.next_cost:
                current.actions.update(actions)
                current.walk_depth = min(
                    current.walk_depth,
                    int(dataset.walk_depths[index]),
                )
                candidate_value = float(dataset.search_value_targets[index])
                if candidate_value < current.search_value:
                    current.search_value = candidate_value
                    current.search_clusters = dataset.search_cluster_targets[
                        index
                    ].copy()

    ordered = sorted(
        merged_rows.items(),
        key=lambda item: (item[0][0], item[1].walk_depth, item[0][1]),
    )
    maximum_labels = max(len(row.actions) for _, row in ordered)
    states = np.stack([row.state for _, row in ordered])
    labels = np.full((len(ordered), maximum_labels), -1, dtype=np.int32)
    counts = np.empty(len(ordered), dtype=np.int16)
    for index, (_, row) in enumerate(ordered):
        actions = sorted(row.actions)
        labels[index, : len(actions)] = actions
        counts[index] = len(actions)
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
        search_value_targets=np.asarray(
            [row.search_value for _, row in ordered],
            dtype=np.float32,
        ),
        search_cluster_targets=np.stack(
            [row.search_clusters for _, row in ordered]
        ).astype(np.float32, copy=False),
    )
    dataset.validate(merged_table.action_count)
    dataset.save(args.out_dir / "teacher.npz")
    group_path = args.out_dir / "source_state_ids.npy"
    temporary = group_path.with_suffix(group_path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, np.asarray([key[0] for key, _ in ordered], dtype=np.int32))
    temporary.replace(group_path)

    report = {
        "action_count": merged_table.action_count,
        "action_digest": merged_table.digest,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "raw_samples": raw_samples,
        "samples": dataset.sample_count,
        "non_improving_samples": int(
            np.count_nonzero(
                dataset.teacher_next_costs
                >= dataset.cluster_cost_targets.sum(axis=1)
            )
        ),
        "source_dirs": [str(path) for path, *_ in sources],
        "source_pids": len({key[0] for key, _ in ordered}),
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
