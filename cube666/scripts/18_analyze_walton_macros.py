"""Analyze Walton 6x6 phase words as exact corner-safe six-cluster macros."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import (  # noqa: E402
    Cube666Decomposition,
    build_decomposition,
    cluster_permutation,
    parity_repair_path,
    permutation_cycles,
    permutation_parity,
    residual_report,
)
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402
from cube666.macro_data import (  # noqa: E402
    load_macro_action_library,
    save_macro_action_library,
)
from cube666.macros import (  # noqa: E402
    MacroEffect,
    analyze_corner_fixing_macro,
    enumerate_basic_corner_fixing_commutators,
    enumerate_conjugated_macros,
    invert_path,
    path_effect,
    reduce_commuting_quarter_turn_path,
    state_cluster_permutations,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube666.walton import (  # noqa: E402
    exact_state_is_colour_solved,
    split_marked_solution,
    translate_walton_path,
)


DEFAULT_PIDS = (597, 808, 854, 906)


@dataclass
class CandidateOrigin:
    pid: int
    start: int
    stop: int
    phase_labels: tuple[str, ...]
    inverse: bool = False


@dataclass
class Candidate:
    macro: MacroEffect
    occurrences: int = 0
    origins: list[CandidateOrigin] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube")
    parser.add_argument(
        "--walton-json",
        type=Path,
        default=PROJECT
        / "cube666"
        / "kaggle_walton_macro_mining"
        / "output"
        / "walton_runs.json",
    )
    parser.add_argument("--pids", default=",".join(str(pid) for pid in DEFAULT_PIDS))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "training" / "walton_macro4",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "walton_macro4.json",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "WALTON_MACRO4.md",
    )
    return parser.parse_args()


def _restriction(effect: Sequence[int], orbit: Sequence[int]) -> tuple[int, ...]:
    local = {position: index for index, position in enumerate(orbit)}
    try:
        return tuple(local[effect[position]] for position in orbit)
    except KeyError as exc:
        raise ValueError("effect does not preserve a discovered orbit") from exc


def _effect_signature(
    effect: Sequence[int],
    decomposition: Cube666Decomposition,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    corner = _restriction(effect, decomposition.corner_orbit)
    parities = tuple(
        permutation_parity(_restriction(effect, orbit))
        for orbit in decomposition.physical_clusters
    )
    return corner, parities


def _effect_analysis(
    path: Sequence[str],
    puzzle: Cube666Puzzle,
    decomposition: Cube666Decomposition,
) -> dict[str, object]:
    effect = path_effect(puzzle.generators, path)
    corner = _restriction(effect, decomposition.corner_orbit)
    clusters = tuple(
        _restriction(effect, orbit)
        for orbit in decomposition.physical_clusters
    )
    cycles = tuple(permutation_cycles(permutation) for permutation in clusters)
    parity = tuple(permutation_parity(permutation) for permutation in clusters)
    corner_fixed = corner == tuple(range(len(corner)))
    return {
        "active_cluster_mask": "".join("1" if item else "0" for item in cycles),
        "cluster_cycle_lengths": [
            [len(cycle) for cycle in item]
            for item in cycles
        ],
        "cluster_parity": list(parity),
        "corner_cycle_lengths": [len(cycle) for cycle in permutation_cycles(corner)],
        "corner_fixed": corner_fixed,
        "even_bulk_macro": corner_fixed and not any(parity),
    }


def _phase_labels_for_interval(
    start: int,
    stop: int,
    intervals: Sequence[tuple[int, int, str]],
) -> tuple[str, ...]:
    return tuple(
        label
        for left, right, label in intervals
        if left < stop and start < right
    )


def _record_candidate(
    candidates: dict[bytes, Candidate],
    macro: MacroEffect,
    origin: CandidateOrigin,
) -> None:
    key = np.asarray(macro.cluster_permutations, dtype=np.uint8).tobytes()
    candidate = candidates.get(key)
    if candidate is None:
        candidate = Candidate(macro)
        candidates[key] = candidate
    elif (len(macro.path), macro.path) < (len(candidate.macro.path), candidate.macro.path):
        candidate.macro = macro
    candidate.occurrences += 1
    if len(candidate.origins) < 12:
        candidate.origins.append(origin)


def _normalize_states(
    puzzle: Cube666Puzzle,
    decomposition: Cube666Decomposition,
    states_by_pid: dict[int, tuple[int, ...]],
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, object]]]:
    solver = ExactCornerSolver.build(
        CornerCoordinateSystem.discover(puzzle.generators, decomposition)
    )
    normalized: dict[int, np.ndarray] = {}
    reports: dict[int, dict[str, object]] = {}
    for pid, state in states_by_pid.items():
        corner_path = solver.solve(state, puzzle.solved_state)
        corner_state = puzzle.apply_path(state, corner_path)
        corner_report = residual_report(corner_state, puzzle.solved_state, decomposition)
        parity_path = parity_repair_path(corner_report.parity_vector, decomposition)
        final_state = puzzle.apply_path(corner_state, parity_path)
        final_report = residual_report(final_state, puzzle.solved_state, decomposition)
        if not final_report.all_even or final_report.unrestricted_three_cycles is None:
            raise AssertionError(f"PID {pid} did not normalize to six even clusters")
        if any(
            final_state[position] != puzzle.solved_state[position]
            for position in decomposition.corner_orbit
        ):
            raise AssertionError(f"PID {pid} corners are not solved after normalization")
        normalized[pid] = np.asarray(
            state_cluster_permutations(final_state, puzzle.solved_state, decomposition),
            dtype=np.uint8,
        )
        reports[pid] = {
            "corner_path": list(corner_path),
            "corner_path_length": len(corner_path),
            "initial_exact_cost": final_report.unrestricted_three_cycles,
            "parity_path": list(parity_path),
            "parity_path_length": len(parity_path),
        }
    return normalized, reports


def _load_existing_effects(
    puzzle: Cube666Puzzle,
    decomposition: Cube666Decomposition,
) -> tuple[dict[str, set[bytes]], dict[str, int]]:
    effects: dict[str, set[bytes]] = {}
    counts: dict[str, int] = {}
    base = enumerate_basic_corner_fixing_commutators(puzzle.generators, decomposition)
    expanded = enumerate_conjugated_macros(
        base,
        puzzle.generators,
        decomposition,
        max_conjugator_depth=1,
    )
    effects["depth1_commutators"] = {
        np.asarray(macro.cluster_permutations, dtype=np.uint8).tobytes()
        for macro in expanded
    }
    counts["depth1_commutators"] = len(effects["depth1_commutators"])
    for name, path in (
        (
            "kmc_teacher",
            PROJECT / "cube666" / "training" / "kmc_macro_teacher_v1" / "action_library.json",
        ),
        (
            "factorized_finisher",
            PROJECT / "cube666" / "training" / "finisher_policy_v1" / "action_library.json",
        ),
    ):
        if not path.exists():
            continue
        _, table = load_macro_action_library(path, puzzle.generators, decomposition)
        effects[name] = {effect.tobytes() for effect in table.effects}
        counts[name] = table.action_count
    return effects, counts


def _write_markdown(report: dict[str, object], path: Path) -> None:
    runs = report["runs"]
    top = report["top_candidates"]
    lines = [
        "# Walton structured-macro mining — four longest 666 PIDs",
        "",
        f"Date: {report['date']}",
        "",
        "## Walton paths",
        "",
        "| PID | Walton moves | Primitive moves | Colour solved | Phases |",
        "|---:|---:|---:|:---:|---:|",
    ]
    for pid in sorted(runs, key=int):
        run = runs[pid]
        lines.append(
            f"| {pid} | {run['standard_move_count']} | {run['primitive_move_count']} | "
            f"{'yes' if run['colour_solved'] else 'no'} | {len(run['phases'])} |"
        )
    mining = report["mining"]
    phase_aggregate = report["phase_aggregate"]
    lines.extend(
        [
            "",
            "## What the full Walton words solve",
            "",
            "All four translated words replay to the conventional six-colour goal, but none "
            "solves the unique-sticker target exactly. They are therefore macro sources, not "
            "submission paths.",
            "",
            f"Walton emitted **{phase_aggregate['generated_phase_words']}** named phase words. "
            f"Of these, **{phase_aggregate['all_clusters_even']}** are even on all six exact "
            f"clusters, **{phase_aggregate['corner_fixed']}** fix the 24 corner stickers, and "
            f"**{phase_aggregate['even_corner_safe']}** satisfy both conditions. Every phase "
            "word and its six cluster restrictions remain recorded in the JSON report.",
            "",
            "## Exact six-cluster mining",
            "",
            f"- Repeated corner/parity-signature intervals examined: "
            f"**{mining['raw_safe_subwords']:,}**.",
            f"- Identity effects discarded: **{mining['identity_subwords']:,}**.",
            f"- Unique non-identity effects before inverse closure: "
            f"**{mining['unique_forward_effects']:,}**.",
            f"- Inverse-closed action library: **{mining['inverse_closed_actions']:,}**.",
            f"- Novel versus the 19,680 depth-1 commutator effects: "
            f"**{mining['novel_vs_depth1_commutators']:,}**.",
            "",
            "Every retained word fixes all 24 corner stickers pointwise and has an even "
            "permutation on each of the six exact 24-piece clusters.",
            "",
            "## Novelty against existing action libraries",
            "",
            "| Library | Existing actions | Overlap | Novel Walton actions |",
            "|:---|---:|---:|---:|",
        ]
    )
    for name, item in report["novelty"].items():
        lines.append(
            f"| `{name}` | {item['existing_actions']:,} | {item['overlapping_actions']:,} | "
            f"{item['novel_actions']:,} |"
        )
    lines.extend(
        [
            "",
            "## Utility on the four normalized exact residuals",
            "",
            "| PID | Initial 3-cycle cost | Improving actions | Best action | Best Δ | "
            "Length | 8Δ-L |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pid in sorted(report["per_pid_utility"], key=int):
        item = report["per_pid_utility"][pid]
        lines.append(
            f"| {pid} | {item['normalized_initial_cost']} | {item['actions_improving']} | "
            f"{item['best_action']} | {item['best_delta']} | {item['best_path_length']} | "
            f"{item['best_proxy_value']} |"
        )
    lines.extend(
        [
            "",
            "## Best one-step candidates",
            "",
            "The score below is exact unrestricted 3-cycle residual reduction; `8Δ-L` is a "
            "conservative primitive-turn value proxy using the shortest finisher cost of "
            "eight turns per isolated 3-cycle.",
            "",
            "| Action | Exact primitive word | Length | Mask | Best PID | Best Δ | 8Δ-L |",
            "|---:|:---|---:|:---:|---:|---:|---:|",
        ]
    )
    for item in top[:10]:
        word = ".".join(item["path"])
        lines.append(
            f"| {item['action']} | `{word}` | {item['length']} | "
            f"`{item['active_cluster_mask']}` | {item['best_pid']} | "
            f"{item['best_delta']} | {item['best_proxy_value']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Action library: `{report['artifacts']['action_library']}`",
            f"- Machine-readable report: `{report['artifacts']['report_json']}`",
            f"- Raw Walton output: `{report['artifacts']['walton_json']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    pids = tuple(int(token) for token in args.pids.split(",") if token.strip())
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    all_states = {
        int(state_id): state
        for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    states_by_pid = {pid: all_states[pid] for pid in pids}
    normalized, normalization_reports = _normalize_states(
        puzzle,
        decomposition,
        states_by_pid,
    )

    walton = json.loads(args.walton_json.read_text(encoding="utf-8"))
    if walton.get("format_version") != 1:
        raise ValueError("unsupported Walton output format")
    candidates: dict[bytes, Candidate] = {}
    run_reports: dict[str, dict[str, object]] = {}
    raw_safe_subwords = 0
    identity_subwords = 0

    identity_effect = tuple(range(puzzle.size))
    for pid in pids:
        run = walton["runs"].get(str(pid))
        if not run or run.get("status") != "solved":
            raise RuntimeError(f"Walton PID {pid} did not solve: {run}")
        clean_standard = tuple(run["solution"])
        phases = split_marked_solution(run["marked_solution"])
        reconstructed = tuple(
            move
            for phase in phases
            for move in phase.standard_moves
        )
        if reconstructed != clean_standard:
            raise ValueError(f"PID {pid} phase markers do not reconstruct the Walton solution")

        primitive: list[str] = []
        phase_intervals: list[tuple[int, int, str]] = []
        phase_reports: list[dict[str, object]] = []
        for phase in phases:
            phase_path = translate_walton_path(phase.standard_moves)
            left = len(primitive)
            primitive.extend(phase_path)
            right = len(primitive)
            phase_intervals.append((left, right, phase.label))
            phase_report = _effect_analysis(phase_path, puzzle, decomposition)
            phase_report.update(
                {
                    "label": phase.label,
                    "primitive_move_count": len(phase_path),
                    "standard_move_count": len(phase.standard_moves),
                }
            )
            phase_reports.append(phase_report)
        primitive_path = tuple(primitive)
        reduced_path = reduce_commuting_quarter_turn_path(primitive_path)
        if path_effect(puzzle.generators, primitive_path) != path_effect(
            puzzle.generators, reduced_path
        ):
            raise AssertionError("quarter-turn reducer changed a Walton path effect")
        colour_end = puzzle.apply_path(states_by_pid[pid], primitive_path)
        colour_solved = exact_state_is_colour_solved(colour_end, puzzle.solved_state)
        if not colour_solved:
            raise ValueError(f"translated Walton path for PID {pid} is not colour solved")

        prefix_effects: list[tuple[int, ...]] = [identity_effect]
        signature_indices: defaultdict[
            tuple[tuple[int, ...], tuple[int, ...]], list[int]
        ] = defaultdict(list)
        signature_indices[_effect_signature(identity_effect, decomposition)].append(0)
        current_effect = identity_effect
        for index, move in enumerate(primitive_path, start=1):
            current_effect = puzzle.apply_move(current_effect, move)
            prefix_effects.append(current_effect)
            signature_indices[_effect_signature(current_effect, decomposition)].append(index)

        pid_raw = 0
        pid_identity = 0
        for indices in signature_indices.values():
            for left_offset, start in enumerate(indices):
                for stop in indices[left_offset + 1 :]:
                    raw_safe_subwords += 1
                    pid_raw += 1
                    segment = reduce_commuting_quarter_turn_path(primitive_path[start:stop])
                    macro = analyze_corner_fixing_macro(
                        segment,
                        puzzle.generators,
                        decomposition,
                    )
                    if macro.effect == identity_effect:
                        identity_subwords += 1
                        pid_identity += 1
                        continue
                    if macro.unrestricted_three_cycle_units is None:
                        raise AssertionError("matching parity signatures produced an odd macro")
                    _record_candidate(
                        candidates,
                        macro,
                        CandidateOrigin(
                            pid=pid,
                            start=start,
                            stop=stop,
                            phase_labels=_phase_labels_for_interval(
                                start,
                                stop,
                                phase_intervals,
                            ),
                        ),
                    )

        run_reports[str(pid)] = {
            "colour_solved": colour_solved,
            "elapsed_seconds": run["elapsed_seconds"],
            "exact_solved": colour_end == puzzle.solved_state,
            "normalized": normalization_reports[pid],
            "phases": phase_reports,
            "primitive_move_count": len(primitive_path),
            "reduced_primitive_move_count": len(reduced_path),
            "safe_subwords": pid_raw,
            "identity_subwords": pid_identity,
            "standard_move_count": len(clean_standard),
        }

    unique_forward_effects = len(candidates)
    for key, candidate in tuple(candidates.items()):
        inverse_macro = analyze_corner_fixing_macro(
            invert_path(candidate.macro.path),
            puzzle.generators,
            decomposition,
        )
        inverse_key = np.asarray(
            inverse_macro.cluster_permutations,
            dtype=np.uint8,
        ).tobytes()
        inverse_candidate = candidates.get(inverse_key)
        inverse_origins = [
            CandidateOrigin(
                pid=origin.pid,
                start=origin.start,
                stop=origin.stop,
                phase_labels=origin.phase_labels,
                inverse=True,
            )
            for origin in candidate.origins
        ]
        if inverse_candidate is None:
            candidates[inverse_key] = Candidate(
                macro=inverse_macro,
                occurrences=0,
                origins=inverse_origins,
            )
        else:
            if (len(inverse_macro.path), inverse_macro.path) < (
                len(inverse_candidate.macro.path),
                inverse_candidate.macro.path,
            ):
                inverse_candidate.macro = inverse_macro

    ordered_candidates = tuple(
        sorted(
            candidates.values(),
            key=lambda item: (
                len(item.macro.path),
                item.macro.path,
                item.macro.cluster_permutations,
            ),
        )
    )
    if not ordered_candidates:
        raise RuntimeError("Walton paths produced no corner/parity-safe non-identity macros")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    action_library = args.out_dir / "action_library.json"
    table = save_macro_action_library(action_library, [item.macro for item in ordered_candidates])

    initial_costs = {
        pid: int(sum(
            len(cycle) // 2
            for permutation in normalized[pid]
            for cycle in permutation_cycles(permutation)
        ))
        for pid in pids
    }
    costs_after = {
        pid: table.action_costs(state)
        for pid, state in normalized.items()
    }
    existing_effects, existing_counts = _load_existing_effects(puzzle, decomposition)

    candidate_rows: list[dict[str, object]] = []
    for action, candidate in enumerate(ordered_candidates):
        macro = candidate.macro
        deltas = {
            str(pid): initial_costs[pid] - int(costs_after[pid][action])
            for pid in pids
        }
        best_pid = max(pids, key=lambda pid: (deltas[str(pid)], -pid))
        best_delta = deltas[str(best_pid)]
        length = len(macro.path)
        source_pids = sorted({origin.pid for origin in candidate.origins})
        candidate_rows.append(
            {
                "action": action,
                "active_cluster_mask": macro.active_cluster_mask,
                "best_delta": best_delta,
                "best_pid": best_pid,
                "best_proxy_value": 8 * best_delta - length,
                "cluster_cycle_lengths": [
                    [len(cycle) for cycle in cycles]
                    for cycles in macro.cluster_cycles
                ],
                "deltas": deltas,
                "effect_three_cycle_units": macro.unrestricted_three_cycle_units,
                "improves_pid_count": sum(delta > 0 for delta in deltas.values()),
                "inverse_action": int(table.inverse_indices[action]),
                "length": length,
                "novel_vs": {
                    name: table.effects[action].tobytes() not in effects
                    for name, effects in existing_effects.items()
                },
                "occurrences": candidate.occurrences,
                "origins": [
                    {
                        "inverse": origin.inverse,
                        "phase_labels": list(origin.phase_labels),
                        "pid": origin.pid,
                        "start": origin.start,
                        "stop": origin.stop,
                    }
                    for origin in candidate.origins
                ],
                "path": list(macro.path),
                "source_pids": source_pids,
            }
        )
    top_candidates = sorted(
        candidate_rows,
        key=lambda row: (
            -row["best_proxy_value"],
            -row["best_delta"],
            row["length"],
            row["action"],
        ),
    )
    useful = [row for row in candidate_rows if row["best_delta"] > 0]
    positive_proxy = [row for row in candidate_rows if row["best_proxy_value"] > 0]
    all_phase_reports = [
        phase
        for run in run_reports.values()
        for phase in run["phases"]
    ]
    phase_aggregate = {
        "all_clusters_even": sum(
            not any(phase["cluster_parity"])
            for phase in all_phase_reports
        ),
        "corner_fixed": sum(phase["corner_fixed"] for phase in all_phase_reports),
        "even_corner_safe": sum(phase["even_bulk_macro"] for phase in all_phase_reports),
        "generated_phase_words": len(all_phase_reports),
    }
    novelty = {
        name: {
            "existing_actions": existing_counts[name],
            "novel_actions": sum(row["novel_vs"].get(name, True) for row in candidate_rows),
            "overlapping_actions": sum(
                not row["novel_vs"].get(name, True)
                for row in candidate_rows
            ),
        }
        for name in existing_counts
    }
    per_pid_utility: dict[str, dict[str, object]] = {}
    for pid in pids:
        pid_text = str(pid)
        best = max(
            candidate_rows,
            key=lambda row: (
                row["deltas"][pid_text],
                8 * row["deltas"][pid_text] - row["length"],
                -row["length"],
                -row["action"],
            ),
        )
        per_pid_utility[pid_text] = {
            "actions_improving": sum(row["deltas"][pid_text] > 0 for row in candidate_rows),
            "best_action": best["action"],
            "best_delta": best["deltas"][pid_text],
            "best_path": best["path"],
            "best_path_length": best["length"],
            "best_proxy_value": 8 * best["deltas"][pid_text] - best["length"],
            "normalized_initial_cost": initial_costs[pid],
        }

    report = {
        "artifacts": {
            "action_library": str(action_library.relative_to(PROJECT)),
            "report_json": str(args.report_json.relative_to(PROJECT)),
            "walton_json": str(args.walton_json.relative_to(PROJECT)),
        },
        "date": "2026-08-22",
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "existing_action_counts": existing_counts,
        "format_version": 1,
        "mining": {
            "actions_improving_at_least_one_pid": len(useful),
            "actions_with_positive_8x_proxy": len(positive_proxy),
            "identity_subwords": identity_subwords,
            "inverse_closed_actions": table.action_count,
            "novel_vs_depth1_commutators": sum(
                row["novel_vs"].get("depth1_commutators", True)
                for row in candidate_rows
            ),
            "raw_safe_subwords": raw_safe_subwords,
            "unique_forward_effects": unique_forward_effects,
        },
        "normalized_initial_costs": {str(pid): cost for pid, cost in initial_costs.items()},
        "novelty": novelty,
        "per_pid_utility": per_pid_utility,
        "phase_aggregate": phase_aggregate,
        "pids": list(pids),
        "runs": run_reports,
        "summary": {
            "best_delta": max(row["best_delta"] for row in candidate_rows),
            "best_proxy_value": max(row["best_proxy_value"] for row in candidate_rows),
            "macro_lengths": {
                "minimum": min(len(item.macro.path) for item in ordered_candidates),
                "mean": round(statistics.fmean(len(item.macro.path) for item in ordered_candidates), 4),
                "median": statistics.median(len(item.macro.path) for item in ordered_candidates),
                "maximum": max(len(item.macro.path) for item in ordered_candidates),
            },
            "source_pid_distribution": dict(
                sorted(
                    Counter(
                        origin.pid
                        for candidate in candidates.values()
                        for origin in candidate.origins
                        if not origin.inverse
                    ).items()
                )
            ),
        },
        "top_candidates": top_candidates[:200],
        "walton_commit": walton.get("walton_commit"),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report_json.with_suffix(args.report_json.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.report_json)
    _write_markdown(report, args.report_md)
    print(json.dumps({
        "action_count": table.action_count,
        "best_delta": report["summary"]["best_delta"],
        "best_proxy_value": report["summary"]["best_proxy_value"],
        "colour_solved": {pid: run_reports[str(pid)]["colour_solved"] for pid in pids},
        "report": str(args.report_json),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
