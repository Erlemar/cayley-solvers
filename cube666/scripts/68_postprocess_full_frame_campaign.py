"""Candidate-local Walton + exact radius-4 post-processing for the full cube666 campaign.

The runner is deliberately restartable.  Walton and radius-4 results are cached by
``(pid, input-path hash)`` before the next candidate is attempted.  Every input,
cached result, accepted rewrite, and final submission row is replayed on the exact
216-sticker puzzle representation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
GATE_PATH = Path(__file__).with_name("67_postprocess_frame_gate.py")
DEFAULT_DATA = PROJECT / "cayley-py-666-cube"
DEFAULT_PRODUCTION = PROJECT / "cube666" / "kaggle_kmc_campaign" / "production_outputs"
DEFAULT_HARVEST = DEFAULT_PRODUCTION / "resume_wave1" / "harvest_report.json"
DEFAULT_INCUMBENT = PROJECT / "submissions" / "cube666_kmc_wave1_complete_merged_v1.csv"
DEFAULT_SEED = PROJECT / "submissions" / "cube666_kmc_postprocess_gate10_v1.csv"
DEFAULT_REPORT_DIR = PROJECT / "cube666" / "reports" / "postprocess_full_wave1_v1"
DEFAULT_OUTPUT = PROJECT / "submissions" / "cube666_kmc_wave1_complete_pp_v1.csv"
DEFAULT_GATE_CACHE = PROJECT / "cube666" / "reports" / "postprocess_gate10_v1"
FRAME_NAMES = ("rot03", "rot18", "rot35")


def load_gate_module():
    spec = importlib.util.spec_from_file_location("cube666_postprocess_gate", GATE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {GATE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = load_gate_module()
Candidate = gate.Candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--harvest-report", type=Path, default=DEFAULT_HARVEST)
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--seed-submission", type=Path, action="append", default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-cache", type=Path, action="append", default=None)
    parser.add_argument("--polisher", type=Path, default=gate.DEFAULT_POLISHER)
    parser.add_argument("--kmc-dir", type=Path, default=gate.DEFAULT_KMC_DIR)
    parser.add_argument("--pids", type=str, default="")
    parser.add_argument("--walton-workers", type=int, default=4)
    parser.add_argument("--walton-timeout", type=int, default=300)
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument("--window-rounds", type=int, default=4)
    parser.add_argument("--window-chunk", type=int, default=5000)
    parser.add_argument("--window-device", type=str, default="cuda")
    parser.add_argument("--skip-walton", action="store_true")
    parser.add_argument("--skip-window", action="store_true")
    parser.add_argument("--skip-positive-controls", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.seed_submission is None:
        args.seed_submission = [DEFAULT_SEED]
    if args.reuse_cache is None:
        args.reuse_cache = [DEFAULT_GATE_CACHE]
    return args


def status(path: Path, stage: str, **payload: object) -> None:
    record = {
        "format_version": 1,
        "pid": os.getpid(),
        "stage": stage,
        "updated_unix": time.time(),
        **payload,
    }
    gate.atomic_write(path, json.dumps(record, indent=2, sort_keys=True) + "\n")


def requested_pids(raw: str, available: set[int]) -> list[int]:
    if not raw.strip():
        return sorted(available)
    selected = [int(token) for token in raw.replace(",", " ").split()]
    if len(selected) != len(set(selected)):
        raise ValueError("--pids contains duplicates")
    missing = sorted(set(selected).difference(available))
    if missing:
        raise ValueError(f"unknown pids requested: {missing}")
    return selected


def discover_frame_paths(args: argparse.Namespace) -> tuple[dict[int, dict[str, Path]], list[str]]:
    original_dirs = sorted(args.production_dir.glob("slot*/extracted/w1_rot*_tl40_s*_b20000"))
    harvest = json.loads(args.harvest_report.read_text(encoding="utf-8"))
    if not harvest.get("coverage_complete"):
        raise ValueError("resume harvest is not marked coverage_complete")
    resume_dirs = [Path(value) for value in harvest["candidate_directories"]]
    directories = original_dirs + resume_dirs
    if not directories or any(not directory.is_dir() for directory in directories):
        absent = [str(directory) for directory in directories if not directory.is_dir()]
        raise FileNotFoundError(f"candidate directories missing: {absent}")

    found: dict[int, dict[str, Path]] = {}
    for directory in directories:
        match = re.search(r"(rot03|rot18|rot35)", directory.name)
        if match is None:
            raise ValueError(f"cannot identify frame for {directory}")
        frame = match.group(1)
        for path_file in directory.glob("*.path.txt"):
            pid = int(path_file.name.split(".")[0])
            previous = found.setdefault(pid, {}).get(frame)
            if previous is not None:
                raise ValueError(f"duplicate task pid={pid} frame={frame}: {previous}, {path_file}")
            found[pid][frame] = path_file
    return found, [str(directory.resolve()) for directory in directories]


def import_caches(report_dir: Path, cache_roots: list[Path], radius: int) -> dict[str, int]:
    imported: dict[str, int] = {}
    for stage in ("walton_cache", f"window_r{radius}_cache"):
        target = report_dir / stage
        target.mkdir(parents=True, exist_ok=True)
        count = 0
        for root in cache_roots:
            source = root / stage
            if not source.is_dir():
                continue
            for source_file in source.glob("*"):
                if not source_file.is_file():
                    continue
                destination = target / source_file.name
                if destination.exists():
                    continue
                shutil.copy2(source_file, destination)
                count += 1
        imported[stage] = count
    return imported


def cache_count(report_dir: Path, stage: str) -> int:
    directory = report_dir / stage
    return sum(1 for _ in directory.glob("*.path.txt")) if directory.is_dir() else 0


def build_candidates(
    selected: list[int],
    frames: dict[int, dict[str, Path]],
    submissions: list[tuple[str, dict[int, tuple[str, ...]]]],
    puzzle,
    states,
) -> tuple[dict[int, list[Candidate]], list[dict[str, object]], dict[int, tuple[str, ...]], dict[int, str]]:
    raw_by_pid: dict[int, list[Candidate]] = {}
    inventory: list[dict[str, object]] = []
    baseline: dict[int, tuple[str, ...]] = {}
    baseline_source: dict[int, str] = {}
    for pid in selected:
        raw: list[Candidate] = []
        submission_lengths: dict[str, int] = {}
        for source, rows in submissions:
            path = rows[pid]
            gate.replay_verify(puzzle, states, pid, path, source)
            submission_lengths[source] = len(path)
            raw.append(Candidate(pid, (source,), path))
        ranked_submission_paths = [
            (source, rows[pid], rank)
            for rank, (source, rows) in enumerate(submissions)
        ]
        best_source, best_path, _ = min(
            ranked_submission_paths,
            key=lambda item: (len(item[1]), item[2]),
        )
        baseline[pid] = best_path
        baseline_source[pid] = best_source

        frame_lengths: dict[str, int] = {}
        for frame in FRAME_NAMES:
            path = gate.parse_path(frames[pid][frame].read_text(encoding="utf-8"))
            gate.replay_verify(puzzle, states, pid, path, frame)
            frame_lengths[frame] = len(path)
            raw.append(Candidate(pid, (frame,), path))
        commuting = [
            Candidate(candidate.pid, candidate.sources, gate.reduce_commuting_quarter_turn_path(candidate.path))
            for candidate in raw
        ]
        for candidate in commuting:
            gate.replay_verify(puzzle, states, pid, candidate.path, "commuting input")
        raw_by_pid[pid] = gate.deduplicate(commuting)
        inventory.append(
            {
                "pid": pid,
                "submission_lengths": submission_lengths,
                "baseline": len(best_path),
                "baseline_source": best_source,
                "frame_lengths": frame_lengths,
                "unique_after_commuting": len(raw_by_pid[pid]),
            }
        )
    return raw_by_pid, inventory, baseline, baseline_source


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    status_file = args.report_dir / "status.json"
    journal = args.report_dir / "journal.jsonl"
    status(status_file, "loading")

    puzzle = gate.Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    puzzle.verify_inverse_pairs()
    states = {int(pid): state for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")}
    pathkit_puzzle = gate.PathkitPuzzle.from_puzzle_info(args.data_dir / "puzzle_info.json")
    pathkit_states = {pid: np.asarray(state, dtype=np.uint8) for pid, state in states.items()}

    primary = gate.load_csv_paths(args.incumbent)
    submissions: list[tuple[str, dict[int, tuple[str, ...]]]] = [("incumbent", primary)]
    for index, seed_path in enumerate(args.seed_submission, start=1):
        if seed_path.is_file():
            submissions.append((f"seed{index}", gate.load_csv_paths(seed_path)))
    for source, rows in submissions:
        if set(rows) != set(states):
            raise ValueError(f"{source} coverage does not match test.csv")

    frames, candidate_directories = discover_frame_paths(args)
    if set(frames) != set(states):
        missing = sorted(set(states).difference(frames))
        extra = sorted(set(frames).difference(states))
        raise ValueError(f"frame PID coverage mismatch: missing={missing}, extra={extra}")
    malformed = {pid: sorted(set(FRAME_NAMES).difference(values)) for pid, values in frames.items() if set(values) != set(FRAME_NAMES)}
    if malformed:
        raise ValueError(f"incomplete frame coverage: {malformed}")
    selected = requested_pids(args.pids, set(states))
    raw_by_pid, inventory, selected_baseline, selected_baseline_source = build_candidates(
        selected, frames, submissions, puzzle, states
    )
    all_candidates = [candidate for pid in selected for candidate in raw_by_pid[pid]]
    baseline_all: dict[int, tuple[str, ...]] = {}
    baseline_all_source: dict[int, str] = {}
    for pid in sorted(states):
        choices = [(source, rows[pid], rank) for rank, (source, rows) in enumerate(submissions)]
        source, path, _ = min(choices, key=lambda item: (len(item[1]), item[2]))
        baseline_all[pid] = path
        baseline_all_source[pid] = source

    inventory_report = {
        "format_version": 1,
        "candidate_directories": candidate_directories,
        "candidate_directory_count": len(candidate_directories),
        "frame_tasks": sum(len(values) for values in frames.values()),
        "full_pid_count": len(states),
        "selected_pid_count": len(selected),
        "selected_pids": selected,
        "input_submissions": [{"source": source, "path": str(path.resolve())} for source, path in [("incumbent", args.incumbent)] + [(f"seed{i}", value) for i, value in enumerate(args.seed_submission, start=1) if value.is_file()]],
        "baseline_total": sum(len(path) for path in baseline_all.values()),
        "unique_candidate_count": len(all_candidates),
        "per_pid": inventory,
    }
    gate.atomic_write(args.report_dir / "inventory.json", json.dumps(inventory_report, indent=2, sort_keys=True) + "\n")
    print(
        f"inventory: {len(states)} pids, {inventory_report['frame_tasks']} frame tasks, "
        f"{len(all_candidates)} unique selected candidates, baseline={inventory_report['baseline_total']}",
        flush=True,
    )
    if args.inventory_only:
        status(status_file, "inventory_complete", **{key: inventory_report[key] for key in ("frame_tasks", "selected_pid_count", "unique_candidate_count", "baseline_total")})
        return

    imported = import_caches(args.report_dir, args.reuse_cache, args.window_radius)
    if args.skip_positive_controls:
        controls: dict[str, object] = {"skipped": True}
    else:
        existing_controls = args.report_dir / "positive_controls.json"
        if existing_controls.is_file():
            controls = json.loads(existing_controls.read_text(encoding="utf-8"))
            controls["reused"] = True
        else:
            controls = gate.positive_controls(
                selected, selected_baseline, puzzle, states, pathkit_puzzle, pathkit_states, args
            )
            gate.atomic_write(existing_controls, json.dumps(controls, indent=2, sort_keys=True) + "\n")
    print("positive controls: " + json.dumps(controls, sort_keys=True), flush=True)

    status(status_file, "walton", candidates=len(all_candidates), cache_files=cache_count(args.report_dir, "walton_cache"), imported=imported)
    walton_started = time.perf_counter()
    walton = gate.walton_stage(all_candidates, puzzle, states, args, journal)
    walton_seconds = time.perf_counter() - walton_started

    window_stage_name = f"window_r{args.window_radius}_cache"
    status(status_file, "radius4", candidates=len(walton), cache_files=cache_count(args.report_dir, window_stage_name), walton_seconds=round(walton_seconds, 3))
    window_started = time.perf_counter()
    windowed = gate.window_stage(
        walton, puzzle, states, pathkit_puzzle, pathkit_states, args, journal
    )
    window_seconds = time.perf_counter() - window_started

    processed_by_pid: dict[int, list[Candidate]] = {pid: [] for pid in selected}
    for candidate in windowed:
        processed_by_pid[candidate.pid].append(candidate)
    merged = dict(baseline_all)
    pid_reports: list[dict[str, object]] = []
    winner_attribution: Counter[str] = Counter()
    for pid in selected:
        choices = [(selected_baseline[pid], selected_baseline_source[pid])]
        choices.extend((candidate.path, "+".join(candidate.sources)) for candidate in processed_by_pid[pid])
        winner, source = min(choices, key=lambda item: (len(item[0]), item[1]))
        if len(winner) < len(selected_baseline[pid]):
            merged[pid] = winner
            winner_attribution[source] += 1
        else:
            source = selected_baseline_source[pid]
            winner_attribution[source] += 1
        gate.replay_verify(puzzle, states, pid, merged[pid], "selected final")
        pid_reports.append(
            {
                "pid": pid,
                "baseline": len(selected_baseline[pid]),
                "walton_best": min(len(candidate.path) for candidate in walton if candidate.pid == pid),
                "radius4_best": min(len(candidate.path) for candidate in processed_by_pid[pid]),
                "final": len(merged[pid]),
                "saved": len(selected_baseline[pid]) - len(merged[pid]),
                "winner": source,
            }
        )

    if set(merged) != set(states):
        raise ValueError("final coverage does not match test.csv")
    for pid in sorted(states):
        gate.replay_verify(puzzle, states, pid, merged[pid], "full final submission")
    gate.write_submission(args.output, merged)

    baseline_total = sum(len(path) for path in baseline_all.values())
    output_total = sum(len(path) for path in merged.values())
    report = {
        "format_version": 1,
        "selected_pids": selected,
        "selected_pid_count": len(selected),
        "raw_unique_candidates": len(all_candidates),
        "walton_output_candidates": len(walton),
        "radius4_output_candidates": len(windowed),
        "walton_seconds": round(walton_seconds, 3),
        "radius4_seconds": round(window_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "baseline_total": baseline_total,
        "output_total": output_total,
        "moves_saved": baseline_total - output_total,
        "strict_winning_pids": sum(item["saved"] > 0 for item in pid_reports),
        "winner_attribution": dict(sorted(winner_attribution.items())),
        "cache_imports": imported,
        "pid_results": pid_reports,
        "replay_verified_full_submission": len(states),
        "output": str(args.output.resolve()),
    }
    gate.atomic_write(args.report_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    status(status_file, "complete", output_total=output_total, moves_saved=baseline_total - output_total, output=str(args.output.resolve()))
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
