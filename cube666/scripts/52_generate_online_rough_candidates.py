"""Generate rough-only KMC candidates on fresh PIDs for online ranking."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import build_decomposition, residual_report  # noqa: E402
from cube666.macros import (  # noqa: E402
    reduce_commuting_quarter_turn_path,
    state_cluster_permutations,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


CONDITIONS = (
    {
        "alpha": 3.0,
        "anneal_keep_best": False,
        "anneal_seed": 50001,
        "anneal_steps": 20_000_000,
        "cache_tag": "cayley_v4_dfr_s50001_n20000000_a3",
        "label": "control",
    },
    {
        "alpha": 3.0,
        "anneal_keep_best": False,
        "anneal_seed": 50002,
        "anneal_steps": 20_000_000,
        "label": "seed50002",
    },
    {
        "alpha": 3.0,
        "anneal_keep_best": False,
        "anneal_seed": 50003,
        "anneal_steps": 20_000_000,
        "label": "seed50003",
    },
    {
        "alpha": 2.5,
        "anneal_keep_best": False,
        "anneal_seed": 50001,
        "anneal_steps": 20_000_000,
        "label": "alpha2p5",
    },
    {
        "alpha": 3.5,
        "anneal_keep_best": False,
        "anneal_seed": 50001,
        "anneal_steps": 20_000_000,
        "label": "alpha3p5",
    },
    {
        "alpha": 3.0,
        "anneal_keep_best": True,
        "anneal_seed": 50001,
        "anneal_steps": 20_000_000,
        "label": "keepbest",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=(
            PROJECT / "external/third_party/kmcoders_santa2023/solution"
        ),
    )
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=(
            PROJECT / "submissions/cube666_path_context_ranker_strictwin_v1.csv"
        ),
    )
    parser.add_argument(
        "--exclude-report",
        type=Path,
        action="append",
        default=[
            PROJECT / "cube666/reports/KMC_PARAM_GATE16.json",
            PROJECT / "cube666/reports/PATH_CONTEXT_CORPUS32_MANIFEST.json",
        ],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666/training/path_context_online24_v1",
    )
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rayon-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--conditions", help="optional comma-separated labels")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def load_lengths(path: Path) -> dict[int, int]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): len(parse_path(row["path"]))
            for row in csv.DictReader(handle)
        }


def stratified_pids(
    lengths: dict[int, int], excluded: set[int], count: int
) -> list[int]:
    if count <= 0 or count % 4:
        raise ValueError("count must be positive and divisible by four")
    candidates = sorted(
        (length, pid)
        for pid, length in lengths.items()
        if 210 <= pid <= 1011 and pid not in excluded
    )
    selected = []
    per_quartile = count // 4
    for quartile in range(4):
        bucket = candidates[
            quartile * len(candidates) // 4 : (quartile + 1) * len(candidates) // 4
        ]
        for sample in range(per_quartile):
            index = ((2 * sample + 1) * len(bucket)) // (2 * per_quartile)
            selected.append(bucket[min(index, len(bucket) - 1)][1])
    if len(set(selected)) != count:
        raise AssertionError("fresh PID selection produced duplicates")
    return sorted(selected)


def solver_input(puzzle: Cube666Puzzle, state_id: str, state: tuple[int, ...]) -> str:
    positive_moves = tuple(f"{axis}{layer}" for axis in "dfr" for layer in range(6))
    tokens = [state_id, "cube_6/6/6", str(puzzle.size), str(len(positive_moves)), "0"]
    for name in positive_moves:
        tokens.append(name)
        tokens.extend(str(value) for value in puzzle.generators[name])
    tokens.extend(str(value) for value in puzzle.solved_state)
    tokens.extend(str(value) for value in state)
    return " ".join(tokens) + "\n"


def digest_tokens(tokens: tuple[str, ...]) -> str:
    return hashlib.sha256(".".join(tokens).encode("utf-8")).hexdigest()


def path_context_features(
    path: tuple[str, ...], move_names: tuple[str, ...]
) -> dict[str, object]:
    move_to_index = {move: index for index, move in enumerate(move_names)}
    ids = [move_to_index[move] for move in path]
    move_counts = [0] * len(move_names)
    axis_counts = [0, 0, 0]
    layer_counts = [0] * 6
    direction_counts = [0, 0]
    same_axis_transitions = 0
    same_layer_transitions = 0
    inverse_transitions = 0
    max_axis_run = 0
    current_axis_run = 0
    previous_axis = ""
    for index, move in enumerate(path):
        move_counts[move_to_index[move]] += 1
        base = move.removeprefix("-")
        axis = base[0]
        axis_counts["frd".index(axis)] += 1
        layer_counts[int(base[1:])] += 1
        direction_counts[int(move.startswith("-"))] += 1
        if axis == previous_axis:
            current_axis_run += 1
        else:
            current_axis_run = 1
            previous_axis = axis
        max_axis_run = max(max_axis_run, current_axis_run)
        if index:
            previous = path[index - 1]
            previous_base = previous.removeprefix("-")
            same_axis_transitions += int(previous_base[0] == axis)
            same_layer_transitions += int(previous_base == base)
            inverse_transitions += int(
                previous_base == base
                and previous.startswith("-") != move.startswith("-")
            )
    return {
        "action_ids": ids,
        "axis_counts": axis_counts,
        "direction_counts": direction_counts,
        "inverse_transitions": inverse_transitions,
        "layer_counts": layer_counts,
        "max_axis_run": max_axis_run,
        "move_counts": move_counts,
        "same_axis_transitions": same_axis_transitions,
        "same_layer_transitions": same_layer_transitions,
    }


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def recover_one(
    *,
    pid: int,
    state: tuple[int, ...],
    condition: dict[str, object],
    puzzle: Cube666Puzzle,
    decomposition: object,
    solution_dir: Path,
    executable: Path,
    corner_cache: Path,
    cache_dir: Path,
    rayon_threads: int,
    timeout_seconds: float,
    force: bool,
) -> dict[str, object]:
    label = str(condition["label"])
    cached = cache_dir / f"{pid:04d}_{label}.json"
    if cached.exists() and not force:
        return json.loads(cached.read_text(encoding="utf-8"))

    anneal_seed = int(condition["anneal_seed"])
    anneal_steps = int(condition["anneal_steps"])
    anneal_keep_best = bool(condition["anneal_keep_best"])
    alpha = float(condition["alpha"])
    cache_tag = str(
        condition.get(
            "cache_tag",
            (
                f"cayley_v4_dfr_s{anneal_seed}_n{anneal_steps}_a{alpha:g}_"
                f"kb{int(anneal_keep_best)}_identity"
            ),
        )
    )
    corner_path = parse_path(
        (corner_cache / f"{pid:04d}.path.txt").read_text(encoding="utf-8")
    )
    environment = os.environ.copy()
    environment.update(
        {
            "KMC_ALPHA": f"{alpha:g}",
            "KMC_ANNEAL_KEEP_BEST": "1" if anneal_keep_best else "0",
            "KMC_ANNEAL_STEPS": str(anneal_steps),
            "KMC_BEAM": "4000",
            "KMC_CACHE_TAG": cache_tag,
            "KMC_CANDIDATE_CAP": "1000",
            "KMC_CORNER_PATH": ".".join(corner_path),
            "KMC_SEED": str(anneal_seed),
            "KMC_SOLUTION_DIR": str(solution_dir.resolve()),
            "KMC_STOP_AFTER_ROUGH": "1",
            "RAYON_NUM_THREADS": str(rayon_threads),
        }
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [str(executable)],
        input=solver_input(puzzle, str(pid), state),
        text=True,
        capture_output=True,
        cwd=solution_dir,
        env=environment,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"PID {pid} {label}: rough solve exited {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"PID {pid} {label}: no rough path returned")
    rough_path = parse_path(lines[-1])
    rough_state = puzzle.apply_path(state, rough_path)
    corner_mismatches = sum(
        rough_state[position] != puzzle.solved_state[position]
        for position in decomposition.corner_orbit
    )
    report = residual_report(rough_state, puzzle.solved_state, decomposition)
    if corner_mismatches or not report.all_even:
        raise AssertionError(
            f"PID {pid} {label}: invalid rough state; corners={corner_mismatches}, "
            f"parity={report.parity_vector}"
        )
    cluster_permutations = state_cluster_permutations(
        rough_state, puzzle.solved_state, decomposition
    )
    cluster_rows = []
    for cluster, permutation in zip(report.clusters, cluster_permutations, strict=True):
        cluster_rows.append(
            {
                "cycles": [list(cycle) for cycle in cluster.cycles],
                "misplaced": sum(index != value for index, value in enumerate(permutation)),
                "three_cycles": cluster.unrestricted_three_cycles,
            }
        )
    row = {
        "alpha": alpha,
        "anneal_keep_best": anneal_keep_best,
        "anneal_seed": anneal_seed,
        "anneal_steps": anneal_steps,
        "cluster_permutations": [list(values) for values in cluster_permutations],
        "clusters": cluster_rows,
        "condition": label,
        "elapsed_seconds": elapsed,
        "mismatches": sum(
            left != right
            for left, right in zip(rough_state, puzzle.solved_state, strict=True)
        ),
        "path_features": path_context_features(rough_path, puzzle.move_names),
        "pid": pid,
        "residual_three_cycles": report.unrestricted_three_cycles,
        "rough_moves": len(rough_path),
        "rough_path": list(rough_path),
        "rough_path_digest": digest_tokens(rough_path),
        "rough_reduced_moves": len(reduce_commuting_quarter_turn_path(rough_path)),
        "rough_state_digest": hashlib.sha256(
            np.asarray(rough_state, dtype=np.uint8).tobytes()
        ).hexdigest(),
    }
    atomic_write(cached, row)
    return row


def main() -> None:
    args = parse_args()
    excluded: set[int] = set()
    for report_path in args.exclude_report:
        excluded.update(
            int(pid)
            for pid in json.loads(report_path.read_text(encoding="utf-8"))["pids"]
        )
    lengths = load_lengths(args.incumbent)
    pids = stratified_pids(lengths, excluded, args.count)
    conditions = list(CONDITIONS)
    if args.conditions:
        requested = {value.strip() for value in args.conditions.split(",") if value.strip()}
        conditions = [row for row in conditions if str(row["label"]) in requested]
    if not conditions:
        raise ValueError("condition selection is empty")

    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    states = {
        int(pid): state for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    executable = args.solution_dir / "target/release/solve_cube_beam.exe"
    corner_cache = PROJECT / "cube666/artifacts/corner_paths_cayley_v1"
    cache_dir = args.out_dir / "rough_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(pid, condition) for pid in pids for condition in conditions]
    rows = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                recover_one,
                pid=pid,
                state=states[pid],
                condition=condition,
                puzzle=puzzle,
                decomposition=decomposition,
                solution_dir=args.solution_dir,
                executable=executable,
                corner_cache=corner_cache,
                cache_dir=cache_dir,
                rayon_threads=args.rayon_threads,
                timeout_seconds=args.timeout_seconds,
                force=args.force,
            ): (pid, str(condition["label"]))
            for pid, condition in jobs
        }
        for completed_count, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            pid, label = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"rough={completed_count}/{len(jobs)} pid={pid} condition={label} "
                f"len={row['rough_moves']} residual={row['residual_three_cycles']}",
                flush=True,
            )
    rows.sort(key=lambda row: (int(row["pid"]), str(row["condition"])))
    elapsed = time.perf_counter() - started
    atomic_write(args.out_dir / "rough_rows.json", rows)
    report = {
        "conditions": [dict(condition) for condition in conditions],
        "elapsed_seconds": elapsed,
        "incumbent": str(args.incumbent),
        "pids": pids,
        "rough_candidates": len(rows),
        "rough_elapsed_seconds_sum": sum(float(row["elapsed_seconds"]) for row in rows),
        "rough_length_mean": statistics.fmean(int(row["rough_moves"]) for row in rows),
    }
    atomic_write(args.out_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
