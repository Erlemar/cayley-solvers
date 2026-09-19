"""Recover KMC rough words and pair them with replay-verified final path costs.

This is the first gate for an insertion-aware model.  It deliberately labels
the same pre-finisher object that a future model will rank: the complete rough
word, its exact residual state, and the final primitive length achieved by the
native exact insertion solver.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.classical import (  # noqa: E402
    build_decomposition,
    residual_report,
)
from cube666.macros import (  # noqa: E402
    reduce_commuting_quarter_turn_path,
    state_cluster_permutations,
)
from cube666.puzzle import Cube666Puzzle  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"
DEFAULT_SOLUTION = (
    PROJECT / "external" / "third_party" / "kmcoders_santa2023" / "solution"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--solution-dir", type=Path, default=DEFAULT_SOLUTION)
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=PROJECT / "cube666" / "reports" / "KMC_PARAM_GATE16.json",
    )
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=PROJECT / "submissions" / "cube666_model_hybrid_strictwin_v1.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666" / "training" / "path_context_oracle_gate16_v1",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rayon-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--pids", help="optional comma-separated PID subset")
    parser.add_argument("--conditions", help="optional comma-separated condition labels")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_tokens(tokens: tuple[str, ...]) -> str:
    return hashlib.sha256(".".join(tokens).encode("utf-8")).hexdigest()


def solver_input(puzzle: Cube666Puzzle, state_id: str, state: tuple[int, ...]) -> str:
    positive_moves = tuple(f"{axis}{layer}" for axis in "dfr" for layer in range(6))
    if set(positive_moves) != {
        name for name in puzzle.move_names if not name.startswith("-")
    }:
        raise ValueError("unexpected cube666 generator alphabet")
    tokens: list[str] = [
        state_id,
        "cube_6/6/6",
        str(puzzle.size),
        str(len(positive_moves)),
        "0",
    ]
    for name in positive_moves:
        tokens.append(name)
        tokens.extend(str(value) for value in puzzle.generators[name])
    tokens.extend(str(value) for value in puzzle.solved_state)
    tokens.extend(str(value) for value in state)
    return " ".join(tokens) + "\n"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_submission_lengths(path: Path) -> dict[int, int]:
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): len(parse_path(row["path"]))
            for row in csv.DictReader(handle)
        }


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
        axis_id = "frd".index(axis)
        layer = int(base[1:])
        axis_counts[axis_id] += 1
        layer_counts[layer] += 1
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


def metadata_for_condition(directory: Path, pid: int) -> dict[str, object]:
    return json.loads((directory / f"{pid:04d}.json").read_text(encoding="utf-8"))


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
    directory = Path(str(condition["directory"]))
    metadata = metadata_for_condition(directory, pid)
    anneal_seed = int(metadata["anneal_seed"])
    anneal_steps = int(metadata["anneal_steps"])
    anneal_keep_best = bool(metadata.get("anneal_keep_best", False))
    alpha = float(metadata["alpha"])
    beam = int(metadata["beam"])
    candidate_cap = int(metadata.get("candidate_cap", 1000))
    cache_key = (
        f"{pid:04d}_{label}_s{anneal_seed}_n{anneal_steps}_a{alpha:g}_"
        f"kb{int(anneal_keep_best)}"
    )
    cached = cache_dir / f"{cache_key}.json"
    if cached.exists() and not force:
        recovered = json.loads(cached.read_text(encoding="utf-8"))
    else:
        corner_path = parse_path(
            (corner_cache / f"{pid:04d}.path.txt").read_text(encoding="utf-8")
        )
        environment = os.environ.copy()
        environment["KMC_BEAM"] = str(beam)
        environment["KMC_CACHE_TAG"] = str(
            condition.get(
                "cache_tag",
                (
                    f"cayley_v4_dfr_s{anneal_seed}_n{anneal_steps}_a{alpha:g}_"
                    f"kb{int(anneal_keep_best)}_identity"
                ),
            )
        )
        environment["KMC_CORNER_PATH"] = ".".join(corner_path)
        environment["KMC_SEED"] = str(anneal_seed)
        environment["KMC_ANNEAL_STEPS"] = str(anneal_steps)
        environment["KMC_ANNEAL_KEEP_BEST"] = "1" if anneal_keep_best else "0"
        environment["KMC_ALPHA"] = f"{alpha:g}"
        environment["KMC_CANDIDATE_CAP"] = str(candidate_cap)
        environment["KMC_SOLUTION_DIR"] = str(solution_dir.resolve())
        environment["KMC_STOP_AFTER_ROUGH"] = "1"
        environment["RAYON_NUM_THREADS"] = str(rayon_threads)
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
                f"PID {pid} {label}: rough recovery exited {completed.returncode}: "
                f"{completed.stderr[-1000:]}"
            )
        stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            raise RuntimeError(f"PID {pid} {label}: no rough path returned")
        rough_path = parse_path(stdout_lines[-1])
        rough_state = puzzle.apply_path(state, rough_path)
        corner_mismatches = sum(
            rough_state[position] != puzzle.solved_state[position]
            for position in decomposition.corner_orbit
        )
        report = residual_report(rough_state, puzzle.solved_state, decomposition)
        if corner_mismatches or not report.all_even:
            raise AssertionError(
                f"PID {pid} {label}: invalid rough state; "
                f"corners={corner_mismatches}, parity={report.parity_vector}"
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
        recovered = {
            "elapsed_seconds": elapsed,
            "rough_path": list(rough_path),
            "rough_state": list(rough_state),
            "rough_path_digest": digest_tokens(rough_path),
            "rough_state_digest": hashlib.sha256(
                np.asarray(rough_state, dtype=np.uint8).tobytes()
            ).hexdigest(),
            "rough_moves": len(rough_path),
            "rough_reduced_moves": len(reduce_commuting_quarter_turn_path(rough_path)),
            "residual_three_cycles": report.unrestricted_three_cycles,
            "mismatches": sum(
                left != right
                for left, right in zip(rough_state, puzzle.solved_state, strict=True)
            ),
            "cluster_permutations": [list(values) for values in cluster_permutations],
            "clusters": cluster_rows,
            "path_features": path_context_features(rough_path, puzzle.move_names),
        }
        atomic_write(cached, json.dumps(recovered, indent=2, sort_keys=True) + "\n")

    final_path_file = directory / f"{pid:04d}.path.txt"
    final_path = parse_path(final_path_file.read_text(encoding="utf-8"))
    reduced_final_path = reduce_commuting_quarter_turn_path(final_path)
    if puzzle.apply_path(state, reduced_final_path) != puzzle.solved_state:
        raise AssertionError(f"PID {pid} {label}: final label path failed exact replay")
    return {
        **recovered,
        "alpha": alpha,
        "anneal_keep_best": anneal_keep_best,
        "anneal_seed": anneal_seed,
        "anneal_steps": anneal_steps,
        "beam": beam,
        "candidate_cap": candidate_cap,
        "condition": label,
        "final_moves": len(reduced_final_path),
        "final_path_sha256": sha256_file(final_path_file),
        "pid": pid,
        "source_directory": str(directory),
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.rayon_threads <= 0:
        raise ValueError("workers and rayon-threads must be positive")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    decomposition = build_decomposition(puzzle.generators)
    states = {
        int(pid): state
        for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    pids = [int(pid) for pid in gate["pids"]]
    conditions = list(gate["conditions"])
    if args.pids:
        selected_pids = {int(value) for value in args.pids.split(",") if value.strip()}
        pids = [pid for pid in pids if pid in selected_pids]
    if args.conditions:
        selected_conditions = {
            value.strip() for value in args.conditions.split(",") if value.strip()
        }
        conditions = [
            condition
            for condition in conditions
            if str(condition["label"]) in selected_conditions
        ]
    if not pids or not conditions:
        raise ValueError("PID/condition selection is empty")
    incumbent_lengths = load_submission_lengths(args.incumbent)
    executable = args.solution_dir / "target" / "release" / "solve_cube_beam.exe"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    corner_cache = PROJECT / "cube666" / "artifacts" / "corner_paths_cayley_v1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "rough_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(pid, condition) for pid in pids for condition in conditions]
    rows: list[dict[str, object]] = []
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
                f"recovered={completed_count}/{len(jobs)} pid={pid} condition={label} "
                f"rough={row['rough_moves']} residual={row['residual_three_cycles']} "
                f"final={row['final_moves']}",
                flush=True,
            )

    rows.sort(key=lambda row: (int(row["pid"]), str(row["condition"])))
    atomic_write(
        args.out_dir / "condition_rows.json",
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
    )

    unique: dict[tuple[int, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row["pid"]), str(row["rough_path_digest"]))
        current = unique.get(key)
        if current is None:
            current = {
                key_name: value
                for key_name, value in row.items()
                if key_name not in {
                    "beam",
                    "candidate_cap",
                    "condition",
                    "final_moves",
                    "final_path_sha256",
                    "source_directory",
                }
            }
            current["conditions"] = []
            current["final_labels"] = []
            unique[key] = current
        current["conditions"].append(str(row["condition"]))
        current["final_labels"].append(int(row["final_moves"]))
    unique_rows = []
    for row in unique.values():
        labels = list(row["final_labels"])
        row["best_final_moves"] = min(labels)
        row["mean_final_moves"] = statistics.fmean(labels)
        unique_rows.append(row)
    unique_rows.sort(key=lambda row: (int(row["pid"]), str(row["rough_path_digest"])))
    atomic_write(
        args.out_dir / "unique_rough_rows.json",
        json.dumps(unique_rows, indent=2, sort_keys=True) + "\n",
    )

    per_pid = []
    condition_by_pid: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    unique_by_pid: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        condition_by_pid[int(row["pid"])].append(row)
    for row in unique_rows:
        unique_by_pid[int(row["pid"])].append(row)
    for pid in pids:
        control_rows = [
            int(row["final_moves"])
            for row in condition_by_pid[pid]
            if row["condition"] == gate["control"]
        ]
        control = control_rows[0] if control_rows else incumbent_lengths[pid]
        oracle = min(int(row["best_final_moves"]) for row in unique_by_pid[pid])
        incumbent = incumbent_lengths[pid]
        per_pid.append(
            {
                "control_moves": control,
                "incumbent_moves": incumbent,
                "oracle_moves": oracle,
                "oracle_saving_vs_control": control - oracle,
                "oracle_saving_vs_incumbent": incumbent - oracle,
                "pid": pid,
                "unique_rough_paths": len(unique_by_pid[pid]),
            }
        )
    report = {
        "condition_rows": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "gate_report": str(args.gate_report),
        "gate_report_sha256": sha256_file(args.gate_report),
        "incumbent": str(args.incumbent),
        "incumbent_sha256": sha256_file(args.incumbent),
        "oracle_total": sum(int(row["oracle_moves"]) for row in per_pid),
        "control_total": sum(int(row["control_moves"]) for row in per_pid),
        "incumbent_total": sum(int(row["incumbent_moves"]) for row in per_pid),
        "oracle_saving_vs_control": sum(
            int(row["oracle_saving_vs_control"]) for row in per_pid
        ),
        "oracle_saving_vs_incumbent": sum(
            int(row["oracle_saving_vs_incumbent"]) for row in per_pid
        ),
        "per_pid": per_pid,
        "pids": pids,
        "replay_verified_final_labels": len(rows),
        "unique_rough_rows": len(unique_rows),
    }
    atomic_write(
        args.out_dir / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "per_pid"}, indent=2))


if __name__ == "__main__":
    main()
