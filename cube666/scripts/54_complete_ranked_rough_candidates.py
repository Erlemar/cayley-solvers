"""Run the native exact insertion beam only on pre-ranked rough candidates."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranked",
        type=Path,
        default=PROJECT / "cube666/reports/path_context_online24_ranked_v1.json",
    )
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=PROJECT / "submissions/cube666_path_context_ranker_strictwin_v1.csv",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=PROJECT / "external/third_party/kmcoders_santa2023/solution",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666/results/path_context_online24_selected_b20000",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=PROJECT / "submissions/path_context_online24_candidates_v1.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "cube666/reports/path_context_online24_completed_v1.json",
    )
    parser.add_argument("--beam", type=int, default=20_000)
    parser.add_argument("--candidate-cap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rayon-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--pids", help="optional comma-separated PID subset")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def load_submission(path: Path) -> dict[int, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): parse_path(row["path"])
            for row in csv.DictReader(handle)
        }


def solver_input(puzzle: Cube666Puzzle, state_id: str, state: tuple[int, ...]) -> str:
    positive_moves = tuple(f"{axis}{layer}" for axis in "dfr" for layer in range(6))
    tokens = [state_id, "cube_6/6/6", str(puzzle.size), str(len(positive_moves)), "0"]
    for name in positive_moves:
        tokens.append(name)
        tokens.extend(str(value) for value in puzzle.generators[name])
    tokens.extend(str(value) for value in puzzle.solved_state)
    tokens.extend(str(value) for value in state)
    return " ".join(tokens) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def complete_one(
    *,
    selection: dict[str, object],
    state: tuple[int, ...],
    puzzle: Cube666Puzzle,
    solution_dir: Path,
    executable: Path,
    corner_cache: Path,
    out_dir: Path,
    beam: int,
    candidate_cap: int,
    rayon_threads: int,
    timeout_seconds: float,
    force: bool,
) -> dict[str, object]:
    pid = int(selection["pid"])
    path_file = out_dir / f"{pid:04d}.path.txt"
    metadata_file = out_dir / f"{pid:04d}.json"
    if path_file.exists() and metadata_file.exists() and not force:
        path = parse_path(path_file.read_text(encoding="utf-8"))
        if puzzle.apply_path(state, path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid}: cached completion failed replay")
        return json.loads(metadata_file.read_text(encoding="utf-8"))

    rough_path = tuple(str(move) for move in selection["rough_path"])
    corner_path = parse_path(
        (corner_cache / f"{pid:04d}.path.txt").read_text(encoding="utf-8")
    )
    environment = os.environ.copy()
    environment.update(
        {
            "KMC_BEAM": str(beam),
            "KMC_CANDIDATE_CAP": str(candidate_cap),
            "KMC_CORNER_PATH": ".".join(corner_path),
            "KMC_ROUGH_PATH": ".".join(rough_path),
            "KMC_SOLUTION_DIR": str(solution_dir.resolve()),
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
    atomic_write_text(out_dir / f"{pid:04d}.stderr.log", completed.stderr)
    atomic_write_text(out_dir / f"{pid:04d}.stdout.log", completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"PID {pid}: exact completion exited {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"PID {pid}: exact completion returned no path")
    path = reduce_commuting_quarter_turn_path(parse_path(lines[-1]))
    if puzzle.apply_path(state, path) != puzzle.solved_state:
        raise AssertionError(f"PID {pid}: completed path failed exact replay")
    atomic_write_text(path_file, ".".join(path) + "\n")
    metadata = {
        "beam": beam,
        "candidate_cap": candidate_cap,
        "condition": selection["condition"],
        "elapsed_seconds": elapsed,
        "model_prediction": selection["model_prediction"],
        "moves": len(path),
        "pid": pid,
        "proxy_condition": selection["proxy_condition"],
        "replay_verified": True,
        "residual_three_cycles": selection["residual_three_cycles"],
        "rough_moves": selection["rough_moves"],
        "rough_path_digest": selection["rough_path_digest"],
    }
    atomic_write_text(
        metadata_file, json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main() -> None:
    args = parse_args()
    if args.beam <= 0 or args.candidate_cap <= 0 or args.workers <= 0:
        raise ValueError("beam, candidate-cap, and workers must be positive")
    ranked = json.loads(args.ranked.read_text(encoding="utf-8"))
    selections = list(ranked["selections"])
    if args.pids:
        requested = {int(value) for value in args.pids.split(",") if value.strip()}
        selections = [row for row in selections if int(row["pid"]) in requested]
    if not selections:
        raise ValueError("selection set is empty")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    states = {
        int(pid): state for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    incumbent = load_submission(args.incumbent)
    executable = args.solution_dir / "target/release/solve_cube_beam.exe"
    corner_cache = PROJECT / "cube666/artifacts/corner_paths_cayley_v1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                complete_one,
                selection=selection,
                state=states[int(selection["pid"])],
                puzzle=puzzle,
                solution_dir=args.solution_dir,
                executable=executable,
                corner_cache=corner_cache,
                out_dir=args.out_dir,
                beam=args.beam,
                candidate_cap=args.candidate_cap,
                rayon_threads=args.rayon_threads,
                timeout_seconds=args.timeout_seconds,
                force=args.force,
            ): int(selection["pid"])
            for selection in selections
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            pid = futures[future]
            print(
                f"completed={count}/{len(selections)} pid={pid} "
                f"condition={row['condition']} moves={row['moves']} "
                f"elapsed={row['elapsed_seconds']:.2f}s",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["pid"]))

    args.candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.candidate_csv.with_suffix(args.candidate_csv.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        for row in rows:
            pid = int(row["pid"])
            path = parse_path((args.out_dir / f"{pid:04d}.path.txt").read_text(encoding="utf-8"))
            writer.writerow({"initial_state_id": pid, "path": ".".join(path)})
    temporary.replace(args.candidate_csv)

    selected_total = sum(int(row["moves"]) for row in rows)
    incumbent_total = sum(len(incumbent[int(row["pid"])]) for row in rows)
    report = {
        "beam": args.beam,
        "candidate_csv": str(args.candidate_csv),
        "completed_pids": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "executable": str(executable),
        "executable_sha256": sha256_file(executable),
        "incumbent_total": incumbent_total,
        "model_selected_improvement_before_strict_merge": incumbent_total - selected_total,
        "model_selected_total": selected_total,
        "per_pid": rows,
        "replay_verified": len(rows),
        "solver_seconds_sum": sum(float(row["elapsed_seconds"]) for row in rows),
        "strict_merge_savings": sum(
            max(len(incumbent[int(row["pid"])]) - int(row["moves"]), 0)
            for row in rows
        ),
        "strict_wins": sum(
            int(row["moves"]) < len(incumbent[int(row["pid"])]) for row in rows
        ),
        "would_regress": sum(
            int(row["moves"]) > len(incumbent[int(row["pid"])]) for row in rows
        ),
    }
    atomic_write_text(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "per_pid"}, indent=2))


if __name__ == "__main__":
    main()
