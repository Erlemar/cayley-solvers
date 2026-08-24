"""Relabel every rough candidate with one uniform native insertion-beam budget."""

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


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_corpus32_v1/"
            "unique_rough_rows_insertion_context.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "cube666/results/path_context_corpus32_uniform_b20000",
    )
    parser.add_argument(
        "--out-dataset",
        type=Path,
        default=(
            PROJECT
            / "cube666/training/path_context_corpus32_v1/"
            "unique_rough_rows_uniform_b20000.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT / "cube666/reports/path_context_corpus32_uniform_b20000.json",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT / "cayley-py-666-cube"
    )
    parser.add_argument(
        "--solution-dir",
        type=Path,
        default=PROJECT / "external/third_party/kmcoders_santa2023/solution",
    )
    parser.add_argument("--beam", type=int, default=20_000)
    parser.add_argument("--candidate-cap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rayon-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--seed-completion-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Reuse matching replay-verified beam completions from a prior result "
            "directory; repeatable."
        ),
    )
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def condition_label(row: dict[str, object]) -> str:
    direct = row.get("condition")
    if direct is not None:
        return str(direct)
    conditions = [str(value) for value in row.get("conditions", ["unknown"])]
    if "control" in conditions:
        return "control"
    return conditions[0]


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


def seed_completion_cache(
    *,
    rows: list[dict[str, object]],
    states: dict[int, tuple[int, ...]],
    puzzle: Cube666Puzzle,
    directories: list[Path],
    out_dir: Path,
    beam: int,
    candidate_cap: int,
) -> int:
    rows_by_key = {
        (int(row["pid"]), str(row["rough_path_digest"])): row for row in rows
    }
    seeded = 0
    for directory in directories:
        for source_metadata_file in sorted(directory.glob("*.json")):
            source = json.loads(source_metadata_file.read_text(encoding="utf-8"))
            if int(source.get("beam", -1)) != beam:
                continue
            if int(source.get("candidate_cap", -1)) != candidate_cap:
                continue
            pid = int(source["pid"])
            digest = str(source["rough_path_digest"])
            row = rows_by_key.get((pid, digest))
            if row is None:
                continue
            source_path_file = directory / f"{pid:04d}.path.txt"
            if not source_path_file.exists():
                raise FileNotFoundError(source_path_file)
            path = parse_path(source_path_file.read_text(encoding="utf-8"))
            if len(path) != int(source["moves"]):
                raise AssertionError(
                    f"PID {pid} {digest[:8]}: seeded metadata/path length mismatch"
                )
            if puzzle.apply_path(states[pid], path) != puzzle.solved_state:
                raise AssertionError(
                    f"PID {pid} {digest[:8]}: seeded completion failed replay"
                )
            stem = f"{pid:04d}_{digest[:16]}"
            path_file = out_dir / f"{stem}.path.txt"
            metadata_file = out_dir / f"{stem}.json"
            if path_file.exists() and metadata_file.exists():
                continue
            atomic_write_text(path_file, ".".join(path) + "\n")
            metadata = {
                "beam": beam,
                "candidate_cap": candidate_cap,
                "condition": condition_label(row),
                "elapsed_seconds": float(source.get("elapsed_seconds", 0.0)),
                "original_best_final_moves": row.get("best_final_moves"),
                "path_file": str(path_file),
                "pid": pid,
                "replay_verified": True,
                "rough_path_digest": digest,
                "seeded_from": str(directory),
                "uniform_final_moves": len(path),
            }
            atomic_write_text(
                metadata_file, json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )
            seeded += 1
    return seeded


def complete_one(
    *,
    index: int,
    row: dict[str, object],
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
) -> tuple[int, dict[str, object]]:
    pid = int(row["pid"])
    digest = str(row["rough_path_digest"])[:16]
    stem = f"{pid:04d}_{digest}"
    path_file = out_dir / f"{stem}.path.txt"
    metadata_file = out_dir / f"{stem}.json"
    if path_file.exists() and metadata_file.exists() and not force:
        path = parse_path(path_file.read_text(encoding="utf-8"))
        if puzzle.apply_path(state, path) != puzzle.solved_state:
            raise AssertionError(f"PID {pid} {digest}: cached label failed replay")
        return index, json.loads(metadata_file.read_text(encoding="utf-8"))

    rough_path = tuple(str(move) for move in row["rough_path"])
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
    atomic_write_text(out_dir / f"{stem}.stderr.log", completed.stderr)
    atomic_write_text(out_dir / f"{stem}.stdout.log", completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"PID {pid} {digest}: uniform completion exited {completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"PID {pid} {digest}: no completed path returned")
    path = reduce_commuting_quarter_turn_path(parse_path(lines[-1]))
    if puzzle.apply_path(state, path) != puzzle.solved_state:
        raise AssertionError(f"PID {pid} {digest}: uniform label failed exact replay")
    atomic_write_text(path_file, ".".join(path) + "\n")
    metadata = {
        "beam": beam,
        "candidate_cap": candidate_cap,
        "condition": condition_label(row),
        "elapsed_seconds": elapsed,
        "original_best_final_moves": row.get("best_final_moves"),
        "path_file": str(path_file),
        "pid": pid,
        "replay_verified": True,
        "rough_path_digest": row["rough_path_digest"],
        "uniform_final_moves": len(path),
    }
    atomic_write_text(
        metadata_file, json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return index, metadata


def main() -> None:
    args = parse_args()
    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("max-rows must be positive")
        rows = rows[: args.max_rows]
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    states = {
        int(pid): state for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")
    }
    executable = args.solution_dir / "target/release/solve_cube_beam.exe"
    corner_cache = PROJECT / "cube666/artifacts/corner_paths_cayley_v1"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeded = seed_completion_cache(
        rows=rows,
        states=states,
        puzzle=puzzle,
        directories=args.seed_completion_dir,
        out_dir=args.out_dir,
        beam=args.beam,
        candidate_cap=args.candidate_cap,
    )
    if seeded:
        print(f"seeded={seeded}/{len(rows)} exact completions", flush=True)
    started = time.perf_counter()
    labels: list[dict[str, object] | None] = [None] * len(rows)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                complete_one,
                index=index,
                row=row,
                state=states[int(row["pid"])],
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
            ): (int(row["pid"]), str(row["rough_path_digest"])[:8])
            for index, row in enumerate(rows)
        }
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            index, metadata = future.result()
            labels[index] = metadata
            pid, digest = futures[future]
            print(
                f"labeled={count}/{len(rows)} pid={pid} rough={digest} "
                f"moves={metadata['uniform_final_moves']} "
                f"elapsed={metadata['elapsed_seconds']:.2f}s",
                flush=True,
            )
    if any(label is None for label in labels):
        raise AssertionError("missing uniform label")
    enriched = []
    for row, label in zip(rows, labels, strict=True):
        enriched.append({**row, **label})
    atomic_write_text(
        args.out_dataset,
        json.dumps(enriched, indent=2, sort_keys=True) + "\n",
    )

    deltas_by_condition: defaultdict[str, list[int]] = defaultdict(list)
    for row in enriched:
        condition = condition_label(row)
        if row.get("original_best_final_moves") is not None:
            deltas_by_condition[condition].append(
                int(row["uniform_final_moves"]) - int(row["original_best_final_moves"])
            )
    report = {
        "beam": args.beam,
        "candidate_cap": args.candidate_cap,
        "dataset": str(args.dataset),
        "elapsed_seconds": time.perf_counter() - started,
        "executable_sha256": sha256_file(executable),
        "out_dataset": str(args.out_dataset),
        "rows": len(enriched),
        "seeded_completions": seeded,
        "solver_seconds_sum": sum(float(row["elapsed_seconds"]) for row in enriched),
        "uniform_minus_original_by_condition": {
            condition: {
                "count": len(values),
                "mean": statistics.fmean(values),
                "sum": sum(values),
            }
            for condition, values in sorted(deltas_by_condition.items())
        },
    }
    atomic_write_text(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
