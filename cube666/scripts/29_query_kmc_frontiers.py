"""Solve replay-verified DAgger frontier states with the callable KMC solver.

The input frontiers already have solved corners and even cluster parity, so the
external solver is explicitly given an empty corner path.  Every returned path
is replayed on the complete 216-sticker state before it is written as a teacher
artifact.  Existing verified outputs are reused, making the query batch resumable.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "cube_nnn" / "src"))

from cube666.frames import FrameTransform  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.symmetry import build_symmetries  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"
DEFAULT_SOLUTION_DIR = PROJECT / "external" / "third_party" / "kmcoders_santa2023" / "solution"


@dataclass(frozen=True)
class QueryResult:
    index: int
    pid: int
    query_id: str
    path: tuple[str, ...]
    elapsed_seconds: float
    reused: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--frontier-dir", type=Path, required=True)
    parser.add_argument("--solution-dir", type=Path, default=DEFAULT_SOLUTION_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--indices", help="comma-separated zero-based frontier indices")
    parser.add_argument(
        "--indices-file",
        type=Path,
        help="text file containing comma- or whitespace-separated frontier indices",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--beam", type=int, default=1000)
    parser.add_argument("--candidate-cap", type=int, default=1000)
    parser.add_argument("--rayon-threads", type=int, default=4)
    parser.add_argument("--anneal-seed", type=int, default=89043280)
    parser.add_argument("--anneal-steps", type=int, default=1_000_000)
    parser.add_argument("--anneal-keep-best", action="store_true")
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--cache-tag", default="cayley_dagger_pilot_v1")
    parser.add_argument("--symmetry-index", type=int)
    parser.add_argument(
        "--use-prefix-context",
        action="store_true",
        help=(
            "solve from the original scramble and supply each row's setup+macro "
            "path as KMC_CORNER_PATH insertion context"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def solver_input(puzzle: Cube666Puzzle, state_id: str, state: tuple[int, ...]) -> str:
    available_positive = tuple(name for name in puzzle.move_names if not name.startswith("-"))
    positive_moves = tuple(f"{axis}{layer}" for axis in "dfr" for layer in range(6))
    if set(available_positive) != set(positive_moves):
        raise ValueError("KMC requires the 18 d/f/r positive generators")
    tokens: list[str] = [state_id, "cube_6/6/6", str(puzzle.size), str(len(positive_moves)), "0"]
    for name in positive_moves:
        tokens.append(name)
        tokens.extend(str(value) for value in puzzle.generators[name])
    tokens.extend(str(value) for value in puzzle.solved_state)
    tokens.extend(str(value) for value in state)
    return " ".join(tokens) + "\n"


def replay_verify(
    puzzle: Cube666Puzzle,
    query_id: str,
    state: tuple[int, ...],
    path: tuple[str, ...],
) -> None:
    unknown = sorted(set(path).difference(puzzle.generators))
    if unknown:
        raise ValueError(f"{query_id}: unknown moves {unknown}")
    final = puzzle.apply_path(state, path)
    if final != puzzle.solved_state:
        mismatches = sum(
            left != right for left, right in zip(final, puzzle.solved_state, strict=True)
        )
        raise ValueError(f"{query_id}: full replay failed with {mismatches} mismatches")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def query_one(
    *,
    index: int,
    pid: int,
    query_id: str,
    state: tuple[int, ...],
    solver_state: tuple[int, ...],
    corner_context: tuple[str, ...],
    puzzle: Cube666Puzzle,
    frame: FrameTransform,
    solution_dir: Path,
    executable: Path,
    out_dir: Path,
    beam: int,
    candidate_cap: int,
    rayon_threads: int,
    anneal_seed: int,
    anneal_steps: int,
    anneal_keep_best: bool,
    alpha: float,
    timeout_seconds: float,
    cache_tag: str,
    force: bool,
) -> QueryResult:
    path_file = out_dir / "paths" / f"{query_id}.path.txt"
    metadata_file = out_dir / "metadata" / f"{query_id}.json"
    if path_file.exists() and not force:
        path = tuple(token for token in path_file.read_text(encoding="utf-8").strip().split(".") if token)
        replay_verify(puzzle, query_id, state, path)
        elapsed = 0.0
        if metadata_file.exists():
            elapsed = float(json.loads(metadata_file.read_text(encoding="utf-8"))["elapsed_seconds"])
        return QueryResult(index, pid, query_id, path, elapsed, True)

    environment = os.environ.copy()
    environment["KMC_BEAM"] = str(beam)
    environment["KMC_CANDIDATE_CAP"] = str(candidate_cap)
    environment["KMC_SEED"] = str(anneal_seed)
    environment["KMC_ANNEAL_STEPS"] = str(anneal_steps)
    environment["KMC_ANNEAL_KEEP_BEST"] = "1" if anneal_keep_best else "0"
    environment["KMC_ALPHA"] = str(alpha)
    environment["RAYON_NUM_THREADS"] = str(rayon_threads)
    environment["KMC_CORNER_PATH"] = ".".join(corner_context)
    context_digest = hashlib.sha256(
        ".".join(corner_context).encode("utf-8")
    ).hexdigest()[:16]
    environment["KMC_CACHE_TAG"] = (
        f"{cache_tag}_s{anneal_seed}_n{anneal_steps}_a{alpha:g}_"
        f"kb{int(anneal_keep_best)}_{frame.label}_ctx{context_digest}"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [str(executable)],
        input=solver_input(puzzle, str(pid), solver_state),
        text=True,
        capture_output=True,
        cwd=solution_dir,
        env=environment,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    atomic_text(out_dir / "logs" / f"{query_id}.stdout.log", completed.stdout)
    atomic_text(out_dir / "logs" / f"{query_id}.stderr.log", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"{query_id}: KMC exited {completed.returncode}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{query_id}: KMC returned no path")
    framed_path = tuple(token for token in lines[-1].split(".") if token)
    path = frame.restore_path(framed_path)
    replay_verify(puzzle, query_id, state, path)
    atomic_text(path_file, ".".join(path) + "\n")
    metadata = {
        "alpha": alpha,
        "anneal_keep_best": anneal_keep_best,
        "anneal_seed": anneal_seed,
        "anneal_steps": anneal_steps,
        "beam": beam,
        "candidate_cap": candidate_cap,
        "elapsed_seconds": round(elapsed, 6),
        "index": index,
        "moves": len(path),
        "pid": pid,
        "query_id": query_id,
        "rayon_threads": rayon_threads,
        "replay_verified": True,
        "symmetry_index": None if frame.label == "identity" else int(frame.label.removeprefix("sym")),
        "context_moves": len(corner_context),
        "context_digest": context_digest,
    }
    atomic_text(metadata_file, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return QueryResult(index, pid, query_id, path, elapsed, False)


def main() -> None:
    args = parse_args()
    if min(args.workers, args.beam, args.candidate_cap, args.rayon_threads, args.anneal_steps) <= 0:
        raise ValueError("workers, beam, candidate-cap, rayon-threads, and anneal-steps must be positive")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    if args.symmetry_index is None:
        frame = FrameTransform(label="identity", move_names=puzzle.move_names)
    else:
        geometric_cube = NCube.from_puzzle_info(args.data_dir / "puzzle_info.json")
        symmetry_table = build_symmetries(geometric_cube)
        if args.symmetry_index < 0 or args.symmetry_index >= len(symmetry_table.perms):
            raise ValueError("symmetry-index is outside the verified frame table")
        frame = FrameTransform(
            label=f"sym{args.symmetry_index:02d}",
            move_names=puzzle.move_names,
            sticker_permutation=symmetry_table.perms[args.symmetry_index],
            move_to_transformed=symmetry_table.relabel[args.symmetry_index],
        )
    with np.load(args.frontier_dir / "frontiers.npz", allow_pickle=False) as payload:
        full_states = payload["full_states"].astype(np.uint8, copy=False)
        source_pids = payload["source_pids"].astype(np.int32, copy=False)
    frontier_report = json.loads((args.frontier_dir / "report.json").read_text(encoding="utf-8"))
    rows = frontier_report["rows"]
    states_by_pid = (
        {
            int(state_id): state
            for state_id, state in puzzle.iter_test_states(args.data_dir / "test.csv")
        }
        if args.use_prefix_context
        else {}
    )
    if full_states.shape != (len(rows), puzzle.size) or source_pids.shape != (len(rows),):
        raise ValueError("frontier artifact shapes do not match its report")
    if args.indices and args.indices_file:
        raise ValueError("use only one of --indices and --indices-file")
    if args.indices_file:
        raw_indices = args.indices_file.read_text(encoding="utf-8").replace(",", " ")
        indices = sorted({int(token) for token in raw_indices.split() if token.strip()})
    elif args.indices:
        indices = sorted(
            {
                int(token)
                for token in args.indices.replace("_", ",").split(",")
                if token.strip()
            }
        )
    else:
        indices = list(range(len(rows)))
        if args.limit is not None:
            indices = indices[: args.limit]
    if not indices or any(index < 0 or index >= len(rows) for index in indices):
        raise IndexError("selected frontier index is outside the artifact")
    release_dir = args.solution_dir / "target" / "release"
    executable = next(
        (
            candidate
            for candidate in (release_dir / "solve_cube_beam.exe", release_dir / "solve_cube_beam")
            if candidate.is_file()
        ),
        None,
    )
    if executable is None:
        raise FileNotFoundError("KMC solve_cube_beam release executable is missing")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: list[QueryResult] = []
    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index in indices:
            query_id = str(rows[index]["query_id"])
            pid = int(source_pids[index])
            endpoint_state = tuple(int(value) for value in full_states[index])
            if args.use_prefix_context:
                setup_path = tuple(
                    str(move)
                    for move in rows[index].get(
                        "setup_path", frontier_report.get("setup_path", [])
                    )
                )
                macro_path = tuple(str(move) for move in rows[index]["macro_path"])
                original_state = states_by_pid[pid]
                query_state = original_state
                solver_state = frame.transform_state(original_state)
                corner_context = frame.transform_path(setup_path + macro_path)
                if puzzle.apply_path(original_state, setup_path + macro_path) != endpoint_state:
                    raise AssertionError(
                        f"{query_id}: prefix context does not reach the stored frontier"
                    )
            else:
                query_state = endpoint_state
                solver_state = frame.transform_state(endpoint_state)
                corner_context = ()
            future = executor.submit(
                query_one,
                index=index,
                pid=pid,
                query_id=query_id,
                state=query_state,
                solver_state=solver_state,
                corner_context=corner_context,
                puzzle=puzzle,
                frame=frame,
                solution_dir=args.solution_dir,
                executable=executable,
                out_dir=args.out_dir,
                beam=args.beam,
                candidate_cap=args.candidate_cap,
                rayon_threads=args.rayon_threads,
                anneal_seed=args.anneal_seed,
                anneal_steps=args.anneal_steps,
                anneal_keep_best=args.anneal_keep_best,
                alpha=args.alpha,
                timeout_seconds=args.timeout_seconds,
                cache_tag=args.cache_tag,
                force=args.force,
            )
            futures[future] = query_id
        for future in concurrent.futures.as_completed(futures):
            query_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"query={query_id} moves={len(result.path)} "
                    f"elapsed={result.elapsed_seconds:.2f}s reused={result.reused}",
                    flush=True,
                )
            except Exception as exc:
                failures.append((query_id, repr(exc)))
                print(f"query={query_id} FAILED {exc!r}", file=sys.stderr, flush=True)

    ordered = sorted(results, key=lambda result: result.index)
    lengths = [len(result.path) for result in ordered]
    report = {
        "alpha": args.alpha,
        "anneal_keep_best": args.anneal_keep_best,
        "anneal_seed": args.anneal_seed,
        "anneal_steps": args.anneal_steps,
        "beam": args.beam,
        "candidate_cap": args.candidate_cap,
        "completed": len(ordered),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "failures": failures,
        "frontier_dir": str(args.frontier_dir),
        "mean_moves": statistics.fmean(lengths) if lengths else None,
        "query_ids": [result.query_id for result in ordered],
        "rayon_threads": args.rayon_threads,
        "replay_verified": len(ordered) == len(indices) and not failures,
        "requested": len(indices),
        "symmetry_index": args.symmetry_index,
        "use_prefix_context": args.use_prefix_context,
    }
    atomic_text(args.out_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} KMC frontier queries failed")


if __name__ == "__main__":
    main()
