"""Run the public KMCoders classical 6x6 solver on local 666 states.

The runner is resumable and treats the external solver as an untrusted proposal
generator: every returned move is checked against the local generator table and
the full 216-sticker path is replayed before an artifact is accepted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "cube_nnn" / "src"))

from cube666.puzzle import Cube666Puzzle  # noqa: E402
from cube666.classical import Cube666Decomposition, build_decomposition, residual_report  # noqa: E402
from cube666.corners import CornerCoordinateSystem, ExactCornerSolver  # noqa: E402
from cube666.frames import FrameTransform  # noqa: E402
from cube666.macros import (  # noqa: E402
    MacroEffect,
    analyze_corner_fixing_macro,
    finish_with_best_inserted_three_cycles,
    finish_with_lookahead_inserted_three_cycles,
    load_three_cycle_library,
    state_cluster_permutations,
)
from cube_nnn.puzzle import NCube  # noqa: E402
from cube_nnn.symmetry import build_symmetries  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"
DEFAULT_SOLUTION_DIR = PROJECT / "external" / "third_party" / "kmcoders_santa2023" / "solution"


@dataclass(frozen=True)
class SolveResult:
    index: int
    state_id: str
    path: tuple[str, ...]
    elapsed_seconds: float
    reused: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--solution-dir", type=Path, default=DEFAULT_SOLUTION_DIR)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--indices",
        help="comma-separated explicit row indices (overrides start-index/limit)",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--beam", type=int, default=1000)
    parser.add_argument(
        "--candidate-cap",
        type=int,
        default=1000,
        help="maximum exact insertion candidates retained per beam state",
    )
    parser.add_argument("--rayon-threads", type=int, default=2)
    parser.add_argument("--anneal-seed", type=int, default=89043280)
    parser.add_argument("--anneal-steps", type=int, default=20_000_000)
    parser.add_argument(
        "--anneal-keep-best",
        action="store_true",
        help="seed downstream search from the best annealing state visited instead of the final state",
    )
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument(
        "--corner-source",
        choices=("exact", "kmc"),
        default="exact",
        help="use our optimal move-derived corner path or KMCoders' native 2x2 corner solver",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--tag", default="kmc_beam")
    parser.add_argument(
        "--symmetry-index",
        type=int,
        help="0-based spatial frame from cube_nnn's verified 48-frame table",
    )
    parser.add_argument(
        "--inverse-frame",
        action="store_true",
        help="solve the inverse permutation after any spatial conjugation",
    )
    parser.add_argument(
        "--hybrid-finisher",
        choices=("none", "best", "lookahead"),
        default="none",
        help="stop the Rust solver after rough alignment and use the move-derived exact finisher",
    )
    parser.add_argument("--lookahead-width", type=int, default=12)
    parser.add_argument("--lookahead-window", type=int, default=4)
    parser.add_argument("--lookahead-global-width", type=int, default=1)
    parser.add_argument(
        "--kmc-alternatives",
        action="store_true",
        help="add replay-derived alternate exact 3-cycle words from KMCoders rotate_all.txt",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--prepare-corners-only",
        action="store_true",
        help="populate exact base-corner caches for the selected states and exit",
    )
    return parser.parse_args()


def solver_input(puzzle: Cube666Puzzle, state_id: str, state: tuple[int, ...]) -> str:
    available_positive = tuple(name for name in puzzle.move_names if not name.startswith("-"))
    # KMCoders' Rot3 tables encode operation numbers in d/f/r axis order.
    # The Cayley JSON happens to list them f/r/d; preserving that file order
    # leaves ordinary moves replayable but attaches every exact macro lookup to
    # the wrong axis family.
    positive_moves = tuple(f"{axis}{layer}" for axis in "dfr" for layer in range(6))
    if set(available_positive) != set(positive_moves):
        raise ValueError(
            "KMCoders' cube coordinate code requires the 18 d/f/r generators; "
            f"got {available_positive}"
        )
    tokens: list[str] = [state_id, "cube_6/6/6", str(puzzle.size), str(len(positive_moves)), "0"]
    for name in positive_moves:
        tokens.append(name)
        tokens.extend(str(value) for value in puzzle.generators[name])
    tokens.extend(str(value) for value in puzzle.solved_state)
    tokens.extend(str(value) for value in state)
    return " ".join(tokens) + "\n"


def replay_verify(
    puzzle: Cube666Puzzle,
    state_id: str,
    state: tuple[int, ...],
    path: tuple[str, ...],
) -> None:
    unknown = sorted(set(path).difference(puzzle.generators))
    if unknown:
        raise ValueError(f"state {state_id}: solver returned unknown moves {unknown}")
    final = puzzle.apply_path(state, path)
    if final != puzzle.solved_state:
        mismatches = sum(left != right for left, right in zip(final, puzzle.solved_state, strict=True))
        raise ValueError(f"state {state_id}: full replay failed with {mismatches} mismatches")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def cached_corner_path(
    cache_dir: Path,
    state_id: str,
    state: tuple[int, ...],
    puzzle: Cube666Puzzle,
    solver: ExactCornerSolver,
) -> tuple[str, ...]:
    path_file = cache_dir / f"{int(state_id):04d}.path.txt"
    if path_file.exists():
        return tuple(token for token in path_file.read_text(encoding="utf-8").strip().split(".") if token)
    path = solver.solve(state, puzzle.solved_state)
    atomic_write(path_file, ".".join(path) + "\n")
    return path


def load_kmc_alternative_paths(
    solution_dir: Path,
    puzzle: Cube666Puzzle,
    decomposition: Cube666Decomposition,
    library: dict[tuple[int, tuple[int, int, int]], MacroEffect],
) -> dict[tuple[int, tuple[int, int, int]], tuple[tuple[str, ...], ...]]:
    """Load n=6 alternate macro words and classify them by exact local effect.

    The upstream lookup coordinates do not agree with Cayley's local cluster
    numbering, so this deliberately ignores the file's labels.  Every word is
    replayed through Cayley's generator permutations and retained only when it
    is an isolated directed 3-cycle on one discovered physical cluster.
    """

    rotate_all = solution_dir / "src" / "rotate_all.txt"
    alternatives: dict[
        tuple[int, tuple[int, int, int]], set[tuple[str, ...]]
    ] = {key: {macro.path} for key, macro in library.items()}
    block = -1
    for line in rotate_all.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "#":
            block += 1
            continue
        if block != 2:  # the native 6x6 (general-center) block
            continue
        for encoded in fields[3:]:
            path = tuple(encoded.split("."))
            macro = analyze_corner_fixing_macro(path, puzzle.generators, decomposition)
            if macro.total_nontrivial_cycles != 1 or macro.total_three_cycles != 1:
                continue
            cluster = next(
                index for index, cycles in enumerate(macro.cluster_cycles) if cycles
            )
            cycle = macro.cluster_cycles[cluster][0]
            key = (cluster, cycle)
            if key in alternatives:
                alternatives[key].add(path)
    return {
        key: tuple(sorted(paths, key=lambda path: (len(path), path)))
        for key, paths in alternatives.items()
    }


def solve_one(
    *,
    index: int,
    state_id: str,
    state: tuple[int, ...],
    solver_state: tuple[int, ...],
    corner_path: tuple[str, ...] | None,
    frame: FrameTransform,
    puzzle: Cube666Puzzle,
    decomposition: Cube666Decomposition,
    finisher_library: dict[tuple[int, tuple[int, int, int]], MacroEffect] | None,
    alternative_paths: dict[
        tuple[int, tuple[int, int, int]], tuple[tuple[str, ...], ...]
    ] | None,
    solution_dir: Path,
    executable: Path,
    artifact_dir: Path,
    beam: int,
    candidate_cap: int,
    rayon_threads: int,
    anneal_seed: int,
    anneal_steps: int,
    anneal_keep_best: bool,
    alpha: float,
    timeout_seconds: float,
    hybrid_finisher: str,
    lookahead_width: int,
    lookahead_window: int,
    lookahead_global_width: int,
    force: bool,
) -> SolveResult:
    path_file = artifact_dir / f"{int(state_id):04d}.path.txt"
    metadata_file = artifact_dir / f"{int(state_id):04d}.json"
    if path_file.exists() and not force:
        text = path_file.read_text(encoding="utf-8").strip()
        path = tuple(token for token in text.split(".") if token)
        replay_verify(puzzle, state_id, state, path)
        elapsed = 0.0
        if metadata_file.exists():
            elapsed = float(json.loads(metadata_file.read_text(encoding="utf-8")).get("elapsed_seconds", 0.0))
        return SolveResult(index, state_id, path, elapsed, True)

    environment = os.environ.copy()
    environment["KMC_BEAM"] = str(beam)
    cache_family = "cayley_v4_dfr" if corner_path is not None else "cayley_v4_dfr_kmc"
    environment["KMC_CACHE_TAG"] = (
        f"{cache_family}_s{anneal_seed}_n{anneal_steps}_a{alpha:g}_"
        f"kb{int(anneal_keep_best)}_{frame.label}"
    )
    if corner_path is not None:
        environment["KMC_CORNER_PATH"] = ".".join(corner_path)
    environment["KMC_SEED"] = str(anneal_seed)
    environment["KMC_ANNEAL_STEPS"] = str(anneal_steps)
    environment["KMC_ANNEAL_KEEP_BEST"] = "1" if anneal_keep_best else "0"
    environment["KMC_ALPHA"] = str(alpha)
    environment["KMC_CANDIDATE_CAP"] = str(candidate_cap)
    environment["KMC_SOLUTION_DIR"] = str(solution_dir.resolve())
    environment["RAYON_NUM_THREADS"] = str(rayon_threads)
    if hybrid_finisher != "none":
        environment["KMC_STOP_AFTER_ROUGH"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        [str(executable)],
        input=solver_input(puzzle, state_id, solver_state),
        text=True,
        capture_output=True,
        cwd=solution_dir,
        env=environment,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_file = artifact_dir / f"{int(state_id):04d}.stderr.log"
    atomic_write(log_file, completed.stderr)
    stdout_file = artifact_dir / f"{int(state_id):04d}.stdout.log"
    atomic_write(stdout_file, completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"state {state_id}: solver exited {completed.returncode}; see {log_file}"
        )
    stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise RuntimeError(f"state {state_id}: solver returned no path; see {log_file}")
    rough_path = tuple(token for token in stdout_lines[-1].split(".") if token)
    rough_moves: int | None = None
    rough_residual: int | None = None
    if hybrid_finisher == "none":
        path = rough_path
    else:
        if finisher_library is None:
            raise AssertionError("hybrid finisher requested without a 3-cycle library")
        rough_state = puzzle.apply_path(solver_state, rough_path)
        corner_mismatches = sum(
            rough_state[position] != puzzle.solved_state[position]
            for position in decomposition.corner_orbit
        )
        if corner_mismatches:
            raise ValueError(f"state {state_id}: rough path leaves {corner_mismatches} corner stickers wrong")
        rough_report = residual_report(rough_state, puzzle.solved_state, decomposition)
        if not rough_report.all_even or rough_report.unrestricted_three_cycles is None:
            raise ValueError(f"state {state_id}: rough path leaves odd cluster parity")
        rough_moves = len(rough_path)
        rough_residual = rough_report.unrestricted_three_cycles
        clusters = state_cluster_permutations(rough_state, puzzle.solved_state, decomposition)
        if hybrid_finisher == "best":
            finished = finish_with_best_inserted_three_cycles(
                clusters,
                rough_path,
                finisher_library,
                puzzle.generators,
                decomposition,
                alternative_paths=alternative_paths,
            )
        else:
            finished = finish_with_lookahead_inserted_three_cycles(
                clusters,
                rough_path,
                finisher_library,
                puzzle.generators,
                decomposition,
                first_width=lookahead_width,
                local_window=lookahead_window,
                global_width=lookahead_global_width,
            )
        path = finished.path
    replay_verify(puzzle, state_id + ":" + frame.label, solver_state, path)
    path = frame.restore_path(path)
    replay_verify(puzzle, state_id, state, path)

    atomic_write(path_file, ".".join(path) + "\n")
    metadata = {
        "beam": beam,
        "candidate_cap": candidate_cap,
        "anneal_seed": anneal_seed,
        "anneal_steps": anneal_steps,
        "anneal_keep_best": anneal_keep_best,
        "alpha": alpha,
        "elapsed_seconds": round(elapsed, 6),
        "formulation": frame.label,
        "index": index,
        "moves": len(path),
        "rough_moves": rough_moves,
        "rough_residual_three_cycles": rough_residual,
        "rayon_threads": rayon_threads,
        "replay_verified": True,
        "state_id": state_id,
    }
    atomic_write(metadata_file, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return SolveResult(index, state_id, path, elapsed, False)


def describe(values: list[int]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "total": 0}
    return {
        "count": len(values),
        "maximum": max(values),
        "mean": round(statistics.fmean(values), 4),
        "median": statistics.median(values),
        "minimum": min(values),
        "total": sum(values),
    }


def write_aggregate(
    results: list[SolveResult],
    artifact_dir: Path,
    total_states: int,
    wall_seconds: float,
) -> None:
    ordered = sorted(results, key=lambda result: result.index)
    csv_path = artifact_dir / "solutions.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        for result in ordered:
            writer.writerow({"initial_state_id": result.state_id, "path": ".".join(result.path)})
    temporary.replace(csv_path)

    lengths = [len(result.path) for result in ordered]
    report = {
        "completed_states": len(ordered),
        "coverage_fraction": round(len(ordered) / total_states, 8),
        "projected_total_at_completed_mean": (
            round(statistics.fmean(lengths) * total_states, 2) if lengths else None
        ),
        "solution_lengths": describe(lengths),
        "solver_seconds_sum": round(sum(result.elapsed_seconds for result in ordered), 4),
        "total_states": total_states,
        "wall_seconds": round(wall_seconds, 4),
    }
    atomic_write(artifact_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.beam <= 0 or args.candidate_cap <= 0 or args.rayon_threads <= 0:
        raise ValueError("workers, beam, candidate-cap, and rayon-threads must be positive")
    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    if puzzle.solved_state != tuple(range(puzzle.size)):
        raise ValueError("the external colorful-cube solver requires identity sticker labels")
    rows = list(puzzle.iter_test_states(args.data_dir / "test.csv"))
    spatial_permutation = None
    move_to_transformed = None
    frame_label = "identity"
    if args.symmetry_index is not None:
        geometric_cube = NCube.from_puzzle_info(args.data_dir / "puzzle_info.json")
        symmetry_table = build_symmetries(geometric_cube)
        if len(symmetry_table) != 48:
            raise ValueError(f"expected 48 verified spatial frames, got {len(symmetry_table)}")
        if not 0 <= args.symmetry_index < len(symmetry_table):
            raise IndexError("symmetry-index must be in 0..47")
        spatial_permutation = symmetry_table.perms[args.symmetry_index]
        move_to_transformed = symmetry_table.relabel[args.symmetry_index]
        frame_label = f"sym{args.symmetry_index:02d}"
    if args.inverse_frame:
        frame_label += "_inv"
    frame = FrameTransform(
        label=frame_label,
        move_names=puzzle.move_names,
        sticker_permutation=spatial_permutation,
        move_to_transformed=move_to_transformed,
        use_inverse=args.inverse_frame,
    )
    decomposition = build_decomposition(puzzle.generators)
    corner_solver = (
        ExactCornerSolver.build(
            CornerCoordinateSystem.discover(puzzle.generators, decomposition)
        )
        if args.corner_source == "exact"
        else None
    )
    stop = len(rows) if args.limit is None else min(len(rows), args.start_index + args.limit)
    if args.indices:
        selected_indices = sorted({int(value) for value in args.indices.split(",") if value.strip()})
        if any(index < 0 or index >= len(rows) for index in selected_indices):
            raise IndexError("an explicit row index is outside the test set")
    else:
        selected_indices = list(range(args.start_index, stop))
    corner_cache = PROJECT / "cube666" / "artifacts" / "corner_paths_cayley_v1"
    selected = [
        (
            index,
            rows[index][0],
            rows[index][1],
            frame.transform_state(rows[index][1]),
            (
                frame.transform_path(
                    cached_corner_path(
                        corner_cache,
                        rows[index][0],
                        rows[index][1],
                        puzzle,
                        corner_solver,
                    )
                )
                if corner_solver is not None
                else None
            ),
        )
        for index in selected_indices
    ]
    if not selected:
        raise ValueError("selected state range is empty")
    if args.prepare_corners_only:
        if corner_solver is None:
            raise ValueError("--prepare-corners-only requires --corner-source exact")
        print(f"prepared {len(selected)} exact corner paths in {corner_cache}")
        return

    finisher_library = None
    alternative_paths = None
    if args.hybrid_finisher != "none":
        finisher_library = load_three_cycle_library(
            PROJECT / "cube666" / "artifacts" / "three_cycle_library.json",
            puzzle.generators,
            decomposition,
        )
        if args.kmc_alternatives:
            alternative_paths = load_kmc_alternative_paths(
                args.solution_dir,
                puzzle,
                decomposition,
                finisher_library,
            )

    release_dir = args.solution_dir / "target" / "release"
    executable_candidates = (
        release_dir / "solve_cube_beam.exe",
        release_dir / "solve_cube_beam",
    )
    executable = next((path for path in executable_candidates if path.is_file()), None)
    if executable is None:
        raise FileNotFoundError(
            "build the solver first; expected one of: "
            + ", ".join(str(path) for path in executable_candidates)
        )
    artifact_dir = PROJECT / "cube666" / "results" / f"{args.tag}_b{args.beam}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    results: list[SolveResult] = []
    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                solve_one,
                index=index,
                state_id=state_id,
                state=state,
                solver_state=solver_state,
                corner_path=corner_path,
                frame=frame,
                puzzle=puzzle,
                decomposition=decomposition,
                finisher_library=finisher_library,
                alternative_paths=alternative_paths,
                solution_dir=args.solution_dir,
                executable=executable,
                artifact_dir=artifact_dir,
                beam=args.beam,
                candidate_cap=args.candidate_cap,
                rayon_threads=args.rayon_threads,
                anneal_seed=args.anneal_seed,
                anneal_steps=args.anneal_steps,
                anneal_keep_best=args.anneal_keep_best,
                alpha=args.alpha,
                timeout_seconds=args.timeout_seconds,
                hybrid_finisher=args.hybrid_finisher,
                lookahead_width=args.lookahead_width,
                lookahead_window=args.lookahead_window,
                lookahead_global_width=args.lookahead_global_width,
                force=args.force,
            ): state_id
            for index, state_id, state, solver_state, corner_path in selected
        }
        for future in concurrent.futures.as_completed(futures):
            state_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                lengths = [len(item.path) for item in results]
                print(
                    f"state={state_id} moves={len(result.path)} elapsed={result.elapsed_seconds:.2f}s "
                    f"completed={len(results)}/{len(selected)} mean={statistics.fmean(lengths):.2f}",
                    flush=True,
                )
                write_aggregate(results, artifact_dir, len(rows), time.perf_counter() - started)
            except Exception as exc:  # keep independent states running and report every failure
                failures.append((state_id, repr(exc)))
                print(f"state={state_id} FAILED {exc!r}", file=sys.stderr, flush=True)

    write_aggregate(results, artifact_dir, len(rows), time.perf_counter() - started)
    if failures:
        atomic_write(
            artifact_dir / "failures.json",
            json.dumps(failures, indent=2, sort_keys=True) + "\n",
        )
        raise RuntimeError(f"{len(failures)} states failed; see {artifact_dir / 'failures.json'}")
    print((artifact_dir / "report.json").read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
