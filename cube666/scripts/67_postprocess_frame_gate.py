"""Replay-verified post-processing gate over completed cube666 frame trajectories.

The gate deliberately keeps every available trajectory alive until all candidate-local
rewrites have run.  It then performs exact shared-state crossover, re-polishes any novel
hybrid, and strict-min merges only verified wins into the full incumbent submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "pathkit"))

from cube666.macros import reduce_commuting_quarter_turn_path  # noqa: E402
from cube666.puzzle import Cube666Puzzle  # noqa: E402
from pathkit.backend import get_backend  # noqa: E402
from pathkit.balls import Zobrist  # noqa: E402
from pathkit.puzzle import Puzzle as PathkitPuzzle  # noqa: E402
from pathkit.window import reduce_path_to_fixpoint  # noqa: E402


DEFAULT_DATA = PROJECT / "cayley-py-666-cube"
DEFAULT_PRODUCTION = PROJECT / "cube666" / "kaggle_kmc_campaign" / "production_outputs"
DEFAULT_RESUME_MANIFEST = (
    PROJECT / "cube666" / "kaggle_kmc_campaign" / "resume_wave1" / "campaign_manifest.json"
)
DEFAULT_INCUMBENT = DEFAULT_PRODUCTION / "wave1_all_harvested_merged.csv"
DEFAULT_REPORT_DIR = PROJECT / "cube666" / "reports" / "postprocess_gate10_v1"
DEFAULT_OUTPUT = PROJECT / "submissions" / "cube666_kmc_postprocess_gate10_v1.csv"
DEFAULT_POLISHER = (
    PROJECT
    / "external"
    / "third_party"
    / "kmcoders_santa2023"
    / "solution"
    / "target"
    / "release"
    / "polish_cube.exe"
)
DEFAULT_KMC_DIR = PROJECT / "external" / "third_party" / "kmcoders_santa2023" / "solution"
FRAME_NAMES = ("rot03", "rot18", "rot35")
COMPLETE_SLOTS = ("01", "08")


@dataclass(frozen=True)
class Candidate:
    pid: int
    sources: tuple[str, ...]
    path: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--production-dir", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--resume-manifest", type=Path, default=DEFAULT_RESUME_MANIFEST)
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--polisher", type=Path, default=DEFAULT_POLISHER)
    parser.add_argument("--kmc-dir", type=Path, default=DEFAULT_KMC_DIR)
    parser.add_argument("--pids", type=str, default="")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--walton-workers", type=int, default=4)
    parser.add_argument("--walton-timeout", type=int, default=300)
    parser.add_argument("--window-radius", type=int, default=4)
    parser.add_argument("--window-rounds", type=int, default=4)
    parser.add_argument("--window-chunk", type=int, default=5000)
    parser.add_argument("--window-device", type=str, default="cuda")
    parser.add_argument("--skip-walton", action="store_true")
    parser.add_argument("--skip-window", action="store_true")
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def path_hash(path: Iterable[str]) -> str:
    return hashlib.sha256(".".join(path).encode("utf-8")).hexdigest()[:16]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def load_csv_paths(path: Path) -> dict[int, tuple[str, ...]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): parse_path(row["path"])
            for row in csv.DictReader(handle)
        }


def scheduled_pids(path: Path) -> set[int]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result: set[int] = set()
    for shard in manifest["shards"]:
        for group in shard["groups"]:
            raw = group["pids"]
            values = raw.split() if isinstance(raw, str) else raw
            result.update(int(value) for value in values)
    return result


def completed_frame_dirs(production_dir: Path) -> dict[int, tuple[Path, ...]]:
    covered: dict[int, tuple[Path, ...]] = {}
    for slot in COMPLETE_SLOTS:
        directories = tuple(
            production_dir / f"slot{slot}" / "extracted" / f"w1_{frame}_tl40_s{slot}_b20000"
            for frame in FRAME_NAMES
        )
        if not all(directory.is_dir() for directory in directories):
            continue
        pid_sets = [
            {int(path.name.split(".")[0]) for path in directory.glob("*.path.txt")}
            for directory in directories
        ]
        for pid in set.intersection(*pid_sets):
            covered[pid] = directories
    return covered


def choose_pids(
    requested: str,
    count: int,
    incumbent: dict[int, tuple[str, ...]],
    frame_dirs: dict[int, tuple[Path, ...]],
    scheduled: set[int],
) -> list[int]:
    eligible = set(frame_dirs).difference(scheduled)
    if requested.strip():
        chosen = [int(token) for token in requested.replace(",", " ").split()]
        invalid = sorted(set(chosen).difference(eligible))
        if invalid:
            raise ValueError(f"requested pids are incomplete or scheduled: {invalid}")
        return chosen
    return sorted(eligible, key=lambda pid: (-len(incumbent[pid]), pid))[:count]


def replay_verify(
    puzzle: Cube666Puzzle,
    states: dict[int, tuple[int, ...]],
    pid: int,
    path: tuple[str, ...],
    label: str,
) -> None:
    unknown = sorted(set(path).difference(puzzle.generators))
    if unknown:
        raise ValueError(f"pid {pid} {label}: unknown moves {unknown}")
    final = puzzle.apply_path(states[pid], path)
    if final != puzzle.solved_state:
        mismatches = sum(a != b for a, b in zip(final, puzzle.solved_state, strict=True))
        raise ValueError(f"pid {pid} {label}: replay failed with {mismatches} mismatches")


def solver_input(puzzle: Cube666Puzzle, pid: int, state: tuple[int, ...]) -> str:
    available = tuple(name for name in puzzle.move_names if not name.startswith("-"))
    positive = tuple(f"{axis}{layer}" for axis in "dfr" for layer in range(6))
    if set(available) != set(positive):
        raise ValueError("KMCoders requires the 18 d/f/r positive generators")
    tokens = [str(pid), "cube_6/6/6", str(puzzle.size), str(len(positive)), "0"]
    for name in positive:
        tokens.append(name)
        tokens.extend(str(value) for value in puzzle.generators[name])
    tokens.extend(str(value) for value in puzzle.solved_state)
    tokens.extend(str(value) for value in state)
    return " ".join(tokens) + "\n"


def walton_polish(
    puzzle: Cube666Puzzle,
    states: dict[int, tuple[int, ...]],
    pid: int,
    path: tuple[str, ...],
    executable: Path,
    kmc_dir: Path,
    timeout: int,
) -> tuple[tuple[str, ...], float, str]:
    environment = os.environ.copy()
    environment["KMC_PATH"] = ".".join(path)
    environment["RAYON_NUM_THREADS"] = "1"
    started = time.perf_counter()
    completed = subprocess.run(
        [str(executable.resolve())],
        input=solver_input(puzzle, pid, states[pid]),
        text=True,
        capture_output=True,
        cwd=kmc_dir,
        env=environment,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"pid {pid}: polish_cube exited {completed.returncode}: {completed.stderr[-1000:]}"
        )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"pid {pid}: polish_cube returned no path")
    polished = parse_path(lines[-1])
    replay_verify(puzzle, states, pid, polished, "Walton polished")
    if len(polished) > len(path):
        raise ValueError(f"pid {pid}: Walton polisher lengthened {len(path)} -> {len(polished)}")
    return polished, elapsed, completed.stderr


def deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    by_path: dict[tuple[str, ...], set[str]] = {}
    for candidate in candidates:
        by_path.setdefault(candidate.path, set()).update(candidate.sources)
    return [
        Candidate(candidates[0].pid, tuple(sorted(sources)), path)
        for path, sources in by_path.items()
    ]


def walton_stage(
    candidates: list[Candidate],
    puzzle: Cube666Puzzle,
    states: dict[int, tuple[int, ...]],
    args: argparse.Namespace,
    journal: Path,
) -> list[Candidate]:
    if args.skip_walton:
        return candidates
    cache_dir = args.report_dir / "walton_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def run(candidate: Candidate) -> tuple[Candidate, dict[str, object]]:
        digest = path_hash(candidate.path)
        cache = cache_dir / f"{candidate.pid:04d}_{digest}.path.txt"
        if cache.exists():
            polished = parse_path(cache.read_text(encoding="utf-8"))
            replay_verify(puzzle, states, candidate.pid, polished, "cached Walton")
            if len(polished) > len(candidate.path):
                raise ValueError(f"pid {candidate.pid}: invalid lengthening Walton cache")
            elapsed = 0.0
            cached = True
        else:
            polished, elapsed, stderr = walton_polish(
                puzzle,
                states,
                candidate.pid,
                candidate.path,
                args.polisher,
                args.kmc_dir,
                args.walton_timeout,
            )
            polished = reduce_commuting_quarter_turn_path(polished)
            replay_verify(puzzle, states, candidate.pid, polished, "Walton plus commute")
            atomic_write(cache, ".".join(polished) + "\n")
            atomic_write(cache.with_suffix(".stderr.log"), stderr)
            cached = False
        result = Candidate(candidate.pid, candidate.sources, polished)
        record: dict[str, object] = {
            "stage": "walton",
            "pid": candidate.pid,
            "sources": list(candidate.sources),
            "input_hash": digest,
            "before": len(candidate.path),
            "after": len(polished),
            "saved": len(candidate.path) - len(polished),
            "seconds": round(elapsed, 3),
            "cached": cached,
        }
        return result, record

    output: list[Candidate] = []
    with ThreadPoolExecutor(max_workers=args.walton_workers) as executor:
        futures = [executor.submit(run, candidate) for candidate in candidates]
        done = 0
        for future in as_completed(futures):
            candidate, record = future.result()
            output.append(candidate)
            append_jsonl(journal, record)
            done += 1
            print(
                f"walton {done}/{len(futures)} pid={candidate.pid} "
                f"{record['before']}->{record['after']}",
                flush=True,
            )
    return sorted(output, key=lambda candidate: (candidate.pid, candidate.sources))


def window_stage(
    candidates: list[Candidate],
    cube_puzzle: Cube666Puzzle,
    cube_states: dict[int, tuple[int, ...]],
    pathkit_puzzle: PathkitPuzzle,
    pathkit_states: dict[int, np.ndarray],
    args: argparse.Namespace,
    journal: Path,
) -> list[Candidate]:
    if args.skip_window:
        return candidates
    cache_dir = args.report_dir / f"window_r{args.window_radius}_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    backend = get_backend(args.window_device)
    if args.window_device.startswith("cuda") and backend.name != "torch":
        raise RuntimeError("CUDA window stage requested but CUDA is unavailable")
    zobrist = Zobrist(pathkit_puzzle.state_size, pathkit_puzzle.n_labels, backend)
    output: list[Candidate] = []
    for index, candidate in enumerate(candidates, start=1):
        digest = path_hash(candidate.path)
        cache = cache_dir / f"{candidate.pid:04d}_{digest}.path.txt"
        if cache.exists():
            shortened = parse_path(cache.read_text(encoding="utf-8"))
            replay_verify(cube_puzzle, cube_states, candidate.pid, shortened, "cached window")
            if len(shortened) > len(candidate.path):
                raise ValueError(f"pid {candidate.pid}: invalid lengthening window cache")
            elapsed = 0.0
            cached = True
        else:
            word = pathkit_puzzle.parse(".".join(candidate.path))
            started = time.perf_counter()
            rewritten = reduce_path_to_fixpoint(
                pathkit_puzzle,
                pathkit_states[candidate.pid],
                word,
                args.window_radius,
                max_rounds=args.window_rounds,
                bk=backend,
                zob=zobrist,
                chunk=args.window_chunk,
            )
            elapsed = time.perf_counter() - started
            shortened = tuple(pathkit_puzzle.names[int(move)] for move in rewritten)
            shortened = reduce_commuting_quarter_turn_path(shortened)
            replay_verify(cube_puzzle, cube_states, candidate.pid, shortened, "exact window")
            if len(shortened) > len(candidate.path):
                raise ValueError(f"pid {candidate.pid}: exact window stage lengthened path")
            atomic_write(cache, ".".join(shortened) + "\n")
            cached = False
        record: dict[str, object] = {
            "stage": f"window_r{args.window_radius}",
            "pid": candidate.pid,
            "sources": list(candidate.sources),
            "input_hash": digest,
            "before": len(candidate.path),
            "after": len(shortened),
            "saved": len(candidate.path) - len(shortened),
            "seconds": round(elapsed, 3),
            "cached": cached,
        }
        append_jsonl(journal, record)
        output.append(Candidate(candidate.pid, candidate.sources, shortened))
        print(
            f"window {index}/{len(candidates)} pid={candidate.pid} "
            f"{len(candidate.path)}->{len(shortened)} ({elapsed:.1f}s)",
            flush=True,
        )
    return output


def union_shortest_path(
    puzzle: Cube666Puzzle,
    state: tuple[int, ...],
    paths: list[tuple[str, ...]],
) -> tuple[tuple[str, ...], dict[str, int]]:
    start_key = bytes(state)
    goal_key = bytes(puzzle.solved_state)
    adjacency: dict[bytes, list[tuple[bytes, str]]] = {}
    owners: dict[bytes, set[int]] = {}
    repeated_within = 0
    for owner, path in enumerate(paths):
        current = state
        current_key = bytes(current)
        seen = {current_key}
        owners.setdefault(current_key, set()).add(owner)
        for move in path:
            nxt = puzzle.apply_move(current, move)
            next_key = bytes(nxt)
            adjacency.setdefault(current_key, []).append((next_key, move))
            owners.setdefault(next_key, set()).add(owner)
            if next_key in seen:
                repeated_within += 1
            seen.add(next_key)
            current, current_key = nxt, next_key
        if current_key != goal_key:
            raise ValueError("union graph received a non-solving path")

    queue = deque([start_key])
    parent: dict[bytes, tuple[bytes, str] | None] = {start_key: None}
    while queue:
        current = queue.popleft()
        if current == goal_key:
            break
        for nxt, move in adjacency.get(current, ()):
            if nxt not in parent:
                parent[nxt] = (current, move)
                queue.append(nxt)
    if goal_key not in parent:
        raise ValueError("union graph did not connect start to solved")
    reversed_path: list[str] = []
    cursor = goal_key
    while cursor != start_key:
        edge = parent[cursor]
        if edge is None:
            raise AssertionError("broken union parent chain")
        cursor, move = edge
        reversed_path.append(move)
    result = tuple(reversed(reversed_path))
    stats = {
        "union_states": len(owners),
        "shared_states_across_paths": sum(len(value) > 1 for value in owners.values()),
        "repeated_states_within_paths": repeated_within,
    }
    return result, stats


def positive_controls(
    selected: list[int],
    incumbent: dict[int, tuple[str, ...]],
    puzzle: Cube666Puzzle,
    states: dict[int, tuple[int, ...]],
    pathkit_puzzle: PathkitPuzzle,
    pathkit_states: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, object]:
    pid = selected[0]
    base = incumbent[pid]
    inflated = base + ("f0", "f0", "f0", "f0")
    replay_verify(puzzle, states, pid, inflated, "inflated positive control")
    report: dict[str, object] = {"pid": pid, "base": len(base), "inflated": len(inflated)}

    commuting = reduce_commuting_quarter_turn_path(inflated)
    replay_verify(puzzle, states, pid, commuting, "commuting positive control")
    if len(commuting) > len(base):
        raise AssertionError("commuting positive control did not remove neutral f0^4")
    report["commuting_after"] = len(commuting)

    hybrid, _ = union_shortest_path(puzzle, states[pid], [base, inflated])
    replay_verify(puzzle, states, pid, hybrid, "union positive control")
    if len(hybrid) > len(base):
        raise AssertionError("union positive control did not recover the base path")
    report["union_after"] = len(hybrid)

    if not args.skip_walton:
        polished, elapsed, _ = walton_polish(
            puzzle,
            states,
            pid,
            inflated,
            args.polisher,
            args.kmc_dir,
            args.walton_timeout,
        )
        if len(polished) > len(base):
            raise AssertionError("Walton positive control did not remove neutral f0^4")
        report["walton_after"] = len(polished)
        report["walton_seconds"] = round(elapsed, 3)

    if not args.skip_window:
        backend = get_backend(args.window_device)
        if args.window_device.startswith("cuda") and backend.name != "torch":
            raise RuntimeError("CUDA positive control requested but CUDA is unavailable")
        zobrist = Zobrist(pathkit_puzzle.state_size, pathkit_puzzle.n_labels, backend)
        word = pathkit_puzzle.parse(".".join(inflated))
        started = time.perf_counter()
        rewritten = reduce_path_to_fixpoint(
            pathkit_puzzle,
            pathkit_states[pid],
            word,
            args.window_radius,
            max_rounds=args.window_rounds,
            bk=backend,
            zob=zobrist,
            chunk=args.window_chunk,
        )
        elapsed = time.perf_counter() - started
        windowed = tuple(pathkit_puzzle.names[int(move)] for move in rewritten)
        replay_verify(puzzle, states, pid, windowed, "window positive control")
        if len(windowed) > len(base):
            raise AssertionError("window positive control did not remove neutral f0^4")
        report["window_after"] = len(windowed)
        report["window_seconds"] = round(elapsed, 3)
    return report


def write_submission(path: Path, rows: dict[int, tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("initial_state_id", "path"))
        writer.writeheader()
        for pid in sorted(rows):
            writer.writerow({"initial_state_id": pid, "path": ".".join(rows[pid])})
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    journal = args.report_dir / "journal.jsonl"

    puzzle = Cube666Puzzle.load(args.data_dir / "puzzle_info.json")
    puzzle.verify_inverse_pairs()
    states = {int(pid): state for pid, state in puzzle.iter_test_states(args.data_dir / "test.csv")}
    pathkit_puzzle = PathkitPuzzle.from_puzzle_info(args.data_dir / "puzzle_info.json")
    pathkit_states = {pid: np.asarray(state, dtype=np.uint8) for pid, state in states.items()}
    incumbent = load_csv_paths(args.incumbent)
    frame_dirs = completed_frame_dirs(args.production_dir)
    scheduled = scheduled_pids(args.resume_manifest)
    selected = choose_pids(args.pids, args.count, incumbent, frame_dirs, scheduled)
    if len(selected) != args.count and not args.pids.strip():
        raise ValueError(f"requested {args.count} pids but selected {len(selected)}")

    selection = []
    raw_by_pid: dict[int, list[Candidate]] = {}
    for pid in selected:
        raw = [Candidate(pid, ("incumbent",), incumbent[pid])]
        frame_lengths: dict[str, int] = {}
        for frame, directory in zip(FRAME_NAMES, frame_dirs[pid], strict=True):
            path = parse_path((directory / f"{pid:04d}.path.txt").read_text(encoding="utf-8"))
            replay_verify(puzzle, states, pid, path, frame)
            frame_lengths[frame] = len(path)
            raw.append(Candidate(pid, (frame,), path))
        commuting = [
            Candidate(candidate.pid, candidate.sources, reduce_commuting_quarter_turn_path(candidate.path))
            for candidate in raw
        ]
        for candidate in commuting:
            replay_verify(puzzle, states, pid, candidate.path, "commuting")
        raw_by_pid[pid] = deduplicate(commuting)
        selection.append(
            {
                "pid": pid,
                "incumbent": len(incumbent[pid]),
                "frames": frame_lengths,
                "unique_after_commuting": len(raw_by_pid[pid]),
                "scheduled_for_resume": pid in scheduled,
            }
        )
    atomic_write(args.report_dir / "selection.json", json.dumps(selection, indent=2) + "\n")
    print("selected " + " ".join(str(pid) for pid in selected), flush=True)
    print(json.dumps(selection, indent=2), flush=True)

    controls = positive_controls(
        selected,
        incumbent,
        puzzle,
        states,
        pathkit_puzzle,
        pathkit_states,
        args,
    )
    atomic_write(args.report_dir / "positive_controls.json", json.dumps(controls, indent=2) + "\n")
    print("positive controls passed: " + json.dumps(controls, sort_keys=True), flush=True)

    all_candidates = [candidate for pid in selected for candidate in raw_by_pid[pid]]
    raw_unique_count = len(all_candidates)
    t_walton = time.perf_counter()
    walton = walton_stage(all_candidates, puzzle, states, args, journal)
    walton_seconds = time.perf_counter() - t_walton
    t_window = time.perf_counter()
    windowed = window_stage(
        walton,
        puzzle,
        states,
        pathkit_puzzle,
        pathkit_states,
        args,
        journal,
    )
    window_seconds = time.perf_counter() - t_window

    final_by_pid: dict[int, tuple[str, ...]] = {}
    pid_reports: list[dict[str, object]] = []
    winner_attribution: Counter[str] = Counter()
    for pid in selected:
        raw_candidates = raw_by_pid[pid]
        walton_candidates = [candidate for candidate in walton if candidate.pid == pid]
        window_candidates = [candidate for candidate in windowed if candidate.pid == pid]
        hybrid, union_stats = union_shortest_path(
            puzzle,
            states[pid],
            [candidate.path for candidate in window_candidates],
        )
        replay_verify(puzzle, states, pid, hybrid, "shared-state hybrid")
        hybrid_source = "union"

        if hybrid not in {candidate.path for candidate in window_candidates}:
            novel = [Candidate(pid, ("union",), reduce_commuting_quarter_turn_path(hybrid))]
            novel = walton_stage(novel, puzzle, states, args, journal)
            novel = window_stage(
                novel,
                puzzle,
                states,
                pathkit_puzzle,
                pathkit_states,
                args,
                journal,
            )
            hybrid = novel[0].path
            replay_verify(puzzle, states, pid, hybrid, "re-polished novel hybrid")
            hybrid_source = "union_novel_repolished"
        else:
            for candidate in window_candidates:
                if candidate.path == hybrid:
                    hybrid_source = "+".join(candidate.sources)
                    break

        choices = [(incumbent[pid], "incumbent"), (hybrid, hybrid_source)]
        choices.extend((candidate.path, "+".join(candidate.sources)) for candidate in window_candidates)
        winner, source = min(choices, key=lambda item: (len(item[0]), item[0], item[1]))
        replay_verify(puzzle, states, pid, winner, "final selected")
        final_by_pid[pid] = winner
        winner_attribution[source] += 1
        pid_reports.append(
            {
                "pid": pid,
                "incumbent": len(incumbent[pid]),
                "raw_best": min(len(candidate.path) for candidate in raw_candidates),
                "walton_best": min(len(candidate.path) for candidate in walton_candidates),
                "window_best": min(len(candidate.path) for candidate in window_candidates),
                "union": len(hybrid),
                "final": len(winner),
                "saved_vs_incumbent": len(incumbent[pid]) - len(winner),
                "winner": source,
                **union_stats,
            }
        )

    merged = dict(incumbent)
    for pid, path in final_by_pid.items():
        if len(path) < len(merged[pid]):
            merged[pid] = path
    for pid, state in states.items():
        if pid not in merged:
            raise ValueError(f"merged submission missing pid {pid}")
        replay_verify(puzzle, states, pid, merged[pid], "full merged submission")
    if set(merged) != set(states):
        raise ValueError("merged submission PID coverage does not match test.csv")
    write_submission(args.output, merged)

    incumbent_total = sum(len(path) for path in incumbent.values())
    output_total = sum(len(path) for path in merged.values())
    selected_before = sum(len(incumbent[pid]) for pid in selected)
    selected_after = sum(len(final_by_pid[pid]) for pid in selected)
    report = {
        "selected_pids": selected,
        "selection": selection,
        "positive_controls": controls,
        "pid_results": pid_reports,
        "raw_unique_candidates": raw_unique_count,
        "walton_seconds": round(walton_seconds, 3),
        "window_seconds": round(window_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "selected_subtotal_before": selected_before,
        "selected_subtotal_after": selected_after,
        "selected_moves_saved": selected_before - selected_after,
        "incumbent_total": incumbent_total,
        "output_total": output_total,
        "full_score_improvement": incumbent_total - output_total,
        "winner_attribution": dict(sorted(winner_attribution.items())),
        "replay_verified_selected": len(selected),
        "replay_verified_full_submission": len(states),
        "output": str(args.output.resolve()),
        "mean_final_selected": statistics.fmean(len(final_by_pid[pid]) for pid in selected),
    }
    atomic_write(args.report_dir / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
