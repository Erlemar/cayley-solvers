"""Use the exact C++ twsearch heuristic to shorten incumbent subpaths.

Each selected window is treated as a small independent scramble.  Twsearch's
depth-10 pruning table can therefore look for the important L -> L-2 rewrite
that a fixed-radius meet-in-the-middle pass cannot cover once L > 14.  Every
reported replacement is replayed as a 72-position permutation before a
maximum-saving set of non-overlapping rewrites is applied.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube
from cayley.verify import load_submission, verify_submission

DEFAULT_EXE = (
    PROJECT
    / "third_party/twips/third_party/twsearch_legacy/build/bin/twsearch.exe"
)
DEFAULT_TWS = PROJECT / "data/picture_cube_pieces_v2_sym.tws"
WINLIBS_BIN = Path(
    r"C:\Users\and-l\AppData\Local\Microsoft\WinGet\Packages"
    r"\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\mingw64\bin"
)


def invert_official(move: str) -> str:
    return move[1:] if move.startswith("-") else "-" + move


def official_to_tws(move: str) -> str:
    return move[1:] + "'" if move.startswith("-") else move


def solution_to_scramble(path: list[str]) -> str:
    return " ".join(
        official_to_tws(invert_official(move)) for move in reversed(path)
    )


def tws_token_to_official(token: str) -> str:
    prime = token.endswith("'")
    base = token[:-1] if prime else token
    if base not in {"f0", "f1", "f2", "r0", "r1", "r2", "d0", "d1", "d2"}:
        raise ValueError(f"unknown twsearch move {token!r}")
    return ("-" if prime else "") + base


def perm_of(path: list[str], generators: dict[str, np.ndarray]) -> np.ndarray:
    state = np.arange(72, dtype=np.int64)
    for move in path:
        state = state[generators[move]]
    return state


def select_non_overlapping(
    path_len: int, candidates: list[tuple[int, int, list[str]]]
) -> list[tuple[int, int, list[str]]]:
    by_start: dict[int, list[tuple[int, int, list[str]]]] = defaultdict(list)
    for candidate in candidates:
        by_start[candidate[0]].append(candidate)
    score = [0] * (path_len + 1)
    choice: list[tuple[int, int, list[str]] | None] = [None] * (path_len + 1)
    for pos in range(path_len - 1, -1, -1):
        score[pos] = score[pos + 1]
        for candidate in by_start.get(pos, ()):
            start, end, word = candidate
            value = end - start - len(word) + score[end]
            if value > score[pos]:
                score[pos] = value
                choice[pos] = candidate
    selected = []
    pos = 0
    while pos < path_len:
        candidate = choice[pos]
        if candidate is None:
            pos += 1
        else:
            selected.append(candidate)
            pos = candidate[1]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window-length", required=True, type=int)
    parser.add_argument("--min-path-length", type=int, default=0)
    parser.add_argument("--pids", type=int, nargs="*")
    parser.add_argument(
        "--skip-windows",
        type=int,
        default=0,
        help="Skip this many windows in deterministic pid/start order",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--tws", type=Path, default=DEFAULT_TWS)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT / "data/twsearch_cache")
    parser.add_argument("--memory-mib", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--start-prune-depth", type=int, default=10)
    parser.add_argument("--min-depth", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--time-limit", type=int, default=2)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quiet-native", action="store_true")
    args = parser.parse_args()

    args.exe = args.exe.resolve()
    args.tws = args.tws.resolve()
    args.cache_dir = args.cache_dir.resolve()
    max_depth = args.max_depth if args.max_depth is not None else args.window_length - 2

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    generators = {
        name: np.asarray(values, dtype=np.int64)
        for name, values in puzzle.generators.items()
    }
    report0 = verify_submission(puzzle, PROJECT / "data/test.csv", args.baseline)
    if not report0.all_valid:
        raise SystemExit(f"invalid baseline: {report0.failures[:5]}")
    rows = load_submission(args.baseline)
    original_lengths = {pid: len(path) for pid, path in rows.items()}
    pid_filter = set(args.pids or ())

    metadata: list[tuple[int, int, list[str], np.ndarray]] = []
    for pid in sorted(rows):
        path = rows[pid]
        if len(path) < max(args.min_path_length, args.window_length):
            continue
        if pid_filter and pid not in pid_filter:
            continue
        for start in range(len(path) - args.window_length + 1):
            window = path[start : start + args.window_length]
            metadata.append((pid, start, window, perm_of(window, generators)))
    stop = (
        None
        if args.max_windows is None
        else args.skip_windows + args.max_windows
    )
    metadata = metadata[args.skip_windows:stop]
    if not metadata:
        raise SystemExit("no windows matched")

    command = [
        str(args.exe),
        "-q",
        "-si",
        "-M",
        str(args.memory_mib),
        "-t",
        str(args.threads),
        "--maxdepth",
        str(max_depth),
        "--startprunedepth",
        str(args.start_prune_depth),
        "--timelimit",
        str(args.time_limit),
        "--cachedir",
        str(args.cache_dir),
    ]
    if args.min_depth is not None:
        command.extend(["--mindepth", str(args.min_depth)])
    if args.random_start:
        command.append("--randomstart")
    if args.seed is not None:
        command.extend(["-R", str(args.seed)])
    command.append(str(args.tws))

    env = os.environ.copy()
    env["PATH"] = str(WINLIBS_BIN) + os.pathsep + env.get("PATH", "")
    print(
        f"selected {len(metadata):,} windows of length {args.window_length}; "
        f"search depths {args.min_depth or 'auto'}..{max_depth}, "
        f"limit={args.time_limit}s/window",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=args.exe.parent.parent.parent,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None
    # Large campaigns exceed both Windows pipe buffers.  Feed stdin on a
    # separate thread while the main thread continuously drains stdout.
    feed_errors: list[BaseException] = []

    def feed_scrambles() -> None:
        try:
            assert process.stdin is not None
            for item in metadata:
                process.stdin.write(solution_to_scramble(item[2]) + "\n")
            process.stdin.close()
        except BaseException as exc:  # surfaced after the child exits
            feed_errors.append(exc)

    feeder = threading.Thread(target=feed_scrambles, name="twsearch-stdin", daemon=True)
    feeder.start()

    blocks: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    completed = 0
    timed_out = 0
    t0 = time.time()
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        if line == "Solving":
            current = []
            blocks.append(current)
            completed = len(blocks) - 1
            if completed and completed % 25 == 0:
                print(
                    f"processed {completed:,}/{len(metadata):,}; "
                    f"timeouts={timed_out:,}; {time.time() - t0:.1f}s",
                    flush=True,
                )
            continue
        if line.startswith("Search timed out"):
            timed_out += 1
        if not args.quiet_native:
            print(line, flush=True)
        if current is None or not raw_line.startswith(" "):
            continue
        try:
            current.append([tws_token_to_official(token) for token in line.split()])
        except ValueError:
            continue
    return_code = process.wait()
    feeder.join()
    if return_code:
        raise SystemExit(f"twsearch exited with code {return_code}")
    if feed_errors:
        raise SystemExit(f"failed to stream scrambles: {feed_errors[0]}")
    if len(blocks) != len(metadata):
        raise SystemExit(f"expected {len(metadata)} solve blocks, got {len(blocks)}")

    candidates: dict[int, list[tuple[int, int, list[str]]]] = defaultdict(list)
    rejected = 0
    for item, solutions in zip(metadata, blocks):
        pid, start, window, target_perm = item
        valid = [
            word
            for word in solutions
            if len(word) < len(window)
            and np.array_equal(perm_of(word, generators), target_perm)
        ]
        rejected += len(solutions) - len(valid)
        if valid:
            winner = min(valid, key=lambda word: (len(word), word))
            candidates[pid].append((start, start + len(window), winner))

    saved = 0
    for pid, options in candidates.items():
        selected = select_non_overlapping(len(rows[pid]), options)
        for start, end, word in reversed(selected):
            saved += end - start - len(word)
            rows[pid][start:end] = word

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            writer.writerow([pid, ".".join(rows[pid])])
    report = verify_submission(puzzle, PROJECT / "data/test.csv", args.out)
    print(
        f"search: blocks={len(blocks):,}, timeouts={timed_out:,}, "
        f"valid raw hits={sum(map(len, candidates.values())):,}, rejected={rejected:,}",
        flush=True,
    )
    for pid in sorted(rows):
        if len(rows[pid]) < original_lengths[pid]:
            print(f"  pid {pid}: {original_lengths[pid]} -> {len(rows[pid])}")
    print(
        f"output: {report.n_valid}/{report.n_total} valid, "
        f"{report0.total_moves:,} -> {report.total_moves:,} "
        f"({report.total_moves - report0.total_moves:+,}; selected savings={saved})",
        flush=True,
    )
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
