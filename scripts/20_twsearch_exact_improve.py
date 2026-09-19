"""Run legacy C++ twsearch against selected IHES incumbents and replay wins.

The input to twsearch is the inverse of each known solution (identity ->
scramble).  ``-si`` searches only below that input length.  Output move words
are translated from the `.tws` naming convention and accepted only after exact
72-position replay with the official generators.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission

DEFAULT_EXE = (
    PROJECT
    / "third_party/twips/third_party/twsearch_legacy/build/bin/twsearch.exe"
)
DEFAULT_TWS = Path(
    r"C:\Users\and-l\kaggle_research\cayleypy-ihes-cube\picture_cube_pieces.tws"
)
WINLIBS_BIN = Path(
    r"C:\Users\and-l\AppData\Local\Microsoft\WinGet\Packages"
    r"\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\mingw64\bin"
)


def invert_official(move: str) -> str:
    return move[1:] if move.startswith("-") else "-" + move


def official_to_tws(move: str, prime_inverses: bool = False) -> str:
    if not move.startswith("-"):
        return move
    return move[1:] + "'" if prime_inverses else "i" + move[1:]


def solution_to_scramble(path: list[str], prime_inverses: bool = False) -> str:
    return " ".join(
        official_to_tws(invert_official(move), prime_inverses)
        for move in reversed(path)
    )


def tws_token_to_official(token: str) -> str:
    prime = token.endswith("'")
    if prime:
        token = token[:-1]
    inverse_base = token.startswith("i")
    base = token[1:] if inverse_base else token
    if base not in {"f0", "f1", "f2", "r0", "r1", "r2", "d0", "d1", "d2"}:
        raise ValueError(f"unknown twsearch move {token!r}")
    negative = inverse_base ^ prime
    return ("-" if negative else "") + base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--tws", type=Path, default=DEFAULT_TWS)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT / "data/twsearch_cache")
    parser.add_argument("--memory-mib", type=int, default=16_384)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--start-prune-depth",
        type=int,
        help="force initial pruning-table construction through this depth",
    )
    parser.add_argument("--path-length", type=int, required=True)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--min-depth", type=int)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--time-limit",
        type=int,
        help="maximum seconds per selected pid; requires deadline-enabled twsearch",
    )
    parser.add_argument(
        "--quiet-native",
        action="store_true",
        help="suppress routine native progress while still parsing all output",
    )
    parser.add_argument("--pids", type=int, nargs="*")
    parser.add_argument("--nowrite", action="store_true")
    parser.add_argument(
        "--prime-inverses",
        action="store_true",
        help="encode official inverse moves as base primes (for direct v2 TWS)",
    )
    args = parser.parse_args()
    args.exe = args.exe.resolve()
    args.tws = args.tws.resolve()
    args.cache_dir = args.cache_dir.resolve()

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    states = load_test_states(PROJECT / "data/test.csv")
    baseline_report = verify_submission(puzzle, PROJECT / "data/test.csv", args.baseline)
    if not baseline_report.all_valid:
        raise SystemExit(f"invalid baseline: {baseline_report.failures[:5]}")
    baseline = load_submission(args.baseline)
    selected = [
        pid for pid in sorted(baseline)
        if len(baseline[pid]) == args.path_length
        and (not args.pids or pid in set(args.pids))
    ]
    if not selected:
        raise SystemExit("no rows matched the requested path length/pids")

    scrambles = [
        solution_to_scramble(baseline[pid], args.prime_inverses)
        for pid in selected
    ]
    max_depth = args.max_depth if args.max_depth is not None else args.path_length - 2
    args.cache_dir.mkdir(parents=True, exist_ok=True)
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
    ]
    if args.start_prune_depth is not None:
        command.extend(["--startprunedepth", str(args.start_prune_depth)])
    if args.min_depth is not None:
        command.extend(["--mindepth", str(args.min_depth)])
    if args.random_start:
        command.append("--randomstart")
    if args.seed is not None:
        command.extend(["-R", str(args.seed)])
    if args.time_limit is not None:
        command.extend(["--timelimit", str(args.time_limit)])
    if args.nowrite:
        command.append("--nowrite")
    else:
        command.extend(["--cachedir", str(args.cache_dir)])
    command.append(str(args.tws))

    env = os.environ.copy()
    env["PATH"] = str(WINLIBS_BIN) + os.pathsep + env.get("PATH", "")
    print(
        f"selected {len(selected)} pids at length {args.path_length}; "
        f"search max depth={max_depth}, memory={args.memory_mib} MiB",
        flush=True,
    )
    print("pids: " + " ".join(map(str, selected)), flush=True)

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
    process.stdin.write("\n".join(scrambles) + "\n")
    process.stdin.close()

    blocks: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        if not args.quiet_native or raw_line.startswith(" ") or line.startswith("Found "):
            print(line, flush=True)
        if line == "Solving":
            current = []
            blocks.append(current)
            continue
        if current is None or not raw_line.startswith(" "):
            continue
        tokens = line.split()
        try:
            official = [tws_token_to_official(token) for token in tokens]
        except ValueError:
            continue
        current.append(official)
    return_code = process.wait()
    if return_code:
        raise SystemExit(f"twsearch exited with code {return_code}")
    if len(blocks) != len(selected):
        raise SystemExit(f"expected {len(selected)} solve blocks, got {len(blocks)}")

    output = dict(baseline)
    for pid, solutions in zip(selected, blocks):
        valid = []
        for path in solutions:
            result = verify_path(puzzle, states[pid], path)
            if result.ok and len(path) < len(output[pid]):
                valid.append(path)
        if valid:
            winner = min(valid, key=lambda path: (len(path), path))
            print(f"ACCEPT pid {pid}: {len(output[pid])} -> {len(winner)}", flush=True)
            output[pid] = winner

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(output):
            writer.writerow([pid, ".".join(output[pid])])
    report = verify_submission(puzzle, PROJECT / "data/test.csv", args.out)
    print(
        f"output: {report.n_valid}/{report.n_total} valid, "
        f"{baseline_report.total_moves:,} -> {report.total_moves:,} "
        f"({report.total_moves - baseline_report.total_moves:+,})",
        flush=True,
    )
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
