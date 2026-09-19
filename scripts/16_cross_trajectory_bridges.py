"""Splice IHES solution trajectories with exact short bridges.

For an incumbent length L, a prefix ending at state A and an alternative
solution suffix beginning at state B form a shorter solution whenever an exact
bridge A -> B has length d and

    prefix_length + d + suffix_length <= L - 2.

The bridge oracle is an existing exact BFS word table (normally d5).  This is
structurally different from replacing a window inside one path: its endpoints
come from two independently generated trajectories.  Every accepted splice and
the final CSV are replay-verified.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.bfs_table import BfsTable
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_submission


def load_paths(path: Path, name_to_idx: dict[str, int]) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"initial_state_id", "path"} <= set(reader.fieldnames or ()):
                return out
            for row in reader:
                text = row.get("path", "").strip()
                if not text:
                    continue
                try:
                    out[int(row["initial_state_id"])] = tuple(
                        name_to_idx[name] for name in text.split(".")
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    except (OSError, UnicodeError, csv.Error):
        return {}
    return out


def trace_path(start: np.ndarray, path: tuple[int, ...], gens: np.ndarray) -> np.ndarray:
    trace = np.empty((len(path) + 1, start.shape[0]), dtype=np.uint8)
    trace[0] = start
    for i, move in enumerate(path):
        trace[i + 1] = trace[i][gens[move]]
    return trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bfs-table", required=True, type=Path)
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-source-length", type=int, default=50)
    args = parser.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states0 = load_test_states(PROJECT / "data" / "test.csv")
    move_names = list(puzzle.generators)
    name_to_idx = {name: i for i, name in enumerate(move_names)}
    gens = np.asarray([puzzle.generators[name] for name in move_names], dtype=np.int64)
    solved = np.arange(gens.shape[1], dtype=np.uint8)

    baseline_named = load_submission(args.baseline)
    baseline = {
        pid: tuple(name_to_idx[name] for name in baseline_named[pid]) for pid in states0
    }
    baseline_report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.baseline)
    if not baseline_report.all_valid:
        raise SystemExit(f"invalid baseline: {baseline_report.failures[:5]}")

    t_table = time.time()
    print(f"bridge table: loading {args.bfs_table}...", flush=True)
    table_obj = BfsTable.load(args.bfs_table)
    print(
        f"bridge table: loaded {len(table_obj.table):,} tuple keys; compacting...",
        flush=True,
    )
    # Compact exact keys are substantially faster to construct in the inner
    # loop than 72-element Python tuples.  The stored word remains exact.
    table = {
        bytes(perm): word
        for perm, word in table_obj.table.items()
    }
    del table_obj
    print(
        f"bridge table: {len(table):,} exact elements at d<={max(map(len, table.values()))} "
        f"({time.time() - t_table:.1f}s)",
        flush=True,
    )

    files: list[Path] = []
    for root in args.roots:
        if root.is_file() and root.suffix.lower() == ".csv":
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.csv"))
    files = sorted(set(path.resolve() for path in files))

    candidates: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    for pid, path in baseline.items():
        candidates[pid].add(path)
    rows_seen = invalid = 0
    for file in files:
        for pid, path in load_paths(file, name_to_idx).items():
            if pid not in states0 or len(path) > args.max_source_length:
                continue
            rows_seen += 1
            trace = trace_path(np.asarray(states0[pid], dtype=np.uint8), path, gens)
            if np.array_equal(trace[-1], solved):
                candidates[pid].add(path)
            else:
                invalid += 1
    print(
        f"sources: {len(files)} CSVs, {rows_seen:,} rows, {invalid:,} invalid, "
        f"{sum(map(len, candidates.values())):,} distinct per-pid paths",
        flush=True,
    )

    output: dict[int, tuple[int, ...]] = {}
    improvements: list[tuple[int, int, int, int, int, int]] = []
    total_probes = 0
    t_search = time.time()
    for number, pid in enumerate(sorted(states0), start=1):
        start = np.asarray(states0[pid], dtype=np.uint8)
        source_paths = sorted(candidates[pid], key=lambda p: (len(p), p))[: args.top_k]
        traces = [trace_path(start, path, gens) for path in source_paths]
        inverse_traces = [np.argsort(trace, axis=1) for trace in traces]
        best = baseline[pid]
        best_detail: tuple[int, int, int, int] | None = None

        changed = True
        while changed:
            changed = False
            # Include a newly found splice as another trajectory on the next pass.
            if best not in source_paths:
                source_paths.append(best)
                traces.append(trace_path(start, best, gens))
                inverse_traces.append(np.argsort(traces[-1], axis=1))

            for a, path_a in enumerate(source_paths):
                trace_a = traces[a]
                inv_a = inverse_traces[a]
                for b, path_b in enumerate(source_paths):
                    if a == b:
                        continue
                    trace_b = traces[b]
                    len_b = len(path_b)
                    for i in range(min(len(path_a), len(best) - 2) + 1):
                        # Need at least a two-move saving; parity makes a one-move
                        # improvement impossible for this generator set.
                        max_suffix = len(best) - 2 - i
                        if max_suffix < 0:
                            continue
                        j_min = max(0, len_b - max_suffix)
                        if j_min > len_b:
                            continue
                        relative = inv_a[i][trace_b[j_min:]]
                        total_probes += len(relative)
                        for offset, perm in enumerate(relative):
                            j = j_min + offset
                            word = table.get(perm.astype(np.uint8, copy=False).tobytes())
                            if word is None:
                                continue
                            candidate_len = i + len(word) + (len_b - j)
                            if candidate_len >= len(best):
                                continue
                            candidate = path_a[:i] + word + path_b[j:]
                            candidate_trace = trace_path(start, candidate, gens)
                            if not np.array_equal(candidate_trace[-1], solved):
                                raise RuntimeError(f"pid {pid}: exact bridge failed replay")
                            old_len = len(best)
                            best = candidate
                            best_detail = (a, i, b, j)
                            changed = True
                            improvements.append(
                                (pid, old_len, len(best), a, i, b)
                            )
                            break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break

        output[pid] = best
        if number % 50 == 0:
            saved = baseline_report.total_moves - (
                sum(len(output[p]) for p in output)
                + sum(len(baseline[p]) for p in states0 if p not in output)
            )
            print(
                f"  processed {number}/{len(states0)}; "
                f"improved={len({row[0] for row in improvements})}, "
                f"saved={saved}, probes={total_probes:,}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(output):
            writer.writerow([pid, ".".join(move_names[m] for m in output[pid])])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print("improvements:")
    for pid in sorted({row[0] for row in improvements}):
        print(f"  pid {pid}: {len(baseline[pid])} -> {len(output[pid])}")
    print(
        f"output: {report.n_valid}/{report.n_total} valid, "
        f"{baseline_report.total_moves:,} -> {report.total_moves:,} "
        f"({report.total_moves - baseline_report.total_moves:+,}); "
        f"{total_probes:,} exact bridge probes in {time.time() - t_search:.1f}s",
        flush=True,
    )
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
