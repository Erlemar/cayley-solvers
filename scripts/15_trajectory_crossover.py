"""Exact IHES crossover over the union of all verified solution trajectories.

For each puzzle, every state visited by every source solution becomes a graph
vertex.  The graph contains the observed path edges in both directions and,
optionally, every official one-move edge between known vertices.  A BFS then
finds the shortest path from the original scramble to solved through this
shared waypoint graph.  The result is replay-verified before it is written.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--no-closure-one", action="store_true")
    parser.add_argument("--max-source-length", type=int, default=50)
    args = parser.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states0 = load_test_states(PROJECT / "data" / "test.csv")
    move_names = list(puzzle.generators)
    name_to_idx = {name: i for i, name in enumerate(move_names)}
    gens = np.asarray([puzzle.generators[name] for name in move_names], dtype=np.int64)
    inv_idx = np.asarray(
        [name_to_idx[puzzle.inverse_name(name)] for name in move_names], dtype=np.int16
    )
    solved = np.arange(gens.shape[1], dtype=np.uint8)
    solved_key = solved.tobytes()

    baseline_named = load_submission(args.baseline)
    baseline = {
        pid: tuple(name_to_idx[name] for name in baseline_named[pid]) for pid in states0
    }
    baseline_report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.baseline)
    if not baseline_report.all_valid:
        raise SystemExit(f"invalid baseline: {baseline_report.failures[:5]}")

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
            cur = np.asarray(states0[pid], dtype=np.uint8)
            for move in path:
                cur = cur[gens[move]]
            if np.array_equal(cur, solved):
                candidates[pid].add(path)
            else:
                invalid += 1

    print(
        f"sources: {len(files)} CSVs, {rows_seen:,} rows, {invalid:,} invalid, "
        f"{sum(map(len, candidates.values())):,} distinct per-pid paths",
        flush=True,
    )

    output: dict[int, tuple[int, ...]] = {}
    improvements: list[tuple[int, int, int, int, int]] = []
    node_hist = Counter()
    for number, pid in enumerate(sorted(states0), start=1):
        start = np.asarray(states0[pid], dtype=np.uint8)
        start_key = start.tobytes()
        nodes: dict[bytes, int] = {}
        arrays: list[np.ndarray] = []
        adjacency: list[dict[int, int]] = []

        def node_id(state: np.ndarray) -> int:
            key = state.tobytes()
            found = nodes.get(key)
            if found is not None:
                return found
            idx = len(arrays)
            nodes[key] = idx
            arrays.append(state.copy())
            adjacency.append({})
            return idx

        for path in candidates[pid]:
            cur = start
            u = node_id(cur)
            for move in path:
                nxt = cur[gens[move]]
                v = node_id(nxt)
                adjacency[u].setdefault(v, move)
                adjacency[v].setdefault(u, int(inv_idx[move]))
                cur, u = nxt, v
            assert np.array_equal(cur, solved)

        if not args.no_closure_one:
            # A legal edge may connect waypoints from two paths even when that
            # edge was not used by either source solution.
            original_n = len(arrays)
            for u in range(original_n):
                state = arrays[u]
                for move, generator in enumerate(gens):
                    v = nodes.get(state[generator].tobytes())
                    if v is not None:
                        adjacency[u].setdefault(v, move)

        source = nodes[start_key]
        goal = nodes[solved_key]
        parent: list[tuple[int, int] | None] = [None] * len(arrays)
        parent[source] = (-1, -1)
        queue = deque([source])
        while queue and parent[goal] is None:
            u = queue.popleft()
            for v, move in adjacency[u].items():
                if parent[v] is None:
                    parent[v] = (u, move)
                    queue.append(v)
        assert parent[goal] is not None

        path_rev: list[int] = []
        cur_id = goal
        while cur_id != source:
            link = parent[cur_id]
            assert link is not None
            prev, move = link
            path_rev.append(move)
            cur_id = prev
        best = tuple(reversed(path_rev))

        cur = start
        for move in best:
            cur = cur[gens[move]]
        if not np.array_equal(cur, solved):
            raise RuntimeError(f"pid {pid}: crossover reconstruction failed")
        output[pid] = best
        node_hist[len(arrays)] += 1
        if len(best) < len(baseline[pid]):
            improvements.append(
                (pid, len(baseline[pid]), len(best), len(candidates[pid]), len(arrays))
            )
        if number % 100 == 0:
            print(
                f"  processed {number}/{len(states0)}; "
                f"wins={len(improvements)}, saved="
                f"{sum(before-after for _, before, after, _, _ in improvements)}",
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
    for pid, before, after, n_paths, n_nodes in improvements:
        print(
            f"  pid {pid}: {before} -> {after}; "
            f"{n_paths} source paths, {n_nodes} waypoint states"
        )
    print(
        f"output: {report.n_valid}/{report.n_total} valid, "
        f"{baseline_report.total_moves:,} -> {report.total_moves:,} "
        f"({report.total_moves - baseline_report.total_moves:+,})",
        flush=True,
    )
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
