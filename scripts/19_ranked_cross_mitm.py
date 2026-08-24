"""Neural-ranked exact radius-11 bridges between IHES solution trajectories.

All-pairs exact waypoint joining is too large (the d6 experiment made 87M
probes).  This script generates only splice-feasible waypoint pairs, ranks their
relative states with the trained IHES value model, and sends the best few per
puzzle to the replay-verified d5+d6 GPU MITM oracle from ``18_mitm_gpu.py``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube
from cayley.search import load_model_checkpoint
from cayley.verify import (
    load_submission,
    load_test_states,
    verify_path,
    verify_submission,
)


def load_exact_joiner_class():
    path = PROJECT / "scripts/18_mitm_gpu.py"
    spec = importlib.util.spec_from_file_location("ihes_mitm_gpu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExactJoiner


def load_paths(path: Path, name_to_idx: dict[str, int]) -> dict[int, tuple[int, ...]]:
    out = {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"initial_state_id", "path"} <= set(reader.fieldnames or ()):
                return out
            for row in reader:
                text = (row.get("path") or "").strip()
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


def trace_path(start: np.ndarray, path: tuple[int, ...], gen: np.ndarray) -> np.ndarray:
    trace = np.empty((len(path) + 1, start.size), dtype=np.uint8)
    trace[0] = start
    for i, move in enumerate(path):
        trace[i + 1] = trace[i][gen[move]]
    return trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "models/small_e5/epoch_7999.pt")
    parser.add_argument("--front-table", type=Path, default=PROJECT / "data/bfs_table_d5.pkl")
    parser.add_argument("--endgame-hash", type=Path, default=PROJECT / "data/bfs_table_d6_hash.npz")
    parser.add_argument("--top-k-paths", type=int, default=5)
    parser.add_argument("--exact-per-pid", type=int, default=20)
    parser.add_argument("--rank-batch", type=int, default=16_384)
    parser.add_argument("--exact-batch", type=int, default=64)
    parser.add_argument("--max-source-length", type=int, default=50)
    parser.add_argument("--min-path-length", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    states0 = load_test_states(PROJECT / "data/test.csv")
    names = list(puzzle.move_names)
    name_to_idx = {name: i for i, name in enumerate(names)}
    gen = np.asarray([puzzle.generators[name] for name in names], dtype=np.int64)
    solved = np.arange(gen.shape[1], dtype=np.uint8)

    baseline_report = verify_submission(puzzle, PROJECT / "data/test.csv", args.baseline)
    if not baseline_report.all_valid:
        raise SystemExit(f"invalid baseline: {baseline_report.failures[:5]}")
    baseline_named = load_submission(args.baseline)
    baseline = {
        pid: tuple(name_to_idx[name] for name in path)
        for pid, path in baseline_named.items()
    }
    target_pids = {
        pid for pid, path in baseline.items() if len(path) >= args.min_path_length
    }

    files: list[Path] = []
    for root in args.roots:
        if root.is_file() and root.suffix.lower() == ".csv":
            files.append(root.resolve())
        elif root.is_dir():
            files.extend(path.resolve() for path in root.rglob("*.csv"))
    files = sorted(set(files))

    candidates: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    for pid, path in baseline.items():
        candidates[pid].add(path)
    rows_seen = invalid = 0
    for file in files:
        for pid, path in load_paths(file, name_to_idx).items():
            if pid not in target_pids or len(path) > args.max_source_length:
                continue
            rows_seen += 1
            trace = trace_path(np.asarray(states0[pid], dtype=np.uint8), path, gen)
            if np.array_equal(trace[-1], solved):
                candidates[pid].add(path)
            else:
                invalid += 1
    print(
        f"sources: {len(files)} CSVs, {rows_seen:,} rows, {invalid:,} invalid; "
        f"target pids={len(target_pids)}",
        flush=True,
    )

    # Each metadata row is pid, path_a, prefix_i, path_b, suffix_j, max_bridge.
    relative_states: list[np.ndarray] = []
    metadata: list[tuple[int, int, int, int, int, int]] = []
    selected_paths: dict[int, list[tuple[int, ...]]] = {}
    t0 = time.time()
    for number, pid in enumerate(sorted(target_pids), start=1):
        paths = sorted(candidates[pid], key=lambda p: (len(p), p))[: args.top_k_paths]
        selected_paths[pid] = paths
        start = np.asarray(states0[pid], dtype=np.uint8)
        traces = [trace_path(start, path, gen) for path in paths]
        inverse = [np.argsort(trace, axis=1) for trace in traces]
        incumbent = len(baseline[pid])
        for a, path_a in enumerate(paths):
            for b, path_b in enumerate(paths):
                if a == b:
                    continue
                len_b = len(path_b)
                for i in range(len(path_a) + 1):
                    # max_bridge = incumbent - 2 - i - (len_b - j) >= 7
                    j_min = max(0, len_b - (incumbent - 2 - i - 7))
                    if j_min > len_b:
                        continue
                    for j in range(j_min, len_b + 1):
                        max_bridge = incumbent - 2 - i - (len_b - j)
                        if max_bridge < 7:
                            continue
                        relative_states.append(inverse[a][i][traces[b][j]].copy())
                        metadata.append((pid, a, i, b, j, max_bridge))
        if number % 100 == 0:
            print(
                f"  generated {number}/{len(target_pids)} pids, "
                f"{len(metadata):,} splice-feasible pairs",
                flush=True,
            )
    rel = np.asarray(relative_states, dtype=np.uint8)
    del relative_states
    meta = np.asarray(metadata, dtype=np.int16)
    del metadata
    print(f"ranking {rel.shape[0]:,} relative states ({time.time() - t0:.1f}s setup)", flush=True)

    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = load_model_checkpoint(args.checkpoint, device=args.device, dtype=dtype)
    model.eval()
    scores = np.empty(rel.shape[0], dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, rel.shape[0], args.rank_batch):
            hi = min(lo + args.rank_batch, rel.shape[0])
            tensor = torch.from_numpy(rel[lo:hi]).to(args.device)
            scores[lo:hi] = model(tensor).float().reshape(-1).cpu().numpy()
            if hi % (args.rank_batch * 20) == 0 or hi == rel.shape[0]:
                print(f"  ranked {hi:,}/{rel.shape[0]:,}", flush=True)
    del model
    torch.cuda.empty_cache()

    # Rank by predicted feasibility margin.  The exact oracle reaches at most 11.
    allowance = np.minimum(meta[:, 5].astype(np.float32), 11.0)
    margin = scores - allowance
    chosen: list[int] = []
    for pid in sorted(target_pids):
        idx = np.flatnonzero(meta[:, 0] == pid)
        if idx.size == 0:
            continue
        take = min(args.exact_per_pid, idx.size)
        local = np.argpartition(margin[idx], take - 1)[:take]
        local = local[np.argsort(margin[idx][local])]
        chosen.extend(idx[local].tolist())
    chosen_idx = np.asarray(chosen, dtype=np.int64)
    print(
        f"exact stage: {chosen_idx.size:,} candidates; "
        f"margin range {margin[chosen_idx].min():.2f}..{margin[chosen_idx].max():.2f}",
        flush=True,
    )

    ExactJoiner = load_exact_joiner_class()
    joiner = ExactJoiner(puzzle, args.front_table, args.endgame_hash, args.device)
    best = dict(baseline)
    raw_hits = 0
    t_exact = time.time()
    for lo in range(0, chosen_idx.size, args.exact_batch):
        hi = min(lo + args.exact_batch, chosen_idx.size)
        idx = chosen_idx[lo:hi]
        best_lengths = np.minimum(meta[idx, 5].astype(np.int64), 11) + 1
        bridges = joiner.solve_batch(rel[idx], best_lengths)
        for row_i, bridge in zip(idx, bridges):
            if bridge is None:
                continue
            raw_hits += 1
            pid, a, i, b, j, max_bridge = map(int, meta[row_i])
            paths = selected_paths[pid]
            candidate = paths[a][:i] + tuple(bridge) + paths[b][j:]
            if len(candidate) >= len(best[pid]):
                continue
            named_candidate = [names[m] for m in candidate]
            result = verify_path(puzzle, states0[pid], named_candidate)
            if not result.ok:
                raise RuntimeError(f"pid {pid}: exact bridge splice failed replay")
            best[pid] = candidate
            print(
                f"  HIT pid {pid}: {len(baseline[pid])} -> {len(candidate)} "
                f"(bridge {len(bridge)} <= {max_bridge})",
                flush=True,
            )
        if hi % (args.exact_batch * 20) == 0 or hi == chosen_idx.size:
            print(
                f"  exact {hi:,}/{chosen_idx.size:,}; raw hits={raw_hits}, "
                f"improved={sum(len(best[p]) < len(baseline[p]) for p in best)}, "
                f"{time.time() - t_exact:.1f}s",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(best):
            writer.writerow([pid, ".".join(names[m] for m in best[pid])])
    report = verify_submission(puzzle, PROJECT / "data/test.csv", args.out)
    print("improvements:")
    for pid in sorted(best):
        if len(best[pid]) < len(baseline[pid]):
            print(f"  pid {pid}: {len(baseline[pid])} -> {len(best[pid])}")
    print(
        f"output: {report.n_valid}/{report.n_total} valid, "
        f"{baseline_report.total_moves:,} -> {report.total_moves:,} "
        f"({report.total_moves - baseline_report.total_moves:+,}); "
        f"raw exact hits={raw_hits}, phantoms={joiner.phantoms}",
        flush=True,
    )
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
