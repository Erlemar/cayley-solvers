"""Mine exact word relations from IHES submission paths and shorten a baseline.

Every contiguous word in every source submission is interpreted as an exact
72-position group permutation.  If a source word realizes the same permutation
as a longer baseline window, it is a safe replacement.  Inverse words are mined
as well.  Candidate rewrites are selected with a shortest-path dynamic program,
so overlapping rewrites are optimized globally rather than accepted greedily.

All source paths and the final output are replay-verified against the official
test states.  Exact permutation bytes are dictionary keys; no probabilistic hash
match is ever trusted.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission


def prefix_permutations(path: tuple[int, ...], gens: np.ndarray) -> np.ndarray:
    """Return identity and every prefix permutation in puzzle-info convention."""
    out = np.empty((len(path) + 1, gens.shape[1]), dtype=np.uint8)
    out[0] = np.arange(gens.shape[1], dtype=np.uint8)
    for i, move in enumerate(path):
        out[i + 1] = out[i][gens[move]]
    return out


def relative_windows(prefix: np.ndarray):
    """Yield (i, j, exact relative permutation bytes) for all nonempty windows."""
    length = prefix.shape[0] - 1
    for i in range(length):
        inv = np.argsort(prefix[i])
        rel = inv[prefix[i + 1 :]]
        for offset, perm in enumerate(rel, start=1):
            # argsort promotes to the platform integer dtype.  Canonicalize to
            # one byte per position so keys are compact and inversion/symmetry
            # code sees a 72-entry permutation rather than its raw int64 bytes.
            yield i, i + offset, perm.astype(np.uint8, copy=False).tobytes()


def inverse_word(word: tuple[int, ...], inv_idx: np.ndarray) -> tuple[int, ...]:
    return tuple(int(inv_idx[m]) for m in reversed(word))


def zobrist_hash_batch(perms: np.ndarray, ztab: np.ndarray) -> np.ndarray:
    perms = np.atleast_2d(perms).astype(np.int64, copy=False)
    return np.bitwise_xor.reduce(ztab[np.arange(perms.shape[1]), perms], axis=1)


def load_csv_paths(path: Path, name_to_idx: dict[str, int]) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"initial_state_id", "path"} <= fields:
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
    return out


def best_rewrite(
    path: tuple[int, ...],
    gens: np.ndarray,
    rules: dict[bytes, tuple[int, ...]],
) -> tuple[tuple[int, ...], list[tuple[int, int, tuple[int, ...]]]]:
    """Globally maximize savings among exact endpoint-preserving rewrites."""
    length = len(path)
    edges: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    prefix = prefix_permutations(path, gens)
    for i, j, key in relative_windows(prefix):
        replacement = rules.get(key)
        if replacement is not None and len(replacement) < j - i:
            edges[i].append((j, replacement))

    # Shortest word length through the interval DAG.  Ordinary path moves are
    # unit edges; a rewrite is an exact macro edge between the same states.
    cost = [10**9] * (length + 1)
    parent: list[tuple[int, tuple[int, ...]] | None] = [None] * (length + 1)
    cost[0] = 0
    for i in range(length):
        if cost[i] + 1 < cost[i + 1]:
            cost[i + 1] = cost[i] + 1
            parent[i + 1] = (i, (path[i],))
        for j, replacement in edges.get(i, ()):
            candidate = cost[i] + len(replacement)
            if candidate < cost[j]:
                cost[j] = candidate
                parent[j] = (i, replacement)

    chunks: list[tuple[int, int, tuple[int, ...]]] = []
    pos = length
    while pos:
        link = parent[pos]
        assert link is not None
        prev, word = link
        chunks.append((prev, pos, word))
        pos = prev
    chunks.reverse()
    rewritten = tuple(move for _, _, word in chunks for move in word)
    used = [(i, j, word) for i, j, word in chunks if j - i != 1 or word != (path[i],)]
    return rewritten, used


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=[PROJECT / "submissions"],
        help="directories or CSV files containing alternative verified paths",
    )
    parser.add_argument("--max-source-length", type=int, default=40)
    parser.add_argument(
        "--symmetry-dir",
        type=Path,
        default=None,
        help="directory containing cube_symmetries.npy and cube_symmetries_inv.npy; "
        "enables all 48 conjugacies in addition to inversion",
    )
    args = parser.parse_args()

    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    states = load_test_states(PROJECT / "data" / "test.csv")
    move_names = list(puzzle.generators)
    name_to_idx = {name: i for i, name in enumerate(move_names)}
    gens = np.asarray([puzzle.generators[name] for name in move_names], dtype=np.int64)
    inv_idx = np.asarray(
        [name_to_idx[puzzle.inverse_name(name)] for name in move_names], dtype=np.int16
    )

    baseline_named = load_submission(args.baseline)
    baseline = {
        pid: tuple(name_to_idx[name] for name in baseline_named[pid]) for pid in states
    }
    baseline_report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.baseline)
    if not baseline_report.all_valid:
        raise SystemExit(f"baseline is invalid: {baseline_report.failures[:5]}")

    print(
        f"baseline: {baseline_report.n_valid}/{baseline_report.n_total} valid, "
        f"{baseline_report.total_moves:,} moves",
        flush=True,
    )

    # Build the exact set of group elements that can actually rewrite the
    # baseline.  Mining only these targets keeps the relation corpus compact.
    target_max_length: dict[bytes, int] = {}
    t0 = time.time()
    for path in baseline.values():
        prefix = prefix_permutations(path, gens)
        for i, j, key in relative_windows(prefix):
            old = target_max_length.get(key, 0)
            if j - i > old:
                target_max_length[key] = j - i
    print(
        f"baseline target elements: {len(target_max_length):,} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    sym = sym_inv = conjugated_move = orbit_hashes = ztab = None
    if args.symmetry_dir is not None:
        sym = np.load(args.symmetry_dir / "cube_symmetries.npy").astype(np.int64)
        sym_inv = np.load(args.symmetry_dir / "cube_symmetries_inv.npy").astype(np.int64)
        if sym.shape != sym_inv.shape or sym.shape[1] != gens.shape[1]:
            raise SystemExit(f"bad symmetry artifact shapes: {sym.shape}, {sym_inv.shape}")

        gen_by_perm = {tuple(g.tolist()): i for i, g in enumerate(gens)}
        conjugated_move = np.empty((len(sym), len(gens)), dtype=np.int16)
        for s, (perm, perm_inv) in enumerate(zip(sym, sym_inv)):
            for move, generator in enumerate(gens):
                transformed = perm[generator[perm_inv]]
                conjugated_move[s, move] = gen_by_perm[tuple(transformed.tolist())]

        rng = np.random.default_rng(20260804)
        ztab = rng.integers(
            0, np.iinfo(np.uint64).max, size=(gens.shape[1], gens.shape[1]), dtype=np.uint64
        )
        target_perms = np.stack(
            [np.frombuffer(key, dtype=np.uint8) for key in target_max_length]
        )
        n_targets = len(target_perms)
        orbit_hashes = np.empty(n_targets * len(sym) * 2, dtype=np.uint64)
        batch_size = 1024
        cursor = 0
        t_orbit = time.time()
        for start in range(0, n_targets, batch_size):
            base = target_perms[start : start + batch_size]
            inverse = np.argsort(base, axis=1).astype(np.uint8, copy=False)
            for family in (base, inverse):
                for perm, perm_inv in zip(sym, sym_inv):
                    transformed = perm[family[:, perm_inv]]
                    hashes = zobrist_hash_batch(transformed, ztab)
                    orbit_hashes[cursor : cursor + len(hashes)] = hashes
                    cursor += len(hashes)
        assert cursor == len(orbit_hashes)
        orbit_hashes = np.unique(orbit_hashes)
        print(
            f"symmetry orbit filter: {len(sym)} conjugacies x inversion -> "
            f"{len(orbit_hashes):,} distinct 64-bit probes "
            f"({time.time() - t_orbit:.1f}s; exact bytes checked after every hit)",
            flush=True,
        )

    files: list[Path] = []
    for root in args.roots:
        if root.is_file() and root.suffix.lower() == ".csv":
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("*.csv"))
    files = sorted(set(path.resolve() for path in files))

    unique_paths: set[tuple[int, ...]] = set()
    valid_rows = invalid_rows = 0
    for file in files:
        for pid, path in load_csv_paths(file, name_to_idx).items():
            if pid not in states or len(path) > args.max_source_length:
                continue
            named = [move_names[m] for m in path]
            if verify_path(puzzle, states[pid], named).ok:
                valid_rows += 1
                unique_paths.add(path)
            else:
                invalid_rows += 1
    print(
        f"sources: {len(files)} CSVs, {valid_rows:,} valid rows, "
        f"{invalid_rows:,} invalid, {len(unique_paths):,} distinct words",
        flush=True,
    )

    rules: dict[bytes, tuple[int, ...]] = {}
    hits = 0
    t1 = time.time()
    for number, path in enumerate(sorted(unique_paths, key=lambda p: (len(p), p)), start=1):
        prefix = prefix_permutations(path, gens)
        if orbit_hashes is None:
            for i, j, key in relative_windows(prefix):
                word = path[i:j]
                target_length = target_max_length.get(key)
                if target_length is not None and len(word) < target_length:
                    prior = rules.get(key)
                    if prior is None or len(word) < len(prior):
                        rules[key] = word
                        hits += 1

                # Inversion is an exact distance-preserving group anti-automorphism.
                perm = np.frombuffer(key, dtype=np.uint8)
                inverse_key = np.argsort(perm).astype(np.uint8, copy=False).tobytes()
                target_length = target_max_length.get(inverse_key)
                if target_length is not None and len(word) < target_length:
                    inv_word = inverse_word(word, inv_idx)
                    prior = rules.get(inverse_key)
                    if prior is None or len(inv_word) < len(prior):
                        rules[inverse_key] = inv_word
                        hits += 1
        else:
            assert sym is not None and sym_inv is not None
            assert conjugated_move is not None and ztab is not None
            length = len(path)
            for i in range(length):
                inv_prefix = np.argsort(prefix[i])
                relatives = inv_prefix[prefix[i + 1 :]]
                hashes = zobrist_hash_batch(relatives, ztab)
                positions = np.searchsorted(orbit_hashes, hashes)
                positions = np.minimum(positions, len(orbit_hashes) - 1)
                possible = np.nonzero(orbit_hashes[positions] == hashes)[0]
                for offset0 in possible.tolist():
                    j = i + offset0 + 1
                    word = path[i:j]
                    perm0 = relatives[offset0]
                    inv_perm0 = np.argsort(perm0).astype(np.uint8, copy=False)
                    inv_word0 = inverse_word(word, inv_idx)
                    for family_perm, family_word in ((perm0, word), (inv_perm0, inv_word0)):
                        transformed_perms = np.take_along_axis(
                            sym, family_perm[sym_inv], axis=1
                        )
                        for s, transformed_perm in enumerate(transformed_perms):
                            key = transformed_perm.astype(np.uint8, copy=False).tobytes()
                            target_length = target_max_length.get(key)
                            if target_length is None or len(family_word) >= target_length:
                                continue
                            transformed_word = tuple(
                                int(conjugated_move[s, move]) for move in family_word
                            )
                            prior = rules.get(key)
                            if prior is None or len(transformed_word) < len(prior):
                                rules[key] = transformed_word
                                hits += 1
        if number % 2000 == 0:
            print(
                f"  mined {number:,}/{len(unique_paths):,} words; "
                f"{len(rules):,} useful exact rules",
                flush=True,
            )
    print(
        f"mined {len(rules):,} useful rules ({hits:,} improvements while mining) "
        f"in {time.time() - t1:.1f}s",
        flush=True,
    )

    improved: dict[int, tuple[int, ...]] = {}
    rewrite_log: list[tuple[int, int, int, int]] = []
    for pid, original in baseline.items():
        path = original
        while True:
            candidate, used = best_rewrite(path, gens, rules)
            if len(candidate) >= len(path):
                break
            before = len(path)
            path = candidate
            rewrite_log.append((pid, before, len(path), len(used)))
        named = [move_names[m] for m in path]
        if not verify_path(puzzle, states[pid], named).ok:
            raise RuntimeError(f"pid {pid}: exact-rule rewrite failed replay")
        improved[pid] = path

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(improved):
            writer.writerow([pid, ".".join(move_names[m] for m in improved[pid])])

    report = verify_submission(puzzle, PROJECT / "data" / "test.csv", args.out)
    print(f"rewritten pids: {len({row[0] for row in rewrite_log})}", flush=True)
    for pid, before, after, count in rewrite_log:
        print(f"  pid {pid}: {before} -> {after} via {count} macro edge(s)")
    print(
        f"output: {report.n_valid}/{report.n_total} valid, "
        f"{baseline_report.total_moves:,} -> {report.total_moves:,} "
        f"({report.total_moves - baseline_report.total_moves:+,})",
        flush=True,
    )
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
