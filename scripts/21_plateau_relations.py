"""Escape local post-processing plateaus with exact neutral word rewrites.

Equal-length words for the same exact group element are mined from all verified
submission trajectories.  Each neutral substitution preserves the complete
solution, but changes its local word geometry.  Windows crossing either new
boundary are then reduced with the exact BFS depth-6 table.  Hashes are not
used: relation keys and BFS lookups are full 72-position permutations, and
every accepted whole path is replay-verified.
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.bfs_table import BfsTable
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_path, verify_submission


def prefix_permutations(path: tuple[int, ...], gens: np.ndarray) -> np.ndarray:
    out = np.empty((len(path) + 1, gens.shape[1]), dtype=np.uint8)
    out[0] = np.arange(gens.shape[1], dtype=np.uint8)
    for i, move in enumerate(path):
        out[i + 1] = out[i][gens[move]]
    return out


def window_key(prefix: np.ndarray, start: int, end: int) -> bytes:
    inv = np.argsort(prefix[start])
    return inv[prefix[end]].astype(np.uint8, copy=False).tobytes()


def inverse_word(word: tuple[int, ...], inv_idx: np.ndarray) -> tuple[int, ...]:
    return tuple(int(inv_idx[m]) for m in reversed(word))


def load_csv_paths(path: Path, name_to_idx: dict[str, int]) -> dict[int, tuple[int, ...]]:
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


def scan_boundary_reductions(
    path: tuple[int, ...],
    changed_start: int,
    changed_end: int,
    gens: np.ndarray,
    exact: dict[bytes, tuple[int, ...]],
    min_window: int,
    max_window: int,
) -> list[tuple[int, int, tuple[int, ...]]]:
    prefix = prefix_permutations(path, gens)
    length = len(path)
    out: list[tuple[int, int, tuple[int, ...]]] = []
    for start in range(length):
        stop = min(length, start + max_window)
        for end in range(start + min_window, stop + 1):
            crosses = (start < changed_start < end) or (start < changed_end < end)
            if not crosses:
                continue
            replacement = exact.get(window_key(prefix, start, end))
            if replacement is not None and len(replacement) < end - start:
                out.append((start, end, replacement))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bfs-table", type=Path, default=PROJECT / "data/bfs_table_d6.pkl")
    parser.add_argument("--roots", nargs="+", type=Path, default=[PROJECT / "submissions"])
    parser.add_argument("--min-neutral", type=int, default=4)
    parser.add_argument("--max-neutral", type=int, default=12)
    parser.add_argument("--min-repair", type=int, default=7)
    parser.add_argument("--max-repair", type=int, default=18)
    parser.add_argument("--max-alternatives", type=int, default=3)
    parser.add_argument("--max-source-length", type=int, default=30)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--skip-d6", action="store_true")
    parser.add_argument(
        "--symmetry-dir",
        type=Path,
        help="directory containing cube_symmetries.npy and cube_symmetries_inv.npy",
    )
    parser.add_argument("--mitm-boundary", action="store_true")
    parser.add_argument("--front-table", type=Path, default=PROJECT / "data/bfs_table_d5.pkl")
    parser.add_argument(
        "--endgame-hash", type=Path, default=PROJECT / "data/bfs_table_d6_hash.npz"
    )
    parser.add_argument("--mitm-min-window", type=int, default=12)
    parser.add_argument("--mitm-max-window", type=int, default=18)
    parser.add_argument("--mitm-neutral-variants", type=int, default=1)
    parser.add_argument("--mitm-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    states = load_test_states(PROJECT / "data/test.csv")
    move_names = list(puzzle.generators)
    name_to_idx = {name: i for i, name in enumerate(move_names)}
    gens = np.asarray([puzzle.generators[name] for name in move_names], dtype=np.int64)
    inv_idx = np.asarray(
        [name_to_idx[puzzle.inverse_name(name)] for name in move_names], dtype=np.int16
    )

    baseline_report = verify_submission(puzzle, PROJECT / "data/test.csv", args.baseline)
    if not baseline_report.all_valid:
        raise SystemExit(f"invalid baseline: {baseline_report.failures[:5]}")
    named = load_submission(args.baseline)
    rows = {pid: tuple(name_to_idx[name] for name in path) for pid, path in named.items()}
    original_lengths = {pid: len(path) for pid, path in rows.items()}

    target_pairs: set[tuple[bytes, int]] = set()
    for path in rows.values():
        prefix = prefix_permutations(path, gens)
        for size in range(args.min_neutral, min(args.max_neutral, len(path)) + 1):
            for start in range(len(path) - size + 1):
                target_pairs.add((window_key(prefix, start, start + size), size))
    print(f"neutral targets: {len(target_pairs):,}", flush=True)

    sym = sym_inv = conjugated_move = orbit_hashes = ztab = None
    if args.symmetry_dir is not None:
        sym = np.load(args.symmetry_dir / "cube_symmetries.npy").astype(np.int64)
        sym_inv = np.load(args.symmetry_dir / "cube_symmetries_inv.npy").astype(np.int64)
        if sym.shape != sym_inv.shape or sym.shape[1] != gens.shape[1]:
            raise SystemExit(f"bad symmetry shapes: {sym.shape}, {sym_inv.shape}")
        gen_by_perm = {tuple(g.tolist()): i for i, g in enumerate(gens)}
        conjugated_move = np.empty((len(sym), len(gens)), dtype=np.int16)
        for s, (perm, perm_inv) in enumerate(zip(sym, sym_inv)):
            for move, generator in enumerate(gens):
                transformed = perm[generator[perm_inv]]
                conjugated_move[s, move] = gen_by_perm[tuple(transformed.tolist())]

        rng = np.random.default_rng(20260805)
        ztab = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=(gens.shape[1], gens.shape[1]),
            dtype=np.uint64,
        )

        def hash_batch(perms: np.ndarray) -> np.ndarray:
            perms = np.atleast_2d(perms).astype(np.int64, copy=False)
            return np.bitwise_xor.reduce(ztab[np.arange(perms.shape[1]), perms], axis=1)

        target_keys = list({key for key, _ in target_pairs})
        target_perms = np.stack([np.frombuffer(key, dtype=np.uint8) for key in target_keys])
        orbit_raw = np.empty(len(target_perms) * len(sym) * 2, dtype=np.uint64)
        cursor = 0
        t_sym = time.time()
        for lo in range(0, len(target_perms), 1024):
            base = target_perms[lo : lo + 1024]
            inverse = np.argsort(base, axis=1).astype(np.uint8, copy=False)
            for family in (base, inverse):
                for perm, perm_inv in zip(sym, sym_inv):
                    transformed = perm[family[:, perm_inv]]
                    hashes = hash_batch(transformed)
                    orbit_raw[cursor : cursor + len(hashes)] = hashes
                    cursor += len(hashes)
        orbit_hashes = np.unique(orbit_raw[:cursor])
        print(
            f"symmetry filter: {len(sym)} conjugacies x inversion -> "
            f"{len(orbit_hashes):,} hashes ({time.time() - t_sym:.1f}s)",
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
            if path in unique_paths:
                valid_rows += 1
                continue
            if verify_path(puzzle, states[pid], [move_names[m] for m in path]).ok:
                unique_paths.add(path)
                valid_rows += 1
            else:
                invalid_rows += 1
    print(
        f"sources: {len(files)} CSVs, {valid_rows:,} valid rows, "
        f"{invalid_rows:,} invalid, {len(unique_paths):,} distinct words",
        flush=True,
    )

    alternatives: dict[tuple[bytes, int], set[tuple[int, ...]]] = defaultdict(set)
    t0 = time.time()

    def add_word(pair: tuple[bytes, int], word: tuple[int, ...]) -> None:
        if pair not in target_pairs:
            return
        bucket = alternatives[pair]
        if len(bucket) < args.max_alternatives:
            bucket.add(word)

    for number, path in enumerate(unique_paths, start=1):
        prefix = prefix_permutations(path, gens)
        for size in range(args.min_neutral, min(args.max_neutral, len(path)) + 1):
            for start in range(len(path) - size + 1):
                word = path[start : start + size]
                key = window_key(prefix, start, start + size)
                perm = np.frombuffer(key, dtype=np.uint8)
                inverse_perm = np.argsort(perm).astype(np.uint8, copy=False)
                inverse = inverse_word(word, inv_idx)
                if sym is None:
                    add_word((key, size), word)
                    add_word((inverse_perm.tobytes(), size), inverse)
                else:
                    assert sym_inv is not None and conjugated_move is not None
                    assert orbit_hashes is not None and ztab is not None
                    raw_hash = np.bitwise_xor.reduce(
                        ztab[np.arange(gens.shape[1]), perm.astype(np.int64, copy=False)]
                    )
                    pos = int(np.searchsorted(orbit_hashes, raw_hash))
                    if pos >= len(orbit_hashes) or orbit_hashes[pos] != raw_hash:
                        continue
                    for family_perm, family_word in ((perm, word), (inverse_perm, inverse)):
                        transformed_perms = np.take_along_axis(
                            sym, family_perm[sym_inv], axis=1
                        )
                        for s, transformed_perm in enumerate(transformed_perms):
                            transformed_word = tuple(
                                int(conjugated_move[s, move]) for move in family_word
                            )
                            add_word(
                                (transformed_perm.astype(np.uint8, copy=False).tobytes(), size),
                                transformed_word,
                            )
        if number % 5000 == 0:
            print(
                f"  mined {number:,}/{len(unique_paths):,}; "
                f"{sum(len(v) for v in alternatives.values()):,} neutral words",
                flush=True,
            )
    alternatives = {k: v for k, v in alternatives.items() if len(v) >= 2}
    print(
        f"neutral relation classes: {len(alternatives):,}, "
        f"words={sum(len(v) for v in alternatives.values()):,} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    exact: dict[bytes, tuple[int, ...]] = {}
    if not args.skip_d6:
        t0 = time.time()
        table_obj = BfsTable.load(args.bfs_table)
        exact = {
            np.asarray(perm, dtype=np.uint8).tobytes(): tuple(word)
            for perm, word in table_obj.table.items()
        }
        del table_obj
        print(
            f"exact repair table: {len(exact):,} elements ({time.time() - t0:.1f}s)",
            flush=True,
        )
    else:
        print("exact depth-6 repair skipped", flush=True)

    total_neutral = total_repairs = 0
    for pass_no in range(1, args.passes + 1):
        pass_saved = 0
        for pid in sorted(rows):
            incumbent = rows[pid]
            prefix = prefix_permutations(incumbent, gens)
            best = incumbent
            for size in range(args.min_neutral, min(args.max_neutral, len(incumbent)) + 1):
                for start in range(len(incumbent) - size + 1):
                    end = start + size
                    pair = (window_key(prefix, start, end), size)
                    source = incumbent[start:end]
                    for alternative in alternatives.get(pair, ()):
                        if alternative == source:
                            continue
                        total_neutral += 1
                        neutral = incumbent[:start] + alternative + incumbent[end:]
                        for r0, r1, repair in scan_boundary_reductions(
                            neutral,
                            start,
                            end,
                            gens,
                            exact,
                            args.min_repair,
                            args.max_repair,
                        ):
                            total_repairs += 1
                            candidate = neutral[:r0] + repair + neutral[r1:]
                            if len(candidate) < len(best):
                                best = candidate
            if len(best) < len(incumbent):
                if not verify_path(puzzle, states[pid], [move_names[m] for m in best]).ok:
                    raise RuntimeError(f"pid {pid}: plateau candidate failed replay")
                print(f"PASS {pass_no} pid {pid}: {len(incumbent)} -> {len(best)}", flush=True)
                pass_saved += len(incumbent) - len(best)
                rows[pid] = best
        print(
            f"pass {pass_no}: saved={pass_saved}, neutral_attempts={total_neutral:,}, "
            f"repair_hits={total_repairs:,}",
            flush=True,
        )
        if pass_saved == 0:
            break

    if args.mitm_boundary:
        spec = importlib.util.spec_from_file_location(
            "ihes_mitm_gpu", PROJECT / "scripts/18_mitm_gpu.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load scripts/18_mitm_gpu.py")
        mitm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mitm_module)
        joiner = mitm_module.ExactJoiner(
            puzzle, args.front_table, args.endgame_hash, args.device
        )

        metadata: list[tuple[int, tuple[int, ...], int, int]] = []
        states_mitm: list[np.ndarray] = []
        best_lengths: list[int] = []
        selected_neutral = 0
        for pid in sorted(rows):
            incumbent = rows[pid]
            prefix = prefix_permutations(incumbent, gens)
            variants: list[tuple[tuple[int, int, int], tuple[int, ...], int, int]] = []
            for size in range(args.min_neutral, min(args.max_neutral, len(incumbent)) + 1):
                for start in range(len(incumbent) - size + 1):
                    end = start + size
                    pair = (window_key(prefix, start, end), size)
                    source = incumbent[start:end]
                    for alternative in alternatives.get(pair, ()):
                        if alternative == source:
                            continue
                        boundary_change = int(alternative[0] != source[0]) + int(
                            alternative[-1] != source[-1]
                        )
                        variants.append(((boundary_change, size, -start), alternative, start, end))
            variants.sort(reverse=True)
            seen_variants: set[tuple[int, ...]] = set()
            for _, alternative, start, end in variants:
                neutral = incumbent[:start] + alternative + incumbent[end:]
                if neutral in seen_variants:
                    continue
                seen_variants.add(neutral)
                selected_neutral += 1
                neutral_prefix = prefix_permutations(neutral, gens)
                seen_windows: set[tuple[int, int]] = set()
                for boundary in (start, end):
                    for length in range(args.mitm_min_window, args.mitm_max_window + 1):
                        if length > len(neutral):
                            continue
                        r0 = max(0, min(len(neutral) - length, boundary - length // 2))
                        r1 = r0 + length
                        if not (r0 < boundary < r1) or (r0, r1) in seen_windows:
                            continue
                        seen_windows.add((r0, r1))
                        key = window_key(neutral_prefix, r0, r1)
                        states_mitm.append(np.frombuffer(key, dtype=np.uint8).copy())
                        best_lengths.append(length)
                        metadata.append((pid, neutral, r0, r1))
                if len(seen_variants) >= args.mitm_neutral_variants:
                    break

        print(
            f"MITM plateau: {selected_neutral:,} neutral variants, "
            f"{len(metadata):,} centered boundary windows",
            flush=True,
        )
        mitm_hits = 0
        mitm_saved = 0
        states_array = np.asarray(states_mitm, dtype=np.uint8)
        lengths_array = np.asarray(best_lengths, dtype=np.int64)
        for lo in range(0, len(metadata), args.mitm_batch_size):
            hi = min(lo + args.mitm_batch_size, len(metadata))
            answers = joiner.solve_batch(
                states_array[lo:hi], lengths_array[lo:hi], n_candidates=8
            )
            for offset, answer in enumerate(answers):
                if answer is None:
                    continue
                pid, neutral, r0, r1 = metadata[lo + offset]
                candidate = neutral[:r0] + tuple(answer) + neutral[r1:]
                if len(candidate) >= len(rows[pid]):
                    continue
                if not verify_path(puzzle, states[pid], [move_names[m] for m in candidate]).ok:
                    raise RuntimeError(f"pid {pid}: MITM plateau candidate failed replay")
                old = len(rows[pid])
                rows[pid] = candidate
                mitm_hits += 1
                mitm_saved += old - len(candidate)
                print(f"MITM pid {pid}: {old} -> {len(candidate)}", flush=True)
            if hi % (args.mitm_batch_size * 50) == 0 or hi == len(metadata):
                print(
                    f"  MITM {hi:,}/{len(metadata):,}; hits={mitm_hits}, "
                    f"saved={mitm_saved}, phantoms={joiner.phantoms}",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            writer.writerow([pid, ".".join(move_names[m] for m in rows[pid])])

    report = verify_submission(puzzle, PROJECT / "data/test.csv", args.out)
    for pid in sorted(rows):
        if len(rows[pid]) < original_lengths[pid]:
            print(f"  pid {pid}: {original_lengths[pid]} -> {len(rows[pid])}")
    print(
        f"output: {report.n_valid}/{report.n_total} valid, "
        f"{baseline_report.total_moves:,} -> {report.total_moves:,} "
        f"({report.total_moves - baseline_report.total_moves:+,})",
        flush=True,
    )
    return 0 if report.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
