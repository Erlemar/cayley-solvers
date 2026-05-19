"""T1.1 Phase 1 — curate hand-picked speedcubing algorithms into a macro library.

Pipeline:
  1. Parse a YAML file of algorithm strings in speedcuber notation
     (tokens: R, U, F, L, D, BR, BL with optional ' or 2 suffixes).
  2. Translate each token to our 24-gen notation. R is our 'R', R' is '-R',
     R2 is 'R . R' (megaminx face order is 5, so R2 ≠ R'^3 in general but our
     post-processing handles same-face runs cleanly; emit R . R for clarity).
  3. Apply the resulting word to a solved megaminx state to compute the
     permutation P_i it produces.
  4. For each P_i, check the BFS-d6 table for a known shortest word; if found,
     compare to the original word length and keep the shorter.
  5. Save as a macro library compatible with MacroInsert in scripts/27_path_sa.py.

For Phase 1 we accept any non-identity, non-trivial-cancel-collapse macro
regardless of length — the goal is to test "do curated > brute-force?",
not to optimize for the macros' own length.

Run: PYTHONUTF8=1 .venv/Scripts/python.exe megaminx/scripts/31_curate_macros.py \\
       --in megaminx/data/named_macros_phase1.yaml \\
       --out megaminx/data/curated_macros_phase1.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
import yaml
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "src"))
sys.path.insert(0, str(PROJECT / "src"))

from megaminx.bfs_bytes import BfsBytesTable
from megaminx.post_process import cancel_adjacent_inverses, reduce_same_face_runs
from megaminx.puzzle import Megaminx


# Speedcuber-notation → our generator names. Maps the LL-visible faces only.
# y/y' (whole-puzzle rotations) are NOT face turns and would need rotations.npy
# handling; algorithms containing them are excluded for Phase 1.
SPEEDCUBER_FACE_MAP = {
    "U": "U", "D": "D", "F": "F", "B": "B", "L": "L", "R": "R",
    "BR": "BR", "BL": "BL", "FR": "FR", "FL": "FL", "DR": "DR", "DL": "DL",
}


def parse_token(tok: str) -> list[str]:
    """One speedcuber token → list of our generator names (CW expansion).

    Examples:
      'R'   -> ['R']
      "R'"  -> ['-R']
      'R2'  -> ['R', 'R']
      "R2'" -> ['-R', '-R']
    """
    if tok in {"y", "y'", "y2"}:
        raise ValueError(f"y-rotation not supported in Phase 1 parser: {tok}")
    # Strip suffix
    suffix = ""
    base = tok
    if tok.endswith("2'"):
        suffix = "2'"
        base = tok[:-2]
    elif tok.endswith("'"):
        suffix = "'"
        base = tok[:-1]
    elif tok.endswith("2"):
        suffix = "2"
        base = tok[:-1]
    if base not in SPEEDCUBER_FACE_MAP:
        raise ValueError(f"unknown face token: {base} (full: {tok})")
    name = SPEEDCUBER_FACE_MAP[base]
    if suffix == "":
        return [name]
    if suffix == "'":
        return [f"-{name}"]
    if suffix == "2":
        return [name, name]
    if suffix == "2'":
        return [f"-{name}", f"-{name}"]
    raise ValueError(f"unhandled suffix: {suffix}")


def parse_sequence(s: str) -> list[str]:
    """Speedcuber-notation string → list of our generator names."""
    out: list[str] = []
    for tok in s.split():
        out.extend(parse_token(tok))
    return out


def compute_perm(puzzle: Megaminx, word: list[str]) -> tuple[int, ...]:
    """Apply word to solved state and return the resulting permutation tuple."""
    return puzzle.apply_path(puzzle.solved_state, word)


def is_identity(perm: tuple[int, ...], puzzle: Megaminx) -> bool:
    return perm == puzzle.solved_state


def hamming_distance(perm: tuple[int, ...], solved: tuple[int, ...]) -> int:
    return sum(1 for a, b in zip(perm, solved) if a != b)


def shortest_word_via_bfs(perm: tuple[int, ...], bfs: BfsBytesTable | None) -> list[int] | None:
    """If `perm` is in BFS-d6 table, return the shortest gen-index word; else None."""
    if bfs is None:
        return None
    key = bytes(perm)
    pb = bfs.table.get(key)
    return list(pb) if pb is not None else None


def gen_indices(puzzle: Megaminx, word_names: list[str]) -> list[int]:
    """Convert ['R', '-U', ...] → [gen_idx, gen_idx, ...]."""
    name_to_idx = {n: i for i, n in enumerate(puzzle.move_names)}
    return [name_to_idx[n] for n in word_names]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="yaml_in", type=Path,
                    default=PROJECT / "data" / "named_macros_phase1.yaml")
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "data" / "curated_macros_phase1.pkl")
    ap.add_argument("--bfs-table", type=Path,
                    default=PROJECT / "data" / "bfs_bytes_d6.pkl")
    args = ap.parse_args()

    puzzle = Megaminx.load(PROJECT / "data" / "puzzle_info.json")
    print(f"loaded {len(puzzle.move_names)} generators")

    bfs = None
    if args.bfs_table.exists():
        print(f"loading BFS-d6 ({args.bfs_table.stat().st_size / 1024 / 1024:.0f} MB) ...")
        bfs = BfsBytesTable.load(args.bfs_table)
        print(f"  {len(bfs.table):,} entries")
    else:
        print(f"WARN: BFS-d6 missing at {args.bfs_table}; will keep original word lengths")

    with open(args.yaml_in, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    macros: list[dict] = []  # Each {name, word_names, word_idxs, perm, hamming, source_len, found_shorter}
    n_skipped = 0

    print(f"\nparsing {len(spec['algorithms'])} algorithms ...\n")
    for entry in spec["algorithms"]:
        name = entry["name"]
        seq = entry["sequence"]
        try:
            word_names = parse_sequence(seq)
        except ValueError as e:
            print(f"  SKIP {name}: parse error: {e}")
            n_skipped += 1
            continue

        # Reduce via same-face + adjacent-inverse to canonical form
        word_names = reduce_same_face_runs(word_names)
        word_names = cancel_adjacent_inverses(word_names)

        if not word_names:
            print(f"  SKIP {name}: collapses to identity word after reduction")
            n_skipped += 1
            continue

        perm = compute_perm(puzzle, word_names)
        if is_identity(perm, puzzle):
            print(f"  SKIP {name}: word produces IDENTITY permutation")
            n_skipped += 1
            continue

        hd = hamming_distance(perm, puzzle.solved_state)
        # Try BFS-d6 lookup for shorter word
        bfs_word = shortest_word_via_bfs(perm, bfs)
        word_idxs = gen_indices(puzzle, word_names)
        source_len = len(word_idxs)
        if bfs_word is not None and len(bfs_word) < source_len:
            best_idxs = bfs_word
            found_shorter = True
            shorter_note = f" -> BFS-d6 shortened to {len(bfs_word)}"
        else:
            best_idxs = word_idxs
            found_shorter = False
            shorter_note = ""

        macros.append({
            "name": name,
            "word_idxs": tuple(best_idxs),
            "word_names": [puzzle.move_names[i] for i in best_idxs],
            "perm": perm,
            "hamming": hd,
            "source_len": source_len,
            "best_len": len(best_idxs),
            "found_shorter_via_bfs": found_shorter,
            "notes": entry.get("notes", ""),
        })
        print(f"  KEEP {name}: source_len={source_len}, best_len={len(best_idxs)}, "
              f"hamming={hd}/120{shorter_note}")

    # Dedupe by perm: keep the shortest word per unique permutation
    by_perm: dict[tuple[int, ...], dict] = {}
    for m in macros:
        existing = by_perm.get(m["perm"])
        if existing is None or m["best_len"] < existing["best_len"]:
            by_perm[m["perm"]] = m
    deduped = list(by_perm.values())
    print(f"\ndeduped {len(macros)} -> {len(deduped)} unique permutations")

    # Save
    out = {
        "macros": deduped,
        "metadata": {
            "source": str(args.yaml_in),
            "n_input_algorithms": len(spec["algorithms"]),
            "n_kept": len(deduped),
            "n_skipped": n_skipped,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")

    # Summary stats
    print(f"\n=== summary ===")
    print(f"  unique perms: {len(deduped)}")
    if deduped:
        print(f"  min length: {min(m['best_len'] for m in deduped)}")
        print(f"  max length: {max(m['best_len'] for m in deduped)}")
        print(f"  mean length: {sum(m['best_len'] for m in deduped) / len(deduped):.1f}")
        print(f"  hamming distance distribution (perm to solved):")
        from collections import Counter
        hc = Counter(m["hamming"] for m in deduped)
        for h in sorted(hc):
            print(f"    h={h}: {hc[h]} macros")

    return 0


if __name__ == "__main__":
    sys.exit(main())
