"""Derive the physical-piece partition of a puzzle's facelets from its generators.

A PieceTransformer (Vlad Kuznetsov's cayleypy-training-core) needs a layout: which
facelet slots belong to which physical piece. That repo hard-codes layouts for the
megaminx (p900) and the IHES cube only. This script derives one for any puzzle.

Rule: facelets rigidly attached to the same physical piece have the SAME stabilizer
under the generator set -- any move that fixes one facelet of a cubie fixes all of
them. So group slots by their stabilizer, then verify the result is a genuine block
system (every generator maps each block onto some block).

Validated: run with --puzzle ../data/puzzle_info.json (the IHES cube) and the output
reproduces cayleypy-training-core's hand-written 26-piece ihes layout exactly.

Usage:
    python 50_derive_piece_layout.py --puzzle tetraminx/data/puzzle_info.json \
        --out tetraminx/data/piece_layout.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_puzzle(path: Path) -> tuple[list[str], list[list[int]], int]:
    info = json.loads(path.read_text(encoding="utf-8"))
    gens = info["generators"]
    names = list(gens.keys())
    perms = [list(gens[n]) for n in names]
    n_slots = len(info["central_state"])
    for name, perm in zip(names, perms):
        if len(perm) != n_slots:
            raise ValueError(f"generator {name} has length {len(perm)}, expected {n_slots}")
    return names, perms, n_slots


def slot_maps(perms: list[list[int]], n: int) -> list[list[int]]:
    """Convention: new_state[i] = state[g[i]], so the content of slot g[i] lands in slot i."""
    maps = []
    for g in perms:
        s = [0] * n
        for i, gi in enumerate(g):
            s[gi] = i
        maps.append(s)
    return maps


def derive_pieces(perms: list[list[int]], n: int) -> list[list[int]]:
    maps = slot_maps(perms, n)
    by_stab: dict[frozenset[int], list[int]] = defaultdict(list)
    for slot in range(n):
        stab = frozenset(gi for gi, s in enumerate(maps) if s[slot] == slot)
        by_stab[stab].append(slot)
    pieces = [sorted(v) for v in by_stab.values()]
    pieces.sort(key=lambda b: (-len(b), b[0]))
    return pieces


def check_block_system(pieces: list[list[int]], perms: list[list[int]], n: int) -> int:
    """Count generator/piece pairs whose image is not itself a piece. Must be 0."""
    maps = slot_maps(perms, n)
    known = {frozenset(p) for p in pieces}
    bad = 0
    for s in maps:
        for p in pieces:
            if frozenset(s[j] for j in p) not in known:
                bad += 1
    return bad


def build_layout(pieces: list[list[int]]) -> dict:
    """Pad every piece to max_piece_size and assign a type id per distinct piece size."""
    max_size = max(len(p) for p in pieces)
    sizes = sorted({len(p) for p in pieces}, reverse=True)
    type_of_size = {s: i for i, s in enumerate(sizes)}
    positions, mask, types = [], [], []
    for p in pieces:
        pad = max_size - len(p)
        positions.append(list(p) + [0] * pad)
        mask.append([True] * len(p) + [False] * pad)
        types.append(type_of_size[len(p)])
    return {
        "num_pieces": len(pieces),
        "max_piece_size": max_size,
        "num_piece_types": len(sizes),
        "piece_sizes_by_type": sizes,
        "piece_positions": positions,
        "piece_mask": mask,
        "piece_types": types,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzle", required=True, help="path to puzzle_info.json")
    ap.add_argument("--out", default=None, help="write the layout JSON here")
    args = ap.parse_args()

    path = Path(args.puzzle)
    names, perms, n = load_puzzle(path)
    pieces = derive_pieces(perms, n)
    violations = check_block_system(pieces, perms, n)
    covered = sum(len(p) for p in pieces)

    hist: dict[int, int] = defaultdict(int)
    for p in pieces:
        hist[len(p)] += 1

    print(f"puzzle          : {path}")
    print(f"slots / gens    : {n} / {len(names)}")
    print(f"pieces          : {len(pieces)}")
    print(f"size histogram  : {dict(sorted(hist.items(), reverse=True))}")
    print(f"slots covered   : {covered} of {n}")
    print(f"block violations: {violations}")
    if covered != n or violations != 0:
        print("FAILED: the stabilizer partition is not a valid block system for this puzzle.")
        return 1

    layout = build_layout(pieces)
    layout["puzzle_file"] = str(path)
    layout["state_size"] = n
    layout["num_actions"] = len(names)
    layout["move_names"] = names
    print(f"max_piece_size  : {layout['max_piece_size']}")
    print(f"piece types     : {layout['num_piece_types']} (sizes {layout['piece_sizes_by_type']})")
    print(f"embedding rows  : {layout['max_piece_size']} x {n} = {layout['max_piece_size'] * n}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(layout, indent=1) + "\n", encoding="utf-8")
        print(f"wrote           : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
