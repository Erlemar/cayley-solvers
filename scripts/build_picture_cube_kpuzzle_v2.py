"""Generate a twips KPuzzle JSON for the IHES Picture Cube (decomposed v2).

Uses group-compatible orientation conventions derived by brute-force search:
  - CORNERS: slots 2, 5, 6, 7 have swapped last two stickers vs piece_features.CORNERS.
  - EDGES, CENTERS: same as piece_features.

End-to-end verification: apply random 20-move sequences under both true state and
KPuzzle semantics, compare piece+ori extraction.

    python scripts/build_picture_cube_kpuzzle_v2.py --out data/picture_cube_decomp.kpuzzle.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube

# Group-compatible orderings from brute-force search.
CORNERS = (
    (0, 38, 48), (2, 26, 36), (9, 50, 12), (11, 14, 24),
    (21, 59, 60), (23, 62, 33), (35, 71, 45), (47, 69, 57),
)
EDGES = (
    (1, 37), (3, 49), (8, 25), (10, 13), (15, 56),
    (20, 27), (22, 61), (32, 39), (34, 68), (44, 51), (46, 70), (58, 63),
)
CENTERS = (
    (4, 5, 7, 6), (16, 17, 19, 18), (28, 29, 31, 30),
    (40, 41, 43, 42), (52, 53, 55, 54), (64, 65, 67, 66),
)

CARD = {"CORNERS": 3, "EDGES": 2, "CENTERS": 4}
N = {"CORNERS": 8, "EDGES": 12, "CENTERS": 6}

# cubing.js notation parses a trailing digit as a move amount, so an IHES move
# such as ``f0`` becomes the zero-th power of ``f``.  Use single-letter quantum
# names inside twips and translate back to the official names at the boundary.
TWIPS_QUANTUM = {
    "f0": "A", "f1": "B", "f2": "C",
    "r0": "D", "r1": "E", "r2": "F",
    "d0": "G", "d1": "H", "d2": "I",
}
TWIPS_TWIN = {
    "f0": "J", "f1": "K", "f2": "L",
    "r0": "M", "r1": "N", "r2": "O",
    "d0": "P", "d1": "Q", "d2": "R",
}


def twips_name(move, twin=False):
    names = TWIPS_TWIN if twin else TWIPS_QUANTUM
    if move.startswith("-"):
        return names[move[1:]] + "'"
    return names[move]


def piece_ori(state, slots):
    can_sets = [frozenset(t) for t in slots]
    out = []
    for slot in slots:
        st = tuple(state[s] for s in slot)
        sset = frozenset(st)
        pid = can_sets.index(sset)
        cfirst = slots[pid][0]
        ori = st.index(cfirst)
        out.append((pid, ori))
    return out


def derive_move(move, puzzle):
    """For this move, return {orbit: (perm, delta)}."""
    new = puzzle.apply_move(puzzle.solved_state, move)
    return derive_state(new)


def derive_state(state):
    """Return direct orbit transformations for a true 72-facelet state."""
    result = {}
    for name, slots in [("CORNERS", CORNERS), ("EDGES", EDGES), ("CENTERS", CENTERS)]:
        po = piece_ori(state, slots)
        perm = [po[i][0] for i in range(len(slots))]
        delta = [po[i][1] for i in range(len(slots))]
        result[name] = (perm, delta)
    return result


def derive_path(path, puzzle):
    state = puzzle.solved_state
    for move in path:
        state = puzzle.apply_move(state, move)
    return derive_state(state)


def build_tws(puzzle):
    """Build a modern twsearch definition with direct transformations.

    Only the nine positive quarter turns are base moves; twsearch generates
    their primes.  This exactly matches the official 18-generator QTM while
    avoiding duplicate bases and enables the order-24 rotational symmetry.
    """
    lines = [
        "Name PictureCube3x3Direct",
        "",
        "Set CORNER 8 3",
        "Set EDGE 12 2",
        "Set CENTER 6 4",
        "",
        "StartState",
        "CORNER",
        "0 1 2 3 4 5 6 7",
        "0 0 0 0 0 0 0 0",
        "EDGE",
        "0 1 2 3 4 5 6 7 8 9 10 11",
        "0 0 0 0 0 0 0 0 0 0 0 0",
        "CENTER",
        "0 1 2 3 4 5",
        "0 0 0 0 0 0",
        "End",
        "",
    ]

    def append_transformation(kind, name, transform):
        lines.append(f"{kind} {name}")
        for tws_orbit, orbit in (("CORNER", "CORNERS"), ("EDGE", "EDGES"), ("CENTER", "CENTERS")):
            perm, delta = transform[orbit]
            lines.extend((tws_orbit, " ".join(map(str, perm)), " ".join(map(str, delta))))
        lines.extend(("End", ""))

    for move in puzzle.move_names:
        if not move.startswith("-"):
            append_transformation("MoveTransformation", move, derive_move(move, puzzle))

    for name, path in (
        ("x", ("f0", "f1", "f2")),
        ("y", ("r0", "r1", "r2")),
        ("z", ("d0", "d1", "d2")),
    ):
        append_transformation("MoveTransformation", name, derive_path(path, puzzle))
    return "\n".join(lines)


def build_kpuzzle(puzzle):
    out = {
        "name": "PictureCube",
        "orbits": [
            {"orbitName": "CORNERS", "numPieces": 8, "numOrientations": 3},
            {"orbitName": "EDGES", "numPieces": 12, "numOrientations": 2},
            {"orbitName": "CENTERS", "numPieces": 6, "numOrientations": 4},
        ],
        "defaultPattern": {
            "CORNERS": {"pieces": list(range(8)), "orientation": [0] * 8},
            "EDGES": {"pieces": list(range(12)), "orientation": [0] * 12},
            "CENTERS": {"pieces": list(range(6)), "orientation": [0] * 6},
        },
        "moves": {},
    }

    for move in puzzle.move_names:
        mv = derive_move(move, puzzle)
        # A duplicate quantum move makes the QTM search complete: twips' canonical
        # FSM forbids repeating one move class, while order-4 puzzles need two
        # consecutive quarter turns.  The canonical ordering permits one primary
        # followed by one twin, exactly covering the only non-redundant repeat.
        for safe in (twips_name(move), twips_name(move, twin=True)):
            out["moves"][safe] = {}
            for orbit in ("CORNERS", "EDGES", "CENTERS"):
                perm, delta = mv[orbit]
                out["moves"][safe][orbit] = {
                    "permutation": perm,
                    "orientationDelta": delta,
                }
    return out


def verify_roundtrip(puzzle, kp):
    """Apply each move then its inverse under KPuzzle semantics, compare to default."""
    default = {k: {"pieces": list(v["pieces"]), "orientation": list(v["orientation"])}
               for k, v in kp["defaultPattern"].items()}

    def apply(state, move_name):
        m = kp["moves"][move_name]
        new = {}
        for orbit in ("CORNERS", "EDGES", "CENTERS"):
            card = CARD[orbit]
            perm = m[orbit]["permutation"]
            delta = m[orbit]["orientationDelta"]
            op = state[orbit]["pieces"]
            oo = state[orbit]["orientation"]
            np_ = [op[perm[i]] for i in range(len(perm))]
            no = [(oo[perm[i]] + delta[i]) % card for i in range(len(perm))]
            new[orbit] = {"pieces": np_, "orientation": no}
        return new

    for move in puzzle.move_names:
        if move.startswith("-"):
            continue
        inv = puzzle.inverse_name(move)
        s = {k: {"pieces": list(v["pieces"]), "orientation": list(v["orientation"])}
             for k, v in default.items()}
        s = apply(s, twips_name(move))
        s = apply(s, twips_name(inv))
        for orbit in ("CORNERS", "EDGES", "CENTERS"):
            assert s[orbit] == default[orbit], \
                f"{move}.{inv} roundtrip FAIL on {orbit}: {s[orbit]} vs {default[orbit]}"
    print(f"roundtrip: {len(puzzle.move_names)//2} move-pairs OK")


def verify_random_sequences(puzzle, kp, n_seqs=100, seq_len=20, seed=0):
    """Apply random move sequences under KPuzzle semantics + under true-state semantics,
    compare piece+ori extraction."""
    rng = random.Random(seed)

    def apply_k(state, move_name):
        m = kp["moves"][move_name]
        new = {}
        for orbit in ("CORNERS", "EDGES", "CENTERS"):
            card = CARD[orbit]
            perm = m[orbit]["permutation"]
            delta = m[orbit]["orientationDelta"]
            op = state[orbit]["pieces"]
            oo = state[orbit]["orientation"]
            np_ = [op[perm[i]] for i in range(len(perm))]
            no = [(oo[perm[i]] + delta[i]) % card for i in range(len(perm))]
            new[orbit] = {"pieces": np_, "orientation": no}
        return new

    def state_to_kp(state):
        return {
            "CORNERS": {"pieces": [p for p, _ in piece_ori(state, CORNERS)],
                        "orientation": [o for _, o in piece_ori(state, CORNERS)]},
            "EDGES": {"pieces": [p for p, _ in piece_ori(state, EDGES)],
                      "orientation": [o for _, o in piece_ori(state, EDGES)]},
            "CENTERS": {"pieces": [p for p, _ in piece_ori(state, CENTERS)],
                        "orientation": [o for _, o in piece_ori(state, CENTERS)]},
        }

    for seq_i in range(n_seqs):
        seq = [rng.choice(puzzle.move_names) for _ in range(seq_len)]
        # True state
        s = puzzle.solved_state
        for m in seq:
            s = puzzle.apply_move(s, m)
        true_kp = state_to_kp(s)

        # KPuzzle semantics
        default = {k: {"pieces": list(v["pieces"]), "orientation": list(v["orientation"])}
                   for k, v in kp["defaultPattern"].items()}
        ks = default
        for m in seq:
            ks = apply_k(ks, twips_name(m))

        for orbit in ("CORNERS", "EDGES", "CENTERS"):
            if ks[orbit] != true_kp[orbit]:
                print(f"MISMATCH seq {seq_i} orbit {orbit}:")
                print(f"  seq: {seq}")
                print(f"  kp: {ks[orbit]}")
                print(f"  true: {true_kp[orbit]}")
                return False
    print(f"random sequences: {n_seqs} x len {seq_len} all match")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tws-out", type=Path)
    args = ap.parse_args()
    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    kp = build_kpuzzle(puzzle)
    verify_roundtrip(puzzle, kp)
    if not verify_random_sequences(puzzle, kp, n_seqs=50, seq_len=20):
        print("VERIFICATION FAILED — not writing output")
        return 1
    args.out.write_text(json.dumps(kp, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes), {len(kp['moves'])} moves")
    if args.tws_out is not None:
        args.tws_out.write_text(build_tws(puzzle), encoding="utf-8")
        print(f"wrote {args.tws_out} ({args.tws_out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
