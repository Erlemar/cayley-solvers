"""Generate a twips KPuzzle JSON for the IHES Picture Cube.

KPuzzle format:
  orbits[]: {orbitName, numPieces, numOrientations}
  defaultPattern: {orbit: {pieces: [...], orientation: [...]}}
  moves: {name: {orbit: {permutation: [...], orientationDelta: [...]}}}

For picture cube: 8 corners × 3 orientations, 12 edges × 2 orientations, 6 centers × 4 orientations.

    python scripts/build_picture_cube_kpuzzle.py --out data/picture_cube.kpuzzle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.piece_features import (
    CENTERS, CENTER_ORI_CARD, CORNERS, CORNER_ORI_CARD,
    EDGES, EDGE_ORI_CARD, N_CENTERS, N_CORNERS, N_EDGES,
    extract_features_numpy,
)
from cayley.puzzle import PictureCube


def build_kpuzzle(puzzle: PictureCube) -> dict:
    """Flat 72-sticker representation — each sticker is a piece with numOrientations=1.

    This bypasses orientation-encoding complexity of the CORNERS/EDGES/CENTERS
    decomposition (which was slot-dependent and not compatible with KPuzzle's
    orientation semantics). twips performance may be lower than with the
    decomposed representation, but correctness is guaranteed.
    """
    n = len(puzzle.solved_state)
    out = {
        "name": "PictureCube",
        "orbits": [
            {"orbitName": "STICKERS", "numPieces": n, "numOrientations": 1},
        ],
        "defaultPattern": {
            "STICKERS": {"pieces": list(puzzle.solved_state), "orientation": [0] * n},
        },
        "moves": {},
    }

    for move_name in puzzle.move_names:
        # Twips allows apostrophes in names; picture cube uses "-f0" etc. Map "-X" → "X'".
        safe_name = move_name[1:] + "'" if move_name.startswith("-") else move_name
        # Permutation: apply_move convention is new_state[i] = state[gen[i]].
        # For KPuzzle: new.pieces[i] = old.pieces[perm[i]].
        # The generator permutation IS perm.
        perm = list(puzzle.generators[move_name])
        out["moves"][safe_name] = {
            "STICKERS": {"permutation": perm, "orientationDelta": [0] * n},
        }
    return out


def verify_roundtrip(puzzle: PictureCube, kp: dict) -> None:
    """Sanity: apply a generator then its inverse via the KPuzzle — should return to solved."""
    import numpy as np
    # Simulate: start at defaultPattern, apply move, apply inverse, check back to identity.
    def apply(state: dict, move_name: str, kp: dict) -> dict:
        m = kp["moves"][move_name]
        new_state = {}
        for orbit in kp["orbits"]:
            name = orbit["orbitName"]
            no = orbit["numOrientations"]
            perm = m[name]["permutation"]
            delta = m[name]["orientationDelta"]
            old_pieces = state[name]["pieces"]
            old_ori = state[name]["orientation"]
            new_pieces = [old_pieces[perm[i]] for i in range(len(perm))]
            new_ori = [(old_ori[perm[i]] + delta[i]) % no for i in range(len(perm))]
            new_state[name] = {"pieces": new_pieces, "orientation": new_ori}
        return new_state

    default = {k: {"pieces": list(v["pieces"]), "orientation": list(v["orientation"])}
               for k, v in kp["defaultPattern"].items()}

    def sanitize(m):
        return m[1:] + "'" if m.startswith("-") else m

    for name in puzzle.move_names:
        if name.startswith("-"):
            continue
        inv = puzzle.inverse_name(name)
        s = default
        s = apply(s, sanitize(name), kp)
        s = apply(s, sanitize(inv), kp)
        for orbit_name, data in s.items():
            assert data == default[orbit_name], f"{name}.{inv} roundtrip fail on {orbit_name}"
    print(f"verified {len(puzzle.move_names)//2} move+inverse pairs round-trip to identity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    puzzle = PictureCube.load(PROJECT / "data" / "puzzle_info.json")
    kp = build_kpuzzle(puzzle)
    verify_roundtrip(puzzle, kp)
    args.out.write_text(json.dumps(kp, indent=2))
    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes), {len(kp['moves'])} moves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
