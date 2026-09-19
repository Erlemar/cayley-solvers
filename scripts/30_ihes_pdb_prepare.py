"""Export verified IHES move coordinates and search inputs for the PDB solver."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cayley.puzzle import PictureCube
from cayley.verify import load_submission, load_test_states, verify_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=ROOT / "submission_ihes.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/ihes_pdb")
    args = ap.parse_args()
    puzzle = PictureCube.load(ROOT / "data/puzzle_info.json")
    report = verify_submission(puzzle, ROOT / "data/test.csv", args.baseline)
    assert report.all_valid, report
    spec = importlib.util.spec_from_file_location("decomp", ROOT / "scripts/build_picture_cube_kpuzzle_v2.py")
    decomp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decomp)
    names = [axis + str(layer) for axis in "frd" for layer in range(3)]
    names = [m for base in names for m in (base, "-" + base)]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "moves.txt").open("w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
            trans = decomp.derive_move(name, puzzle)
            for key in ("CORNERS", "EDGES", "CENTERS"):
                for row in trans[key]:
                    f.write(" ".join(map(str, row)) + "\n")
    paths = load_submission(args.baseline)
    states = load_test_states(ROOT / "data/test.csv")
    with (args.out_dir / "queries.txt").open("w", encoding="utf-8") as f:
        for pid, path in sorted(paths.items()):
            f.write(f"{pid} {len(path)}\n")
            for key in ("CORNERS", "EDGES", "CENTERS"):
                for row in decomp.derive_state(states[pid])[key]:
                    f.write(" ".join(map(str, row)) + "\n")
    manifest = {"baseline": str(args.baseline.resolve()), "sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                "total": report.total_moves, "verified": report.n_valid, "moves": names}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
