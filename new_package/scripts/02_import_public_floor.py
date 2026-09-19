"""Import the public Tetraminx solution (kernel ka1242/load-tetramix-solution) as our floor.

The kernel embeds a full 1000-pid submission as a literal string in cell 0.
We NEVER submit this file -- it is used only as
  (a) the per-pid merge floor every later result is compared against, and
  (b) the source of policy/value training data for the AZ dataset builder.

Every path is replayed against the competition test states and asserted to reach
the solved state before anything is written.

    .venv/Scripts/python.exe tetraminx/scripts/02_import_public_floor.py \
        --notebook <path>/load-tetramix-solution.ipynb
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def load_paths_from_notebook(nb_path: Path) -> dict[int, list[str]]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    blob = None
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if "initial_state_id,path" in src and '"""' in src:
            blob = src.split('"""')[1].strip()
            break
    if blob is None:
        raise SystemExit("no submission string found in notebook")
    lines = blob.splitlines()
    assert lines[0].strip() == "initial_state_id,path", f"unexpected header: {lines[0]!r}"
    out: dict[int, list[str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        pid, path = line.split(",", 1)
        out[int(pid)] = path.strip().split(".")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", type=Path, required=True)
    ap.add_argument("--puzzle-info", type=Path,
                    default=PROJECT / "tetraminx" / "data" / "puzzle_info.json")
    ap.add_argument("--test", type=Path, default=PROJECT / "tetraminx" / "data" / "test.csv")
    ap.add_argument("--out", type=Path,
                    default=PROJECT / "tetraminx" / "submissions" / "floor_public_29622.csv")
    args = ap.parse_args()

    info = json.loads(args.puzzle_info.read_text(encoding="utf-8"))
    gens = {nm: np.array(p, dtype=np.int64) for nm, p in info["generators"].items()}
    solved = np.array(info["central_state"], dtype=np.int64)
    n = len(solved)

    with open(args.test, encoding="utf-8") as f:
        states = {int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
            for r in csv.DictReader(f)}

    paths = load_paths_from_notebook(args.notebook)
    assert set(paths) == set(states), "pid mismatch between notebook and test.csv"

    total = 0
    for pid, path in paths.items():
        st = states[pid].copy()
        for mv in path:
            if mv not in gens:
                raise SystemExit(f"pid {pid}: unknown move {mv!r}")
            st = st[gens[mv]]
        if not np.array_equal(st, solved):
            raise SystemExit(f"pid {pid}: path does not solve")
        total += len(path)

    assert len(solved) == n
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(paths):
            w.writerow([pid, ".".join(paths[pid])])

    lens = [len(paths[p]) for p in sorted(paths)]
    print(f"verified {len(paths)}/{len(paths)} pids solve", flush=True)
    print(f"total moves = {total}  (min {min(lens)}, max {max(lens)}, "
          f"mean {total / len(lens):.2f})", flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
