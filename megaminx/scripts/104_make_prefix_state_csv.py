#!/usr/bin/env python3
"""Build a one-row test CSV from a prefix of an existing solution path."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def read_path(path_csv: Path, pid: int) -> list[str]:
    with path_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"empty path CSV: {path_csv}")
    header = [c.strip().lower() for c in rows[0]]
    if "path" in header:
        id_col = next((i for i, name in enumerate(header)
                       if name in {"id", "pid", "initial_state_id"}), 0)
        path_col = header.index("path")
        data_rows = rows[1:]
        for row_idx, row in enumerate(data_rows):
            rid = int(row[id_col]) if len(row) > id_col and row[id_col] else row_idx
            if rid == pid:
                return [m for m in row[path_col].split(".") if m]
    else:
        for row_idx, row in enumerate(rows):
            if not row:
                continue
            rid = int(row[0]) if len(row) > 1 and row[0] else row_idx
            if rid == pid:
                return [m for m in row[1].split(".") if m]
    raise ValueError(f"pid {pid} not found in {path_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--prefix-len", type=int, required=True)
    ap.add_argument("--path-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--suffix-out", type=Path, default=None,
                    help="optional id,path CSV containing the remaining suffix for anchored tracing")
    ap.add_argument("--puzzle-info", type=Path, default=PROJECT / "megaminx" / "data" / "puzzle_info.json")
    ap.add_argument("--test-csv", type=Path, default=PROJECT / "megaminx" / "data" / "test.csv")
    args = ap.parse_args()

    pinfo = json.loads(args.puzzle_info.read_text(encoding="utf-8"))
    generators = pinfo["generators"]
    path = read_path(args.path_csv, args.pid)
    prefix = path[:args.prefix_len]

    rows = list(csv.DictReader(args.test_csv.open(encoding="utf-8", newline="")))
    state = [int(x) for x in rows[args.pid]["initial_state"].split(",")]
    for move in prefix:
        perm = generators[move]
        state = [state[i] for i in perm]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "initial_state"])
        w.writeheader()
        w.writerow({"id": 0, "initial_state": ",".join(str(x) for x in state)})

    if args.suffix_out is not None:
        args.suffix_out.parent.mkdir(parents=True, exist_ok=True)
        with args.suffix_out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "path"])
            w.writeheader()
            w.writerow({"id": 0, "path": ".".join(path[args.prefix_len:])})

    print(f"wrote {args.out}")
    if args.suffix_out is not None:
        print(f"wrote suffix trace {args.suffix_out}")
    print(f"pid={args.pid} prefix_len={args.prefix_len} suffix_len={len(path) - len(prefix)}")
    print("prefix=" + ".".join(prefix))


if __name__ == "__main__":
    main()
