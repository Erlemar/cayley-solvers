#!/usr/bin/env python3
"""Splice a known path prefix with a TPU JSON suffix and write a merged CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def read_paths(path_csv: Path) -> tuple[list[str], list[list[str]]]:
    with path_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = rows[1:] if header and header[0] in {"id", "pid", "initial_state_id"} else rows
    out = []
    ids = []
    for row_idx, row in enumerate(data):
        if not row:
            continue
        if len(row) > 1:
            ids.append(row[0])
            out.append([m for m in row[1].split(".") if m])
        else:
            ids.append(str(row_idx))
            out.append([])
    return ids, out


def apply_path(state: list[int], path: list[str], generators: dict[str, list[int]]) -> list[int]:
    cur = list(state)
    for move in path:
        perm = generators[move]
        cur = [cur[i] for i in perm]
    return cur


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--prefix-len", type=int, required=True)
    ap.add_argument("--base-csv", type=Path, required=True)
    ap.add_argument("--suffix-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--puzzle-info", type=Path, default=PROJECT / "megaminx" / "data" / "puzzle_info.json")
    ap.add_argument("--test-csv", type=Path, default=PROJECT / "megaminx" / "data" / "test.csv")
    args = ap.parse_args()

    pinfo = json.loads(args.puzzle_info.read_text(encoding="utf-8"))
    generators = pinfo["generators"]
    solved = list(pinfo["central_state"])
    move_names = list(generators.keys())

    ids, paths = read_paths(args.base_csv)
    suffix_rows = json.loads(args.suffix_json.read_text(encoding="utf-8"))
    suffix_idx = suffix_rows[0]["path_idx"]
    suffix = [move_names[int(i)] for i in suffix_idx]

    old_path = paths[args.pid]
    prefix = old_path[:args.prefix_len]
    candidate = prefix + suffix

    test_rows = list(csv.DictReader(args.test_csv.open(encoding="utf-8", newline="")))
    init = [int(x) for x in test_rows[args.pid]["initial_state"].split(",")]
    ok = apply_path(init, candidate, generators) == solved
    if not ok:
        raise SystemExit("spliced candidate does not verify")
    if len(candidate) <= len(old_path):
        paths[args.pid] = candidate

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for rid, path in zip(ids, paths):
            w.writerow([rid, ".".join(path)])

    print(f"pid={args.pid} old={len(old_path)} candidate={len(candidate)} verify={ok}")
    print(f"saved={len(old_path) - len(candidate)} out={args.out}")
    print("candidate=" + ".".join(candidate))


if __name__ == "__main__":
    main()
