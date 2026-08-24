"""N-way per-pid min over every cube444 result we produced, replay-verified.

Rule 26: judge by the per-pid MIN over ALL available sources, never one run's total. Every
configuration run here beat the floor on a DIFFERENT pid, so the merge is worth strictly
more than any single arm.

Sources are discovered by content, not by name: sqlite progress DBs (GPU arms), TPU result
JSONs, and submission CSVs are all accepted. Every path is replayed against the real
generators before it is allowed into the output -- a shorter path that does not solve is
worse than useless.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
BUNDLE = PROJECT / "cube444" / "kaggle_inference" / "cube444_inference" / "solver"
FLOOR_CSV = BUNDLE.parent / "submissions" / "cube4_submission_46662.csv"


def load_puzzle():
    import torch

    sys.path.insert(0, str(BUNDLE))
    from pilgrim.utils import parse_generator_spec

    spec = json.loads((BUNDLE / "generators" / "p002.json").read_text(encoding="utf-8"))
    moves, names = parse_generator_spec(spec)
    v0 = torch.load(BUNDLE / "targets" / "p002-t000.pt", map_location="cpu",
                    weights_only=False).numpy().astype(np.int64)
    return np.asarray(moves, dtype=np.int64), names, v0


def main() -> int:
    all_moves, names, v0 = load_puzzle()
    idx = {n: i for i, n in enumerate(names)}
    states = {int(r["initial_state_id"]):
              np.asarray([int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
              for r in csv.DictReader(open(BUNDLE / "test.csv", encoding="utf-8"))}

    def solves(pid: int, path: str) -> bool:
        cur = states[pid].copy()
        for tok in path.split("."):
            if tok not in idx:
                return False
            cur = cur[all_moves[idx[tok]]]
        return bool(np.array_equal(cur, v0))

    floor = {int(r["initial_state_id"]): r["path"]
             for r in csv.DictReader(open(FLOOR_CSV, encoding="utf-8"))}
    best = dict(floor)
    origin = {p: "floor" for p in floor}

    scratch = Path("C:/Users/and-l/AppData/Local/Temp/claude/"
                   "C--Users-and-l-cayley/9c7e1036-cc49-48f1-98a7-cd274ab24edd/scratchpad")
    sources: list[tuple[str, dict[int, str]]] = []

    for tag, db in (("gpu-armA", BUNDLE / "results" / "arm_s3_1m.sqlite3"),
                    ("gpu-armB", BUNDLE / "results" / "arm_blend_1m.sqlite3")):
        if db.exists():
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = con.execute("SELECT initial_state_id, path FROM solutions").fetchall()
            con.close()
            sources.append((tag, {int(p): s for p, s in rows}))

    # Extra CSVs given on the command line. Treated as untrusted like everything else:
    # every path is replayed before it is allowed to win a pid.
    for arg in sys.argv[1:]:
        rows = {int(r["initial_state_id"]): r["path"]
                for r in csv.DictReader(open(arg, encoding="utf-8"))}
        sources.append((f"extra:{Path(arg).stem[:22]}", rows))

    for d in sorted(scratch.glob("klog*")):
        j = d / "cube444_q_beam_results.json"
        if j.exists():
            data = json.loads(j.read_text(encoding="utf-8"))
            sources.append((f"tpu-{d.name}",
                            {int(k): v["path"] for k, v in data.items() if v.get("path")}))
        c = d / "submission.csv"
        if c.exists():
            sub = {int(r["initial_state_id"]): r["path"]
                   for r in csv.DictReader(open(c, encoding="utf-8"))}
            # only the rows that differ from the floor carry information
            sub = {p: s for p, s in sub.items() if s != floor.get(p)}
            if sub:
                sources.append((f"sub-{d.name}", sub))

    print(f"floor: {sum(len(v.split('.')) for v in floor.values()):,} moves")
    print(f"{len(sources)} sources\n")
    for tag, rows in sources:
        wins = bad = 0
        for pid, path in rows.items():
            if pid not in best:
                continue
            if len(path.split(".")) >= len(best[pid].split(".")):
                continue
            if not solves(pid, path):
                bad += 1
                continue
            best[pid] = path
            origin[pid] = tag
            wins += 1
        print(f"  {tag:16s} {len(rows):5d} rows -> {wins:3d} improvements"
              + (f"  ({bad} REJECTED: do not replay)" if bad else ""))

    total = sum(len(v.split(".")) for v in best.values())
    floor_total = sum(len(v.split(".")) for v in floor.values())
    print(f"\nmerged total: {total:,}  (floor {floor_total:,}, saved {floor_total-total})")

    improved = [p for p in best if origin[p] != "floor"]
    print(f"improved pids: {len(improved)}")
    for p in sorted(improved):
        print(f"  pid {p:4d}  {len(floor[p].split('.')):3d} -> "
              f"{len(best[p].split('.')):3d}   from {origin[p]}")

    bad = [p for p in best if not solves(p, best[p])]
    print(f"\nfinal replay check: {len(best)} rows, {len(bad)} failures")
    if bad:
        print("NOT writing -- some rows do not solve")
        return 1

    out = PROJECT / "cube444" / "submissions" / f"cube444_merged_{total}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(best):
            w.writerow([pid, best[pid]])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
