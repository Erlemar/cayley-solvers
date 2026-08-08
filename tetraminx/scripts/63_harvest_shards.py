"""Pull the tier-30 beam shards off every GPU box and min-merge them into the floor.

Safe to run at any point during the sweep: `30_solve.py --resume` appends per pid, so a
partial shard is a valid CSV of finished pids. Every row is replayed against its own
scramble before it is allowed to lower the floor -- a shard file that is being written
concurrently can end in a torn line, and a torn path must not silently become a "win".

    python tetraminx/scripts/63_harvest_shards.py --floor tetraminx/submissions/FINAL_tetraminx_28455.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "tetraminx" / "data"
SCRATCH = Path("C:/Users/and-l/AppData/Local/Temp/claude/"
               "C--Users-and-l-cayley/1b82ce05-add1-4ecd-86f7-939120cbe619/scratchpad")

BOXES = [("tetra-a100b", "us-central1-a", 0),
         ("tetra-a100", "us-central1-a", 1),
         ("tetra-latent", "us-central1-a", 2),
         ("cayley-gpu", "us-east1-b", 3)]


def load_csv(p: Path):
    out = {}
    with io.open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            s = (r.get("path") or "").strip()
            if s:
                out[int(r["initial_state_id"])] = s.split(".")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-pull", action="store_true", help="use already-downloaded shards")
    args = ap.parse_args()

    info = json.loads((DATA / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"].keys())
    GEN = np.array([info["generators"][n] for n in names], dtype=np.int64)
    IDX = {n: i for i, n in enumerate(names)}
    IDENT = np.arange(GEN.shape[1], dtype=np.int64)

    starts = {}
    with io.open(DATA / "test.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            starts[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)

    def valid(pid, mv):
        s = starts[pid].copy()
        for m in mv:
            i = IDX.get(m)
            if i is None:
                return False
            s = s[GEN[i]]
        return np.array_equal(s, IDENT)

    # On Windows `gcloud` is a .cmd shim, which CreateProcess will not resolve from a
    # bare name -- shutil.which finds the real entry point.
    gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not args.no_pull and not gcloud:
        raise SystemExit("gcloud not on PATH; rerun with --no-pull")
    if not args.no_pull:
        for vm, zone, s in BOXES:
            dst = SCRATCH / f"tier30_shard{s}.csv"
            cmd = [gcloud, "compute", "scp",
                   f"{vm}:/home/and-l/tier30_shard{s}.csv", str(dst),
                   f"--zone={zone}"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(f"  shard{s} <- {vm}: "
                  f"{'ok' if r.returncode == 0 else 'MISSING/not started'}", flush=True)

    floor = load_csv(args.floor)
    base = sum(map(len, floor.values()))
    print(f"floor {args.floor.name}: {base:,} moves, avg {base/len(floor):.3f}")

    wins, saved, torn, done = [], 0, 0, 0
    for _, _, s in BOXES:
        p = SCRATCH / f"tier30_shard{s}.csv"
        if not p.exists():
            continue
        got = load_csv(p)
        done += len(got)
        for pid, mv in got.items():
            if not valid(pid, mv):
                torn += 1
                continue
            if pid in floor and len(mv) < len(floor[pid]):
                wins.append((pid, len(floor[pid]), len(mv), s))
                saved += len(floor[pid]) - len(mv)
                floor[pid] = mv

    tot = sum(map(len, floor.values()))
    print(f"\n{done} pids finished across shards, {torn} torn/invalid rows skipped")
    print(f"{len(wins)} wins, {saved} moves saved")
    for pid, a, b, s in sorted(wins, key=lambda t: t[1] - t[2], reverse=True):
        print(f"   pid {pid:4d}: {a} -> {b}   (shard {s})")
    print(f"\ntotal {tot:,}  avg {tot/len(floor):.3f}   "
          f"(hit rate {len(wins)}/{done} = {100*len(wins)/max(done,1):.1f}%, "
          f"{saved/max(done,1):.3f} moves/pid)")

    if args.out and saved:
        with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["initial_state_id", "path"])
            for pid in sorted(floor):
                w.writerow([pid, ".".join(floor[pid])])
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
