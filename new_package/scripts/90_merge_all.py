"""N-way per-pid min over EVERY tetraminx result source, with replay verification.

Rule 26 in one command. Scans submission CSVs, beam result JSONs, and any extra
paths given, keeps the shortest VERIFIED path per pid, and writes a full
submission. Every candidate is replayed against the original test state before it
is accepted, and the written file is re-read and re-verified afterwards.

Why this exists: results scatter across the repo, sibling Claude session
scratchpads (AppData/Local/Temp/claude/<uuid>/scratchpad/) and whatever VM was up
that day. On 2026-07-29 the recorded best was 28,821 while the true verified min
over everything was 28,718 -- 103 moves sat unclaimed purely because nothing ever
merged them. See memory [[merge-all-sessions-before-quoting-score]].

    python tetraminx/scripts/90_merge_all.py                    # report only
    python tetraminx/scripts/90_merge_all.py --write            # + write FINAL_<total>.csv
    python tetraminx/scripts/90_merge_all.py --extra <dir_or_file> ...
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=PROJECT / "tetraminx" / "data")
    ap.add_argument("--subs-dir", type=Path, default=PROJECT / "tetraminx" / "submissions")
    ap.add_argument("--results-dir", type=Path, default=PROJECT / "tetraminx" / "results")
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="extra CSV/JSON files or dirs to fold in (sibling session "
                         "scratchpads, freshly pulled VM output, ...)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="CSV to attribute against (default: the merge minus our own "
                         "beam JSONs is NOT used; pass e.g. community_28843.csv)")
    ap.add_argument("--write", action="store_true", help="write FINAL_<total>.csv")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    info = json.loads((args.data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    names = list(info["generators"])
    gi = {n: i for i, n in enumerate(names)}
    gens = np.array([info["generators"][n] for n in names], dtype=np.int64)
    solved = np.array(info["central_state"], dtype=np.int64)

    st0 = {}
    with open(args.data_dir / "test.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            st0[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)

    def valid(pid: int, path: str) -> bool:
        if pid not in st0 or not path:
            return False
        s = st0[pid]
        for m in path.split("."):
            if m not in gi:
                return False
            s = s[gens[gi[m]]]
        return bool(np.array_equal(s, solved))

    best: dict[int, str] = {}
    src: dict[int, str] = {}

    def offer(pid, path, where):
        if not path or pid not in st0:
            return
        if pid in best and len(path.split(".")) >= len(best[pid].split(".")):
            return
        if valid(pid, path):
            best[pid], src[pid] = path, where

    files: list[Path] = []
    files += [Path(p) for p in sorted(glob.glob(str(args.subs_dir / "*.csv")))]
    files += [Path(p) for p in sorted(glob.glob(str(args.results_dir / "**" / "*.json"),
                                                recursive=True))]
    for e in args.extra:
        if e.is_dir():
            files += [Path(p) for p in sorted(glob.glob(str(e / "**" / "*.*"),
                                                        recursive=True))
                      if p.endswith((".csv", ".json"))]
        elif e.exists():
            files.append(e)

    n_csv = n_json = 0
    for p in files:
        try:
            if p.suffix == ".csv":
                rows = list(csv.DictReader(open(p, encoding="utf-8")))
                if not rows or "initial_state_id" not in rows[0]:
                    continue
                n_csv += 1
                for r in rows:
                    offer(int(r["initial_state_id"]), r.get("path", ""), p.name)
            else:
                recs = json.load(open(p, encoding="utf-8"))
                if not isinstance(recs, list):
                    continue
                n_json += 1
                for r in recs:
                    if r.get("found") and r.get("verify_ok"):
                        offer(int(r["pid"]), r.get("path", ""), p.name)
        except Exception as exc:                      # a malformed file must not abort
            print(f"  [skip] {p.name}: {exc}")

    total = sum(len(q.split(".")) for q in best.values())
    print(f"scanned {n_csv} CSVs + {n_json} JSONs")
    print(f"pids covered : {len(best)} / {len(st0)}")
    print(f"TOTAL        : {total:,}")
    missing = sorted(set(st0) - set(best))
    if missing:
        print(f"MISSING {len(missing)} pids -- NOT a submittable file: {missing[:10]}")

    import collections
    print("\ncredited source (first file achieving the min, ties to the earlier name):")
    for k, v in collections.Counter(src.values()).most_common(8):
        print(f"  {v:>5}  {k}")
    h = collections.Counter(len(q.split(".")) for q in best.values())
    print(f"\nlength histogram: {dict(sorted(h.items()))}")

    if args.baseline and args.baseline.exists():
        base = {int(r["initial_state_id"]): r["path"]
                for r in csv.DictReader(open(args.baseline, encoding="utf-8"))
                if r.get("path")}
        wins = [(p, len(base[p].split(".")) - len(best[p].split(".")))
                for p in best if p in base
                and len(best[p].split(".")) < len(base[p].split("."))]
        bt = sum(len(q.split(".")) for q in base.values())
        print(f"\nvs {args.baseline.name}: {bt:,} -> {total:,} "
              f"({total - bt:+,}) over {len(wins)} improved pids")

    if args.write:
        out = args.out or (args.subs_dir / f"FINAL_tetraminx_{total}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["initial_state_id", "path"])
            for pid in sorted(best):
                w.writerow([pid, best[pid]])
        # re-read and re-verify what was actually written, not what we think we wrote
        bad, tot2, n = 0, 0, 0
        for r in csv.DictReader(open(out, encoding="utf-8")):
            n += 1
            tot2 += len(r["path"].split("."))
            if not valid(int(r["initial_state_id"]), r["path"]):
                bad += 1
        print(f"\nwrote {out}")
        print(f"  re-read: {n} rows, {tot2:,} moves, invalid={bad}")
        if bad or tot2 != total:
            print("  *** MISMATCH -- do not submit ***")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
