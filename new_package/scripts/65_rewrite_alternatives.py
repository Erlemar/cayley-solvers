"""Rewrite EVERY path we own per pid, then min-merge -- not the other way round.

WHY THE USUAL ORDER IS LOSSY. We min-merge first and post-process the winner. But the
radius-13 rewriter is PATH-DEPENDENT: it can only shorten a window it actually sees, and
two paths to the same solved state pass through different intermediate states. A shorter
path can be geodesic in every window <= 18 and so immune to rewriting, while a longer one
contains a window that collapses hard (pid 32 gave a 16 -> 13 in one window). Merging
first throws that longer path away before the rewriter ever sees it.

Merging AFTER rewriting is free and can only help. Merging BEFORE permanently destroys
candidates. So the correct order is: rewrite all, then merge.

WHEN A LONGER PATH WINS. If the alternative is k moves longer and the rewriter saves
s_alt on it against s_best on the incumbent, the alternative wins iff

    s_alt > k + s_best

Measured corpus (2026-08-05): 530 pids have a distinct alternative at k=1, 711 within
k=2, 789 within k=3 -- 2,657 extra paths, i.e. 2.66x one full sweep. Since most paths are
already geodesic to 13 (s_best = 0), a single alternative saving 2 flips its pid.

    python tetraminx/scripts/65_rewrite_alternatives.py \\
        --baseline tetraminx/submissions/FINAL_tetraminx_28308.csv \\
        --front tetraminx/data/b6_front.npz --max-extra 3 \\
        --out tetraminx/submissions/alt_rewritten.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "tetraminx" / "data"
HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("m64", HERE / "64_mitm_r13.py")
m64 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m64)


def collect(roots, starts, gen, idx, ident):
    """pid -> set of distinct valid paths (as move-name tuples), from CSVs and JSONs."""
    def valid(pid, mv):
        s = starts[pid].copy()
        for m in mv:
            i = idx.get(m)
            if i is None:
                return False
            s = s[gen[i]]
        return np.array_equal(s, ident)

    per = defaultdict(set)
    for r in roots:
        for f in sorted(glob.glob(f"{r}/**/*.csv", recursive=True)):
            try:
                with io.open(f, encoding="utf-8", newline="", errors="replace") as fh:
                    rd = csv.DictReader(fh)
                    if not rd.fieldnames or "initial_state_id" not in rd.fieldnames:
                        continue
                    for row in rd:
                        s = (row.get("path") or "").strip()
                        if s and int(row["initial_state_id"]) in starts:
                            per[int(row["initial_state_id"])].add(tuple(s.split(".")))
            except Exception:
                continue
        for f in sorted(glob.glob(f"{r}/**/*.json", recursive=True)):
            try:
                recs = json.loads(Path(f).read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if not isinstance(recs, list):
                continue
            for rec in recs:
                if isinstance(rec, dict) and rec.get("path") and "pid" in rec:
                    if int(rec["pid"]) in starts:
                        per[int(rec["pid"])].add(tuple(rec["path"].split(".")))
    return {p: {m for m in v if valid(p, m)} for p, v in per.items()}


def rewrite(pz, jn, w, lo, hi):
    """Window-rewrite a move-index path in place; returns (new_path, saved)."""
    saved, i = 0, 0
    w = list(w)
    while i < len(w):
        top = min(hi, len(w) - i)
        if top < lo:
            i += 1
            continue
        hit = False
        for L in range(lo, top + 1):
            got = jn.solve(pz.perm_of(w[i:i + L]), best_len=L)
            if got is not None:
                w[i:i + L] = got
                saved += L - len(got)
                hit = True
                break
        if not hit:
            i += 1
    return w, saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--front", type=Path, default=DATA / "b6_front.npz")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-extra", type=int, default=3,
                    help="only rewrite alternatives within +k of the pid's best")
    ap.add_argument("--min-window", type=int, default=14)
    ap.add_argument("--max-window", type=int, default=18)
    ap.add_argument("--pids", type=str, default="")
    ap.add_argument("--chunk", type=int, default=4_000_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pz = m64.Puzzle(args.data_dir, args.device)
    jn = m64.Joiner(pz, args.front, True, args.chunk)
    if args.min_window <= 13:
        raise SystemExit("--shell-only front is exact only for windows >= 14")

    starts = {}
    with io.open(args.data_dir / "test.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            starts[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)

    base = {p: list(mv) for p, mv in m64.load_csv(args.baseline).items()}
    roots = [str(PROJECT / "tetraminx"), str(PROJECT / "submissions")]
    per = collect(roots, starts, pz.gen, pz.idx, pz.ident)
    print(f"baseline {args.baseline.name}: "
          f"{sum(map(len, base.values())):,} moves", flush=True)

    pids = sorted(base)
    if args.pids:
        want = {int(x) for x in args.pids.split(",")}
        pids = [p for p in pids if p in want]

    n_alt = wins = saved_tot = 0
    t0 = time.time()
    for n, pid in enumerate(pids):
        best = base[pid]
        blen = len(best)
        alts = [list(m) for m in per.get(pid, ()) if blen < len(m) <= blen + args.max_extra]
        if not alts:
            continue
        n_alt += len(alts)
        for alt in alts:
            w = [pz.idx[m] for m in alt]
            new, sv = rewrite(pz, jn, w, args.min_window, args.max_window)
            if sv and len(new) < len(best):
                s = starts[pid].copy()
                for m in new:
                    s = s[pz.gen[m]]
                if not np.array_equal(s, pz.ident):
                    print(f"  pid {pid}: REPLAY FAILED, discarded", flush=True)
                    continue
                names = [pz.names[m] for m in new]
                print(f"  pid {pid}: alt {len(alt)} -> {len(new)} "
                      f"BEATS best {blen}  (saved {sv})", flush=True)
                saved_tot += blen - len(new)
                base[pid] = names
                best, blen = names, len(new)
                wins += 1
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(pids)} pids, {n_alt} alts rewritten, "
                  f"{wins} wins, {saved_tot} moves, {time.time()-t0:.0f}s", flush=True)

    tot = sum(map(len, base.values()))
    print(f"\n{n_alt} alternatives rewritten, {wins} beat their pid's best, "
          f"{saved_tot} moves saved")
    print(f"total {tot:,}  ({time.time()-t0:.0f}s)")
    print(f"phantom hits rejected: {jn.phantoms:,}")

    if args.out and saved_tot:
        with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["initial_state_id", "path"])
            for pid in sorted(base):
                wr.writerow([pid, ".".join(base[pid])])
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
