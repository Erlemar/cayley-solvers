"""Harvest every exact-search journal, apply the verified hits, report coverage.

Reads all `*.jsonl` journals produced by `24_twsearch_ladder.py` (local runs and
whatever the GCS fleet has synced), folds every replay-verified shortening into
the baseline, and prints what the campaign actually PROVED -- which is the point
of the journal: a window that timed out is unfinished, not optimal, and the two
must never be reported as the same thing.

    python scripts/25_harvest_ladder.py --baseline sub.csv --journals data/ladder \
        --out submissions/harvested.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cayley.puzzle import PictureCube
from cayley.verify import load_submission, verify_submission


def perm_of(path, generators) -> np.ndarray:
    state = np.arange(72, dtype=np.int64)
    for move in path:
        state = state[generators[move]]
    return state


def select_non_overlapping(path_len, candidates):
    by_start = defaultdict(list)
    for cand in candidates:
        by_start[cand[0]].append(cand)
    score = [0] * (path_len + 1)
    choice = [None] * (path_len + 1)
    for pos in range(path_len - 1, -1, -1):
        score[pos] = score[pos + 1]
        for start, end, word in by_start.get(pos, ()):
            value = end - start - len(word) + score[end]
            if value > score[pos]:
                score[pos] = value
                choice[pos] = (start, end, word)
    out, pos = [], 0
    while pos < path_len:
        cand = choice[pos]
        if cand is None:
            pos += 1
        else:
            out.append(cand)
            pos = cand[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--journals", nargs="+", type=Path, default=[])
    ap.add_argument("--status", type=Path,
                    help="fleet status.txt; its 'HIT {json}' lines carry the full "
                         "record, so a win can be applied without downloading any "
                         "journal (multi-object GCS reads truncate on this box)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    puzzle = PictureCube.load(PROJECT / "data/puzzle_info.json")
    generators = {n: np.asarray(v, dtype=np.int64)
                  for n, v in puzzle.generators.items()}
    rows = load_submission(args.baseline)
    original = {pid: len(p) for pid, p in rows.items()}

    files = []
    for j in args.journals:
        files += sorted(j.rglob("*.jsonl")) if j.is_dir() else [j]

    def offer_hit(pid, start, wl, word, hits, window=None):
        """Accept a shortening only after replaying it as a 72-position perm."""
        path = rows.get(pid)
        if path is None:
            return False
        end = start + wl
        if end > len(path) or len(word) >= wl:
            return False
        if window is not None and path[start:end] != window:
            return False
        if not np.array_equal(perm_of(word, generators),
                              perm_of(path[start:end], generators)):
            return False
        hits[pid].append((start, end, word))
        return True

    hits = defaultdict(list)
    stats = Counter()
    if args.status is not None and args.status.exists():
        n_status = 0
        for line in args.status.read_text(encoding="utf-8",
                                          errors="replace").splitlines():
            if not line.startswith("HIT "):
                continue
            try:
                h = json.loads(line[4:])
            except json.JSONDecodeError:
                continue
            if offer_hit(h["pid"], h["start"], h["window_len"], h["word"], hits):
                n_status += 1
        print(f"status file: {n_status} verified hits")
    # (window_len -> verdict counts), and which pids are settled at full path
    by_len = defaultdict(Counter)
    proven_pids: set[int] = set()
    n_rec = 0
    for f in files:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:      # torn last line of a killed run
                continue
            n_rec += 1
            verdict = rec.get("verdict")
            stats[verdict] += 1
            wl = rec.get("window_len", 0)
            by_len[wl][verdict] += 1
            pid = rec["pid"]
            path = rows.get(pid)
            if path is None:
                continue
            # A "none" on a window that IS the whole path certifies optimality
            # at that threshold; a timeout certifies nothing at all.
            if verdict == "none" and rec.get("start") == 0 and wl == len(path):
                proven_pids.add(pid)
            if verdict != "hit":
                continue
            offer_hit(pid, rec["start"], wl, rec["word"], hits, rec.get("window"))

    saved = 0
    for pid, options in hits.items():
        for start, end, word in reversed(select_non_overlapping(len(rows[pid]), options)):
            saved += end - start - len(word)
            rows[pid][start:end] = word

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            writer.writerow([pid, ".".join(rows[pid])])

    base = verify_submission(puzzle, PROJECT / "data/test.csv", args.baseline)
    rep = verify_submission(puzzle, PROJECT / "data/test.csv", args.out)

    print(f"journals   : {len(files)}  records {n_rec:,}")
    print(f"verdicts   : {dict(stats)}")
    print("by window length (verdict counts):")
    for wl in sorted(by_len):
        print(f"  len {wl:3d}: {dict(by_len[wl])}")
    print(f"\npids PROVEN optimal at their searched threshold: {len(proven_pids)}")
    for pid in sorted(rows):
        if len(rows[pid]) < original[pid]:
            print(f"  WIN pid {pid}: {original[pid]} -> {len(rows[pid])}")
    print(
        f"\napplied saving={saved}; {base.total_moves:,} -> {rep.total_moves:,} "
        f"({rep.total_moves - base.total_moves:+,}); "
        f"{rep.n_valid}/{rep.n_total} valid -> {args.out}"
    )
    return 0 if rep.all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
