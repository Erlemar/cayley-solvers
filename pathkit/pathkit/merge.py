"""N-way per-pid min-merge -- the only post-processing method that has never stopped paying.

WHAT IT IS. For every puzzle id, keep the shortest REPLAY-VERIFIED path found in any
source: our runs, old runs, other machines, community files, old Kaggle kernel versions.
It is not glamorous and it beats every clever rewriter in this package on measured moves.

WHY IT KEEPS FINDING MOVES. The source set is not stable and is easy to under-scan:
  * on tetraminx the recorded best was 28,821 while the true verified min over everything
    already on disk was 28,718 -- 103 moves sat unclaimed because nothing had merged them;
  * a stale copy of a live machine caps the merge silently, so re-pull every box each time
    rather than refreshing only the file being watched;
  * a Kaggle kernel serves only its LATEST version through the CLI, yet old versions each
    hold unique wins: one sweep was worth 51 moves and a re-sweep four days later another
    91. Version COUNT is a bad proxy for value though -- a 276-version kernel contributed
    exactly zero because its paths were mostly long fallback. Check the fraction of
    beam-quality paths before spending hundreds of requests on a kernel.

THE ATTRIBUTION TRAP. When reporting an improvement on base B, always compare against the
per-pid min over EVERY other source covering those pids, not against B. A megaminx bridge
run reported 17 moves saved; against the true n-way floor its unique contribution was one
pid and 6 moves -- the rest had already been contributed by a community file nobody had
folded in.
"""
from __future__ import annotations

import collections
from pathlib import Path

from .io import (iter_candidate_files, load_submission, read_result_json,
                 sniff_submission, write_submission)


def merge(puzzle, tests, roots, base=None, log=print, verbose=True, max_report=8):
    """Scan `roots`, keep the shortest verified path per pid. Returns (best, report).

    Every offered path is replayed against the ORIGINAL test state before it can win, so
    a malformed or foreign file costs a skip, never a corrupt row.
    """
    best, src = {}, {}
    counts = {"csv": 0, "json": 0, "scanned": 0, "rejected_invalid": 0, "skipped": 0}

    def offer(pid, path_str, where):
        if pid not in tests or not path_str:
            return
        n = path_str.count(".") + 1
        if pid in best and n >= len(best[pid]):
            return                                    # cannot win; skip the replay cost
        try:
            word = puzzle.parse(path_str)
        except KeyError:
            counts["skipped"] += 1
            return
        if not puzzle.solves(tests[pid], word):
            counts["rejected_invalid"] += 1
            return
        best[pid], src[pid] = word, where

    if base is not None:
        for pid, word in load_submission(base, puzzle).items():
            offer(pid, puzzle.format(word), Path(base).name)

    for p in iter_candidate_files(roots):
        counts["scanned"] += 1
        try:
            if p.suffix == ".csv":
                if not sniff_submission(p, puzzle):
                    continue
                counts["csv"] += 1
                for pid, word in load_submission(p, None).items():
                    offer(pid, word, p.name)
            else:
                recs = read_result_json(p)
                if not recs:
                    continue
                if not puzzle.in_alphabet(recs[0][1]):
                    continue
                counts["json"] += 1
                for pid, path_str in recs:
                    offer(pid, path_str, p.name)
        except Exception as exc:                      # a malformed file must not abort
            counts["skipped"] += 1
            if verbose:
                log("  [skip] %s: %s" % (p.name, exc))

    total = sum(len(w) for w in best.values())
    missing = sorted(set(tests) - set(best))
    report = {
        "total": total, "covered": len(best), "expected": len(tests),
        "missing": missing, "files_csv": counts["csv"], "files_json": counts["json"],
        "files_scanned": counts["scanned"], "rejected_invalid": counts["rejected_invalid"],
        "credit": collections.Counter(src.values()).most_common(max_report),
        "length_histogram": dict(sorted(collections.Counter(
            len(w) for w in best.values()).items())),
    }
    return best, report


def compare(puzzle, best, baseline_paths):
    """Attribute a merge against a baseline: total delta and the per-pid wins.

    Use this, not a remembered floor, whenever quoting an improvement.
    """
    wins, losses = [], []
    for pid, word in best.items():
        if pid not in baseline_paths:
            continue
        b = len(baseline_paths[pid])
        n = len(word)
        if n < b:
            wins.append((pid, b - n))
        elif n > b:
            losses.append((pid, n - b))
    base_total = sum(len(w) for w in baseline_paths.values())
    new_total = sum(len(best[p]) for p in baseline_paths if p in best)
    return {
        "baseline_total": base_total, "merged_total": new_total,
        "delta": new_total - base_total, "wins": sorted(wins, key=lambda t: -t[1]),
        "losses": losses,
    }


def format_report(report, log=print):
    r = report
    log("scanned %d files -> %d submission CSVs + %d result JSONs"
        % (r["files_scanned"], r["files_csv"], r["files_json"]))
    log("pids covered : %d / %d" % (r["covered"], r["expected"]))
    log("TOTAL        : %s" % "{:,}".format(r["total"]))
    if r["rejected_invalid"]:
        log("rejected %d rows that failed replay" % r["rejected_invalid"])
    if r["missing"]:
        log("MISSING %d pids -- NOT a submittable file: %s"
            % (len(r["missing"]), r["missing"][:10]))
    log("credited source (first file achieving the min):")
    for name, n in r["credit"]:
        log("  %5d  %s" % (n, name))
