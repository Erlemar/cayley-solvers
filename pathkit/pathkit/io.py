"""Submission I/O and CONTENT-BASED source discovery.

The discovery half is the load-bearing part. Results scatter: sibling agent scratchpads
under AppData/Local/Temp, whatever VM was up that day, a Downloads folder, a Telegram
export, a kernel output pulled months ago. Two habits recover moves that are already
paid for:

  * find files by CONTENT, not by name or location. A file called submission_dad.csv sat
    in the megaminx folder holding TETRAMINX data, and three submission_publ*.csv sat in
    an unrelated project folder -- together worth 22 moves. The header test alone is not
    enough (every puzzle in this family has the same header), so the move ALPHABET is
    checked too, which makes pointing a scan at unrelated folders safe.
  * replay every row before it may lower the floor. Discovery is worthless if a
    plausible-looking file can inject an invalid path.
"""
from __future__ import annotations

import csv
import io as _io
import json
import os
from pathlib import Path

import numpy as np

SUBMISSION_HEADER = ("initial_state_id", "path")


def load_tests(path, dtype=np.int64):
    """test.csv -> {pid: state array}."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=dtype)
    return out


def load_submission(path, puzzle=None):
    """submission.csv -> {pid: word}. Words are move-index lists when a puzzle is given,
    otherwise raw strings."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            p = r.get("path") or ""
            if not p:
                continue
            pid = int(r["initial_state_id"])
            out[pid] = puzzle.parse(p) if puzzle is not None else p
    return out


def write_submission(path, puzzle, paths, tests=None, verify=True, log=print):
    """Write and then RE-READ and re-verify what actually landed on disk.

    Verifying the in-memory dict proves nothing about the file that gets submitted; a
    formatting slip between the two is exactly the failure this catches.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUBMISSION_HEADER)
        for pid in sorted(paths):
            word = paths[pid]
            w.writerow([pid, word if isinstance(word, str) else puzzle.format(word)])
    if not verify:
        return {"rows": len(paths), "verified": False}
    rows, total, bad = 0, 0, []
    for pid, word in load_submission(path, puzzle).items():
        rows += 1
        total += len(word)
        if tests is not None and not puzzle.solves(tests[pid], word):
            bad.append(pid)
    log("wrote %s: %d rows, %s moves, invalid=%d" % (path, rows, "{:,}".format(total), len(bad)))
    if bad:
        log("  *** INVALID ROWS -- do not submit: %s" % bad[:10])
    return {"rows": rows, "total": total, "invalid": bad, "verified": True}


def iter_candidate_files(roots, exts=(".csv", ".json"), skip_dirs=(".git", ".venv", "__pycache__")):
    """Walk `roots` yielding files that could hold results. Cheap; the real filter is next."""
    seen = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            if root.suffix in exts and str(root) not in seen:
                seen.add(str(root))
                yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                if not fn.endswith(exts):
                    continue
                p = Path(dirpath) / fn
                if str(p) in seen:
                    continue
                seen.add(str(p))
                yield p


def sniff_submission(path, puzzle):
    """Is this a submission CSV for THIS puzzle? Header test AND alphabet test.

    Reads two lines, not the whole file, so scanning tens of thousands of files is cheap.
    """
    try:
        with _io.open(path, encoding="utf-8", newline="", errors="strict") as fh:
            head = fh.readline()
            if "initial_state_id" not in head or "path" not in head:
                return False
            probe = fh.readline().strip()
    except Exception:
        return False
    if not probe or "," not in probe:
        return False
    body = probe.split(",", 1)[1].strip().strip('"')
    return puzzle.in_alphabet(body)


def read_result_json(path):
    """Beam-search result JSONs: a list of records with pid / path / found / verify_ok.

    Only records the producer itself marked verified are offered; they are replayed again
    downstream regardless.
    """
    try:
        recs = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(recs, list):
        return []
    out = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        if r.get("found") is False or r.get("verify_ok") is False:
            continue
        pid, p = r.get("pid", r.get("initial_state_id")), r.get("path")
        if pid is None or not p:
            continue
        out.append((int(pid), p))
    return out
