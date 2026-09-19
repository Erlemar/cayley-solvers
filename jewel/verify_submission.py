"""Independent verifier using only the official 48-position permutations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--puzzle-info", default="jewel/data/puzzle_info.json")
    parser.add_argument("--test", default="jewel/data/test.csv")
    args = parser.parse_args()

    puzzle = json.loads(Path(args.puzzle_info).read_text(encoding="utf-8"))
    central = np.asarray(puzzle["central_state"], dtype=np.uint8)
    generators = {
        name: np.asarray(permutation, dtype=np.uint8)
        for name, permutation in puzzle["generators"].items()
    }
    with Path(args.test).open(encoding="utf-8", newline="") as handle:
        tests = {row["initial_state_id"]: row for row in csv.DictReader(handle)}
    with Path(args.submission).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    ids = [row["initial_state_id"] for row in rows]
    problems: list[str] = []
    if len(ids) != len(set(ids)):
        problems.append("duplicate initial_state_id")
    missing = sorted(set(tests) - set(ids), key=int)
    extra = sorted(set(ids) - set(tests), key=int)
    if missing:
        problems.append(f"missing ids: {missing[:10]}")
    if extra:
        problems.append(f"extra ids: {extra[:10]}")

    score = 0
    valid = 0
    for row in rows:
        pid = row["initial_state_id"]
        if pid not in tests:
            continue
        tokens = row["path"].split(".") if row["path"] else []
        score += len(tokens)
        unknown = [token for token in tokens if token not in generators]
        if unknown:
            problems.append(f"pid {pid}: unknown moves {unknown[:3]}")
            continue
        state = np.fromstring(tests[pid]["initial_state"], sep=",", dtype=np.uint8)
        for token in tokens:
            state = state[generators[token]]
        if np.array_equal(state, central):
            valid += 1
        else:
            problems.append(f"pid {pid}: path does not reach central state")

    report = {
        "submission": args.submission,
        "rows": len(rows),
        "expected_rows": len(tests),
        "score": score,
        "valid_replays": valid,
        "problems": problems,
        "ok": not problems and len(rows) == len(tests) and valid == len(tests),
    }
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
