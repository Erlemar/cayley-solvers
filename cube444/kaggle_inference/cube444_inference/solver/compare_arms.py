"""Compare the two 1M-beam arms against the 46,662 floor.

Reads the RAW beam result from each arm's progress DB, not the output CSV -- the CSV is
already min-merged with the floor, which would hide a regression (same reasoning as
bench_beam.py). Judges by the per-pid MIN against the floor as well as the standalone
total, because a run whose own mean looks bad can still carry the merge.
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOOR = HERE.parent / "submissions" / "cube4_submission_46662.csv"
PIDS = [133, 946, 467, 985, 294, 582, 846, 97, 299,
        487, 685, 881, 118, 346, 610, 848, 193, 707]

ARMS = [
    ("s3 standalone", HERE / "results" / "arm_s3_1m.sqlite3"),
    ("s3 + blend 0.4", HERE / "results" / "arm_blend_1m.sqlite3"),
]


def floor_lengths() -> dict[int, int]:
    with open(FLOOR, encoding="utf-8") as fh:
        return {int(r["initial_state_id"]): len(r["path"].split("."))
                for r in csv.DictReader(fh)}


def arm_lengths(db: Path) -> dict[int, int]:
    if not db.exists():
        return {}
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT initial_state_id, path_length FROM solutions").fetchall()
    finally:
        con.close()
    return {int(p): int(n) for p, n in rows}


def main() -> int:
    floors = floor_lengths()
    floor_total = sum(floors[p] for p in PIDS)
    arms = [(name, arm_lengths(db)) for name, db in ARMS]

    print(f"{len(PIDS)} pids, B=2**20, 100 steps, 1 attempt, endgame depth 6")
    print(f"46,662-file floor on these pids: {floor_total} moves\n")

    header = f"{'pid':>6}{'floor':>8}"
    for name, _ in arms:
        header += f"{name:>17}"
    header += f"{'best':>7}"
    print(header)
    for p in PIDS:
        line = f"{p:>6}{floors[p]:>8}"
        vals = []
        for _, got in arms:
            if p in got:
                n = got[p]
                vals.append(n)
                mark = "*" if n < floors[p] else ("=" if n == floors[p] else " ")
                line += f"{str(n) + mark:>17}"
            else:
                line += f"{'--':>17}"
        line += f"{min(vals + [floors[p]]):>7}"
        print(line)

    print()
    print(f"{'arm':>16}{'solved':>8}{'own tot':>10}{'vs floor':>10}"
          f"{'wins':>6}{'ties':>6}{'merge':>8}{'gain':>6}")
    for name, got in arms:
        common = [p for p in PIDS if p in got]
        if not common:
            print(f"{name:>16}{0:>8}{'--':>10}")
            continue
        own = sum(got[p] for p in common)
        fl = sum(floors[p] for p in common)
        wins = sum(1 for p in common if got[p] < floors[p])
        ties = sum(1 for p in common if got[p] == floors[p])
        merged = sum(min(got[p], floors[p]) if p in got else floors[p] for p in PIDS)
        pct = 100.0 * (own - fl) / max(fl, 1)
        print(f"{name:>16}{len(common):>8}{own:>10}{own - fl:>+10}"
              f"{wins:>6}{ties:>6}{merged:>8}{floor_total - merged:>6}"
              f"   ({pct:+.1f}%)")

    both = {p: min(g[p] for _, g in arms if p in g)
            for p in PIDS if any(p in g for _, g in arms)}
    if both:
        union = sum(min(both.get(p, floors[p]), floors[p]) for p in PIDS)
        print(f"\nper-pid min over BOTH arms merged into the floor: {union} "
              f"(gain {floor_total - union})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
