"""Collapse same-axis runs in a solution word. Free, lossless, no model.

    python cube555/scripts/42_commute_reduce.py in.csv out.csv

All layers of one axis commute with each other (f0 and f3 touch disjoint stickers), so a
maximal run of consecutive same-axis moves can be reordered freely and each layer's moves
inside that run summed mod 4. Net 0 drops out entirely; net 3 becomes a single inverse
turn. Removing moves can merge the runs on either side, so this iterates to a fixed point.

There are no half turns in this generator set, so a net of 2 still costs 2 moves -- pricing
imported cubing algorithms in this metric is a standing trap (`Rw2` is 4 moves here).

EVERY reduced path is replayed against test.csv before it is written. A commuting-run
rewrite is exactly the kind of change that looks obviously correct and is off by one
somewhere, and the whole point of this script is that it is free -- a wrong free thing is
worse than nothing.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from cube555.puzzle import Cube555  # noqa: E402

N_SANTA = 35


def axis_layer_sign(name: str) -> tuple[str, int, int]:
    neg = name.startswith("-")
    body = name[1:] if neg else name
    return body[0], int(body[1:]), (-1 if neg else 1)


def reduce_word(moves: list[str]) -> list[str]:
    cur = list(moves)
    while True:
        out: list[str] = []
        i = 0
        changed = False
        while i < len(cur):
            ax = axis_layer_sign(cur[i])[0]
            j = i
            while j < len(cur) and axis_layer_sign(cur[j])[0] == ax:
                j += 1
            net: dict[int, int] = {}
            order: list[int] = []
            for m in cur[i:j]:
                _, lay, sgn = axis_layer_sign(m)
                if lay not in net:
                    order.append(lay)
                net[lay] = (net.get(lay, 0) + sgn) % 4
            emitted: list[str] = []
            for lay in order:
                n = net[lay]
                if n == 0:
                    continue
                if n == 1:
                    emitted.append(f"{ax}{lay}")
                elif n == 2:
                    emitted.extend([f"{ax}{lay}", f"{ax}{lay}"])
                else:
                    emitted.append(f"-{ax}{lay}")
            if len(emitted) != j - i:
                changed = True
            out.extend(emitted)
            i = j
        cur = out
        if not changed:
            return cur


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: 42_commute_reduce.py <in.csv> <out.csv>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    puz = Cube555.load(PROJECT / "data" / "puzzle_info.json")
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}
    central = np.array(puz.solved_state, dtype=np.int64)
    tests = {
        int(r["initial_state_id"]): np.array(
            [int(x) for x in r["initial_state"].split(",")], dtype=np.int64
        )
        for r in csv.DictReader(open(PROJECT / "data" / "test.csv", encoding="utf-8"))
    }

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    before = after = 0
    s_before = s_after = 0
    out_rows, failed = [], 0
    for r in rows:
        pid = int(r["initial_state_id"])
        mv = [m for m in r["path"].split(".") if m]
        red = reduce_word(mv)
        cur = tests[pid].copy()
        for m in red:
            cur = cur[G[m]]
        if not np.array_equal(cur, central):
            failed += 1
            red = mv  # keep the original rather than ship something that does not solve
        before += len(mv)
        after += len(red)
        if pid < N_SANTA:
            s_before += len(mv)
            s_after += len(red)
        out_rows.append((pid, puz.format_path(red)))

    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid, p in sorted(out_rows):
            w.writerow([pid, p])

    print(
        f"{src.name}: {before:,} -> {after:,} moves  ({100.0*(after-before)/before:+.2f}%)"
    )
    if s_before:
        n = sum(1 for r in rows if int(r["initial_state_id"]) < N_SANTA)
        print(f"  santa subset (n={n}): mean {s_before/n:.3f} -> {s_after/n:.3f}")
    print(f"  replay failures (kept original): {failed}")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
