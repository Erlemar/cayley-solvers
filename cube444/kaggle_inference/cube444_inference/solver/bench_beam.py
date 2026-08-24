"""Matched beam A/B across checkpoints on a fixed hard pid set.

Rule 5: same pid set, machine, flags AND checkpoint before quoting any delta. Rule 4: judge
by the per-pid MIN against the floor, never a standalone mean -- a run whose own mean looks
bad can still be worth hundreds of moves in the merge.

Rule 7 is the reason this script exists at all: the by-depth proxies disagree with each other
(stage 2 gains pair accuracy and loses top-1 in the same bands), so only a real beam settles
which checkpoint to take forward.

The 18 pids are the deepest in the 46,662 file (baseline 49-50 moves each, total 884) --
the band where every solve starts and where the scorer is weakest.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import json
import subprocess
import sys
import time
from pathlib import Path
from pathlib import Path as pathlib_Path

HERE = Path(__file__).resolve().parent
SOLVER = HERE  # bench_beam.py lives inside solver/ in this bundle
PY = os.environ.get("CUBE444_PY", sys.executable)
FLOOR = pathlib_Path(os.environ.get(
    "CUBE444_FLOOR", HERE.parent / "submissions/cube4_submission_46662.csv"))
HARD_PIDS = "60,76,102,134,148,248,250,274,278,297,312,330,342,364,380,420,422,797"


def floor_lengths() -> dict[int, int]:
    return {
        int(r["initial_state_id"]): len(r["path"].split("."))
        for r in csv.DictReader(open(FLOOR, encoding="utf-8"))
    }


def run_arm(
    tag: str,
    info: Path,
    weights: Path,
    pids: str,
    beam: int,
    steps: int,
    attempts: int,
    mlp_weight: float,
    compile_on: bool,
    search_seed: int = 0,
    tail_bfs_depth: int = 0,
    history_depth: int = 0,
) -> dict:
    sfx = f"_s{search_seed}_t{tail_bfs_depth}_h{history_depth}_m{mlp_weight:g}"
    db = SOLVER / "results" / f"bench_{tag}{sfx}.sqlite3"
    out = SOLVER / "results" / f"bench_{tag}{sfx}.csv"
    for f in (db, out, Path(str(db) + "-wal"), Path(str(db) + "-shm")):
        if f.exists():
            f.unlink()
    cmd = [
        PY,
        "solve_ensemble_submission.py",
        "--transformer-info",
        str(info),
        "--transformer-weights",
        str(weights),
        "--mlp-weight",
        str(mlp_weight),
        "--baseline-submission",
        str(FLOOR),
        "--pids",
        pids,
        "--B",
        str(beam),
        "--num-steps",
        str(steps),
        "--num-attempts",
        str(attempts),
        "--search-seed",
        str(search_seed),
        "--tail-bfs-depth",
        str(tail_bfs_depth),
        "--history-depth",
        str(history_depth),
        "--progress-db",
        str(db),
        "--output",
        str(out),
    ]
    if not compile_on:
        cmd.append("--no-compile")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=SOLVER, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise SystemExit(f"arm {tag} failed")

    # The progress DB holds the RAW beam result; the CSV is already min-merged with the
    # floor, which would hide a regression.
    import sqlite3

    con = sqlite3.connect(db)
    rows = dict(
        con.execute("SELECT initial_state_id, path_length FROM solutions").fetchall()
    )
    con.close()
    return {"tag": tag, "wall": wall, "solved": rows, "stdout": proc.stdout}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arms",
        required=True,
        help="tag=dir[|tail=N][|hist=N][|mlp=W],... each dir holding model.json + "
        "model.pth. Per-arm options override the global flags, so a matched A/B of a "
        "search setting runs inside ONE invocation.",
    )
    ap.add_argument(
        "--tail-bfs-depth", type=int, default=0, help="default for all arms"
    )
    ap.add_argument("--history-depth", type=int, default=0, help="default for all arms")
    ap.add_argument("--pids", default=HARD_PIDS)
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--mlp-weight", type=float, default=0.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument(
        "--seeds",
        default="0",
        help="comma-separated --search-seed values. The beam's Zobrist hash vector is "
        "seeded from this, so a different seed re-rolls dedup and top-k tie-breaking: "
        "an independent draw of the SAME arm. Quoting a one-seed delta as an improvement "
        "assumes a noise floor nobody has measured.",
    )
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "beam_bench.json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    floors = floor_lengths()
    want = [int(p) for p in a.pids.split(",")]
    floor_total = sum(floors[p] for p in want)
    print(
        f"{len(want)} pids, B={a.beam}, steps={a.steps}, attempts={a.attempts}, "
        f"mlp_weight={a.mlp_weight}"
    )
    print(f"46,662-file floor on these pids: {floor_total} moves\n")

    results = []
    for spec in a.arms.split(","):
        parts = spec.split("|")
        tag0, d = parts[0].split("=", 1)
        opts = dict(p.split("=", 1) for p in parts[1:])
        tail = int(opts.get("tail", a.tail_bfs_depth))
        hist = int(opts.get("hist", a.history_depth))
        mlpw = float(opts.get("mlp", a.mlp_weight))
        # the solver subprocess runs with cwd=SOLVER, so arm paths must be absolute
        d = Path(d).resolve()
        if not (d / "model.pth").exists():
            raise SystemExit(f"arm {tag0}: no model.pth in {d}")
        for sd in seeds:
            tag = tag0 if len(seeds) == 1 else f"{tag0}@{sd}"
            print(
                f"--- arm {tag}: {d}  (tail_bfs={tail} history={hist} mlp_w={mlpw})",
                flush=True,
            )
            r = run_arm(
                tag0,
                d / "model.json",
                d / "model.pth",
                a.pids,
                a.beam,
                a.steps,
                a.attempts,
                mlpw,
                a.compile,
                search_seed=sd,
                tail_bfs_depth=tail,
                history_depth=hist,
            )
            r["tag"], r["arm"], r["seed"] = tag, tag0, sd
            r["tail_bfs"], r["history"], r["mlp_weight"] = tail, hist, mlpw
            solved = r["solved"]
            common = [p for p in want if p in solved]
            tot = sum(solved[p] for p in common)
            ftot = sum(floors[p] for p in common)
            wins = sum(1 for p in common if solved[p] < floors[p])
            ties = sum(1 for p in common if solved[p] == floors[p])
            merged = sum(
                min(solved[p], floors[p]) if p in solved else floors[p] for p in want
            )
            r.update(
                solved_n=len(common),
                total=tot,
                floor_on_common=ftot,
                wins=wins,
                ties=ties,
                merge_total=merged,
                merge_gain=floor_total - merged,
            )
            results.append(r)
            print(
                f"    solved {len(common)}/{len(want)}  own total {tot} vs floor {ftot} "
                f"({100 * (tot - ftot) / max(ftot, 1):+.1f}%)  wins {wins} ties {ties}  "
                f"merge {merged} (gain {floor_total - merged})  wall {r['wall']:.0f}s\n",
                flush=True,
            )

    print(
        f"{'arm':>12}{'solved':>8}{'own tot':>10}{'vs floor':>10}{'wins':>6}{'merge':>8}{'gain':>7}{'wall s':>9}"
    )
    for r in results:
        d = r["total"] - r["floor_on_common"]
        print(
            f"{r['tag']:>12}{r['solved_n']:>8}{r['total']:>10}{d:>+10}{r['wins']:>6}"
            f"{r['merge_total']:>8}{r['merge_gain']:>7}{r['wall']:>9.0f}"
        )
    print(f"\nfloor total on all {len(want)} pids = {floor_total}")

    # ---- across-seed aggregation -------------------------------------------------------
    # Rule 4 says judge by the per-pid MIN against the floor. The same logic applies across
    # seeds: re-rolling the hash is a legitimate extra attempt, so `min` is what an ensemble
    # would actually bank. `spread` is the thing to compare any claimed gain against -- a
    # delta smaller than one arm's own seed spread is not a result.
    if len(seeds) > 1:
        print(
            f"\n{'arm':>12}{'seeds':>26}{'mean':>8}{'min':>7}{'max':>7}{'spread':>8}"
            f"{'per-pid min':>13}"
        )
        agg = {}
        for tag0 in dict.fromkeys(r["arm"] for r in results):
            rs = [r for r in results if r["arm"] == tag0]
            tots = [r["total"] for r in rs]
            pidmin = {
                p: min(r["solved"][p] for r in rs if p in r["solved"])
                for p in want
                if any(p in r["solved"] for r in rs)
            }
            agg[tag0] = dict(
                totals=tots,
                mean=sum(tots) / len(tots),
                min=min(tots),
                max=max(tots),
                spread=max(tots) - min(tots),
                per_pid_min_total=sum(pidmin.values()),
                per_pid_min_n=len(pidmin),
            )
            g = agg[tag0]
            print(
                f"{tag0:>12}{str(tots):>26}{g['mean']:>8.1f}{g['min']:>7}{g['max']:>7}"
                f"{g['spread']:>8}{g['per_pid_min_total']:>13}"
            )
        worst = max(agg.values(), key=lambda g: g["spread"])["spread"]
        print(
            f"\n  NOISE FLOOR: the widest single-arm seed spread is {worst} moves. Any "
            f"between-arm\n  difference smaller than that is indistinguishable from "
            f"re-rolling the hash seed."
        )
        a.out.with_suffix(".agg.json").write_text(
            json.dumps(agg, indent=2), encoding="utf-8"
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            [{k: v for k, v in r.items() if k != "stdout"} for r in results],
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
