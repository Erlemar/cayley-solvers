"""Beam-evaluate every Nth training checkpoint as it lands; keep a path-length trend table.

Training rule (user, 2026-09-16): after each 200 epochs, measure the path lengths the beam
finds with that checkpoint; keep training while they keep shortening.

For each `epoch_XXXX.pt` in --ckpt-dir with XXXX % --every == 0 and no result yet, this runs
`scripts/84_solve_tf.py` on the fixed gate (--pid-file) and appends one row to
`<out-dir>/trend.csv`. The comparison metric is `score` = sum over ALL gate pids of the
found length, with an unsolved pid charged floor + --miss-penalty, so a checkpoint cannot
look better by failing hard pids. Lower is better. The beam is deterministic, so one run
per checkpoint is the measurement (add pids, not repeats, for resolution).

    python scripts/85_watch_eval.py --ckpt-dir models/ihes_tf_a --out-dir submissions/eval/ihes_tf_a
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TREND_FIELDS = ["epoch", "score", "solved", "n", "total", "floor_same", "excess", "ties",
                "wins", "unsolved", "wall_s", "stamp"]


def score_of(summary: dict, floor: dict[str, int], penalty: int) -> int:
    s = 0
    for pid, row in summary["per_pid"].items():
        s += row["len"] if row["len"] is not None else floor[pid] + penalty
    return s


def verdict(rows: list[dict]) -> str:
    """IMPROVING while the best score was set by one of the last two evals."""
    if len(rows) < 2:
        return "baseline"
    scores = [int(r["score"]) for r in rows]
    best_at = min(range(len(scores)), key=lambda i: (scores[i], i))
    return "IMPROVING" if best_at >= len(scores) - 2 else "FLAT (best at epoch %s)" % rows[best_at]["epoch"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch a checkpoint dir and beam-eval every Nth epoch.")
    ap.add_argument("--ckpt-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--every", type=int, default=200)
    ap.add_argument("--prefix", default="epoch_",
                    help="checkpoint name prefix: epoch_ (stage 1) or step_ (86_train_q_bellman.py)")
    ap.add_argument("--pid-file", type=Path, default=PROJECT / "data" / "ihes_gate54.json")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--sym-frames", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--miss-penalty", type=int, default=10)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--min-age", type=int, default=30, help="seconds a checkpoint must be untouched")
    ap.add_argument("--once", action="store_true", help="evaluate what is there, then exit")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="passed through to 84_solve_tf.py (e.g. --bf16)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trend = args.out_dir / "trend.csv"
    gate = json.loads(args.pid_file.read_text(encoding="utf-8"))
    floor = {str(k): int(v) for k, v in gate["floor"].items()}
    print(f"watching {args.ckpt_dir} every {args.every} epochs | gate {args.pid_file.name} "
          f"({len(gate['pids'])} pids, floor {gate['floor_total']}) | beam {args.beam} "
          f"frames {args.sym_frames} | extra {args.extra}", flush=True)

    while True:
        rows = []
        if trend.exists():
            with open(trend, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
        seen = {int(r["epoch"]) for r in rows}
        todo = []
        for p in args.ckpt_dir.glob(f"{args.prefix}*.pt"):
            m = re.fullmatch(re.escape(args.prefix) + r"(\d+)\.pt", p.name)
            if not m:
                continue
            ep = int(m.group(1))
            if ep % args.every or ep in seen:
                continue
            if time.time() - p.stat().st_mtime < args.min_age:
                continue
            todo.append((ep, p))
        for ep, p in sorted(todo):
            tag = f"e{ep:04d}"
            out_csv = args.out_dir / f"{tag}.csv"
            out_json = args.out_dir / f"{tag}.json"
            cmd = [args.python, "-u", str(PROJECT / "scripts" / "84_solve_tf.py"),
                   "--checkpoint", str(p), "--pid-file", str(args.pid_file),
                   "--beam", str(args.beam), "--sym-frames", str(args.sym_frames),
                   "--max-steps", str(args.max_steps), "--no-merge",
                   "--out", str(out_csv), "--summary-json", str(out_json), *args.extra]
            print(f"[{time.strftime('%H:%M:%S')}] eval epoch {ep}: {' '.join(cmd[2:])}", flush=True)
            with open(args.out_dir / f"{tag}.log", "w", encoding="utf-8") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT).returncode
            if rc != 0 or not out_json.exists():
                print(f"  eval epoch {ep} FAILED rc={rc} (see {tag}.log)", flush=True)
                continue
            s = json.loads(out_json.read_text(encoding="utf-8"))
            row = {"epoch": ep, "score": score_of(s, floor, args.miss_penalty),
                   "solved": s["solved"], "n": s["n"], "total": s["total"],
                   "floor_same": s["floor_same"], "excess": s["excess"], "ties": s["ties"],
                   "wins": s["wins"], "unsolved": len(s["unsolved"]), "wall_s": s["wall_s"],
                   "stamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            rows.append({k: str(v) for k, v in row.items()})
            rows.sort(key=lambda r: int(r["epoch"]))
            with open(trend, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=TREND_FIELDS)
                w.writeheader()
                w.writerows(rows)
            print(f"  TREND epoch {ep}: score {row['score']} (gate floor {gate['floor_total']}) "
                  f"solved {row['solved']}/{row['n']} excess {row['excess']:+d} ties {row['ties']} "
                  f"wins {row['wins']} | {verdict(rows)} | {row['wall_s']:.0f}s", flush=True)
        if args.once:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
