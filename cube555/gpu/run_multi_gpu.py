"""Run the cube555 beam across N GPUs by splitting (pid, frame) TASKS, not the beam.

WHY TASK-PARALLEL AND NOT A SHARDED BEAM. Every sibling kernel in this family shards
one beam across devices to buy width. On cube555 that would buy nothing, because width
is measured NON-MONOTONIC here -- same pids, same frame, only width varying:

    2^20  2/6 solved      2^21  6/6      2^22  3/6      2^23  0/1

with no OOM anywhere; 2^23 ran fine and simply found less. A wider beam admits
candidates the scorer cannot rank and they crowd out the good ones in a global top-B.
So the optimum is 2^21, one 2^21 beam fits a single 16 GB T4, and a second card is
worth more as a second *worker* than as half of a bigger beam.

What the second card actually buys, in order:
  1. FRAMES. Each conjugation frame is close to an independent ~33-50% draw, so k
     frames solve 1-(1-q)^k. This runner keeps the per-pid MINIMUM over every frame it
     runs, which the source project lists as untested and "strictly better" than the
     first-frame-wins the PyTorch solver does.
  2. PIDS. Deeper coverage of the queue in a fixed session.
Both are pure throughput, which is exactly what two independent workers deliver, with
zero cross-device communication and no failure mode where one card stalls the other.

`--devices cuda:0,cuda:0` is legal and is how the orchestration is tested on a
single-GPU box: two workers, same card, smaller beams.

    python run_multi_gpu.py --pids 1034,1033 --frames "0f,7i" --devices cuda:0,cuda:1
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_HANDOFF = PROJECT / "cube555_pull" / "cube555_handoff_2026_08_22" / "cube555"
DEFAULT_ASSETS = PROJECT / "cube555" / "tpu" / "kaggle_dataset"


def parse_frames(spec: str) -> list[tuple[int, bool]]:
    """'0f,7i' -> [(0, False), (7, True)].  f = forward, i = inverse."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok[-1] not in "fi":
            raise SystemExit(f"frame {tok!r} must end in 'f' (forward) or 'i' (inverse)")
        out.append((int(tok[:-1]), tok[-1] == "i"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    ap.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--devices", default="cuda:0,cuda:1")
    ap.add_argument("--pids", default="1034,1033,1032,1031",
                    help="comma list, or 'deepest:N' for the N deepest pids")
    ap.add_argument("--frames", default="0f,7i")
    ap.add_argument("--beams", default="2097152")
    ap.add_argument("--max-steps", default="300")
    ap.add_argument("--endgame-depth", type=int, default=5)
    ap.add_argument("--internal-batch-size", type=int, default=65536)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--history-depth", type=int, default=4)
    ap.add_argument("--no-backtrack", action="store_true")
    ap.add_argument("--test-csv", type=Path, default=None)
    ap.add_argument("--baseline-csv", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=HERE / "_run")
    ap.add_argument("--out", type=Path, default=HERE / "_run" / "submission.csv")
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    frames = parse_frames(args.frames)
    test_csv = args.test_csv or (args.handoff / "data" / "test.csv")

    tests = {}
    with open(test_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tests[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)

    if args.pids.startswith("deepest:"):
        n = int(args.pids.split(":")[1])
        pids = sorted(tests, reverse=True)[:n]
    else:
        pids = [int(p) for p in args.pids.split(",")]
    missing = [p for p in pids if p not in tests]
    if missing:
        raise SystemExit(f"pids not in test.csv: {missing[:10]}")

    # Deepest-first: the shipped baseline for pid p is length p-34, so saving is
    # base_p - L. Ordering this way front-loads the gain and makes any stopping point
    # near-optimal. Round-robin over devices keeps both cards on comparable depths
    # rather than giving one card the whole hard end of the queue.
    tasks = [{"pid": p, "sym": k, "inverted": inv}
             for p in sorted(pids, reverse=True) for (k, inv) in frames]
    shards: list[list[dict]] = [[] for _ in devices]
    for i, t in enumerate(tasks):
        shards[i % len(devices)].append(t)

    # Resolve and validate the baseline BEFORE launching any worker. Doing it at merge
    # time meant a bad path was only discovered after the full solve had run, which on
    # Kaggle is hours. A silently-empty baseline also yields a submission holding only
    # this run's pids, which reads as success and is not submittable.
    bl = args.baseline_csv or (args.handoff / "bench" / "sample_reduced.csv")
    if args.baseline_csv is not None and not bl.exists():
        raise SystemExit(f"--baseline-csv {bl} does not exist")

    args.work.mkdir(parents=True, exist_ok=True)
    procs, outs = [], []
    print(f"{len(tasks)} tasks ({len(pids)} pids x {len(frames)} frames) over "
          f"{len(devices)} device(s): {[len(s) for s in shards]}")
    t0 = time.time()
    for i, (dev, shard) in enumerate(zip(devices, shards)):
        if not shard:
            continue
        tf = args.work / f"tasks_{i}.json"
        of = args.work / f"results_{i}.json"
        tf.write_text(json.dumps(shard), encoding="utf-8")
        outs.append(of)
        cmd = [sys.executable, str(HERE / "gpu_beam.py"),
               "--device", dev, "--tasks", str(tf), "--out", str(of),
               "--handoff", str(args.handoff), "--assets", str(args.assets),
               "--beams", args.beams, "--max-steps", args.max_steps,
               "--endgame-depth", str(args.endgame_depth),
               "--internal-batch-size", str(args.internal_batch_size),
               "--dtype", args.dtype, "--history-depth", str(args.history_depth),
               "--test-csv", str(test_csv)]
        if args.checkpoint:
            cmd += ["--checkpoint", str(args.checkpoint)]
        if args.no_backtrack:
            cmd += ["--no-backtrack"]
        procs.append(subprocess.Popen(cmd))
    rcs = [p.wait() for p in procs]
    print(f"\nworkers returned {rcs} in {time.time() - t0:.0f}s")

    # ---- merge: best-of-frames per pid, replay-verified ----------------------
    recs = []
    for of in outs:
        if of.exists():
            recs += json.loads(of.read_text(encoding="utf-8"))
    (args.work / "results_all.json").write_text(json.dumps(recs, indent=1),
                                                encoding="utf-8")

    info = json.loads((args.assets / "puzzle_info.json").read_text(encoding="utf-8"))
    G = {n: np.asarray(v, dtype=np.int64) for n, v in info["generators"].items()}
    central = np.asarray(info["central_state"], dtype=np.int64)

    def replays(pid, path):
        cur = tests[pid].copy()
        for m in path.split("."):
            if m not in G:
                return False
            cur = cur[G[m]]
        return bool(np.array_equal(cur, central))

    best: dict[int, str] = {}
    for r in recs:
        if "path" not in r:
            continue
        if not replays(r["pid"], r["path"]):
            print(f"  [WARN] pid {r['pid']} k={r['sym']}: path does not replay, dropped")
            continue
        if r["pid"] not in best or r["path_len"] < len(best[r["pid"]].split(".")):
            best[r["pid"]] = r["path"]

    print(f"\n{'pid':>6} {'base':>6} {'best':>6} {'saved':>7}   per-frame")
    print("-" * 62)
    saved = 0
    for pid in sorted(best, reverse=True):
        base = pid - 34 if pid >= 35 else None
        L = len(best[pid].split("."))
        s = (base - L) if base else 0
        saved += max(0, s)
        per = " ".join(f"k{r['sym']}{'i' if r['inverted'] else 'f'}:"
                       + str(r.get("path_len", "-")) for r in recs if r["pid"] == pid)
        print(f"{pid:>6} {str(base or '-'):>6} {L:>6} {s:>7}   {per}")
    print("-" * 62)
    print(f"solved {len(best)}/{len(pids)} pids; moves saved vs baseline {saved:,}")

    # ---- submission: prefill from a baseline, overwrite only where shorter ----
    baseline = {}
    if bl.exists():
        with open(bl, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("path"):
                    baseline[int(r["initial_state_id"])] = r["path"]
        print(f"baseline {bl.name}: {len(baseline)} rows, "
              f"{sum(len(p.split('.')) for p in baseline.values()):,} moves")
    rows = dict(baseline)
    n_better = 0
    for pid, path in best.items():
        if pid not in rows or len(path.split(".")) < len(rows[pid].split(".")):
            rows[pid] = path
            n_better += 1
    bad = [pid for pid, p in rows.items() if pid in tests and not replays(pid, p)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["initial_state_id", "path"])
        for pid in sorted(rows):
            w.writerow([pid, rows[pid]])
    total = sum(len(p.split(".")) for p in rows.values())
    print(f"wrote {args.out}: {len(rows)} rows, {total:,} moves "
          f"({n_better} improved, {len(bad)} invalid)")
    return 0 if not bad and all(rc == 0 for rc in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
