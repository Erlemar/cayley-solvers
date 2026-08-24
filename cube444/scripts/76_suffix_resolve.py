"""Re-solve the SUFFIX of each path with a wide beam (heuristic post-processing).

WHY SUFFIXES, AND WHY THIS IS THE ONLY DIRECTION THAT CAN SCALE. The exact rewriters
(`70`/`74`) prove every window of length <= 10 in the 46,718 file is already geodesic, so
whatever slack remains sits in ranges too long for an exact ball to reach. A beam is not
exact, but it has no reach limit -- and our V model scores DISTANCE TO SOLVED, so the
only sub-problem it can score is a suffix `s_{L-k} -> solved`. Prefixes and interior
windows have an arbitrary target and no heuristic.

The bet: at k ~ 15-30 the suffix state sits far closer to solved than a fresh test state
(distance ~40), which is exactly the regime where our beam is strong -- and where the
public file's own solver had the least reason to be optimal. A win needs the beam to
return STRICTLY fewer than k moves.

Diagnostic first, campaign second: `--report` prints beam length vs k per pid without
touching the file, which answers "can our beam even match this file's suffixes?" before
any GPU is spent at scale. If beam_len == k routinely the method is live; if beam_len > k
always, the beam is too weak and the idea is dead.

    python cube444/scripts/76_suffix_resolve.py --target <csv> \
        --checkpoint cube444/models/c_bells2/epoch_0399.pt --k 16,20,24 --beam 65536 \
        --pids 0:30 --report
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cayley.khoruzhii_search import KhoruzhiiSearchConfig, KhoruzhiiSolver  # noqa: E402
from cube444.models import load_v_model  # noqa: E402
from cube444.puzzle import Cube444  # noqa: E402

DATA = PROJECT / "data"


def load_tests(path: Path):
    out = {}
    with io.open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out[int(r["initial_state_id"])] = np.array(
                [int(x) for x in r["initial_state"].split(",")], dtype=np.int64)
    return out


def load_paths(path: Path):
    out = {}
    with io.open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            p = r["path"].strip()
            out[int(r["initial_state_id"])] = p.split(".") if p else []
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT / "models" / "c_bells2" / "epoch_0399.pt")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--k", type=str, default="16,20,24", help="suffix lengths to retry")
    ap.add_argument("--beam", type=int, default=65536)
    ap.add_argument("--max-steps", type=int, default=0, help="0 => k + 6")
    ap.add_argument("--sym-ensemble", type=int, default=1)
    ap.add_argument("--pids", type=str, default="")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--internal-batch-size", type=int, default=1 << 16)
    ap.add_argument("--report", action="store_true", help="diagnostic only, do not rewrite")
    ap.add_argument("--progress", type=int, default=10)
    args = ap.parse_args()

    puz = Cube444.load(DATA / "puzzle_info.json")
    names = list(puz.move_names)
    G = {n: np.array(v, dtype=np.int64) for n, v in puz.generators.items()}
    central = np.array(puz.solved_state, dtype=np.int64)
    tests = load_tests(DATA / "test.csv")
    rows = load_paths(args.target)
    base_total = sum(len(v) for v in rows.values())

    ks = [int(x) for x in args.k.split(",")]
    todo = sorted(rows)
    if args.pids:
        if ":" in args.pids:
            a, b = args.pids.split(":")
            todo = [p for p in todo if int(a) <= p < int(b)]
        else:
            todo = [int(x) for x in args.pids.split(",")]

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = load_v_model(args.checkpoint, device=args.device, dtype=dtype)
    solver = KhoruzhiiSolver(puz, model, device=args.device,
                            internal_batch_size=args.internal_batch_size)
    print(f"model {args.checkpoint.name}  beam {args.beam}  k={ks}  pids={len(todo)}",
          flush=True)

    saved, tried, matched, beat = 0, 0, 0, 0
    t0 = time.time()
    for n, pid in enumerate(todo):
        w = rows[pid]
        L = len(w)
        s0 = tests[pid]
        # states along the path
        cur = s0.copy()
        states = [cur.copy()]
        for m in w:
            cur = cur[G[m]]
            states.append(cur.copy())
        for k in ks:
            if k >= L:
                continue
            i = L - k
            cfg = KhoruzhiiSearchConfig(beam_width=args.beam,
                                        num_steps=args.max_steps or (k + 6),
                                        num_attempts=1,
                                        internal_batch_size=args.internal_batch_size)
            found, _, path = solver.solve(states[i], cfg)
            tried += 1
            if not found:
                if args.report:
                    print(f"  pid {pid} k={k}: no solution", flush=True)
                continue
            chk = states[i].copy()
            for m in path:
                chk = chk[G[m]]
            if not np.array_equal(chk, central):
                print(f"  pid {pid} k={k}: beam path does NOT solve -- discarded")
                continue
            if len(path) == k:
                matched += 1
            if len(path) < k:
                beat += 1
            if args.report:
                flag = "WIN" if len(path) < k else ("=" if len(path) == k else "")
                print(f"  pid {pid} k={k}: beam {len(path)}  {flag}", flush=True)
            if len(path) < k and not args.report:
                rows[pid] = w[:i] + list(path)
                saved += k - len(path)
                print(f"  HIT pid {pid}: suffix {k} -> {len(path)}  "
                      f"(total {L} -> {len(rows[pid])})", flush=True)
                break
        if args.progress and (n + 1) % args.progress == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)} pids, {tried} solves, matched {matched}, "
                  f"beat {beat}, saved {saved}, {el:.0f}s ({el/(n+1):.1f}s/pid)", flush=True)

    print(f"\nsuffix re-solve: {tried} beam solves, matched-k {matched}, beat-k {beat}, "
          f"saved {saved}")
    if args.report:
        return 0
    total = sum(len(v) for v in rows.values())
    print(f"{base_total:,} -> {total:,}")
    bad = 0
    for pid, path in rows.items():
        cur = tests[pid].copy()
        for m in path:
            cur = cur[G[m]]
        if not np.array_equal(cur, central):
            bad += 1
    print(f"output replay check: {bad} unsolved")
    if bad:
        print("REFUSING TO WRITE -- replay failed")
        return 4
    if args.out and saved:
        with io.open(args.out, "w", encoding="utf-8", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["initial_state_id", "path"])
            for pid in sorted(rows):
                wr.writerow([pid, ".".join(rows[pid])])
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
