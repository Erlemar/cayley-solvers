"""Two-phase 4x4x4 solve, and the phase-1 A/B (margin vs control).

  --phase1-only   report reduction length + reach rate only. This is the clean A/B
                  for the sparse-Q margin term: same beam, same pids, same width,
                  the two arms differ only in `lambda_margin`.
  (default)       full two-phase: reduction beam + classical 3x3x3 finish, with the
                  per-endpoint joint minimisation and a hard verify of every word.

    python cube444/scripts/14_solve_two_phase.py \
        --model cube444/models/p1_margin/epoch_0149.pt --pids 0-19 --beam 16384
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "src"))

from cube444.orbits import reduction_defects  # noqa: E402
from cube444.puzzle import Cube444  # noqa: E402
from cube444.two_phase import Phase1Beam, solve_two_phase  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

load_masked = import_module("12_train_phase1").load_masked


def parse_pids(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--pids", default="0-19")
    ap.add_argument("--beam", type=int, default=16384)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--endpoints", type=int, default=8)
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--p2-timeout", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    puz = Cube444.load(PROJECT / "data" / "puzzle_info.json")
    gens = {k: np.array(v, dtype=np.int64) for k, v in puz.generators.items()}
    solved = np.array(puz.solved_state, dtype=np.int64)

    init = {}
    with open(PROJECT / "data" / "test.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            init[int(row["initial_state_id"])] = np.array(
                [int(x) for x in row["initial_state"].split(",")], dtype=np.int64)

    model = load_masked(args.model, args.device)
    beam = Phase1Beam(puz, model, device=args.device, beam_width=args.beam,
                      max_steps=args.max_steps)
    p2 = None
    if not args.phase1_only:
        from cube444.phase2 import Phase2Solver
        p2 = Phase2Solver(gens, solved, timeout=args.p2_timeout)

    pids = [p for p in parse_pids(args.pids) if p in init]
    print(f"model {args.model}")
    print(f"beam {args.beam}  max_steps {args.max_steps}  endpoints {args.endpoints}  "
          f"pids {len(pids)}")

    rows = []
    t_all = time.time()
    for pid in pids:
        s = init[pid]
        t0 = time.time()
        if args.phase1_only:
            r = beam.run(s, want_endpoints=args.endpoints)
            best = min((len(w) for w, _ in r.endpoints), default=None)
            rows.append({"pid": pid, "p1_len": best, "n_endpoints": len(r.endpoints),
                         "steps": r.steps_run, "sec": time.time() - t0,
                         "start_defects": int(reduction_defects(s))})
            print(f"  pid {pid:4d}  p1={best}  endpoints={len(r.endpoints):2d}  "
                  f"steps={r.steps_run:2d}  ({time.time() - t0:.1f}s)", flush=True)
        else:
            res = solve_two_phase(s, beam, p2, puz, want_endpoints=args.endpoints)
            rows.append({"pid": pid, **{k: v for k, v in res.items() if k != "word"},
                         "sec": time.time() - t0,
                         "path": ".".join(res["word"]) if res["word"] else ""})
            print(f"  pid {pid:4d}  total={res['total']}  p1={res['p1_len']} "
                  f"p2={res['p2_len']}  parity_fail={res['n_parity_fail']}  "
                  f"verified={res['verified']}  ({time.time() - t0:.1f}s)", flush=True)

    print(f"\n=== summary ({time.time() - t_all:.1f}s) ===")
    if args.phase1_only:
        got = [r for r in rows if r["p1_len"] is not None]
        print(f"  reached R: {len(got)}/{len(rows)}")
        if got:
            v = np.array([r["p1_len"] for r in got])
            print(f"  phase-1 length: mean {v.mean():.2f}  min {v.min()}  max {v.max()}")
            print(f"  (counting bound for distance-to-R is ~21.5 moves)")
    else:
        got = [r for r in rows if r["found"]]
        ver = [r for r in got if r["verified"]]
        print(f"  solved {len(got)}/{len(rows)}   verified {len(ver)}/{len(got)}")
        if ver:
            tot = np.array([r["total"] for r in ver])
            p1 = np.array([r["p1_len"] for r in ver])
            p2v = np.array([r["p2_len"] for r in ver])
            print(f"  total mean {tot.mean():.2f}  (p1 {p1.mean():.2f} + "
                  f"p2 {p2v.mean():.2f})")
            print(f"  parity failures: {sum(r['n_parity_fail'] for r in rows)} endpoints")
            print(f"  reference: our per-pid floor mean 46.73, Rokicki 44.39")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
