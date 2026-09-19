"""Matched-control gate for Beam-AVI checkpoints.

Runs every arm through `30_solve.py` in ONE invocation, on the SAME pids, box, flags
and conversion, then reports paired per-pid deltas against the first arm. Everything
about that sentence is load-bearing: CLAUDE.md 28 exists because a total compared
across any of those axes silently attributes the difference to the variable under test.

WHY PAIRED W/L/T AND NOT THE TOTAL. A 15-pid total resolves ~+-2 moves/pid, and the
incumbent's own 800-epoch sweep sat in a 424-431 band with no trend -- so a total delta
under ~8-10 moves is unreadable. The per-pid sign test is far more sensitive: 20W/5L on
the same scrambles is decisive at a total delta the band would swallow. Both are
reported; believe the paired one.

Solving is delegated to `30_solve.py` rather than reimplemented, so every arm gets the
same exact d<=6 endgame splice and the same replay verification against the original
scramble. `--floor none --no-merge` keeps the output a PURE beam result: merging against
a floor would clamp every arm to the same banked paths and hide the difference.

    .venv/Scripts/python.exe tetraminx/scripts/76_gate_avi.py \
        --arm incumbent=tetraminx/models/mx_tf_az/epoch_1500.pt \
        --arm r000=tetraminx/models/avi/r000.pt \
        --beam 262144 --out tetraminx/results/gate_avi_b18.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT / "tetraminx" / "scripts"
PY = sys.executable

# The standard stratified gate set, held out of AVI generation by 73_gen_harvest.py's
# explicit GATE_PIDS exclusion list.
GATE_PIDS = "0,50,100,200,300,400,500,600,700,800,900,950,990,995,999"


def sign_test(w: int, l: int) -> float:
    """Two-sided exact binomial p under H0: a win and a loss are equally likely."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired matched-control gate for AVI checkpoints.")
    ap.add_argument("--arm", action="append", required=True,
                    help="name=path/to/checkpoint.pt; the FIRST arm is the baseline")
    ap.add_argument("--partner", type=Path,
                    default=PROJECT / "tetraminx" / "models" / "mx_resmlp_az" / "best.pt")
    ap.add_argument("--no-blend", action="store_true",
                    help="transformer only. AVI moves Q's level, so the 0.8/0.2 blend "
                         "tuned against the incumbent must be RE-gated, not inherited")
    ap.add_argument("--qv-consistency", type=float, default=0.3)
    ap.add_argument("--history-depth", type=int, default=1)
    ap.add_argument("--beam", type=int, default=262144)
    ap.add_argument("--sym-frames", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=45)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--pids", type=str, default=GATE_PIDS)
    ap.add_argument("--work-dir", type=Path,
                    default=PROJECT / "tetraminx" / "results" / "gate_runs")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    arms = []
    for spec in args.arm:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT / p
        if not p.exists():
            raise SystemExit(f"arm {name}: checkpoint {p} missing")
        arms.append((name, p))
    args.work_dir.mkdir(parents=True, exist_ok=True)

    tag = f"b{args.beam}_qv{args.qv_consistency}_{'solo' if args.no_blend else 'blend'}"
    lens: dict[str, dict[int, int]] = {}
    walls: dict[str, float] = {}

    for name, ckpt in arms:
        out_csv = args.work_dir / f"{name}_{tag}.csv"
        if out_csv.exists():
            out_csv.unlink()
        cmd = [PY, str(SCRIPTS / "30_solve.py"),
               "--checkpoint", str(ckpt),
               "--out", str(out_csv),
               "--floor", "none", "--no-merge",
               "--beam", str(args.beam), "--sym-frames", str(args.sym_frames),
               "--max-steps", str(args.max_steps),
               "--history-depth", str(args.history_depth),
               "--qv-consistency", str(args.qv_consistency),
               "--chunk-size", str(args.chunk_size),
               "--pids", args.pids, "--bf16"]
        if not args.no_blend:
            cmd += ["--blend", str(args.partner), "--blend-weights", "0.8", "0.2"]
        print(f"\n=== arm {name}: {ckpt.name} ===", flush=True)
        t = time.time()
        r = subprocess.run(cmd)
        walls[name] = time.time() - t
        if r.returncode != 0:
            raise SystemExit(f"arm {name} failed with exit {r.returncode}")
        with open(out_csv, encoding="utf-8", newline="") as f:
            lens[name] = {int(row["initial_state_id"]): len(row["path"].split("."))
                          for row in csv.DictReader(f) if row["path"]}
        print(f"[{name}] {len(lens[name])} solved, total {sum(lens[name].values())}, "
              f"{walls[name]:.0f}s", flush=True)

    base_name = arms[0][0]
    base = lens[base_name]
    common = sorted(set.intersection(*(set(v) for v in lens.values())))
    print(f"\n=== gate: {args.beam:,} x {args.sym_frames} frame, qv={args.qv_consistency}, "
          f"hd={args.history_depth}, {'solo' if args.no_blend else 'blend'} ===")
    print(f"{len(common)} pids solved by every arm (baseline = {base_name})\n")
    print(f"{'arm':>12} {'solved':>7} {'total':>7} {'vs base':>8} {'W':>4} {'L':>4} "
          f"{'T':>4} {'p':>8} {'wall':>7}")
    print("-" * 70)
    rows = []
    for name, _ in arms:
        cur = lens[name]
        tot = sum(cur[p] for p in common)
        bt = sum(base[p] for p in common)
        w = sum(1 for p in common if cur[p] < base[p])
        l = sum(1 for p in common if cur[p] > base[p])
        t_ = len(common) - w - l
        p_val = sign_test(w, l) if name != base_name else 1.0
        rows.append(dict(arm=name, solved=len(cur), total=tot, delta=tot - bt,
                         w=w, l=l, t=t_, p=p_val, wall=walls[name],
                         per_pid={str(k): cur.get(k) for k in common}))
        print(f"{name:>12} {len(cur):>7} {tot:>7} {tot-bt:>+8d} {w:>4} {l:>4} {t_:>4} "
              f"{p_val:>8.4f} {walls[name]:>6.0f}s")

    print(f"\nper-pid ({base_name} -> others), pids {common}")
    for name, _ in arms[1:]:
        d = [lens[name][p] - base[p] for p in common]
        print(f"  {name:>10}: {d}")
    print("\nREAD THE PAIRED COLUMNS, NOT THE TOTAL: a 15-pid total resolves ~+-2 "
          "moves/pid.\nThe incumbent's own 800-epoch sweep sat in a 424-431 band with "
          "no trend.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dict(config=dict(beam=args.beam, sym_frames=args.sym_frames,
                                   qv_consistency=args.qv_consistency,
                                   history_depth=args.history_depth,
                                   blend=not args.no_blend, pids=args.pids),
                       baseline=base_name, common=common, arms=rows), f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
