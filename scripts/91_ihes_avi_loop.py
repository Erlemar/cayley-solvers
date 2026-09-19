"""IHES Beam-AVI loop: harvest -> train -> level check -> gate, round after round.

Port of `tetraminx/scripts/75_avi_loop.py` (solo: the target net IS the model being trained)
with the lessons of that failed port built in:
  * every round is followed by `92_ihes_level_check.py` on ONE fixed population (shard r000)
    -- the 2-minute flattening predictor;
  * every round is gated by `84_solve_tf.py` on the fixed gate pids at the gate width, and
    compared PER PID against the incumbent's matched result (paired W/L/T + sign test);
  * stop rule (user, 2026-09-16): continue while the gate score keeps shortening -- stop when
    the best score is older than `--patience` rounds, or at `--rounds`.

Resumable: shards, checkpoints, level checks and gate summaries that already exist are reused.

    python scripts/91_ihes_avi_loop.py --arm-dir runs/ihes_avi/A --rounds 6 \
        --incumbent models/ihes_tf_b_e1300a_s3/step_02000.pt \
        --baseline-json submissions/eval/finalists_4090/BAS_g54_b18.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
PY = sys.executable


def run(cmd: list, tag: str, log: Path) -> None:
    print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}\n" + " ".join(str(c) for c in cmd), flush=True)
    t = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== {tag} ====\n")
        fh.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=PROJECT)
    if r.returncode != 0:
        raise SystemExit(f"{tag} FAILED (exit {r.returncode}); see {log}")
    print(f"[{tag} ok, {time.time() - t:.0f}s]", flush=True)


def sign_test(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run IHES Beam-AVI rounds with per-round gates.")
    ap.add_argument("--arm-dir", required=True, type=Path)
    ap.add_argument("--incumbent", required=True, type=Path)
    ap.add_argument("--baseline-json", type=Path, default=None,
                    help="84_solve_tf summary of the INCUMBENT at the gate settings, on THIS "
                         "machine (per-pid; a superset of the gate pids is fine)")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--shared-r000", type=Path, default=None,
                    help="reuse an existing round-0 shard (same incumbent => same target net)")
    # generation
    ap.add_argument("--gen-beam", type=int, default=262144)
    ap.add_argument("--n-pids", type=int, default=50)
    ap.add_argument("--rows-per-step", type=int, default=8192)
    ap.add_argument("--full-expand", type=int, default=256)
    # training
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--gap-weight", type=float, default=0.0)
    ap.add_argument("--gap-k-max", type=int, default=12)
    ap.add_argument("--gap-batch", type=int, default=256)
    ap.add_argument("--sparse-weight", type=float, default=1.0)
    ap.add_argument("--dense-batch", type=int, default=128)
    ap.add_argument("--rehearse-weight", type=float, default=0.0,
                    help="frozen-teacher distillation on walk states; the teacher is always "
                         "--incumbent, in every round")
    ap.add_argument("--rehearse-batch", type=int, default=512)
    ap.add_argument("--rehearse-k-max", type=int, default=22)
    ap.add_argument("--anchor-weight", type=float, default=2.0)
    # gate
    ap.add_argument("--level-shard", type=Path, default=None,
                    help="fixed level-check population (default: this arm's r000 shard); "
                         "point several arms at ONE shard to compare their rows directly")
    ap.add_argument("--gate-pids", type=Path, default=PROJECT / "data" / "ihes_gate54.json")
    ap.add_argument("--gate-beam", type=int, default=262144)
    ap.add_argument("--no-gate", action="store_true",
                    help="generate + train all --rounds with no in-loop gate and no early stop; "
                         "gate the checkpoints elsewhere (scripts/95_ihes_avi_gate_watch.py)")
    args = ap.parse_args()

    d = args.arm_dir if args.arm_dir.is_absolute() else PROJECT / args.arm_dir
    (d / "shards").mkdir(parents=True, exist_ok=True)
    (d / "models").mkdir(parents=True, exist_ok=True)
    (d / "gate").mkdir(parents=True, exist_ok=True)
    log = d / "loop.log"
    trend_path = d / "trend.json"
    gate_pids = [int(p) for p in json.loads(args.gate_pids.read_text(encoding="utf-8"))["pids"]]
    if not args.no_gate and args.baseline_json is None:
        raise SystemExit("--baseline-json is required unless --no-gate")
    base = ({} if args.no_gate else
            json.loads(args.baseline_json.read_text(encoding="utf-8"))["per_pid"])
    base_len = {p: base[str(p)]["len"] for p in gate_pids} if base else {}
    base_total = sum(base_len.values())
    print(f"arm {d.name}: incumbent {args.incumbent}; baseline gate total {base_total} "
          f"({len(gate_pids)} pids, from {getattr(args.baseline_json, 'name', None)}); "
          f"gap_weight {args.gap_weight}, "
          f"sparse_weight {args.sparse_weight}, full_expand {args.full_expand}, "
          f"dense_batch {args.dense_batch}, rehearse_weight {args.rehearse_weight}",
          flush=True)
    trend = json.loads(trend_path.read_text(encoding="utf-8")) if trend_path.exists() else []
    level_ckpts = [f"incumbent={args.incumbent}"]

    for k in range(args.rounds):
        prev = args.incumbent if k == 0 else d / "models" / f"r{k - 1:03d}.pt"
        shard = d / "shards" / f"r{k:03d}.pt"
        out = d / "models" / f"r{k:03d}.pt"
        if k == 0 and args.shared_r000 is not None and not shard.exists():
            shard = args.shared_r000
        if not shard.exists():
            run([PY, SCRIPTS / "89_ihes_gen_harvest.py", "--checkpoint", prev, "--round", k,
                 "--n-pids", args.n_pids, "--beam", args.gen_beam,
                 "--rows-per-step", args.rows_per_step, "--full-expand", args.full_expand,
                 "--out", shard], f"round {k}: HARVEST with {Path(prev).name}", log)
        if not out.exists():
            run([PY, SCRIPTS / "90_ihes_train_avi.py", "--shard", shard, "--init", prev,
                 "--out", out, "--round", k, "--steps", args.steps, "--lr", args.lr,
                 "--gap-weight", args.gap_weight, "--gap-k-max", args.gap_k_max,
                 "--gap-batch", args.gap_batch, "--sparse-weight", args.sparse_weight,
                 "--dense-batch", args.dense_batch,
                 "--rehearse-weight", args.rehearse_weight,
                 "--rehearse-batch", args.rehearse_batch,
                 "--rehearse-k-max", args.rehearse_k_max, "--teacher", args.incumbent,
                 "--anchor-weight", args.anchor_weight],
                f"round {k}: TRAIN {Path(prev).name} -> {out.name}", log)
        level_ckpts.append(f"r{k:03d}={out}")
        r0_shard = args.level_shard or args.shared_r000 or (d / "shards" / "r000.pt")
        run([PY, SCRIPTS / "92_ihes_level_check.py", "--shard", r0_shard,
             *sum((["--ckpt", c] for c in level_ckpts), []),
             "--json", d / f"level_r{k:03d}.json"], f"round {k}: LEVEL CHECK", log)

        if args.no_gate:
            print(f"  AVI {d.name} round {k}: trained {out.name} (no in-loop gate)", flush=True)
            continue
        g_json = d / "gate" / f"r{k:03d}.json"
        if not g_json.exists():
            run([PY, SCRIPTS / "84_solve_tf.py", "--checkpoint", out, "--pid-file", args.gate_pids,
                 "--beam", args.gate_beam, "--sym-frames", 1, "--max-steps", 40, "--no-merge",
                 "--bf16", "--out", d / "gate" / f"r{k:03d}.csv", "--summary-json", g_json],
                f"round {k}: GATE {out.name}", log)
        g = json.loads(g_json.read_text(encoding="utf-8"))["per_pid"]
        cur = {p: g[str(p)]["len"] for p in gate_pids}
        if any(v is None for v in cur.values()):
            score = sum((v if v is not None else base_len[p] + 10) for p, v in cur.items())
        else:
            score = sum(cur.values())
        w = sum(1 for p in gate_pids if cur[p] is not None and cur[p] < base_len[p])
        l_ = sum(1 for p in gate_pids if cur[p] is None or cur[p] > base_len[p])
        row = {"round": k, "score": score, "delta": score - base_total, "W": w, "L": l_,
               "T": len(gate_pids) - w - l_, "p": sign_test(w, l_),
               "ties_floor": json.loads(g_json.read_text(encoding="utf-8"))["ties"]}
        trend = [t for t in trend if t["round"] != k] + [row]
        trend.sort(key=lambda t: t["round"])
        trend_path.write_text(json.dumps(trend, indent=1) + "\n", encoding="utf-8")
        # the incumbent is "round -1"; ties go to the EARLIER entry, so only a strictly
        # shorter gate total counts as the lengths still shortening
        cands = [{"round": -1, "score": base_total}] + trend
        best = min(cands, key=lambda t: (t["score"], t["round"]))
        who = "incumbent" if best["round"] < 0 else f"r{best['round']:03d}"
        print(f"  AVI {d.name} round {k}: gate {score} vs incumbent {base_total} "
              f"({row['delta']:+d}) W{w}/L{l_}/T{row['T']} p={row['p']:.3f} | best {who} "
              f"{best['score']}", flush=True)
        if k - best["round"] >= args.patience:
            print(f"  STOP: best ({who}) is {k - best['round']} rounds old -- lengths stopped "
                  f"shortening", flush=True)
            break
    print("AVI LOOP DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
