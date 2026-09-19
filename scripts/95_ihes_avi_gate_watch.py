"""Gate Beam-AVI round checkpoints on THIS machine while the loop runs on a remote VM.

The VM runs `91_ihes_avi_loop.py --no-gate` (generate + train only). This watcher copies each
finished round checkpoint and scores it with `84_solve_tf.py` at the deployment-like width,
matched against the incumbent's summary from THIS machine at the same settings (rule 28).
Path length is the only verdict; each gate appends one row to `<local-dir>/gate<tag>/trend.json`.

The loop's stdout must go to `<remote-dir>/driver.log` (its "AVI LOOP DONE" line ends the
watch). Start this watcher only AFTER the loop is running: no loop process and no DONE line
reads as a finished (or crashed) loop.

Offline mode: once `<local-dir>/remote_done.json` exists (written by `96_ihes_vm_reaper.py`
after it copied every finished checkpoint and stopped the VM), no more ssh is used.

A round rNNN counts as finished once the VM has written `level_rNNN.json` (the loop runs the
level check right after training), so a half-written checkpoint is never copied.

Order: while the loop runs, gate the NEWEST finished, ungated round (a 2^20 gate is slower
than a round, so some rounds are skipped). After the loop is done, gate every `--must` round
still missing, then exit with "GATE WATCH DONE".

    python scripts/95_ihes_avi_gate_watch.py --remote and-l@1.2.3.4 \
        --remote-dir /home/and-l/cayley/runs/ihes_avi/A20 --local-dir runs/ihes_avi/A20 \
        --baseline-json submissions/eval/finalists_4090/BAS_g54_b20.json --must 4 9
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
SSH = r"C:\Program Files\Git\usr\bin\ssh.exe"
SCP = r"C:\Program Files\Git\usr\bin\scp.exe"


def sign_test(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate remote Beam-AVI checkpoints locally.")
    ap.add_argument("--remote", required=True, help="user@host")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "google_compute_engine"))
    ap.add_argument("--remote-dir", required=True, help="the arm dir on the VM (absolute)")
    ap.add_argument("--local-dir", required=True, type=Path)
    ap.add_argument("--baseline-json", required=True, type=Path,
                    help="84_solve_tf summary of the incumbent on THIS machine, same settings")
    ap.add_argument("--gate-pids", type=Path, default=PROJECT / "data" / "ihes_gate54.json")
    ap.add_argument("--beam", type=int, default=1048576)
    ap.add_argument("--tag", default="20", help="gate dir suffix: <local-dir>/gate<tag>/")
    ap.add_argument("--must", type=int, nargs="*", default=[],
                    help="rounds to gate once the loop is done, even if skipped while running")
    ap.add_argument("--poll", type=int, default=300)
    args = ap.parse_args()
    if not args.remote_dir.startswith("/") or ":" in args.remote_dir:
        # Git Bash (MSYS) rewrites /home/... into C:/Program Files/Git/home/... when it
        # launches a Windows exe; launch from cmd/.bat or set MSYS_NO_PATHCONV=1.
        raise SystemExit(f"--remote-dir must be a POSIX path on the VM, got "
                         f"{args.remote_dir!r}")

    local = args.local_dir if args.local_dir.is_absolute() else PROJECT / args.local_dir
    models = local / "models"
    gdir = local / f"gate{args.tag}"
    models.mkdir(parents=True, exist_ok=True)
    gdir.mkdir(parents=True, exist_ok=True)
    trend_path = gdir / "trend.json"
    pids = [int(p) for p in json.loads(args.gate_pids.read_text(encoding="utf-8"))["pids"]]
    base = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    base_len = {p: base["per_pid"][str(p)]["len"] for p in pids}
    base_total = sum(base_len.values())
    print(f"gate watch: {args.remote}:{args.remote_dir} -> {local}; B={args.beam:,}; incumbent "
          f"{base_total} on {len(pids)} pids ({args.baseline_json.name}, config "
          f"{base.get('config')}); must {args.must}", flush=True)
    ssh_base = [SSH, "-i", args.key, "-o", "StrictHostKeyChecking=no", "-o",
                "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=30"]

    done_file = local / "remote_done.json"

    def remote_state():
        if done_file.exists():     # 96_ihes_vm_reaper copied everything and stopped the VM
            info = json.loads(done_file.read_text(encoding="utf-8"))
            return sorted(int(k) for k in info["rounds"]), True
        cmd = (f"cd {args.remote_dir} 2>/dev/null && ls level_r*.json 2>/dev/null; "
               f"grep -c 'AVI LOOP DONE' {args.remote_dir}/driver.log 2>/dev/null; "
               f"echo PROCS $(ps -eo cmd | grep -c '[9]1_ihes_avi_loop')")
        r = subprocess.run(ssh_base + [args.remote, cmd], capture_output=True, text=True,
                           timeout=120)
        if r.returncode != 0 and "PROCS" not in r.stdout:
            return None
        done_rounds = sorted(int(x[len("level_r"):-len(".json")]) for x in r.stdout.split()
                             if x.startswith("level_r") and x.endswith(".json"))
        lines = r.stdout.strip().splitlines()
        procs = int(lines[-1].split()[1]) if lines and lines[-1].startswith("PROCS") else -1
        finished = any(ln.strip().isdigit() and int(ln) > 0 for ln in lines[:-1]
                       if not ln.startswith("level_r"))
        return done_rounds, finished or procs == 0

    def fetch(k: int) -> Path:
        dst = models / f"r{k:03d}.pt"
        if not dst.exists():
            tmp = dst.with_suffix(".part")
            src = f"{args.remote}:{args.remote_dir}/models/r{k:03d}.pt"
            subprocess.run([SCP] + ssh_base[1:] + [src, str(tmp)], check=True, timeout=600)
            tmp.replace(dst)
        return dst

    def gate(k: int) -> None:
        ck = fetch(k)
        stem = gdir / f"r{k:03d}"
        cmd = [sys.executable, "-u", str(PROJECT / "scripts" / "84_solve_tf.py"),
               "--checkpoint", str(ck), "--pid-file", str(args.gate_pids),
               "--beam", str(args.beam), "--sym-frames", "1", "--max-steps", "40",
               "--no-merge", "--bf16", "--out", str(stem.with_suffix(".csv")),
               "--summary-json", str(stem.with_suffix(".json"))]
        print(f"GATE{args.tag} r{k:03d}: start", flush=True)
        t = time.time()
        with open(stem.with_suffix(".log"), "w", encoding="utf-8") as fh:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=PROJECT).returncode
        if rc != 0:
            print(f"GATE{args.tag} r{k:03d}: FAILED exit {rc} (see {stem}.log)", flush=True)
            raise SystemExit(rc)
        g = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        cur = {p: g["per_pid"][str(p)]["len"] for p in pids}
        score = sum((v if v is not None else base_len[p] + 10) for p, v in cur.items())
        w = sum(1 for p in pids if cur[p] is not None and cur[p] < base_len[p])
        l_ = sum(1 for p in pids if cur[p] is None or cur[p] > base_len[p])
        row = {"round": k, "score": score, "delta": score - base_total, "W": w, "L": l_,
               "T": len(pids) - w - l_, "p": sign_test(w, l_), "ties_floor": g.get("ties"),
               "wins_floor": g.get("wins"), "config": g.get("config"), "wall_s": g.get("wall_s")}
        trend = json.loads(trend_path.read_text(encoding="utf-8")) if trend_path.exists() else []
        trend = sorted([x for x in trend if x["round"] != k] + [row], key=lambda x: x["round"])
        trend_path.write_text(json.dumps(trend, indent=1) + "\n", encoding="utf-8")
        best = min([{"round": -1, "score": base_total}] + trend,
                   key=lambda x: (x["score"], x["round"]))
        print(f"GATE{args.tag} r{k:03d}: total {score} vs incumbent {base_total} "
              f"({score - base_total:+d}) W{w}/L{l_}/T{row['T']} p={row['p']:.3f} "
              f"floor-wins {g.get('wins')} | best so far "
              f"{'incumbent' if best['round'] < 0 else 'r%03d' % best['round']} {best['score']} "
              f"| {time.time() - t:.0f}s", flush=True)

    fails = 0
    while True:
        try:
            st = remote_state()
        except subprocess.TimeoutExpired:
            st = None
        if st is None:
            fails += 1
            if fails in (3, 12):
                print(f"GATE WATCH WARNING: VM unreachable {fails} polls in a row", flush=True)
            time.sleep(args.poll)
            continue
        fails = 0
        ready, loop_done = st
        gated = {int(x.stem[1:]) for x in gdir.glob("r[0-9][0-9][0-9].json")}
        todo = [k for k in ready if k not in gated]
        pick = None
        if todo and not loop_done:
            pick = max(todo)
        elif loop_done:
            last = max(ready) if ready else None
            wanted = sorted({k for k in args.must if k in ready} | ({last} if last is not None
                                                                      else set()))
            left = [k for k in wanted if k not in gated]
            if not left:
                print("GATE WATCH DONE", flush=True)
                return 0
            pick = left[-1]
        if pick is None:
            time.sleep(args.poll)
            continue
        gate(pick)


if __name__ == "__main__":
    raise SystemExit(main())
