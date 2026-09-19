"""Stop the GCP VM as soon as its Beam-AVI loop ends, after copying what this machine needs.

User rule (2026-09-17): stop the A100 once the loop ends. The local gate watcher
(`95_ihes_avi_gate_watch.py`) still needs round checkpoints after that. So when the remote loop
prints "AVI LOOP DONE", or its process disappears (a crash), this script:
  1. copies every finished round checkpoint (`models/rNNN.pt` with a `level_rNNN.json`) and
     the loop's logs + level files into --local-dir;
  2. writes `<local-dir>/remote_done.json` (finished rounds, crashed flag), which switches the
     gate watcher to offline mode;
  3. runs `gcloud compute instances stop` and confirms the VM reached TERMINATED.
Shards stay on the VM disk; a crashed loop can be resumed after restarting the VM.

    python scripts/96_ihes_vm_reaper.py --remote and-l@1.2.3.4 \
        --remote-dir /home/and-l/cayley/runs/ihes_avi/A20 --local-dir runs/ihes_avi/A20 \
        --instance ihes-tf-b --zone us-central1-f
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SSH = r"C:\Program Files\Git\usr\bin\ssh.exe"
SCP = r"C:\Program Files\Git\usr\bin\scp.exe"


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy results and stop the VM when its loop ends.")
    ap.add_argument("--remote", required=True, help="user@host")
    ap.add_argument("--key", default=str(Path.home() / ".ssh" / "google_compute_engine"))
    ap.add_argument("--remote-dir", required=True)
    ap.add_argument("--local-dir", required=True, type=Path)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--zone", required=True)
    ap.add_argument("--poll", type=int, default=180)
    args = ap.parse_args()
    if not args.remote_dir.startswith("/") or ":" in args.remote_dir:
        # Git Bash (MSYS) rewrites /home/... into C:/Program Files/Git/home/... when it
        # launches a Windows exe; launch from cmd/.bat or set MSYS_NO_PATHCONV=1.
        raise SystemExit(f"--remote-dir must be a POSIX path on the VM, got "
                         f"{args.remote_dir!r}")

    local = args.local_dir if args.local_dir.is_absolute() else PROJECT / args.local_dir
    (local / "models").mkdir(parents=True, exist_ok=True)
    (local / "vm_logs").mkdir(parents=True, exist_ok=True)
    done_file = local / "remote_done.json"
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise SystemExit("gcloud not found on PATH")
    opts = ["-i", args.key, "-o", "StrictHostKeyChecking=no", "-o",
            "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=30"]
    rd = args.remote_dir
    print(f"reaper: watching {args.remote}:{rd}; will stop {args.instance} ({args.zone}) when "
          f"the loop ends", flush=True)

    fails = 0
    while True:
        cmd = (f"cd {rd} 2>/dev/null && ls level_r*.json 2>/dev/null; "
               f"echo DONE $(grep -c 'AVI LOOP DONE' {rd}/driver.log 2>/dev/null || echo 0); "
               f"echo PROCS $(ps -eo cmd | grep -c '[9]1_ihes_avi_loop')")
        try:
            r = subprocess.run([SSH] + opts + [args.remote, cmd], capture_output=True, text=True,
                               timeout=120)
            out = r.stdout
        except subprocess.TimeoutExpired:
            out = ""
        if "PROCS" not in out:
            fails += 1
            if fails in (3, 20):
                print(f"REAPER WARNING: VM unreachable for {fails} polls", flush=True)
            time.sleep(args.poll)
            continue
        fails = 0
        toks = out.split()
        rounds = sorted(int(x[len("level_r"):-len(".json")]) for x in toks
                        if x.startswith("level_r") and x.endswith(".json"))
        done = int(toks[toks.index("DONE") + 1]) > 0
        procs = int(toks[toks.index("PROCS") + 1])
        if not done and procs > 0:
            time.sleep(args.poll)
            continue
        crashed = not done
        print(f"reaper: loop {'CRASHED (no process, no DONE line)' if crashed else 'done'}; "
              f"finished rounds {rounds}", flush=True)
        for k in rounds:
            dst = local / "models" / f"r{k:03d}.pt"
            if not dst.exists():
                tmp = dst.with_suffix(".part")
                subprocess.run([SCP] + opts + [f"{args.remote}:{rd}/models/r{k:03d}.pt",
                                               str(tmp)], check=True, timeout=900)
                tmp.replace(dst)
        subprocess.run([SCP] + opts + [f"{args.remote}:{rd}/driver.log",
                                       f"{args.remote}:{rd}/loop.log",
                                       str(local / "vm_logs")], check=False, timeout=900)
        subprocess.run([SCP] + opts + [f"{args.remote}:{rd}/level_r*.json",
                                       str(local / "vm_logs")], check=False, timeout=900)
        missing = [k for k in rounds if not (local / "models" / f"r{k:03d}.pt").exists()]
        if missing:
            raise SystemExit(f"REAPER ERROR: checkpoints {missing} not copied; VM left running")
        done_file.write_text(json.dumps({"rounds": rounds, "crashed": crashed,
                                         "time": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1)
                             + "\n", encoding="utf-8")
        print(f"reaper: copied {len(rounds)} checkpoints + logs; wrote {done_file.name}; "
              f"stopping {args.instance}", flush=True)
        subprocess.run([gcloud, "compute", "instances", "stop", args.instance,
                        f"--zone={args.zone}", "--quiet"], check=False, timeout=900)
        st = subprocess.run([gcloud, "compute", "instances", "describe", args.instance,
                             f"--zone={args.zone}", "--format=value(status)"],
                            capture_output=True, text=True, timeout=300).stdout.strip()
        print(f"VM STOPPED: {args.instance} status {st}" if st == "TERMINATED" else
              f"REAPER WARNING: {args.instance} status {st!r} after stop", flush=True)
        return 0 if st == "TERMINATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
