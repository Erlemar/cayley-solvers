"""Wait for AZ15 cross-width eviction labels, then train/evaluate a delta V.

This is the first concrete run for Direction 1A:

    score(s) = V_az15_ep4(s) + Delta(s)

The base V is frozen.  The delta is trained mostly by an eviction-ranking hinge:
teacher good-child states should score below the near-cutoff states that survived
instead.  We intentionally keep path-MSE and lower-bound pressure off for this
first run, because those objectives already regressed in the AZ15 follow-up.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
MODELS = ROOT / "megaminx" / "models"
DATA = ROOT / "megaminx" / "data"
SUBMISSIONS = ROOT / "megaminx" / "submissions"

BASE = MODELS / "m_az15_full_az73614_ep0004_v_only.pt"
EVICTIONS = DATA / "cross_width_evictions_az15_ep4_top100_b1024.pt"
DELTA_DIR = MODELS / "m_az15_ep0004_evdelta_top100_b1024"
DELTA_FINAL = DELTA_DIR / "epoch_0023.pt"

HARD4 = "500,600,800,900"
TEN_PID = "0,100,200,300,400,500,600,700,800,900"


def log(msg: str) -> None:
    print(msg, flush=True)


def run_step(name: str, cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[run] {name}")
    log(f"[run] log: {log_path.relative_to(ROOT)}")
    with log_path.open("a", encoding="utf-8", newline="") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"{name}\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"\n[exit] {rc}\n")
    if rc != 0:
        raise SystemExit(f"{name} failed with exit code {rc}; see {log_path}")


def py_cmd(script: str, *args: str) -> list[str]:
    return [str(PY), "-u", str(ROOT / "megaminx" / "scripts" / script), *args]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--settle-seconds", type=int, default=30)
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    if not BASE.exists():
        raise SystemExit(f"missing base checkpoint: {BASE}")

    log(f"[wait] {EVICTIONS.relative_to(ROOT)}")
    while not EVICTIONS.exists():
        time.sleep(args.poll_seconds)
    if args.settle_seconds > 0:
        log(f"[settle] sleeping {args.settle_seconds}s")
        time.sleep(args.settle_seconds)

    if args.force_train or not DELTA_FINAL.exists():
        run_step(
            "AZ15 ep4 eviction-ranking delta",
            py_cmd(
                "97_train_delta_v.py",
                "--base-checkpoint", str(BASE),
                "--output", str(DELTA_DIR),
                "--path-dataset", "megaminx/data/az_dataset_73614.pt",
                "--stag-anchor-path", "megaminx/data/stagnation_anchors_v0.pt",
                "--eviction-path", str(EVICTIONS),
                "--hidden-dims", "1024,256",
                "--num-res-blocks", "1",
                "--epochs", "24",
                "--steps-per-epoch", "80",
                "--path-batch-size", "0",
                "--exact-batch-size", "256",
                "--lb-batch-size", "0",
                "--eviction-batch-size", "64",
                "--preserve-batch-size", "4096",
                "--preserve-k-max", "80",
                "--lr", "1.0e-4",
                "--lambda-path", "0.0",
                "--lambda-exact", "0.05",
                "--lambda-lb", "0.0",
                "--lambda-eviction", "2.0",
                "--lambda-preserve", "0.10",
                "--eviction-margin", "0.50",
                "--checkpoint-every", "4",
                "--seed", "211",
            ),
            MODELS / "m_az15_ep0004_evdelta_top100_b1024_training.log",
        )
    else:
        log(f"[skip] final checkpoint exists: {DELTA_FINAL.relative_to(ROOT)}")

    if args.skip_eval:
        return 0

    run_step(
        "evdelta v_canary",
        py_cmd("v_canary.py", "--checkpoint", str(DELTA_FINAL)),
        MODELS / "m_az15_ep0004_evdelta_top100_b1024_v_canary.log",
    )
    run_step(
        "evdelta near-solved calibration",
        py_cmd("61_eval_v_at_solved.py", "--checkpoints", str(DELTA_FINAL)),
        MODELS / "m_az15_ep0004_evdelta_top100_b1024_eval_v_at_solved.log",
    )
    run_step(
        "evdelta hard4 beam16k",
        py_cmd(
            "03_solve.py",
            "--checkpoint", str(DELTA_FINAL),
            "--out", str(SUBMISSIONS / "m_az15_ep0004_evdelta_top100_b1024_hard4_b16k.csv"),
            "--pids", HARD4,
            "--beams", "16384",
            "--max-steps", "120",
            "--bf16",
        ),
        MODELS / "m_az15_ep0004_evdelta_top100_b1024_hard4_b16k.log",
    )
    run_step(
        "evdelta 10pid beam16k",
        py_cmd(
            "03_solve.py",
            "--checkpoint", str(DELTA_FINAL),
            "--out", str(SUBMISSIONS / "m_az15_ep0004_evdelta_top100_b1024_10pid_b16k.csv"),
            "--pids", TEN_PID,
            "--beams", "16384",
            "--max-steps", "120",
            "--bf16",
        ),
        MODELS / "m_az15_ep0004_evdelta_top100_b1024_10pid_b16k.log",
    )
    log("[done] eviction delta watcher completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
