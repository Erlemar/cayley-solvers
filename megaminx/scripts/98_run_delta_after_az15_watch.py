"""Run the function-preserving delta canary after the 15M AZ pipeline finishes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
MODELS = ROOT / "megaminx" / "models"
SUBMISSIONS = ROOT / "megaminx" / "submissions"

AZ15_EXPORT = MODELS / "m_az15_full_az73614_v_only.pt"
BASE_V = MODELS / "m_az_v5_73614_orbit_v_only.pt"
DELTA_DIR = MODELS / "m_az_v5_delta15_mistakes"
DELTA_FINAL = DELTA_DIR / "epoch_0039.pt"
BENCH_PIDS = "0,100,200,300,400,500,600,700,800,900"


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
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--settle-seconds", type=int, default=300)
    args = ap.parse_args()

    log(f"[wait] {AZ15_EXPORT.relative_to(ROOT)}")
    while not AZ15_EXPORT.exists():
        time.sleep(args.poll_seconds)
    log(f"[found] {AZ15_EXPORT.relative_to(ROOT)}")
    if args.settle_seconds > 0:
        log(f"[settle] sleeping {args.settle_seconds}s before delta training")
        time.sleep(args.settle_seconds)

    if not BASE_V.exists():
        raise SystemExit(f"Missing base V checkpoint: {BASE_V}")

    if not DELTA_FINAL.exists():
        run_step(
            "AZ-v5 function-preserving delta-on-mistakes",
            py_cmd(
                "97_train_delta_v.py",
                "--base-checkpoint", str(BASE_V),
                "--output", str(DELTA_DIR),
                "--path-dataset", "megaminx/data/az_dataset_73614.pt",
                "--stag-anchor-path", "megaminx/data/stagnation_anchors_v0.pt",
                "--hidden-dims", "2048,1024",
                "--num-res-blocks", "4",
                "--epochs", "40",
                "--steps-per-epoch", "96",
                "--path-batch-size", "1024",
                "--exact-batch-size", "512",
                "--lb-batch-size", "1024",
                "--preserve-batch-size", "2048",
                "--lr", "2.0e-4",
                "--lambda-path", "0.10",
                "--lambda-exact", "1.0",
                "--lambda-lb", "1.0",
                "--lambda-preserve", "0.03",
                "--checkpoint-every", "10",
                "--seed", "117",
            ),
            MODELS / "m_az_v5_delta15_mistakes_training.log",
        )
    else:
        log(f"[skip] delta final exists: {DELTA_FINAL.relative_to(ROOT)}")

    run_step(
        "delta v_canary",
        py_cmd("v_canary.py", "--checkpoint", str(DELTA_FINAL)),
        MODELS / "m_az_v5_delta15_mistakes_v_canary.log",
    )
    run_step(
        "delta near-solved calibration",
        py_cmd("61_eval_v_at_solved.py", "--checkpoints", str(DELTA_FINAL)),
        MODELS / "m_az_v5_delta15_mistakes_eval_v_at_solved.log",
    )
    run_step(
        "delta 10-pid beam16k bench",
        py_cmd(
            "03_solve.py",
            "--checkpoint", str(DELTA_FINAL),
            "--out", str(SUBMISSIONS / "m_az_v5_delta15_mistakes_10pid_b16k.csv"),
            "--pids", BENCH_PIDS,
            "--beams", "16384",
            "--max-steps", "120",
            "--bf16",
        ),
        MODELS / "m_az_v5_delta15_mistakes_10pid_b16k.log",
    )
    log("[done] delta watcher completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
