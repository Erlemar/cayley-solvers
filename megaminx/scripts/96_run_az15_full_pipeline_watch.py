"""Watch and continue the 15M full AZ pipeline after pretrain finishes.

This script assumes stage 1 was already launched:

    .venv/Scripts/python.exe -u megaminx/scripts/02_train.py \
        --config megaminx/configs/m_az15_full_pretrain.yaml \
        --output megaminx/models/m_az15_full_pretrain

It waits for the final pretrain checkpoint, runs calibration probes, then runs
Bellman refinement and AZ fine-tuning with the AZ-v4 large-batch recipe. Logs are
written under megaminx/models/ so the run can be audited after a long session.
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
SUBMISSIONS = ROOT / "megaminx" / "submissions"

PRETRAIN_FINAL = MODELS / "m_az15_full_pretrain" / "epoch_3999.pt"
BELL_DIR = MODELS / "m_az15_full_bellman"
BELL_FINAL = BELL_DIR / "epoch_0299.pt"
AZ_DIR = MODELS / "m_az15_full_az73614"
AZ_V_ONLY = MODELS / "m_az15_full_az73614_v_only.pt"

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


def wait_for(path: Path, poll_seconds: int) -> None:
    log(f"[wait] {path.relative_to(ROOT)}")
    while not path.exists():
        time.sleep(poll_seconds)
    log(f"[found] {path.relative_to(ROOT)}")


def py_cmd(script: str, *args: str) -> list[str]:
    return [str(PY), "-u", str(ROOT / "megaminx" / "scripts" / script), *args]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    if not PY.exists():
        raise SystemExit(f"Missing venv python: {PY}")

    wait_for(PRETRAIN_FINAL, args.poll_seconds)

    run_step(
        "pretrain v_canary",
        py_cmd("v_canary.py", "--checkpoint", str(PRETRAIN_FINAL)),
        MODELS / "m_az15_full_pretrain_v_canary.log",
    )
    run_step(
        "pretrain near-solved calibration",
        py_cmd("61_eval_v_at_solved.py", "--checkpoints", str(PRETRAIN_FINAL)),
        MODELS / "m_az15_full_pretrain_eval_v_at_solved.log",
    )

    if not BELL_FINAL.exists():
        run_step(
            "15M Bellman refine",
            py_cmd(
                "05_bellman_refine.py",
                "--config", "megaminx/configs/m_az15_full_bellman.yaml",
                "--output", "megaminx/models/m_az15_full_bellman",
            ),
            MODELS / "m_az15_full_bellman_training.log",
        )
    else:
        log(f"[skip] Bellman final exists: {BELL_FINAL.relative_to(ROOT)}")

    run_step(
        "Bellman v_canary",
        py_cmd("v_canary.py", "--checkpoint", str(BELL_FINAL)),
        MODELS / "m_az15_full_bellman_v_canary.log",
    )
    run_step(
        "Bellman near-solved calibration",
        py_cmd("61_eval_v_at_solved.py", "--checkpoints", str(BELL_FINAL)),
        MODELS / "m_az15_full_bellman_eval_v_at_solved.log",
    )
    if not args.skip_bench:
        run_step(
            "Bellman 10-pid beam16k bench",
            py_cmd(
                "03_solve.py",
                "--checkpoint", str(BELL_FINAL),
                "--out", str(SUBMISSIONS / "m_az15_full_bellman_10pid_b16k.csv"),
                "--pids", BENCH_PIDS,
                "--beams", "16384",
                "--max-steps", "120",
                "--bf16",
            ),
            MODELS / "m_az15_full_bellman_10pid_b16k.log",
        )

    az_final = AZ_DIR / "epoch_0034.pt"
    if not az_final.exists():
        run_step(
            "15M AZ fine-tune on 73614 paths",
            py_cmd(
                "71_train_az_v3.py",
                "--output", str(AZ_DIR),
                "--warmstart-trunk", str(BELL_FINAL),
                "--policy-dataset", "megaminx/data/az_dataset_73614.pt",
                "--bfs-d6-path", "megaminx/data/bfs_d6_train.pt",
                "--epochs", "35",
                "--samples-per-epoch", "500000",
                "--rw-batch-size", "8192",
                "--policy-batch-size", "1024",
                "--anchor-v0", "32",
                "--anchor-d1", "4",
                "--bfs-d6-fraction", "0.10",
                "--k-max", "80",
                "--n-back", "1",
                "--lr", "2.0e-4",
                "--alpha", "1.0",
                "--beta", "1.0",
                "--target-update-every-epochs", "10",
                "--hidden-dims", "2048,1024",
                "--num-res-blocks", "4",
                "--checkpoint-every", "5",
                "--seed", "116",
            ),
            MODELS / "m_az15_full_az73614_training.log",
        )
    else:
        log(f"[skip] AZ final exists: {az_final.relative_to(ROOT)}")

    az_ckpts = [str(p) for p in sorted(AZ_DIR.glob("epoch_*.pt"))]
    if az_ckpts and not args.skip_bench:
        run_step(
            "AZ checkpoint sweep 10-pid beam16k",
            py_cmd(
                "80_bench_az_v_head.py",
                "--checkpoints", *az_ckpts,
                "--pids", BENCH_PIDS,
                "--beam", "16384",
                "--max-steps", "120",
            ),
            MODELS / "m_az15_full_az73614_10pid_b16k_sweep.log",
        )

    if az_final.exists():
        run_step(
            "export final AZ V-only",
            py_cmd(
                "95_export_az_v_only.py",
                "--az-checkpoint", str(az_final),
                "--out", str(AZ_V_ONLY),
            ),
            MODELS / "m_az15_full_az73614_export.log",
        )
        run_step(
            "probe final AZ values",
            py_cmd("92_probe_az_values.py", "--checkpoint", str(az_final)),
            MODELS / "m_az15_full_az73614_probe_values.log",
        )

    log("[done] 15M full-pipeline watcher completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
