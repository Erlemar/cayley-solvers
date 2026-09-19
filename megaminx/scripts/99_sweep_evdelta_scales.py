"""Run a hard4 sweep over pre-scaled eviction-delta checkpoints."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VENVPY = ROOT / ".venv" / "Scripts" / "python.exe"
PY = VENVPY if VENVPY.exists() else Path(sys.executable)
MODELS = ROOT / "megaminx" / "models"
SUBMISSIONS = ROOT / "megaminx" / "submissions"

HARD4 = "500,600,800,900"


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(" ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        f.write(f"\n[exit] {rc}\n")
    if rc != 0:
        raise SystemExit(f"command failed with exit code {rc}; see {log_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scales",
        default="025,01,005",
        help="Scale tags to evaluate, matching model dir suffixes s<tag>.",
    )
    ap.add_argument("--pids", default=HARD4)
    ap.add_argument("--beam", default="16384")
    ap.add_argument("--max-steps", default="120")
    ap.add_argument(
        "--label",
        default="hard4_b16k",
        help="Suffix used for output CSV/log names.",
    )
    ap.add_argument(
        "--model-prefix",
        default="m_az15_ep0004_evdelta_top100_b1024",
        help="Model directory prefix. Scale dirs are <prefix>_s<tag>.",
    )
    args = ap.parse_args()

    for tag in [x.strip() for x in args.scales.split(",") if x.strip()]:
        ckpt = MODELS / f"{args.model_prefix}_s{tag}" / "epoch_0023.pt"
        if not ckpt.exists():
            raise SystemExit(f"missing checkpoint: {ckpt}")
        out_csv = SUBMISSIONS / f"{args.model_prefix}_s{tag}_{args.label}.csv"
        log_path = MODELS / f"{args.model_prefix}_s{tag}_{args.label}.log"
        print(f"[run] s{tag} -> {log_path.relative_to(ROOT)}", flush=True)
        run(
            [
                str(PY),
                "-u",
                str(ROOT / "megaminx" / "scripts" / "03_solve.py"),
                "--checkpoint",
                str(ckpt),
                "--out",
                str(out_csv),
                "--pids",
                args.pids,
                "--beams",
                args.beam,
                "--max-steps",
                args.max_steps,
                "--bf16",
            ],
            log_path,
        )
    print("[done] scale sweep complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
