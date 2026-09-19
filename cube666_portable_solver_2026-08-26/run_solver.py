"""Portable dispatcher for the neural, KMC and merge/verification entry points."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "neural": ROOT / "solver" / "solve_neural.py",
    "kmc": ROOT / "cube666" / "scripts" / "06_run_kmcoders_beam.py",
    "merge": ROOT / "cube666" / "scripts" / "07_merge_verify.py",
    "smoke": ROOT / "smoke_test.py",
    "verify": ROOT / "verify_manifest.py",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(COMMANDS)
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{choices}> [arguments]")
    script = COMMANDS[sys.argv[1]]
    completed = subprocess.run([sys.executable, str(script), *sys.argv[2:]], cwd=ROOT)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

