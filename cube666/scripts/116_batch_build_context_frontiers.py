"""Run the exact-corner macro frontier builder for many PIDs in one process.

This is a Windows-safe campaign wrapper around script 108.  It avoids the venv
launcher failure seen when PowerShell invokes the Python launcher inside a loop,
while preserving one durable per-PID artifact directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--value-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument(
        "--pids", required=True, help="underscore- or comma-separated puzzle IDs"
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--shortlist", type=int, default=4096)
    parser.add_argument("--value-candidates", type=int, default=8)
    parser.add_argument("--policy-candidates", type=int, default=4)
    parser.add_argument("--random-candidates", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260824)
    parser.add_argument("--random-max-cost", type=int, default=14)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pids = [
        int(token)
        for token in args.pids.replace("_", ",").split(",")
        if token.strip()
    ]
    if not pids or len(set(pids)) != len(pids):
        raise ValueError("pids must be non-empty and unique")

    source = Path(__file__).with_name("108_build_root_kmc_rollout_frontiers.py")
    spec = importlib.util.spec_from_file_location("context_frontier_builder_108", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original_argv = sys.argv
    completed = 0
    reused = 0
    try:
        for pid in pids:
            out_dir = args.out_root / f"macro_kmc_root_pid{pid:03d}_v1"
            if (
                not args.force
                and (out_dir / "report.json").is_file()
                and (out_dir / "frontiers.npz").is_file()
            ):
                print(f"pid={pid} reused={out_dir}", flush=True)
                reused += 1
                continue
            sys.argv = [
                str(source),
                "--checkpoint",
                str(args.checkpoint),
                "--value-checkpoint",
                str(args.value_checkpoint),
                "--teacher-dir",
                str(args.teacher_dir),
                "--pid",
                str(pid),
                "--shortlist",
                str(args.shortlist),
                "--value-candidates",
                str(args.value_candidates),
                "--policy-candidates",
                str(args.policy_candidates),
                "--random-candidates",
                str(args.random_candidates),
                "--random-seed",
                str(args.random_seed),
                "--random-max-cost",
                str(args.random_max_cost),
                "--skip-parity-repair",
                "--out-dir",
                str(out_dir),
            ]
            module.main()
            completed += 1
    finally:
        sys.argv = original_argv
    print(
        f"campaign_pids={len(pids)} completed={completed} reused={reused} "
        f"out_root={args.out_root}"
    )


if __name__ == "__main__":
    main()
