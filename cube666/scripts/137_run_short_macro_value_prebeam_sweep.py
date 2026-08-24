"""Run a replay-verified value-prebeam sweep over a fixed short-macro gate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--direct-action-checkpoint", type=Path, action="append", required=True
    )
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--indices-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--weights", default="0.05,0.1,0.25,0.5")
    parser.add_argument("--beam", type=int, default=4096)
    parser.add_argument("--direct-root-branch", type=int, default=0)
    parser.add_argument("--value-prebeam-multiplier", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--wait-timeout-seconds", type=int, default=900)
    return parser.parse_args()


def persist(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    weights = [float(token) for token in args.weights.split(",") if token.strip()]
    if not weights:
        raise ValueError("weights must not be empty")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    summary: dict[str, object] = {
        "beam": args.beam,
        "path_cost_weight": 0.02,
        "policy_nll_weight": 1.0,
        "direct_root_branch": args.direct_root_branch,
        "value_prebeam_multiplier": args.value_prebeam_multiplier,
        "variants": [],
    }
    if args.wait_for is not None:
        wait_started = time.perf_counter()
        while not args.wait_for.exists():
            if time.perf_counter() - wait_started > args.wait_timeout_seconds:
                raise TimeoutError(f"timed out waiting for {args.wait_for}")
            time.sleep(5)
    for weight in weights:
        label = str(weight).replace(".", "p")
        report_path = args.out_dir / f"gate32_prebeam_value_w{label}.json"
        log_path = report_path.with_suffix(".log")
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            row = {
                "status": "existing",
                "value_weight": weight,
                "solved": int(report["solved"]),
                "replay_verified": int(
                    sum(bool(item["replay_verified"]) for item in report["rows"])
                ),
                "report": str(report_path),
            }
            summary["variants"].append(row)
            persist(summary_path, summary)
            continue
        command = [
            sys.executable,
            "-u",
            str(args.eval_script),
            "--factorized-value-checkpoint", str(args.critic),
        ]
        for checkpoint in args.direct_action_checkpoint:
            command.extend(("--direct-action-checkpoint", str(checkpoint)))
        if args.direct_root_branch:
            command.extend(("--direct-root-branch", str(args.direct_root_branch)))
        command.extend(
            (
                "--dataset-dir", str(args.dataset_dir),
                "--indices-file", str(args.indices_file),
                "--beam", str(args.beam),
                "--branch", "0",
                "--direct-branch", "512",
                "--direct-cost-stratified-branch", "128",
                "--direct-cost-maximum", "6",
                "--direct-cost-stratified-depths", "3",
                "--direct-score-mode", "log_probability",
                "--maximum-depth", "6",
                "--path-cost-weight", "0.02",
                "--value-weight", str(weight),
                "--value-prebeam-multiplier", str(args.value_prebeam_multiplier),
                "--policy-nll-weight", "1",
                "--exact-root-two-macro",
                "--one-macro-endgame",
                "--continue-after-solution",
                "--inference-batch-size", "16384",
                "--out", str(report_path),
            )
        )
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout_seconds,
                    check=False,
                )
            if completed.returncode:
                row = {
                    "status": "failed",
                    "returncode": completed.returncode,
                    "value_weight": weight,
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                    "log": str(log_path),
                }
            else:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                row = {
                    "status": "complete",
                    "value_weight": weight,
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                    "solved": int(report["solved"]),
                    "replay_verified": int(
                        sum(
                            bool(item["replay_verified"])
                            for item in report["rows"]
                        )
                    ),
                    "report": str(report_path),
                    "log": str(log_path),
                }
        except subprocess.TimeoutExpired:
            row = {
                "status": "timeout",
                "value_weight": weight,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "log": str(log_path),
            }
        summary["variants"].append(row)
        persist(summary_path, summary)
        print(json.dumps(row, sort_keys=True), flush=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
