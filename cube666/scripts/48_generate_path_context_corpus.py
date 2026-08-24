"""Generate a resumable, PID-disjoint KMC rough-word completion corpus."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


CONDITIONS = (
    {
        "label": "seed50002",
        "tag": "pcorpus32_seed50002_n20m_a3",
        "anneal_seed": 50002,
        "alpha": 3.0,
        "anneal_keep_best": False,
    },
    {
        "label": "seed50003",
        "tag": "pcorpus32_seed50003_n20m_a3",
        "anneal_seed": 50003,
        "alpha": 3.0,
        "anneal_keep_best": False,
    },
    {
        "label": "alpha2p5",
        "tag": "pcorpus32_seed50001_n20m_a2p5",
        "anneal_seed": 50001,
        "alpha": 2.5,
        "anneal_keep_best": False,
    },
    {
        "label": "alpha3p5",
        "tag": "pcorpus32_seed50001_n20m_a3p5",
        "anneal_seed": 50001,
        "alpha": 3.5,
        "anneal_keep_best": False,
    },
    {
        "label": "keepbest",
        "tag": "pcorpus32_seed50001_n20m_a3_keepbest",
        "anneal_seed": 50001,
        "alpha": 3.0,
        "anneal_keep_best": True,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incumbent",
        type=Path,
        default=PROJECT / "submissions/cube666_model_hybrid_strictwin_v1.csv",
    )
    parser.add_argument(
        "--exclude-report",
        type=Path,
        default=PROJECT / "cube666/reports/KMC_PARAM_GATE16.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "cube666/reports/PATH_CONTEXT_CORPUS32_MANIFEST.json",
    )
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--beam", type=int, default=4000)
    parser.add_argument("--candidate-cap", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rayon-threads", type=int, default=1)
    parser.add_argument("--anneal-steps", type=int, default=20_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--conditions", help="optional comma-separated labels")
    return parser.parse_args()


def parse_path(text: str) -> tuple[str, ...]:
    return tuple(token for token in text.strip().split(".") if token)


def load_lengths(path: Path) -> dict[int, int]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["initial_state_id"]): len(parse_path(row["path"]))
            for row in csv.DictReader(handle)
        }


def stratified_pids(
    lengths: dict[int, int], excluded: set[int], count: int
) -> list[int]:
    if count <= 0 or count % 4:
        raise ValueError("count must be positive and divisible by four")
    candidates = sorted(
        (
            (length, pid)
            for pid, length in lengths.items()
            if 210 <= pid <= 1011 and pid not in excluded
        )
    )
    per_quartile = count // 4
    selected: list[int] = []
    for quartile in range(4):
        start = quartile * len(candidates) // 4
        stop = (quartile + 1) * len(candidates) // 4
        bucket = candidates[start:stop]
        for sample in range(per_quartile):
            index = ((2 * sample + 1) * len(bucket)) // (2 * per_quartile)
            index = min(index, len(bucket) - 1)
            selected.append(bucket[index][1])
    if len(set(selected)) != count:
        raise AssertionError("stratified PID selection produced duplicates")
    return sorted(selected)


def atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    excluded = set(
        int(pid)
        for pid in json.loads(args.exclude_report.read_text(encoding="utf-8"))["pids"]
    )
    lengths = load_lengths(args.incumbent)
    pids = stratified_pids(lengths, excluded, args.count)
    requested = None
    if args.conditions:
        requested = {value.strip() for value in args.conditions.split(",") if value.strip()}
    conditions = [
        condition
        for condition in CONDITIONS
        if requested is None or condition["label"] in requested
    ]
    if not conditions:
        raise ValueError("condition selection is empty")

    runner = PROJECT / "cube666/scripts/06_run_kmcoders_beam.py"
    started = time.perf_counter()
    manifest_conditions = [
        {
            "label": "control",
            "directory": str(
                PROJECT / "cube666/results/dfr_20m_b20k_full_a3_b20000"
            ),
        }
    ]
    for index, condition in enumerate(conditions, start=1):
        command = [
            sys.executable,
            str(runner),
            "--indices",
            ",".join(str(pid) for pid in pids),
            "--workers",
            str(args.workers),
            "--beam",
            str(args.beam),
            "--candidate-cap",
            str(args.candidate_cap),
            "--rayon-threads",
            str(args.rayon_threads),
            "--anneal-seed",
            str(condition["anneal_seed"]),
            "--anneal-steps",
            str(args.anneal_steps),
            "--alpha",
            str(condition["alpha"]),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--tag",
            str(condition["tag"]),
        ]
        if condition["anneal_keep_best"]:
            command.append("--anneal-keep-best")
        print(
            f"condition={index}/{len(conditions)} label={condition['label']} "
            f"pids={len(pids)} beam={args.beam}",
            flush=True,
        )
        subprocess.run(command, cwd=PROJECT, check=True)
        manifest_conditions.append(
            {
                "label": condition["label"],
                "directory": str(
                    PROJECT
                    / "cube666/results"
                    / f"{condition['tag']}_b{args.beam}"
                ),
            }
        )
        atomic_write(
            args.manifest,
            {
                "beam": args.beam,
                "candidate_cap": args.candidate_cap,
                "conditions": manifest_conditions,
                "control": "control",
                "elapsed_seconds": time.perf_counter() - started,
                "incumbent": str(args.incumbent),
                "pids": pids,
                "selected_length_range": [
                    min(lengths[pid] for pid in pids),
                    max(lengths[pid] for pid in pids),
                ],
            },
        )

    print(args.manifest.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
