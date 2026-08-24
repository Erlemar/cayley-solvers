"""Generate balanced, resumable Kaggle CPU kernels for the KMC frame campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_INCUMBENT = PROJECT / "submissions" / "cube666_classical_paramgate16_merged.csv"
DATASET_SOURCE = "artgor/cayley-666-kmc-campaign-v1"
EXPECTED_HASHES = {
    "cayley-py-666-cube/puzzle_info.json": "f9d435ba6345f52d68ae52659cbe3acb5f4d5d055e42bd3d55410661c246cc31",
    "cayley-py-666-cube/test.csv": "19a4a192fae217dfb693364512fb7d49cd598acd173c2f2434031cda992887fd",
    "cube666/scripts/06_run_kmcoders_beam.py": "457bb8d627612ae86183d9311263b3003a089aefc66856bcec3b4d950d29d306",
    "external/third_party/kmcoders_santa2023/solution/src/bin/solve_cube_beam.rs": "512c1346da789f693f77bfb4e2aa1c3d8e7cce6d8d71c06dba1ed80ec3896202",
}
BINARY_RELATIVE = "external/third_party/kmcoders_santa2023/solution/target/release/solve_cube_beam"


RUNTIME_TEMPLATE = r'''"""Generated KMC frame-campaign shard for Kaggle CPU."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import traceback

WAVE = __WAVE__
SLOT = __SLOT__
PIDS = __PIDS__
FRAME_SPECS = __FRAME_SPECS__
EXPECTED_HASHES = __EXPECTED_HASHES__
EXPECTED_BINARY_SHA256 = __EXPECTED_BINARY_SHA256__
BINARY_RELATIVE = "external/third_party/kmcoders_santa2023/solution/target/release/solve_cube_beam"
MAX_WALL_SECONDS = 10.5 * 3600
WORK = pathlib.Path("/kaggle/working")
RUN_ROOT = WORK / f"kmc_campaign_w{WAVE}_s{SLOT:02d}"
PROJECT = RUN_ROOT / "cayley"
OUTPUT_ROOT = WORK / "kmc_results"
SUMMARY_PATH = WORK / f"kmc_summary_w{WAVE}_s{SLOT:02d}.json"
ARCHIVE_BASE = WORK / f"kmc_results_w{WAVE}_s{SLOT:02d}"


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_source_project():
    matches = sorted(
        pathlib.Path("/kaggle/input").glob("**/cayley/cayley-py-666-cube/puzzle_info.json")
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one campaign project, found {matches}")
    return matches[0].parents[1]


def run(command, cwd):
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sync_result(tag):
    source = PROJECT / "cube666" / "results" / f"{tag}_b20000"
    if not source.exists():
        return 0
    destination = OUTPUT_ROOT / source.name
    shutil.copytree(source, destination, dirs_exist_ok=True)
    metadata_files = sorted(destination.glob("[0-9][0-9][0-9][0-9].json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_files]
    report = {
        "completed_states": len(rows),
        "moves_total": sum(int(row["moves"]) for row in rows),
        "solver_seconds_sum": round(sum(float(row["elapsed_seconds"]) for row in rows), 4),
        "all_replay_verified": all(row.get("replay_verified") is True for row in rows),
    }
    atomic_json(destination / "campaign_report.json", report)
    return len(rows)


summary = {
    "format_version": 1,
    "wave": WAVE,
    "slot": SLOT,
    "pids": PIDS,
    "frame_specs": FRAME_SPECS,
    "expected_tasks": len(PIDS) * len(FRAME_SPECS),
    "cpu_count": os.cpu_count(),
    "python": sys.version,
    "status": "starting",
    "frame_progress": {},
}
atomic_json(SUMMARY_PATH, summary)
started = time.perf_counter()
error = None

try:
    source_project = find_source_project()
    actual_hashes = {relative: sha256(source_project / relative) for relative in EXPECTED_HASHES}
    summary["source_project"] = str(source_project)
    summary["critical_hashes"] = actual_hashes
    if actual_hashes != EXPECTED_HASHES:
        raise RuntimeError(f"critical file hash mismatch: {actual_hashes}")
    source_binary = source_project / BINARY_RELATIVE
    actual_binary_sha256 = sha256(source_binary)
    summary["binary_sha256"] = actual_binary_sha256
    if actual_binary_sha256 != EXPECTED_BINARY_SHA256:
        raise RuntimeError(
            f"solver binary hash mismatch: {actual_binary_sha256} != {EXPECTED_BINARY_SHA256}"
        )

    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_project, PROJECT)
    binary = PROJECT / BINARY_RELATIVE
    binary.chmod(0o755)

    workers = max(1, min(4, os.cpu_count() or 1))
    batch_size = max(8, workers * 4)
    summary["workers"] = workers
    summary["batch_size"] = batch_size
    runner = PROJECT / "cube666" / "scripts" / "06_run_kmcoders_beam.py"
    common = [
        sys.executable, str(runner),
        "--workers", str(workers),
        "--rayon-threads", "1",
        "--beam", "20000",
        "--candidate-cap", "1000",
        "--anneal-seed", "50001",
        "--alpha", "3",
        "--corner-source", "exact",
        "--timeout-seconds", "3600",
    ]

    deadline_reached = False
    for frame in FRAME_SPECS:
        tag = f"{frame['tag']}_s{SLOT:02d}"
        for offset in range(0, len(PIDS), batch_size):
            if time.perf_counter() - started >= MAX_WALL_SECONDS:
                deadline_reached = True
                break
            batch = PIDS[offset : offset + batch_size]
            command = common + [
                "--indices", ",".join(map(str, batch)),
                "--anneal-steps", str(frame["anneal_steps"]),
                "--tag", tag,
            ] + frame["frame_args"]
            run(command, PROJECT)
            summary["frame_progress"][tag] = sync_result(tag)
            summary["wall_seconds"] = round(time.perf_counter() - started, 4)
            atomic_json(SUMMARY_PATH, summary)
        if deadline_reached:
            break

    completed_tasks = sum(int(value) for value in summary["frame_progress"].values())
    summary["completed_tasks"] = completed_tasks
    summary["status"] = "complete" if completed_tasks == summary["expected_tasks"] else "partial"
except Exception as exc:
    error = exc
    summary["status"] = "error"
    summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["traceback"] = traceback.format_exc()
finally:
    for frame in FRAME_SPECS:
        tag = f"{frame['tag']}_s{SLOT:02d}"
        completed = sync_result(tag)
        if completed:
            summary["frame_progress"][tag] = completed
    summary["completed_tasks"] = sum(int(value) for value in summary["frame_progress"].values())
    summary["wall_seconds"] = round(time.perf_counter() - started, 4)
    atomic_json(SUMMARY_PATH, summary)
    if OUTPUT_ROOT.exists():
        shutil.make_archive(str(ARCHIVE_BASE), "zip", root_dir=OUTPUT_ROOT)
        shutil.rmtree(OUTPUT_ROOT, ignore_errors=True)
    shutil.rmtree(RUN_ROOT, ignore_errors=True)

if error is not None:
    raise error
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", type=int, choices=(1, 2), required=True)
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--out-root", type=Path)
    return parser.parse_args()


def path_length(text: str) -> int:
    return len([token for token in text.split(".") if token])


def frame_specs(wave: int) -> list[dict[str, object]]:
    if wave == 1:
        return [
            {"tag": "w1_rot03_tl40", "anneal_steps": 40_000_000, "frame_args": ["--symmetry-index", "3"]},
            {"tag": "w1_rot18_tl40", "anneal_steps": 40_000_000, "frame_args": ["--symmetry-index", "18"]},
            {"tag": "w1_rot35_tl40", "anneal_steps": 40_000_000, "frame_args": ["--symmetry-index", "35"]},
        ]
    return [
        {"tag": "w2_inverse_tl20", "anneal_steps": 20_000_000, "frame_args": ["--inverse-frame"]},
        {"tag": "w2_ref01_tl20", "anneal_steps": 20_000_000, "frame_args": ["--symmetry-index", "1"]},
        {"tag": "w2_ref40_tl20", "anneal_steps": 20_000_000, "frame_args": ["--symmetry-index", "40"]},
        {
            "tag": "w2_ref25inv_tl20",
            "anneal_steps": 20_000_000,
            "frame_args": ["--symmetry-index", "25", "--inverse-frame"],
        },
    ]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.slots <= 0:
        raise ValueError("slots must be positive")
    out_root = args.out_root or PROJECT / "cube666" / "kaggle_kmc_campaign" / f"production_wave{args.wave}"
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {out_root}")

    with open(args.incumbent, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    weighted = [
        (int(row["initial_state_id"]), max(0.15, path_length(row["path"]) / 195.0))
        for row in rows
    ]
    if sorted(index for index, _ in weighted) != list(range(1012)):
        raise ValueError("incumbent must cover PIDs 0..1011 exactly")

    shards: list[list[int]] = [[] for _ in range(args.slots)]
    loads = [0.0] * args.slots
    for index, weight in sorted(weighted, key=lambda item: (-item[1], item[0])):
        slot = min(range(args.slots), key=lambda value: (loads[value], len(shards[value]), value))
        shards[slot].append(index)
        loads[slot] += weight
    for shard in shards:
        shard.sort()

    specs = frame_specs(args.wave)
    manifest = {
        "format_version": 1,
        "wave": args.wave,
        "slots": args.slots,
        "dataset_source": DATASET_SOURCE,
        "critical_hashes": EXPECTED_HASHES,
        "binary_relative": BINARY_RELATIVE,
        "binary_sha256": args.binary_sha256,
        "frame_specs": specs,
        "incumbent": str(args.incumbent.resolve()),
        "shards": [
            {"slot": slot, "pids": shards[slot], "pid_count": len(shards[slot]), "predicted_load": round(loads[slot], 6)}
            for slot in range(args.slots)
        ],
    }
    write_text(out_root / "campaign_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for slot in range(args.slots):
        slug = f"cayley-666-kmc-w{args.wave}-s{slot:02d}"
        directory = out_root / f"slot{slot:02d}"
        metadata = {
            "code_file": "kmc_campaign.py",
            "competition_sources": [],
            "dataset_sources": [DATASET_SOURCE],
            "enable_gpu": False,
            "enable_internet": False,
            "id": f"artgor/{slug}",
            "is_private": True,
            "kernel_sources": [],
            "kernel_type": "script",
            "language": "python",
            "model_sources": [],
            "title": slug,
        }
        code = (
            RUNTIME_TEMPLATE.replace("__WAVE__", repr(args.wave))
            .replace("__SLOT__", repr(slot))
            .replace("__PIDS__", repr(shards[slot]))
            .replace("__FRAME_SPECS__", repr(specs))
            .replace("__EXPECTED_HASHES__", repr(EXPECTED_HASHES))
            .replace("__EXPECTED_BINARY_SHA256__", repr(args.binary_sha256))
        )
        write_text(directory / "kernel-metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        write_text(directory / "kmc_campaign.py", code)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
