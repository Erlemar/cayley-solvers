"""Build hash-pinned Kaggle CPU kernels for only the missing wave-1 KMC tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = PROJECT / "cube666" / "kaggle_kmc_campaign"
SOURCE_MANIFEST = CAMPAIGN_ROOT / "production_wave1_prebuilt" / "campaign_manifest.json"
SOURCE_OUTPUTS = CAMPAIGN_ROOT / "production_outputs"
DEFAULT_INCUMBENT = PROJECT / "submissions" / "cube666_classical_paramgate16_merged.csv"


RUNTIME_TEMPLATE = r'''"""Generated resume-only KMC frame-campaign shard for Kaggle CPU."""
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

CAMPAIGN = __CAMPAIGN__
SLOT = __SLOT__
TASK_GROUPS = __TASK_GROUPS__
EXPECTED_HASHES = __EXPECTED_HASHES__
EXPECTED_BINARY_SHA256 = __EXPECTED_BINARY_SHA256__
BINARY_RELATIVE = __BINARY_RELATIVE__
MAX_WALL_SECONDS = 10.5 * 3600
WORK = pathlib.Path("/kaggle/working")
RUN_ROOT = WORK / f"kmc_campaign_{CAMPAIGN}_s{SLOT:02d}"
PROJECT = RUN_ROOT / "cayley"
OUTPUT_ROOT = WORK / "kmc_results"
SUMMARY_PATH = WORK / f"kmc_summary_{CAMPAIGN}_s{SLOT:02d}.json"
ARCHIVE_BASE = WORK / f"kmc_results_{CAMPAIGN}_s{SLOT:02d}"


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
    "campaign": CAMPAIGN,
    "slot": SLOT,
    "task_groups": TASK_GROUPS,
    "expected_tasks": sum(len(group["pids"]) for group in TASK_GROUPS),
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
    for group in TASK_GROUPS:
        tag = f"{group['tag']}_s{SLOT:02d}"
        pids = group["pids"]
        for offset in range(0, len(pids), batch_size):
            if time.perf_counter() - started >= MAX_WALL_SECONDS:
                deadline_reached = True
                break
            batch = pids[offset : offset + batch_size]
            command = common + [
                "--indices", ",".join(map(str, batch)),
                "--anneal-steps", str(group["anneal_steps"]),
                "--tag", tag,
            ] + group["frame_args"]
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
    for group in TASK_GROUPS:
        tag = f"{group['tag']}_s{SLOT:02d}"
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
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--source-outputs", type=Path, default=SOURCE_OUTPUTS)
    parser.add_argument("--out-root", type=Path, default=CAMPAIGN_ROOT / "resume_wave1")
    parser.add_argument("--expected-missing", type=int, default=1044)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def frame_key(directory_name: str) -> str:
    for key in ("rot03", "rot18", "rot35"):
        if key in directory_name:
            return key
    raise ValueError(f"cannot infer frame from {directory_name}")


def collect_completed(outputs: Path) -> tuple[set[tuple[str, int]], dict[int, list[float]]]:
    completed: set[tuple[str, int]] = set()
    timings: dict[int, list[float]] = {}
    for slot in range(10):
        extracted = outputs / f"slot{slot:02d}" / "extracted"
        if not extracted.is_dir():
            raise FileNotFoundError(f"missing extracted output directory: {extracted}")
        for result_dir in sorted(path for path in extracted.iterdir() if path.is_dir()):
            frame = frame_key(result_dir.name)
            for metadata_path in sorted(result_dir.glob("[0-9][0-9][0-9][0-9].json")):
                row = json.loads(metadata_path.read_text(encoding="utf-8"))
                pid = int(row["index"])
                if row.get("replay_verified") is not True:
                    raise ValueError(f"unverified completed task: {metadata_path}")
                if not metadata_path.with_suffix(".path.txt").is_file():
                    raise FileNotFoundError(f"missing path for {metadata_path}")
                task = (frame, pid)
                if task in completed:
                    raise ValueError(f"duplicate completed task: {task}")
                completed.add(task)
                timings.setdefault(pid, []).append(float(row["elapsed_seconds"]))
    return completed, timings


def main() -> None:
    args = parse_args()
    if args.slots <= 0:
        raise ValueError("slots must be positive")
    if args.out_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {args.out_root}")

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("wave") != 1:
        raise ValueError("source manifest is not wave 1")
    frames = {
        "rot03": next(spec for spec in source_manifest["frame_specs"] if "rot03" in spec["tag"]),
        "rot18": next(spec for spec in source_manifest["frame_specs"] if "rot18" in spec["tag"]),
        "rot35": next(spec for spec in source_manifest["frame_specs"] if "rot35" in spec["tag"]),
    }
    planned = {
        (frame, int(pid))
        for shard in source_manifest["shards"]
        for pid in shard["pids"]
        for frame in frames
    }
    if len(planned) != 3036:
        raise ValueError(f"expected 3036 original tasks, found {len(planned)}")

    completed, timings = collect_completed(args.source_outputs)
    if not completed <= planned:
        raise ValueError(f"completed set contains {len(completed - planned)} unplanned tasks")
    missing = sorted(planned - completed)
    if len(missing) != args.expected_missing:
        raise ValueError(f"expected {args.expected_missing} missing tasks, found {len(missing)}")
    missing_counts = {frame: sum(item[0] == frame for item in missing) for frame in frames}
    if missing_counts != {"rot03": 0, "rot18": 234, "rot35": 810}:
        raise ValueError(f"unexpected missing-task distribution: {missing_counts}")

    with args.incumbent.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    path_lengths = {
        int(row["initial_state_id"]): len([token for token in row["path"].split(".") if token])
        for row in rows
    }
    if sorted(path_lengths) != list(range(1012)):
        raise ValueError("incumbent must cover PIDs 0..1011 exactly")

    predicted_seconds = {
        pid: statistics.median(values) if values else max(30.0, path_lengths[pid] * 1.1)
        for pid, values in ((pid, timings.get(pid, [])) for pid in range(1012))
    }
    shards: list[list[tuple[str, int]]] = [[] for _ in range(args.slots)]
    loads = [0.0] * args.slots
    for task in sorted(missing, key=lambda item: (-predicted_seconds[item[1]], item[0], item[1])):
        slot = min(range(args.slots), key=lambda value: (loads[value], len(shards[value]), value))
        shards[slot].append(task)
        loads[slot] += predicted_seconds[task[1]]
    for shard in shards:
        shard.sort()

    source_artifacts = {str(args.source_manifest.resolve()): sha256(args.source_manifest)}
    for slot in range(10):
        directory = args.source_outputs / f"slot{slot:02d}"
        for pattern in ("kmc_summary_*.json", "kmc_results_*.zip"):
            matches = sorted(directory.glob(pattern))
            if len(matches) != 1:
                raise ValueError(f"expected one {pattern} in {directory}, found {matches}")
            source_artifacts[str(matches[0].resolve())] = sha256(matches[0])

    task_payload = [{"frame": frame, "pid": pid} for frame, pid in missing]
    task_hash = hashlib.sha256(
        json.dumps(task_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest = {
        "format_version": 1,
        "campaign": "w1r",
        "dataset_source": source_manifest["dataset_source"],
        "critical_hashes": source_manifest["critical_hashes"],
        "binary_relative": source_manifest["binary_relative"],
        "binary_sha256": source_manifest["binary_sha256"],
        "source_artifacts_sha256": source_artifacts,
        "source_completed_tasks": len(completed),
        "missing_task_count": len(missing),
        "missing_counts": missing_counts,
        "missing_tasks_sha256": task_hash,
        "shards": [],
    }

    for slot, tasks in enumerate(shards):
        groups = []
        for frame in ("rot18", "rot35"):
            pids = sorted(pid for task_frame, pid in tasks if task_frame == frame)
            if pids:
                spec = frames[frame]
                groups.append(
                    {
                        "tag": f"w1r_{frame}_tl40",
                        "anneal_steps": int(spec["anneal_steps"]),
                        "frame_args": spec["frame_args"],
                        "pids": pids,
                    }
                )
        slug = f"cayley-666-kmc-w1r-s{slot:02d}"
        directory = args.out_root / f"slot{slot:02d}"
        metadata = {
            "code_file": "kmc_campaign.py",
            "competition_sources": [],
            "dataset_sources": [source_manifest["dataset_source"]],
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
            RUNTIME_TEMPLATE.replace("__CAMPAIGN__", repr("w1r"))
            .replace("__SLOT__", repr(slot))
            .replace("__TASK_GROUPS__", repr(groups))
            .replace("__EXPECTED_HASHES__", repr(source_manifest["critical_hashes"]))
            .replace("__EXPECTED_BINARY_SHA256__", repr(source_manifest["binary_sha256"]))
            .replace("__BINARY_RELATIVE__", repr(source_manifest["binary_relative"]))
        )
        write_text(directory / "kernel-metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        write_text(directory / "kmc_campaign.py", code)
        manifest["shards"].append(
            {
                "slot": slot,
                "task_count": len(tasks),
                "predicted_seconds": round(loads[slot], 3),
                "groups": groups,
            }
        )

    write_text(args.out_root / "campaign_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
