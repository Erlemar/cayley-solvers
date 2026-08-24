"""Validate and safely extract the completed cube666 KMC resume-wave artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath


PROJECT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = PROJECT / "cube666" / "kaggle_kmc_campaign"
DEFAULT_MANIFEST = CAMPAIGN_ROOT / "resume_wave1" / "campaign_manifest.json"
DEFAULT_HARVEST = CAMPAIGN_ROOT / "production_outputs" / "resume_wave1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--harvest-root", type=Path, default=DEFAULT_HARVEST)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_HARVEST / "harvest_report.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def normalized_group(group: dict[str, object]) -> dict[str, object]:
    raw_pids = group["pids"]
    pids = raw_pids.split() if isinstance(raw_pids, str) else raw_pids
    raw_args = group["frame_args"]
    frame_args = raw_args.split() if isinstance(raw_args, str) else raw_args
    return {
        "anneal_steps": int(group["anneal_steps"]),
        "frame_args": [str(value) for value in frame_args],
        "pids": [int(value) for value in pids],
        "tag": str(group["tag"]),
    }


def safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    output: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe ZIP member {member.filename!r}")
        output.append(member)
    return output


def write_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, root: Path) -> None:
    target = root.joinpath(*PurePosixPath(member.filename).parts)
    if member.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return
    payload = archive.read(member)
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f"existing extracted file differs from ZIP: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_shards = {int(shard["slot"]): shard for shard in manifest["shards"]}
    if set(expected_shards) != set(range(10)):
        raise ValueError(f"manifest slots are not exactly 0..9: {sorted(expected_shards)}")

    expected_task_keys: set[tuple[str, int]] = set()
    reports: list[dict[str, object]] = []
    total_completed = 0
    all_candidate_dirs: list[str] = []

    for slot in range(10):
        slot_dir = args.harvest_root / f"slot{slot:02d}"
        summary_path = slot_dir / f"kmc_summary_w1r_s{slot:02d}.json"
        archive_path = slot_dir / f"kmc_results_w1r_s{slot:02d}.zip"
        log_path = slot_dir / f"cayley-666-kmc-w1r-s{slot:02d}.log"
        for required in (summary_path, archive_path, log_path):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty shard artifact: {required}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard = expected_shards[slot]
        expected_groups = [normalized_group(group) for group in shard["groups"]]
        actual_groups = [normalized_group(group) for group in summary["task_groups"]]
        if actual_groups != expected_groups:
            raise ValueError(f"slot {slot}: summary task groups differ from manifest")
        if summary.get("campaign") != manifest["campaign"]:
            raise ValueError(f"slot {slot}: campaign mismatch")
        if int(summary.get("slot", -1)) != slot:
            raise ValueError(f"slot {slot}: summary slot mismatch")
        if summary.get("status") != "complete":
            raise ValueError(f"slot {slot}: summary is not complete")
        if summary.get("binary_sha256") != manifest["binary_sha256"]:
            raise ValueError(f"slot {slot}: binary hash mismatch")
        if summary.get("critical_hashes") != manifest["critical_hashes"]:
            raise ValueError(f"slot {slot}: critical source hash mismatch")

        expected_count = sum(len(group["pids"]) for group in expected_groups)
        if expected_count != int(shard["task_count"]):
            raise ValueError(f"slot {slot}: manifest task count is internally inconsistent")
        if int(summary["expected_tasks"]) != expected_count:
            raise ValueError(f"slot {slot}: summary expected-task mismatch")
        if int(summary["completed_tasks"]) != expected_count:
            raise ValueError(f"slot {slot}: incomplete summary coverage")

        expected_dirs: dict[str, set[int]] = {}
        expected_progress: dict[str, int] = {}
        for group in expected_groups:
            tag = f"{group['tag']}_s{slot:02d}"
            directory = f"{tag}_b20000"
            pids = set(group["pids"])
            expected_dirs[directory] = pids
            expected_progress[tag] = len(pids)
            for pid in pids:
                key = (str(group["tag"]), int(pid))
                if key in expected_task_keys:
                    raise ValueError(f"duplicate resume task across shards: {key}")
                expected_task_keys.add(key)
        if summary.get("frame_progress") != expected_progress:
            raise ValueError(f"slot {slot}: frame progress differs from manifest")

        extracted = slot_dir / "extracted"
        with zipfile.ZipFile(archive_path) as archive:
            members = safe_members(archive)
            path_members: dict[str, set[int]] = {name: set() for name in expected_dirs}
            metadata_members: dict[str, set[int]] = {name: set() for name in expected_dirs}
            stdout_members: dict[str, set[int]] = {name: set() for name in expected_dirs}
            stderr_members: dict[str, set[int]] = {name: set() for name in expected_dirs}
            for member in members:
                path = PurePosixPath(member.filename)
                if member.is_dir():
                    continue
                if len(path.parts) != 2 or path.parts[0] not in expected_dirs:
                    raise ValueError(f"slot {slot}: unexpected ZIP member {member.filename!r}")
                directory, filename = path.parts
                if filename in {"solutions.csv", "report.json", "campaign_report.json"}:
                    continue
                suffix_sets = (
                    (".path.txt", path_members[directory]),
                    (".json", metadata_members[directory]),
                    (".stdout.log", stdout_members[directory]),
                    (".stderr.log", stderr_members[directory]),
                )
                matched = False
                for suffix, destination in suffix_sets:
                    if filename.endswith(suffix):
                        destination.add(int(filename.removesuffix(suffix)))
                        matched = True
                        break
                if not matched:
                    raise ValueError(f"slot {slot}: unexpected result filename {filename!r}")

            for directory, expected_pids in expected_dirs.items():
                for label, found in (
                    ("paths", path_members[directory]),
                    ("metadata", metadata_members[directory]),
                    ("stdout", stdout_members[directory]),
                    ("stderr", stderr_members[directory]),
                ):
                    if found != expected_pids:
                        missing = sorted(expected_pids - found)
                        extra = sorted(found - expected_pids)
                        raise ValueError(
                            f"slot {slot} {directory} {label}: missing={missing} extra={extra}"
                        )
            for member in members:
                write_member(archive, member, extracted)

        for directory, expected_pids in expected_dirs.items():
            candidate_dir = extracted / directory
            actual_pids = {
                int(path.name.removesuffix(".path.txt"))
                for path in candidate_dir.glob("*.path.txt")
            }
            if actual_pids != expected_pids:
                raise ValueError(f"slot {slot}: extracted PID coverage mismatch in {directory}")
            all_candidate_dirs.append(str(candidate_dir.resolve()))

        total_completed += expected_count
        reports.append(
            {
                "slot": slot,
                "status": "complete",
                "completed_tasks": expected_count,
                "expected_tasks": expected_count,
                "frame_progress": expected_progress,
                "wall_seconds": float(summary["wall_seconds"]),
                "archive": str(archive_path.resolve()),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": sha256(archive_path),
                "summary_sha256": sha256(summary_path),
                "log_sha256": sha256(log_path),
                "candidate_dirs": [str((extracted / name).resolve()) for name in expected_dirs],
            }
        )

    if total_completed != int(manifest["missing_task_count"]):
        raise ValueError(
            f"harvested {total_completed} tasks, manifest requires {manifest['missing_task_count']}"
        )
    if len(expected_task_keys) != total_completed:
        raise ValueError("resume task keys are not one-to-one")

    report = {
        "format_version": 1,
        "campaign": manifest["campaign"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "expected_tasks": int(manifest["missing_task_count"]),
        "completed_tasks": total_completed,
        "coverage_complete": True,
        "candidate_directories": all_candidate_dirs,
        "candidate_directory_count": len(all_candidate_dirs),
        "shards": reports,
    }
    atomic_write(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"HARVEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
