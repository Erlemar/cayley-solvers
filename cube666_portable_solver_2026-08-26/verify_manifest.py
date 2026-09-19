"""Create or verify the portable bundle's SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.json"
IGNORED_PARTS = {".venv", "__pycache__", ".pytest_cache", "outputs", "cache"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def distributed_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if "target" in relative.parts and path.name != "solve_cube_beam.exe":
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def write_manifest() -> None:
    rows = [
        {
            "bytes": path.stat().st_size,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": digest(path),
        }
        for path in distributed_files()
    ]
    payload = {
        "bundle_format": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": rows,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST.name}: {len(rows)} files")


def verify_manifest() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError("MANIFEST.json is missing; run verify_manifest.py --write")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in payload["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append(f"missing: {row['path']}")
            continue
        actual_size = path.stat().st_size
        if actual_size != int(row["bytes"]):
            failures.append(f"size: {row['path']} expected {row['bytes']} got {actual_size}")
            continue
        actual_digest = digest(path)
        if actual_digest != row["sha256"]:
            failures.append(f"sha256: {row['path']}")
    if failures:
        raise SystemExit("manifest verification failed:\n" + "\n".join(failures))
    print(f"manifest verified: {len(payload['files'])} files")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    write_manifest() if args.write else verify_manifest()


if __name__ == "__main__":
    main()

