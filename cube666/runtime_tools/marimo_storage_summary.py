from pathlib import Path


root = Path("/marimo/storage")
files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
print(
    {
        "exists": root.exists(),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "top_level": sorted(path.name for path in root.iterdir()) if root.exists() else [],
        "largest": [
            (str(path.relative_to(root)), path.stat().st_size)
            for path in sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:40]
        ],
    }
)
