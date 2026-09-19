from pathlib import Path

_root = Path("/marimo/storage")
print({"exists": _root.exists()})
if _root.exists():
    _rows = []
    for _path in _root.rglob("*"):
        if _path.is_file():
            _rows.append(
                {
                    "path": str(_path),
                    "bytes": _path.stat().st_size,
                }
            )
    print(sorted(_rows, key=lambda _row: _row["path"]))
