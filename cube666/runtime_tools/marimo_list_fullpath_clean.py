from pathlib import Path


root = Path("/marimo/storage/cube666_fullpath_primmoves_clean_v1")
print(
    {
        "exists": root.exists(),
        "files": sorted(
            (str(path.relative_to(root)), path.stat().st_size)
            for path in root.rglob("*")
            if path.is_file()
        ),
    }
)
