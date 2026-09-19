import json
import os
import re

source = open("/marimo/notebook.py", encoding="utf-8").read()
lines = source.splitlines()

markers = []
patterns = (
    "cube666",
    "rung",
    "orbit",
    "corner",
    "macro",
    "train",
    "beam",
    "upload",
    "checkpoint",
    "report",
)
for line_no, line in enumerate(lines, 1):
    low = line.lower()
    if line.lstrip().startswith(("def ", "class ", "#")) or any(p in low for p in patterns):
        markers.append({"line": line_no, "text": line[:220]})

storage = []
for directory, dirs, files in os.walk("/marimo/storage"):
    dirs[:] = sorted(dirs)[:20]
    for filename in sorted(files)[:50]:
        path = os.path.join(directory, filename)
        storage.append(
            {
                "path": os.path.relpath(path, "/marimo"),
                "size": os.path.getsize(path),
            }
        )
    if len(storage) >= 200:
        break

print(
    json.dumps(
        {
            "notebook_lines": len(lines),
            "notebook_bytes": len(source.encode("utf-8")),
            "markers": markers[:300],
            "storage": storage[:200],
        },
        indent=2,
    )
)
