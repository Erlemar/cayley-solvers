import json
import os
import shutil
import subprocess

root_entries = []
for name in sorted(os.listdir("/marimo")):
    path = os.path.join("/marimo", name)
    try:
        size = os.path.getsize(path) if os.path.isfile(path) else None
    except OSError:
        size = None
    root_entries.append(
        {"name": name, "kind": "dir" if os.path.isdir(path) else "file", "size": size}
    )

interesting = {}
for name, value in list(globals().items()):
    if name.startswith("_") or name in {"json", "os", "shutil", "subprocess"}:
        continue
    try:
        description = type(value).__name__
        if hasattr(value, "shape"):
            description += f" shape={tuple(value.shape)}"
        elif isinstance(value, (list, tuple, dict, set)):
            description += f" len={len(value)}"
        interesting[name] = description
    except Exception:
        interesting[name] = type(value).__name__

gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
).stdout.strip()

usage = shutil.disk_usage("/marimo")
print(
    json.dumps(
        {
            "entries": root_entries,
            "globals": interesting,
            "gpu_processes": gpu,
            "disk_gib": {
                "total": round(usage.total / 2**30, 2),
                "used": round(usage.used / 2**30, 2),
                "free": round(usage.free / 2**30, 2),
            },
        },
        indent=2,
    )
)
