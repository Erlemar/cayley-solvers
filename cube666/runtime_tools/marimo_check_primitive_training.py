import json
import subprocess
from pathlib import Path


_root = Path(
    "/marimo/storage/cube666_fullpath_primmoves_clean_v1/"
    "model_warmpolicy_resetv_pro6000_v1"
)
_report = _root / "report.json"
_log = _root / "training.log"
_processes = subprocess.run(
    ["bash", "-lc", "ps -eo pid,etime,cmd | grep 68_remote_train_primitive_effect.py | grep -v grep || true"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
print(
    json.dumps(
        {
            "exists": _root.exists(),
            "files": {
                path.name: path.stat().st_size
                for path in _root.iterdir()
                if path.is_file()
            }
            if _root.exists()
            else {},
            "log_tail": _log.read_text(encoding="utf-8")[-3000:] if _log.exists() else "",
            "processes": _processes,
            "report": json.loads(_report.read_text(encoding="utf-8")) if _report.exists() else None,
        },
        default=str,
    )
)
