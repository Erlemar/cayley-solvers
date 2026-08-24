import json
from pathlib import Path


_root = Path(
    "/marimo/storage/cube666_hardnegative_v1/"
    "model_hardneg128_random32_b64_v1"
)
_report = _root / "report.json"
_log = _root / "training.log"
print(
    json.dumps(
        {
            "files": {
                path.name: path.stat().st_size
                for path in _root.iterdir()
                if path.is_file()
            }
            if _root.exists()
            else {},
            "log_tail": _log.read_text(encoding="utf-8")[-10000:] if _log.exists() else "",
            "report": json.loads(_report.read_text(encoding="utf-8")) if _report.exists() else None,
        }
    )
)
