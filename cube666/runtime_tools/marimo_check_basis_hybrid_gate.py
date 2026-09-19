import json
from pathlib import Path


_root = Path(
    "/marimo/storage/cube666_basis_hybrid_gate_v1/"
    "eval_basis_b1024_p128_e112_v1"
)
_report = _root / "report.json"
_log = _root / "evaluation.log"
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
