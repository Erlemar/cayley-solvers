import json
import os
import subprocess
from pathlib import Path


root = Path("/marimo/storage/cube666_verified_d6_v1")
status_path = root / "pipeline_status.json"
process_path = root / "pipeline_process.json"
log_path = root / "pipeline.log"
process = json.loads(process_path.read_text()) if process_path.exists() else None
process_alive = False
if process is not None:
    try:
        os.kill(int(process["pid"]), 0)
        process_alive = True
    except OSError:
        pass
gpu = subprocess.run(
    [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ],
    capture_output=True,
    text=True,
).stdout.strip()
print(
    {
        "status": json.loads(status_path.read_text()) if status_path.exists() else None,
        "process": process,
        "process_alive": process_alive,
        "gpu_processes": gpu,
        "log_tail": log_path.read_text(encoding="utf-8")[-1800:] if log_path.exists() else "",
        "report_exists": (root / "eval_verified_d6_gate32_b4096_portfolio_v1.json").exists(),
    }
)
