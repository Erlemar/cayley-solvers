import json
from pathlib import Path

print(cube666_transversal_pro6000_result)
_report_path = Path(
    "/marimo/storage/cube666_stabilizer_v1/model_transversal_pro6000_v1/report.json"
)
_report = json.loads(_report_path.read_text(encoding="utf-8"))
print(
    {
        "beam_width": _report["beam_width"],
        "branch_width": _report["branch_width"],
        "elapsed_seconds": _report["elapsed_seconds"],
        "eval_trials": _report["eval_trials"],
        "gate_passed": _report["gate_passed"],
        "heldout_metrics": _report["heldout_metrics"],
        "solved": _report["solved"],
        "step_min": min(row["steps"] for row in _report["results"]),
        "step_mean": sum(row["steps"] for row in _report["results"])
        / len(_report["results"]),
        "step_max": max(row["steps"] for row in _report["results"]),
    }
)
