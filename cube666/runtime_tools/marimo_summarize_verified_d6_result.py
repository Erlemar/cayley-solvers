import json
from pathlib import Path


root = Path("/marimo/storage/cube666_verified_d6_v1")
gate = json.loads(
    (root / "eval_verified_d6_gate32_b4096_portfolio_v1.json").read_text()
)
models = {}
for name in (
    "model_direct_d6_unconditioned_seed127672",
    "model_direct_d6_depthconditioned_seed127673",
):
    report = json.loads((root / name / "report.json").read_text())
    models[name] = {
        "before": report["before"],
        "after": report["after"],
        "elapsed_seconds": report["elapsed_seconds"],
        "last_loss": report["last_loss"],
        "model_config": report["model_config"],
    }
rows = gate["rows"]
print(
    json.dumps(
        {
            "models": models,
            "gate": {
                "solved": gate["solved"],
                "cases": len(rows),
                "solved_rows": [row for row in rows if row["solved"]],
                "teacher_upper_bounds": [row["teacher_upper_bound"] for row in rows],
                "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows),
                "expanded_children": sum(row["expanded_children"] for row in rows),
            },
        }
    )
)
