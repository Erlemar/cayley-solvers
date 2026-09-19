import json
import statistics
from pathlib import Path


report = json.loads(
    Path(
        "/marimo/storage/cube666_verified_d6_v1/"
        "eval_verified_d6_gate32_oracle_audit_v2.json"
    ).read_text()
)
layers = {}
for row in report["rows"]:
    for audit in row["oracle_audit"]:
        layer = layers.setdefault(
            audit["layer"],
            {"cases": 0, "proposed": 0, "retained": 0, "ranks": []},
        )
        layer["cases"] += 1
        layer["proposed"] += int(audit["oracle_proposed"])
        layer["retained"] += int(audit["oracle_retained"])
        if audit["oracle_global_rank"] is not None:
            layer["ranks"].append(audit["oracle_global_rank"])
for layer in layers.values():
    ranks = layer.pop("ranks")
    layer["median_global_rank"] = statistics.median(ranks) if ranks else None
    layer["p90_global_rank"] = (
        sorted(ranks)[min(len(ranks) - 1, int(0.9 * len(ranks)))] if ranks else None
    )
print({"solved": report["solved"], "cases": len(report["rows"]), "layers": layers})
