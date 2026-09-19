import json
from pathlib import Path

import marimo._code_mode as cm


report_path = Path(
    "/marimo/storage/cube666_verified_d6_v1/"
    "eval_verified_d6_gate32_oracle_audit_v2.json"
)
async with cm.get_context() as ctx:
    cells = [cell for cell in ctx.cells if cell.name == "cube666_verified_d6_oracle_audit_pro6000"]
    print(
        {
            "cell": (
                {
                    "id": cells[0].id,
                    "status": str(cells[0].status),
                    "errors": [str(error) for error in cells[0].errors],
                    "output": str(cells[0].output)[:1200],
                }
                if cells
                else None
            ),
            "report_exists": report_path.exists(),
            "report_solved": (
                json.loads(report_path.read_text())["solved"] if report_path.exists() else None
            ),
        }
    )
