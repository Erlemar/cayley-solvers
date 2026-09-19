import json
from pathlib import Path

import marimo._code_mode as cm


root = Path("/marimo/storage/cube666_verified_d6_v1")
summary_path = root / "verified_d6_scoring_sweep_summary_v1.json"
async with cm.get_context() as ctx:
    cells = [cell for cell in ctx.cells if cell.name == "cube666_verified_d6_scoring_sweep_pro6000"]
    reports = sorted(path.name for path in root.glob("eval_verified_d6_gate32_*_v1.json"))
    print(
        {
            "cell": (
                {
                    "id": cells[0].id,
                    "status": str(cells[0].status),
                    "errors": [str(error) for error in cells[0].errors],
                }
                if cells
                else None
            ),
            "reports": reports,
            "summary": json.loads(summary_path.read_text()) if summary_path.exists() else None,
        }
    )
