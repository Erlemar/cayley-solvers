import json

import marimo._code_mode as cm


async with cm.get_context() as ctx:
    cells = list(ctx.cells)[-8:]
    print(
        json.dumps(
            [
                {
                    "id": cell.id,
                    "name": cell.name,
                    "status": str(cell.status),
                    "errors": [str(error) for error in cell.errors],
                    "code": cell.code[:1600],
                }
                for cell in cells
            ]
        )
    )
