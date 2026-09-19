import json

import marimo._code_mode as cm


async with cm.get_context() as ctx:
    _cells = [
        cell
        for cell in ctx.cells
        if cell.name == "cube666_clean_primitive_training_pro6000"
    ]
    print(
        json.dumps(
            [
                {
                    "id": cell.id,
                    "name": cell.name,
                    "status": str(cell.status),
                    "output": str(cell.output)[:12000],
                }
                for cell in _cells
            ]
        )
    )
