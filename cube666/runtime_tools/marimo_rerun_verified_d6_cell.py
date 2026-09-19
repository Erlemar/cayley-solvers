import marimo._code_mode as cm


async with cm.get_context() as ctx:
    cells = [cell for cell in ctx.cells if cell.name == "cube666_verified_d6_training_pro6000"]
    if len(cells) != 1:
        raise RuntimeError(f"expected one durable cell, found {len(cells)}")
    _ = cells[0].code
    ctx.run_cell(cells[0].id)
    print({"status": "queued", "id": cells[0].id})
