import marimo._code_mode as cm


async with cm.get_context() as ctx:
    cells = [cell for cell in ctx.cells if cell.name == "cube666_verified_d6_training_pro6000"]
    if len(cells) != 1:
        raise RuntimeError(f"expected one cell, found {len(cells)}")
    code = cells[0].code
    for marker in ("_shutil.copy2", "if _process_path.exists"):
        position = code.find(marker)
        print({"marker": marker, "position": position, "fragment": code[max(0, position - 260): position + 620]})
