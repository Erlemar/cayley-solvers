import marimo._code_mode as cm


cell_name = "cube666_verified_d6_training_pro6000"
async with cm.get_context() as ctx:
    cells = [cell for cell in ctx.cells if cell.name == cell_name]
    if len(cells) != 1:
        raise RuntimeError(f"expected one {cell_name} cell, found {len(cells)}")
    cell = cells[0]
    current = cell.code
    copy_block = '''    for _name in (
        "__init__.py",
        "macro_policy.py",
        "macro_action_policy.py",
        "macro_factorized_value.py",
    ):
        _shutil.copy2(_root / _name, _src / _name)
'''
    fixed_copy_block = '''    (_src / "__init__.py").write_text("", encoding="utf-8")
    for _name in (
        "macro_policy.py",
        "macro_action_policy.py",
        "macro_factorized_value.py",
    ):
        _shutil.copy2(_root / _name, _src / _name)
'''
    process_block = '''    if _process_path.exists():
        _old_pid = int(_json.loads(_process_path.read_text())["pid"])
        try:
            _os.kill(_old_pid, 0)
        except OSError:
            pass
        else:
            return {
                "status": "running",
                "pid": _old_pid,
                "status_file": str(_status_path),
                "log": str(_log_path),
            }
'''
    fixed_process_block = '''    _prior_failed = False
    if _status_path.exists():
        _prior_failed = _json.loads(_status_path.read_text()).get("state") == "failed"
    if _process_path.exists() and not _prior_failed:
        _old_pid = int(_json.loads(_process_path.read_text())["pid"])
        try:
            _os.kill(_old_pid, 0)
        except OSError:
            pass
        else:
            return {
                "status": "running",
                "pid": _old_pid,
                "status_file": str(_status_path),
                "log": str(_log_path),
            }
'''
    if copy_block not in current or process_block not in current:
        raise RuntimeError("durable cell no longer matches the expected source")
    updated = current.replace(copy_block, fixed_copy_block, 1).replace(
        process_block, fixed_process_block, 1
    )
    ctx.edit_cell(cell.id, code=updated)
    ctx.run_cell(cell.id)
    print(
        {
            "status": "patched_and_queued",
            "id": cell.id,
            "old_chars": len(current),
            "new_chars": len(updated),
        }
    )
