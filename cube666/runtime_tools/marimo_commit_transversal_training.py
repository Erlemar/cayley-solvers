import marimo._code_mode as cm

_cell_code = r'''
from pathlib import Path as _Path
import json as _json
import subprocess as _subprocess
import sys as _sys


def _run_cube666_transversal_pro6000():
    _root = _Path("/marimo/storage/cube666_stabilizer_v1")
    _output = _root / "model_transversal_pro6000_v1"
    _output.mkdir(parents=True, exist_ok=True)
    _command = [
        _sys.executable,
        str(_root / "63_train_stabilizer_factor_policy.py"),
        "--teacher",
        str(_root / "teacher.npz"),
        "--output-dir",
        str(_output),
        "--steps",
        "3000",
        "--batch-size",
        "4096",
        "--eval-trials",
        "400",
        "--beam-width",
        "128",
        "--branch-width",
        "16",
        "--beam-steps",
        "30",
        "--compile",
    ]
    _completed = _subprocess.run(
        _command,
        capture_output=True,
        text=True,
    )
    (_output / "training.log").write_text(
        _completed.stdout + _completed.stderr,
        encoding="utf-8",
    )
    if _completed.returncode:
        raise RuntimeError(
            f"transversal training failed with {_completed.returncode}: "
            f"{_completed.stderr[-4000:]}"
        )
    _report = _json.loads((_output / "report.json").read_text(encoding="utf-8"))
    return {
        "checkpoint": _report["checkpoint"],
        "elapsed_seconds": _report["elapsed_seconds"],
        "eval_trials": _report["eval_trials"],
        "gate_passed": _report["gate_passed"],
        "heldout_metrics": _report["heldout_metrics"],
        "solved": _report["solved"],
    }


cube666_transversal_pro6000_result = _run_cube666_transversal_pro6000()
cube666_transversal_pro6000_result
'''

async with cm.get_context() as ctx:
    _existing = [cell for cell in ctx.cells if cell.name == "cube666_transversal_pro6000_gate"]
    if _existing:
        _target = _existing[0].id
        _ = _existing[0].code
        ctx.edit_cell(_target, code=_cell_code, hide_code=False)
    else:
        _target = ctx.create_cell(
            _cell_code,
            hide_code=False,
            name="cube666_transversal_pro6000_gate",
        )
    ctx.run_cell(_target)
print(_target)
