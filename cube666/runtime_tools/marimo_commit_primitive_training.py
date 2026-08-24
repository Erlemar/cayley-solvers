import marimo._code_mode as cm


_cell_code = r'''
from pathlib import Path as _Path
import json as _json
import os as _os
import subprocess as _subprocess
import sys as _sys


def _run_cube666_clean_primitive_training():
    _root = _Path("/marimo/storage/cube666_fullpath_primmoves_clean_v1")
    _old = _Path("/marimo/storage/cube666_classical_fullpath_searchv_v1")
    _output = _root / "model_warmpolicy_resetv_pro6000_v1"
    _output.mkdir(parents=True, exist_ok=True)
    _command = [
        _sys.executable,
        str(_root / "68_remote_train_primitive_effect.py"),
        "--teacher",
        str(_root / "teacher.npz"),
        "--group-ids",
        str(_root / "source_state_ids.npy"),
        "--action-effects",
        str(_root / "action_effects.npy"),
        "--train-module",
        str(_old / "train.py"),
        "--init-checkpoint",
        str(_old / "model_pro6000_seed500666" / "checkpoint.pt"),
        "--reset-value-heads",
        "--out-dir",
        str(_output),
        "--action-digest",
        "f0c2907a8f2df77c7b6add6e3df26d7d5966b37dac25a828195fb15b0922cd3e",
        "--steps",
        "3000",
        "--batch-size",
        "512",
        "--negative-actions",
        "512",
        "--learning-rate",
        "0.0001",
        "--value-weight",
        "1.0",
        "--cluster-value-weight",
        "1.0",
        "--high-value-fraction",
        "0.5",
        "--eval-samples",
        "4096",
        "--policy-eval-samples",
        "64",
        "--policy-topk",
        "16",
        "--policy-chunk-size",
        "1024",
        "--compile",
    ]
    _report_path = _output / "report.json"
    _pid_path = _output / "process.json"
    if _report_path.exists():
        _report = _json.loads(_report_path.read_text(encoding="utf-8"))
        return {
            "after_policy": _report["after_policy"],
            "after_values": _report["after_values"],
            "before_policy": _report["before_policy"],
            "before_values": _report["before_values"],
            "checkpoint": str(_output / "checkpoint.pt"),
            "elapsed_seconds": _report["elapsed_seconds"],
            "initialization": _report["initialization"],
            "status": "completed",
            "value_unit": _report["value_unit"],
        }
    if _pid_path.exists():
        _old_pid = int(_json.loads(_pid_path.read_text(encoding="utf-8"))["pid"])
        try:
            _os.kill(_old_pid, 0)
        except OSError:
            pass
        else:
            return {
                "log": str(_output / "training.log"),
                "pid": _old_pid,
                "status": "running",
            }
    _log = open(_output / "training.log", "a", encoding="utf-8")
    _process = _subprocess.Popen(
        _command,
        stdout=_log,
        stderr=_subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    _log.close()
    _pid_path.write_text(
        _json.dumps({"command": _command, "pid": _process.pid}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "log": str(_output / "training.log"),
        "pid": _process.pid,
        "status": "launched",
    }


cube666_clean_primitive_training_result = _run_cube666_clean_primitive_training()
cube666_clean_primitive_training_result
'''

async with cm.get_context() as ctx:
    _existing = [
        cell
        for cell in ctx.cells
        if cell.name == "cube666_clean_primitive_training_pro6000"
    ]
    if _existing:
        _target = _existing[0].id
        _ = _existing[0].code
        ctx.edit_cell(_target, code=_cell_code, hide_code=False)
    else:
        _target = ctx.create_cell(
            _cell_code,
            hide_code=False,
            name="cube666_clean_primitive_training_pro6000",
        )
    ctx.run_cell(_target)
print(_target)
