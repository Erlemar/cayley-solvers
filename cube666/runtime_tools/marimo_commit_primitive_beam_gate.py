import marimo._code_mode as cm


_cell_code = r'''
from pathlib import Path as _Path
import json as _json
import os as _os
import subprocess as _subprocess
import sys as _sys


def _run_cube666_primitive_beam_gate():
    _data = _Path("/marimo/storage/cube666_fullpath_primmoves_clean_v1")
    _gate = _Path("/marimo/storage/cube666_primitive_beam_gate_v1")
    _old = _Path("/marimo/storage/cube666_classical_fullpath_searchv_v1")
    _output = _gate / "eval_b64_branch16_v1"
    _output.mkdir(parents=True, exist_ok=True)
    _report_path = _output / "report.json"
    _pid_path = _output / "process.json"
    if _report_path.exists():
        _report = _json.loads(_report_path.read_text(encoding="utf-8"))
        return {
            "elapsed_seconds": _report["elapsed_seconds"],
            "selected_pids": _report["selected_pids"],
            "status": "completed",
            "summaries": _report["summaries"],
            "teacher_audit": _report["teacher_audit"],
        }
    if _pid_path.exists():
        _old_pid = int(_json.loads(_pid_path.read_text(encoding="utf-8"))["pid"])
        try:
            _os.kill(_old_pid, 0)
        except OSError:
            pass
        else:
            return {"pid": _old_pid, "status": "running"}
    _command = [
        _sys.executable,
        str(_gate / "69_remote_eval_primitive_beam.py"),
        "--teacher",
        str(_data / "teacher.npz"),
        "--group-ids",
        str(_data / "source_state_ids.npy"),
        "--action-effects",
        str(_data / "action_effects.npy"),
        "--action-costs",
        str(_gate / "action_costs.npy"),
        "--train-module",
        str(_old / "train.py"),
        "--control-checkpoint",
        str(_old / "model_pro6000_seed500666" / "checkpoint.pt"),
        "--candidate-checkpoint",
        str(_data / "model_warmpolicy_resetv_pro6000_v1" / "checkpoint.pt"),
        "--out-dir",
        str(_output),
        "--action-digest",
        "f0c2907a8f2df77c7b6add6e3df26d7d5966b37dac25a828195fb15b0922cd3e",
        "--eval-puzzles",
        "4",
        "--beam-width",
        "64",
        "--branch-width",
        "16",
        "--max-steps",
        "30",
        "--policy-nll-weight",
        "0.02",
    ]
    _log = open(_output / "evaluation.log", "a", encoding="utf-8")
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
        "log": str(_output / "evaluation.log"),
        "pid": _process.pid,
        "status": "launched",
    }


cube666_primitive_beam_gate_result = _run_cube666_primitive_beam_gate()
cube666_primitive_beam_gate_result
'''

async with cm.get_context() as ctx:
    _existing = [
        cell
        for cell in ctx.cells
        if cell.name == "cube666_primitive_beam_gate_pro6000"
    ]
    if _existing:
        _target = _existing[0].id
        _ = _existing[0].code
        ctx.edit_cell(_target, code=_cell_code, hide_code=False)
    else:
        _target = ctx.create_cell(
            _cell_code,
            hide_code=False,
            name="cube666_primitive_beam_gate_pro6000",
        )
    ctx.run_cell(_target)
print(_target)
