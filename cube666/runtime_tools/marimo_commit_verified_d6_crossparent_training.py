import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_crossparent_value_training_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys


    def _cube666_crossparent_value_training_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _base = _Path("/marimo/storage/cube666_verified_d6_certified_value_v1")
        _run = _Path("/marimo/storage/cube666_verified_d6_crossparent_train_v1")
        _work = _root / "work"
        _output = _run / "model_crossparent_seed139666"
        _report_path = _output / "report.json"
        _log_path = _output / "training.log"
        _process_path = _output / "process.json"
        _output.mkdir(parents=True, exist_ok=True)
        if _report_path.exists():
            _report = _json.loads(_report_path.read_text(encoding="utf-8"))
            return {
                "status": "complete",
                "before_pairs": _report["before_pairs"],
                "after_pairs": _report["after_pairs"],
                "before_returns": _report["before_returns"],
                "after_returns": _report["after_returns"],
                "checkpoint": str(_output / "checkpoint.pt"),
            }
        if _process_path.exists():
            _process = _json.loads(_process_path.read_text(encoding="utf-8"))
            _pid = int(_process["pid"])
            try:
                _os.kill(_pid, 0)
                return {"status": "running", "pid": _pid, "log": str(_log_path)}
            except ProcessLookupError:
                pass
        _env = dict(_os.environ)
        _env["PYTHONPATH"] = str(_work / "src") + _os.pathsep + _env.get(
            "PYTHONPATH", ""
        )
        _command = [
            _sys.executable,
            "-u",
            str(_run / "139_finetune_short_macro_crossparent_value.py"),
            "--dataset-dir", str(_root / "dataset_d6_2m_seed126669"),
            "--direct-action-checkpoint", str(
                _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
            ),
            "--direct-action-checkpoint", str(
                _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
            ),
            "--utilities-script", str(
                _Path("/marimo/storage/cube666_verified_d6_certified_value_patch_v3")
                / "136_train_short_macro_certified_interval_value.py"
            ),
            "--init-checkpoint", str(
                _base / "model_interval_return4_seed136667_v3" / "checkpoint.pt"
            ),
            "--out-dir", str(_output),
            "--steps", "2000",
            "--root-batch-size", "4",
            "--root-branch", "128",
            "--second-branch", "32",
            "--hard-pairs", "32",
            "--anchor-batch-size", "256",
            "--eval-roots", "512",
            "--learning-rate", "5e-5",
            "--ranking-weight", "4",
            "--interval-weight", "1",
            "--return-weight", "2",
            "--seed", "139666",
            "--compile",
        ]
        _log_handle = _log_path.open("w", encoding="utf-8")
        _child = _subprocess.Popen(
            _command,
            cwd=str(_work),
            env=_env,
            stdout=_log_handle,
            stderr=_subprocess.STDOUT,
            start_new_session=True,
        )
        _log_handle.close()
        _process_path.write_text(
            _json.dumps({"pid": _child.pid, "command": _command}, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"status": "launched", "pid": _child.pid, "log": str(_log_path)}


    cube666_verified_d6_crossparent_value_training = (
        _cube666_crossparent_value_training_status()
    )
    cube666_verified_d6_crossparent_value_training
    """
)

async with cm.get_context() as ctx:
    existing = [cell for cell in ctx.cells if cell.name == cell_name]
    if existing:
        print({"status": "already_exists", "id": existing[0].id})
    else:
        cell_id = ctx.create_cell(cell_code, name=cell_name, hide_code=False)
        ctx.run_cell(cell_id)
        print({"status": "created_and_queued", "id": cell_id})
