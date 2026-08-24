import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_root_policy_training_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys


    def _cube666_root_policy_training_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _run = _Path("/marimo/storage/cube666_verified_d6_root_policy_v1")
        _work = _root / "work"
        _dataset = _root / "dataset_d6_2m_seed126669"
        _init = (
            _root
            / "model_direct_d6_unconditioned_seed127672"
            / "checkpoint.pt"
        )
        _output = _run / "model_root_depth6_seed127674"
        _report_path = _output / "report.json"
        _log_path = _output / "training.log"
        _process_path = _output / "process.json"
        _output.mkdir(parents=True, exist_ok=True)
        if _report_path.exists():
            _report = _json.loads(_report_path.read_text(encoding="utf-8"))
            return {
                "status": "complete",
                "before": _report["before"],
                "after": _report["after"],
                "elapsed_seconds": _report["elapsed_seconds"],
                "checkpoint": str(_output / "checkpoint.pt"),
            }
        if _process_path.exists():
            _process = _json.loads(_process_path.read_text(encoding="utf-8"))
            _pid = int(_process["pid"])
            try:
                _os.kill(_pid, 0)
                return {
                    "status": "running",
                    "pid": _pid,
                    "log": str(_log_path),
                    "checkpoint": str(_output / "checkpoint.pt"),
                }
            except ProcessLookupError:
                pass
        _env = dict(_os.environ)
        _env["PYTHONPATH"] = str(_work / "src") + _os.pathsep + _env.get(
            "PYTHONPATH", ""
        )
        _command = [
            _sys.executable,
            "-u",
            str(_run / "127_train_short_macro_action_policy.py"),
            "--dataset-dir", str(_dataset),
            "--init-checkpoint", str(_init),
            "--out-dir", str(_output),
            "--steps", "3000",
            "--batch-size", "2048",
            "--learning-rate", "1e-4",
            "--warmup-steps", "100",
            "--minimum-train-depth", "6",
            "--maximum-train-depth", "6",
            "--fixed-prefix", "0",
            "--evaluation-samples", "8192",
            "--label-smoothing", "0.01",
            "--seed", "127674",
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
        return {
            "status": "launched",
            "pid": _child.pid,
            "log": str(_log_path),
            "checkpoint": str(_output / "checkpoint.pt"),
        }


    cube666_verified_d6_root_policy_training = _cube666_root_policy_training_status()
    cube666_verified_d6_root_policy_training
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
