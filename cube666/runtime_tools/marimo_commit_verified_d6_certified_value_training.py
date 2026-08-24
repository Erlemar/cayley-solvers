import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_certified_value_training_v3_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys


    def _cube666_certified_value_training_v3_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _run = _Path("/marimo/storage/cube666_verified_d6_certified_value_v1")
        _script = _Path(
            "/marimo/storage/cube666_verified_d6_certified_value_patch_v3/"
            "136_train_short_macro_certified_interval_value.py"
        )
        _work = _root / "work"
        _dataset = _root / "dataset_d6_2m_seed126669"
        _model_a = _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
        _model_b = _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
        _init = (
            _run
            / "model_interval256x4_global768x5_seed136666_v2"
            / "checkpoint.pt"
        )
        _output = _run / "model_interval_return4_seed136667_v3"
        _report_path = _output / "report.json"
        _log_path = _output / "training.log"
        _process_path = _output / "process.json"
        _output.mkdir(parents=True, exist_ok=True)
        if _report_path.exists():
            _report = _json.loads(_report_path.read_text(encoding="utf-8"))
            return {
                "status": "complete",
                "checkpoint": str(_output / "checkpoint.pt"),
                "before": _report["before"],
                "after": _report["after"],
                "certified_pairs_seen": _report["certified_pairs_seen"],
                "elapsed_seconds": _report["elapsed_seconds"],
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
            str(_script),
            "--dataset-dir", str(_dataset),
            "--direct-action-checkpoint", str(_model_a),
            "--direct-action-checkpoint", str(_model_b),
            "--out-dir", str(_output),
            "--init-checkpoint", str(_init),
            "--steps", "2000",
            "--batch-size", "64",
            "--proposal-branch", "128",
            "--hard-pairs", "32",
            "--eval-samples", "2048",
            "--learning-rate", "1e-4",
            "--ranking-margin", "1",
            "--interval-weight", "1",
            "--ranking-weight", "4",
            "--trace-return-weight", "4",
            "--minimum-depth", "2",
            "--maximum-depth", "6",
            "--local-dim", "256",
            "--local-blocks", "4",
            "--global-dim", "768",
            "--global-blocks", "5",
            "--seed", "136667",
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


    cube666_verified_d6_certified_value_training_v3 = (
        _cube666_certified_value_training_v3_status()
    )
    cube666_verified_d6_certified_value_training_v3
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
