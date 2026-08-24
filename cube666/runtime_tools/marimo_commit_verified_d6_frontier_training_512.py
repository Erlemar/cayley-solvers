import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_frontier_return_training_512_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys


    def _cube666_verified_d6_frontier_return_training_512_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _critic_root = _Path(
            "/marimo/storage/cube666_verified_d6_certified_value_v1"
        )
        _returns = _Path(
            "/marimo/storage/cube666_verified_d6_frontier_returns_v1"
        )
        _worker = _Path("/marimo/storage/cube666_verified_d6_frontier_train_v1")
        _run = _Path("/marimo/storage/cube666_verified_d6_frontier_model_v1")
        _work = _root / "work"
        _completion = (
            _returns / "frontier_completion_512roots_top4_b512"
            / "solved_completions.npz"
        )
        _output = _run / "model_frontier512_seed141666"
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
        if not _completion.exists():
            return {"status": "waiting_for_completions", "path": str(_completion)}
        _env = dict(_os.environ)
        _env["PYTHONPATH"] = str(_work / "src") + _os.pathsep + _env.get(
            "PYTHONPATH", ""
        )
        _command = [
            _sys.executable,
            "-u",
            str(_worker / "141_finetune_short_macro_frontier_returns.py"),
            "--dataset-dir", str(_root / "dataset_d6_2m_seed126669"),
            "--solved-completions", str(_completion),
            "--init-checkpoint", str(
                _critic_root / "model_interval_return4_seed136667_v3" / "checkpoint.pt"
            ),
            "--out-dir", str(_output),
            "--steps", "1500",
            "--return-batch-size", "256",
            "--pair-batch-size", "128",
            "--anchor-batch-size", "256",
            "--anchor-pool-size", "8192",
            "--learning-rate", "2e-5",
            "--return-weight", "4",
            "--ranking-weight", "2",
            "--anchor-weight", "0.5",
            "--heldout-root-fraction", "0.2",
            "--seed", "141666",
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


    cube666_verified_d6_frontier_return_training_512 = (
        _cube666_verified_d6_frontier_return_training_512_status()
    )
    cube666_verified_d6_frontier_return_training_512
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
