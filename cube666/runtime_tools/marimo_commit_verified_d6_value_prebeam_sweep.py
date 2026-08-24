import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_root2048_prebeam128_gate_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys


    def _cube666_root2048_prebeam128_gate_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _run = _Path("/marimo/storage/cube666_verified_d6_certified_value_v1")
        _patch = _Path("/marimo/storage/cube666_verified_d6_value_prebeam_patch_v2")
        _sweep = _Path("/marimo/storage/cube666_verified_d6_value_prebeam_sweep_v2")
        _work = _root / "work"
        _output = _sweep / "return4_root2048_prebeam128_gate32"
        _summary_path = _output / "summary.json"
        _log_path = _output / "sweep.log"
        _process_path = _output / "process.json"
        _output.mkdir(parents=True, exist_ok=True)
        if _summary_path.exists():
            _summary = _json.loads(_summary_path.read_text(encoding="utf-8"))
            _complete = len(_summary["variants"]) == 1
            if _complete:
                return {
                    "status": "complete",
                    "variants": _summary["variants"],
                    "summary": str(_summary_path),
                }
        if _process_path.exists():
            _process = _json.loads(_process_path.read_text(encoding="utf-8"))
            _pid = int(_process["pid"])
            try:
                _os.kill(_pid, 0)
                return {
                    "status": "running",
                    "pid": _pid,
                    "partial": (
                        _json.loads(_summary_path.read_text(encoding="utf-8"))
                        if _summary_path.exists()
                        else None
                    ),
                    "log": str(_log_path),
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
            str(_sweep / "137_run_short_macro_value_prebeam_sweep.py"),
            "--eval-script", str(_patch / "78_eval_short_macro_value_beam.py"),
            "--dataset-dir", str(_root / "dataset_d6_2m_seed126669"),
            "--direct-action-checkpoint", str(
                _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
            ),
            "--direct-action-checkpoint", str(
                _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
            ),
            "--critic", str(
                _run / "model_interval_return4_seed136667_v3" / "checkpoint.pt"
            ),
            "--indices-file", str(_root / "verified_d6_gate32_indices.txt"),
            "--out-dir", str(_output),
            "--weights", "0.25",
            "--beam", "4096",
            "--direct-root-branch", "2048",
            "--value-prebeam-multiplier", "128",
            "--timeout-seconds", "300",
            "--wait-for", str(
                _run / "eval_certified_return4_gate32_b4096_f1_v1.json"
            ),
            "--wait-timeout-seconds", "900",
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
            "summary": str(_summary_path),
        }


    cube666_verified_d6_root2048_prebeam128_gate = (
        _cube666_root2048_prebeam128_gate_status()
    )
    cube666_verified_d6_root2048_prebeam128_gate
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
