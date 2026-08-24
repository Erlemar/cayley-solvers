import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_frontier_completion_512_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys


    def _cube666_verified_d6_frontier_completion_512_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _critic_root = _Path(
            "/marimo/storage/cube666_verified_d6_certified_value_v1"
        )
        _eval = _Path("/marimo/storage/cube666_verified_d6_frontier_eval_v1")
        _worker = _Path("/marimo/storage/cube666_verified_d6_frontier_complete_v1")
        _run = _Path("/marimo/storage/cube666_verified_d6_frontier_returns_v1")
        _work = _root / "work"
        _output = _run / "frontier_completion_512roots_top4_b512"
        _report_path = _output / "report.json"
        _log_path = _output / "completion.log"
        _process_path = _output / "process.json"
        _frontier_path = _run / "frontier_train512_d2_top4.npz"
        _output.mkdir(parents=True, exist_ok=True)
        if _report_path.exists():
            return {
                "status": "complete",
                **_json.loads(_report_path.read_text(encoding="utf-8")),
            }
        if _process_path.exists():
            _process = _json.loads(_process_path.read_text(encoding="utf-8"))
            _pid = int(_process["pid"])
            try:
                _os.kill(_pid, 0)
                _attempt_path = _output / "attempts.jsonl"
                _lines = (
                    _attempt_path.read_text(encoding="utf-8").splitlines()
                    if _attempt_path.exists()
                    else []
                )
                _solved = sum(
                    bool(_json.loads(_line).get("solved"))
                    for _line in _lines
                    if _line.strip()
                )
                return {
                    "status": "running",
                    "pid": _pid,
                    "attempted": len(_lines),
                    "solved": _solved,
                    "log": str(_log_path),
                }
            except ProcessLookupError:
                pass
        if not _frontier_path.exists():
            return {"status": "waiting_for_frontier", "frontier": str(_frontier_path)}
        _env = dict(_os.environ)
        _env["PYTHONPATH"] = str(_work / "src") + _os.pathsep + _env.get(
            "PYTHONPATH", ""
        )
        _command = [
            _sys.executable,
            "-u",
            str(_worker / "140_complete_short_macro_frontiers.py"),
            "--eval-script", str(_eval / "78_eval_short_macro_value_beam.py"),
            "--dataset-dir", str(_root / "dataset_d6_2m_seed126669"),
            "--frontier", str(_frontier_path),
            "--direct-action-checkpoint", str(
                _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
            ),
            "--direct-action-checkpoint", str(
                _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
            ),
            "--critic", str(
                _critic_root / "model_interval_return4_seed136667_v3" / "checkpoint.pt"
            ),
            "--out-dir", str(_output),
            "--beam", "512",
            "--maximum-completion-depth", "4",
            "--value-prebeam-multiplier", "32",
            "--checkpoint-every", "32",
            "--seed", "140670",
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


    cube666_verified_d6_frontier_completion_512 = (
        _cube666_verified_d6_frontier_completion_512_status()
    )
    cube666_verified_d6_frontier_completion_512
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
