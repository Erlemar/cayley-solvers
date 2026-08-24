import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_certified_pair_audit_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys


    def _cube666_certified_pair_audit_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _audit = _Path("/marimo/storage/cube666_verified_d6_certified_pair_v1")
        _work = _root / "work"
        _dataset = _root / "dataset_d6_2m_seed126669"
        _model_a = _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
        _model_b = _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
        _report_path = _audit / "certified_pair_audit_8192_b512_v1.json"
        _log_path = _audit / "certified_pair_audit_8192_b512_v1.log"
        _process_path = _audit / "certified_pair_audit_8192_b512_v1.process.json"
        if _report_path.exists():
            _report = _json.loads(_report_path.read_text(encoding="utf-8"))
            return {
                "status": "complete",
                "report": str(_report_path),
                "samples": _report["samples"],
                "total_certified_pairs": _report["total_certified_pairs"],
                "parents_with_certified_pair_rate": _report[
                    "parents_with_certified_pair_rate"
                ],
                "constructive_action_proposal_recall": _report[
                    "constructive_action_proposal_recall"
                ],
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
                    "report": str(_report_path),
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
            str(_audit / "135_audit_short_macro_certified_pairs.py"),
            "--dataset-dir", str(_dataset),
            "--direct-action-checkpoint", str(_model_a),
            "--direct-action-checkpoint", str(_model_b),
            "--samples", "8192",
            "--proposal-branch", "512",
            "--batch-size", "128",
            "--minimum-depth", "2",
            "--maximum-depth", "6",
            "--margin", "0",
            "--out", str(_report_path),
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
            "report": str(_report_path),
        }


    cube666_verified_d6_certified_pair_audit = _cube666_certified_pair_audit_status()
    cube666_verified_d6_certified_pair_audit
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
