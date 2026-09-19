import textwrap

import marimo._code_mode as cm


cell_name = "cube666_verified_d6_root2048_prebeam128_oracle_audit_pro6000"
cell_code = textwrap.dedent(
    r"""
    from pathlib import Path as _Path
    import json as _json
    import os as _os
    import subprocess as _subprocess
    import sys as _sys


    def _cube666_root2048_prebeam128_oracle_audit_status():
        _root = _Path("/marimo/storage/cube666_verified_d6_v1")
        _run = _Path("/marimo/storage/cube666_verified_d6_certified_value_v1")
        _work = _root / "work"
        _dataset = _root / "dataset_d6_2m_seed126669"
        _model_a = _root / "model_direct_d6_unconditioned_seed127672" / "checkpoint.pt"
        _model_b = _root / "model_direct_d6_depthconditioned_seed127673" / "checkpoint.pt"
        _critic = (
            _run
            / "model_interval_return4_seed136667_v3"
            / "checkpoint.pt"
        )
        _patch = _Path("/marimo/storage/cube666_verified_d6_value_prebeam_patch_v2")
        _report_path = _run / "eval_root2048_prebeam128_gate32_oracle_audit_v1.json"
        _log_path = _run / "eval_root2048_prebeam128_gate32_oracle_audit_v1.log"
        _process_path = _run / "eval_root2048_prebeam128_gate32_oracle_audit_v1.process.json"
        if _report_path.exists():
            _report = _json.loads(_report_path.read_text(encoding="utf-8"))
            return {
                "status": "complete",
                "solved": _report["solved"],
                "cases": len(_report["rows"]),
                "replay_verified": sum(
                    int(_row["replay_verified"]) for _row in _report["rows"]
                ),
                "report": str(_report_path),
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
            str(_patch / "78_eval_short_macro_value_beam.py"),
            "--factorized-value-checkpoint", str(_critic),
            "--direct-action-checkpoint", str(_model_a),
            "--direct-action-checkpoint", str(_model_b),
            "--dataset-dir", str(_dataset),
            "--indices-file", str(_root / "verified_d6_gate32_indices.txt"),
            "--beam", "4096",
            "--branch", "0",
            "--direct-branch", "512",
            "--direct-root-branch", "2048",
            "--direct-cost-stratified-branch", "128",
            "--direct-cost-maximum", "6",
            "--direct-cost-stratified-depths", "3",
            "--direct-score-mode", "log_probability",
            "--maximum-depth", "6",
            "--path-cost-weight", "0.02",
            "--value-weight", "0.25",
            "--value-prebeam-multiplier", "128",
            "--policy-nll-weight", "1",
            "--exact-root-two-macro",
            "--one-macro-endgame",
            "--continue-after-solution",
            "--audit-oracle-retention",
            "--inference-batch-size", "16384",
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


    cube666_verified_d6_root2048_prebeam128_oracle_audit = (
        _cube666_root2048_prebeam128_oracle_audit_status()
    )
    cube666_verified_d6_root2048_prebeam128_oracle_audit
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
